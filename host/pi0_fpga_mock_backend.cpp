#include "pi0_fpga_mock_backend.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "../scripts/llama3_8b_runtime_host_regs.h"

namespace pi0 {

void MockDeviceMemory::write(uint64_t address, const void* source, size_t bytes) {
    const uint8_t* src = static_cast<const uint8_t*>(source);
    while (bytes > 0) {
        const uint64_t page = address / kPageBytes;
        const uint64_t offset = address % kPageBytes;
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(bytes, kPageBytes - offset));
        auto it = pages_.find(page);
        if (it == pages_.end()) {
            it = pages_.emplace(page, std::vector<uint8_t>(kPageBytes, 0)).first;
        }
        std::memcpy(it->second.data() + offset, src, chunk);
        address += chunk;
        src += chunk;
        bytes -= chunk;
    }
}

void MockDeviceMemory::read(uint64_t address, void* destination, size_t bytes) const {
    uint8_t* dst = static_cast<uint8_t*>(destination);
    while (bytes > 0) {
        const uint64_t page = address / kPageBytes;
        const uint64_t offset = address % kPageBytes;
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(bytes, kPageBytes - offset));
        const auto it = pages_.find(page);
        if (it == pages_.end()) {
            std::memset(dst, 0, chunk);
        } else {
            std::memcpy(dst, it->second.data() + offset, chunk);
        }
        address += chunk;
        dst += chunk;
        bytes -= chunk;
    }
}

// ---------------------------------------------------------------------------

class MockBackend::Impl : public RegisterAccess, public BulkTransfer {
public:
    explicit Impl(const MockConfig& config) : config_(config) {
        // RESET_ENABLE reads back low until the host sets it, matching the
        // active-high enable in top_ctrl.sv:724-725.
        registers_[LLAMA3_RT_REG_TC_CONTROL] = 0;
        registers_[LLAMA3_RT_REG_GENERIC_ABI_STATUS] =
            LLAMA3_RT_GENERIC_ABI_STATUS_VALUE;
        registers_[LLAMA3_RT_REG_PRECISION_STATUS] =
            LLAMA3_RT_PRECISION_COMPILED_INT8;
    }

    // -- RegisterAccess ----------------------------------------------------

    void write32(uint64_t offset, uint32_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offset == LLAMA3_RT_REG_GENERIC_CTRL) {
            write_generic_ctrl(value);
            return;
        }
        if (offset == LLAMA3_RT_REG_CTRL) {
            write_paged_ctrl(value);
            return;
        }
        registers_[offset] = value;
    }

    uint32_t read32(uint64_t offset) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offset == LLAMA3_RT_REG_GENERIC_STATUS) {
            settle_generic();
            return generic_status();
        }
        if (offset == LLAMA3_RT_REG_STATUS) {
            settle_paged();
            return paged_status();
        }
        const auto it = registers_.find(offset);
        return it == registers_.end() ? 0u : it->second;
    }

    // -- BulkTransfer ------------------------------------------------------

    void h2d(const void* source, size_t bytes, uint64_t device_address) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            memory_.write(device_address, source, bytes);
        }
        account_h2d(bytes);
    }

    void d2h(void* destination, size_t bytes, uint64_t device_address) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            memory_.read(device_address, destination, bytes);
        }
        account_d2h(bytes);
    }

    // -- observation -------------------------------------------------------

    uint64_t accepted_launches() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepted_launches_;
    }

    uint64_t swallowed_starts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return swallowed_starts_;
    }

    MockDeviceMemory& memory() { return memory_; }

    uint64_t launched_package_base() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return launched_package_base_;
    }

    const MockConfig& config() const { return config_; }

private:
    uint32_t register_or_zero(uint64_t offset) const {
        const auto it = registers_.find(offset);
        return it == registers_.end() ? 0u : it->second;
    }

    // -- generic command path ---------------------------------------------

    void write_generic_ctrl(uint32_t value) {
        // CLEAR is evaluated before START in the RTL, so a word carrying both
        // clears and does not start.
        if ((value & LLAMA3_RT_GENERIC_CTRL_CLEAR) != 0u) {
            if (!config_.clear_is_ineffective) {
                generic_busy_ = false;
                generic_done_ = false;
                generic_fault_ = false;
                generic_result_ = false;
                generic_commands_ = 0;
                generic_dma_read_ = 0;
                generic_dma_write_ = 0;
            }
            return;
        }
        if ((value & LLAMA3_RT_GENERIC_CTRL_START) == 0u) {
            return;
        }
        // start_accept = i_start && !busy && !done && !fault
        if (generic_busy_ || generic_done_ || generic_fault_) {
            // Silently swallowed. This is the whole point of the model: no
            // error is raised, and the stale DONE/RESULT/counters stay put.
            ++swallowed_starts_;
            return;
        }
        generic_busy_ = true;
        generic_done_ = false;
        generic_fault_ = false;
        generic_result_ = false;
        ++accepted_launches_;
        ++run_index_;
        // Latch, do not interpret. Recording which package base the host armed
        // lets a test prove the engine launched the package it meant to,
        // without the mock becoming a second implementation of the RTL.
        launched_package_base_ =
            static_cast<uint64_t>(register_or_zero(LLAMA3_RT_REG_PACKAGE_BASE_LO)) |
            (static_cast<uint64_t>(
                 register_or_zero(LLAMA3_RT_REG_PACKAGE_BASE_HI) &
                 LLAMA3_RT_GENERIC_PACKAGE_HI_MASK)
             << 32);
        launch_time_ = std::chrono::steady_clock::now();
    }

    void settle_generic() {
        if (!generic_busy_ || config_.never_completes) {
            return;
        }
        if (std::chrono::steady_clock::now() - launch_time_ <
            config_.execute_latency) {
            return;
        }
        generic_busy_ = false;
        if (config_.fault_on_run != 0 && run_index_ == config_.fault_on_run) {
            generic_fault_ = true;
            registers_[LLAMA3_RT_REG_GENERIC_FAULT_CODE] = config_.fault_code;
            registers_[LLAMA3_RT_REG_GENERIC_FAULT_PC] = 0x40;
            registers_[LLAMA3_RT_REG_GENERIC_FAULT_ENGINE] = 1;
            registers_[LLAMA3_RT_REG_GENERIC_FAULT_DESC] = 2;
            return;
        }
        generic_done_ = true;
        generic_result_ = true;
        generic_commands_ = config_.commands_per_run * run_index_;
        generic_dma_read_ = 64 * run_index_;
        generic_dma_write_ = 32 * run_index_;
        registers_[LLAMA3_RT_REG_GENERIC_COMMANDS_LO] =
            static_cast<uint32_t>(generic_commands_);
        registers_[LLAMA3_RT_REG_GENERIC_COMMANDS_HI] =
            static_cast<uint32_t>(generic_commands_ >> 32);
        registers_[LLAMA3_RT_REG_GENERIC_DMA_READ_LO] =
            static_cast<uint32_t>(generic_dma_read_);
        registers_[LLAMA3_RT_REG_GENERIC_DMA_READ_HI] =
            static_cast<uint32_t>(generic_dma_read_ >> 32);
        registers_[LLAMA3_RT_REG_GENERIC_DMA_WRITE_LO] =
            static_cast<uint32_t>(generic_dma_write_);
        registers_[LLAMA3_RT_REG_GENERIC_DMA_WRITE_HI] =
            static_cast<uint32_t>(generic_dma_write_ >> 32);
    }

    uint32_t generic_status() const {
        uint32_t status = 0;
        if (generic_busy_)   status |= LLAMA3_RT_GENERIC_STATUS_BUSY;
        if (generic_done_)   status |= LLAMA3_RT_GENERIC_STATUS_DONE;
        if (generic_fault_)  status |= LLAMA3_RT_GENERIC_STATUS_FAULT;
        if (generic_result_) status |= LLAMA3_RT_GENERIC_STATUS_RESULT;
        return status;
    }

    // -- paged UOP path ----------------------------------------------------

    void write_paged_ctrl(uint32_t value) {
        const uint32_t previous = paged_ctrl_;
        paged_ctrl_ = value;
        const uint32_t rising = value & ~previous;

        if ((rising & LLAMA3_RT_CTRL_CLEAR) != 0u) {
            if (!config_.clear_is_ineffective) {
                paged_running_ = false;
                paged_done_ = false;
            }
            return;
        }
        if ((rising & LLAMA3_RT_CTRL_START) != 0u) {
            // START is taken only while not already running; otherwise it is
            // dropped, exactly as in llama3_8b_runtime_ctrl.sv:457.
            if (paged_running_) {
                ++swallowed_starts_;
                return;
            }
            paged_running_ = true;
            paged_done_ = false;
            ++accepted_launches_;
            paged_launch_time_ = std::chrono::steady_clock::now();
        }
    }

    void settle_paged() {
        if (!paged_running_ || config_.never_completes) {
            return;
        }
        if (std::chrono::steady_clock::now() - paged_launch_time_ <
            config_.execute_latency) {
            return;
        }
        paged_running_ = false;
        paged_done_ = true;
    }

    uint32_t paged_status() const {
        uint32_t status = 0;
        if (paged_running_) status |= LLAMA3_RT_STATUS_RUNNING;
        if (paged_done_)    status |= LLAMA3_RT_STATUS_DONE;
        // Bits 2/3/4 mirror the CTRL level bits back to the host.
        if ((paged_ctrl_ & LLAMA3_RT_CTRL_USE_UOP_RAM) != 0u) {
            status |= LLAMA3_RT_STATUS_USE_UOP;
        }
        if ((paged_ctrl_ & LLAMA3_RT_CTRL_AUTO_ADV) != 0u) {
            status |= LLAMA3_RT_STATUS_AUTO_ADV;
        }
        if ((paged_ctrl_ & LLAMA3_RT_CTRL_MODE_DECODE) != 0u) {
            status |= LLAMA3_RT_STATUS_DECODE;
        }
        return status;
    }

    MockConfig config_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, uint32_t> registers_;
    MockDeviceMemory memory_;

    bool generic_busy_ = false;
    bool generic_done_ = false;
    bool generic_fault_ = false;
    bool generic_result_ = false;
    uint64_t generic_commands_ = 0;
    uint64_t generic_dma_read_ = 0;
    uint64_t generic_dma_write_ = 0;
    std::chrono::steady_clock::time_point launch_time_{};
    uint64_t launched_package_base_ = 0;

    uint32_t paged_ctrl_ = 0;
    bool paged_running_ = false;
    bool paged_done_ = false;
    std::chrono::steady_clock::time_point paged_launch_time_{};

    uint64_t accepted_launches_ = 0;
    uint64_t swallowed_starts_ = 0;
    uint32_t run_index_ = 0;
};

MockBackend::MockBackend(const MockConfig& config)
    : impl_(new Impl(config)) {}

MockBackend::~MockBackend() = default;

RegisterAccess& MockBackend::registers() { return *impl_; }

BulkTransfer& MockBackend::bulk() { return *impl_; }

std::string MockBackend::description() const {
    std::ostringstream out;
    out << "mock backend (no hardware), execute_latency="
        << impl_->config().execute_latency.count() << "us";
    if (impl_->config().clear_is_ineffective) out << " clear_is_ineffective";
    if (impl_->config().never_completes)      out << " never_completes";
    if (impl_->config().fault_on_run != 0) {
        out << " fault_on_run=" << impl_->config().fault_on_run;
    }
    return out.str();
}

uint64_t MockBackend::accepted_launches() const {
    return impl_->accepted_launches();
}

uint64_t MockBackend::swallowed_starts() const {
    return impl_->swallowed_starts();
}

uint64_t MockBackend::launched_package_base() const {
    return impl_->launched_package_base();
}

MockDeviceMemory& MockBackend::memory() { return impl_->memory(); }

std::unique_ptr<Backend> make_mock_backend(const MockConfig& config) {
    return std::unique_ptr<Backend>(new MockBackend(config));
}

}  // namespace pi0

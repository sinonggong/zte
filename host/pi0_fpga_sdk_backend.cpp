#include "pi0_fpga_sdk_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include "Achronix_DMA.h"
#include "Achronix_IP/Achronix_DBI_interface.h"
#include "Achronix_IP/Achronix_IP_config.h"
#include "Achronix_IP_block.h"
#include "Achronix_MMIO.h"
#include "Achronix_device.h"
#include "Achronix_register_maps/Achronix_DMA_registers.h"

#include "pi0_fpga_mock_backend.h"

namespace pi0 {
namespace {

// Hard ceiling on a single-buffer DMA transfer.
//
// The driver backs every DMA buffer with dma_alloc_coherent()
// (SDK/drivers/acxpcie/ioctl.c), which is physically contiguous and therefore
// bounded by the kernel's MAX_ORDER -- 4 MiB on a stock x86-64 kernel. The
// SDK's own DMA_example enforces the same number and says why:
//
//     #define BUFFER_MAX_SIZE 0x400000UL
//     // DMA engine does not support transfers > one VM Page.
//     // Will require linked-list support (unimplemented)
//
// This is not a tuning knob. A request above it fails to allocate. Every
// transfer is split to respect it, on both the staged and fallback paths --
// the previous code handed acx_dma_start_xfer_h2d chunks of up to 256 MiB,
// which would have failed on the first transfer against real hardware.
constexpr uint64_t kMaxSingleTransferBytes = 0x400000ull;  // 4 MiB

void check_sdk(ACX_SDK_STATUS status, const char* what) {
    if (status != ACX_SDK_STATUS_OK) {
        std::ostringstream error;
        error << what << " failed, sdk_status=" << static_cast<int>(status);
        throw Error(ErrorKind::Device, error.str());
    }
}

// Releases a DMA transaction however the scope exits. The bring-up code left
// acx_dma_cleanup_tactn after the status check, so every failed transfer
// leaked its transaction -- on a resident engine that runs for hours, a leak
// per failure is a real one.
class ScopedTransaction {
public:
    explicit ScopedTransaction(ACX_DMA_transaction* transaction)
        : transaction_(transaction) {}

    ~ScopedTransaction() {
        if (transaction_ != nullptr) {
            acx_dma_cleanup_tactn(transaction_);
        }
    }

    ScopedTransaction(const ScopedTransaction&) = delete;
    ScopedTransaction& operator=(const ScopedTransaction&) = delete;

    ACX_DMA_transaction* get() const { return transaction_; }

private:
    ACX_DMA_transaction* transaction_ = nullptr;
};

class SdkRegisters : public RegisterAccess {
public:
    SdkRegisters(ACX_DEV_PCIe_device* device,
                 unsigned bar_index,
                 uint64_t bar_offset,
                 uint64_t bar_length) {
        ACX_MMIO_attributes attributes = {};
        attributes.name = "pi0_runtime_ctrl";
        attributes.trace_mask = 0;
        check_sdk(
            acx_mmio_open_hand_constant(
                device, bar_index, bar_offset, bar_length, &attributes, &handle_),
            "acx_mmio_open_hand_constant");
    }

    ~SdkRegisters() override {
        if (handle_ != nullptr) {
            acx_mmio_close_hand(handle_);
            handle_ = nullptr;
        }
    }

    void write32(uint64_t offset, uint32_t value) override {
        check_sdk(acx_mmio_write_uint32(handle_, offset, value),
                  "acx_mmio_write_uint32");
    }

    uint32_t read32(uint64_t offset) override {
        uint32_t value = 0;
        check_sdk(acx_mmio_read_uint32(handle_, offset, &value),
                  "acx_mmio_read_uint32");
        return value;
    }

private:
    ACX_MMIO_handle handle_ = nullptr;
};

// Bulk movement through a BAR window instead of the DMA engine: one 32-bit
// PCIe transaction per word. See SdkBackendConfig::bulk_mode for why this
// exists. Byte-granular at the edges via read-modify-write, because plan
// payloads are not required to start or end on a word boundary.
class PioBulk : public BulkTransfer {
public:
    PioBulk(ACX_DEV_PCIe_device* device,
            unsigned bar,
            uint64_t noc_base,
            uint64_t window_bytes)
        : bar_(bar), noc_base_(noc_base), window_(window_bytes) {
        ACX_MMIO_attributes attributes = {};
        attributes.name = "pi0_pio_bulk";
        attributes.trace_mask = 0;
        check_sdk(acx_mmio_open_hand_constant(
                      device, bar, 0, window_bytes, &attributes, &handle_),
                  "acx_mmio_open_hand_constant(pio)");
    }

    ~PioBulk() override {
        if (handle_ != nullptr) {
            acx_mmio_close_hand(handle_);
            handle_ = nullptr;
        }
    }

    void h2d(const void* source, size_t bytes, uint64_t device_address) override {
        copy(true, const_cast<void*>(source), bytes, device_address);
        account_h2d(bytes);
    }

    void d2h(void* destination, size_t bytes, uint64_t device_address) override {
        copy(false, destination, bytes, device_address);
        account_d2h(bytes);
    }

    unsigned bar() const { return bar_; }
    uint64_t noc_base() const { return noc_base_; }
    uint64_t window() const { return window_; }

private:
    uint32_t read_word(uint64_t offset) {
        uint32_t value = 0;
        check_sdk(acx_mmio_read_uint32(handle_, offset, &value),
                  "acx_mmio_read_uint32(pio)");
        return value;
    }

    void write_word(uint64_t offset, uint32_t value) {
        check_sdk(acx_mmio_write_uint32(handle_, offset, value),
                  "acx_mmio_write_uint32(pio)");
    }

    void copy(bool to_device, void* host, size_t bytes, uint64_t device_address) {
        if (bytes == 0) {
            return;
        }
        if (device_address < noc_base_ ||
            device_address - noc_base_ + bytes > window_) {
            std::ostringstream out;
            out << "PIO transfer outside the BAR" << bar_ << " window: address=0x"
                << std::hex << device_address << " bytes=0x" << bytes
                << " window=[0x" << noc_base_ << ",0x" << (noc_base_ + window_)
                << ")" << std::dec
                << " -- rebase the plan (--gddr-base) or use the DMA backend";
            throw Error(ErrorKind::Config, out.str());
        }

        uint64_t offset = device_address - noc_base_;
        uint8_t* cursor = static_cast<uint8_t*>(host);
        size_t left = bytes;

        // Head: partial first word.
        const uint64_t head = offset & 3u;
        if (head != 0u) {
            const uint64_t word_offset = offset - head;
            const size_t n = std::min<size_t>(4u - head, left);
            uint32_t word = read_word(word_offset);
            uint8_t lanes[4];
            std::memcpy(lanes, &word, 4);
            if (to_device) {
                std::memcpy(lanes + head, cursor, n);
                std::memcpy(&word, lanes, 4);
                write_word(word_offset, word);
            } else {
                std::memcpy(cursor, lanes + head, n);
            }
            offset += n;
            cursor += n;
            left -= n;
        }

        // Body: whole words.
        while (left >= 4u) {
            if (to_device) {
                uint32_t word = 0;
                std::memcpy(&word, cursor, 4);
                write_word(offset, word);
            } else {
                const uint32_t word = read_word(offset);
                std::memcpy(cursor, &word, 4);
            }
            offset += 4;
            cursor += 4;
            left -= 4;
        }

        // Tail: partial last word.
        if (left != 0u) {
            uint32_t word = read_word(offset);
            uint8_t lanes[4];
            std::memcpy(lanes, &word, 4);
            if (to_device) {
                std::memcpy(lanes, cursor, left);
                std::memcpy(&word, lanes, 4);
                write_word(offset, word);
            } else {
                std::memcpy(cursor, lanes, left);
            }
        }
    }

    ACX_MMIO_handle handle_ = nullptr;
    unsigned bar_ = 1;
    uint64_t noc_base_ = 0;
    uint64_t window_ = 0;
};

// NOT thread-safe, and deliberately so: one staging buffer and one transaction
// are the whole point. The allocate-per-transfer path it replaces was
// incidentally safe to call concurrently, so this is a real contract change.
// It holds because every caller is serialised -- Pi0FpgaEngine runs all device
// access on its worker thread, and load_static() only touches the device after
// waiting for that worker to go idle. The CLI is single-threaded.
class SdkBulk : public BulkTransfer {
public:
    SdkBulk(ACX_DEV_PCIe_device* device,
            ACX_DMA_engine_context* engine,
            unsigned channel,
            uint32_t timeout_ms,
            uint64_t staging_bytes)
        : device_(device), engine_(engine), channel_(channel),
          timeout_ms_(timeout_ms) {
        if (staging_bytes == 0) {
            return;  // allocate-per-transfer fallback
        }
        if (staging_bytes > kMaxSingleTransferBytes) {
            throw Error(ErrorKind::Config,
                        "dma_staging_bytes=" + std::to_string(staging_bytes) +
                            " exceeds the " +
                            std::to_string(kMaxSingleTransferBytes) +
                            " byte single-transfer limit; the driver allocates "
                            "physically contiguous memory and cannot satisfy it");
        }
        staging_ = acx_dma_alloc_buf(device, staging_bytes);
        if (staging_ == nullptr) {
            throw Error(ErrorKind::Device,
                        "failed to allocate a " + std::to_string(staging_bytes) +
                            " byte pinned DMA staging buffer");
        }
        transaction_ = acx_dma_alloc_tactn(engine_);
        if (transaction_ == nullptr) {
            acx_dma_free_buf(staging_);
            staging_ = nullptr;
            throw Error(ErrorKind::Device, "failed to allocate a DMA transaction");
        }
    }

    ~SdkBulk() override {
        if (transaction_ != nullptr) {
            // _no_payload: the staging buffer is ours, not the transaction's,
            // and is freed separately below.
            acx_dma_cleanup_tactn_no_payload(transaction_);
        }
        if (staging_ != nullptr) {
            acx_dma_free_buf(staging_);
        }
    }

    void h2d(const void* source, size_t bytes, uint64_t device_address) override {
        const uint8_t* cursor = static_cast<const uint8_t*>(source);
        for_each_dma_chunk(bytes, chunk_limit(), [&](uint64_t off, uint64_t n) {
            if (staging_ != nullptr) {
                staged(ACX_DMA_HOST_TO_DEVICE, const_cast<uint8_t*>(cursor) + off,
                       n, device_address + off);
            } else {
                h2d_per_transfer(cursor + off, static_cast<size_t>(n),
                                 device_address + off);
            }
        });
        account_h2d(bytes);
    }

    void d2h(void* destination, size_t bytes, uint64_t device_address) override {
        uint8_t* cursor = static_cast<uint8_t*>(destination);
        for_each_dma_chunk(bytes, chunk_limit(), [&](uint64_t off, uint64_t n) {
            if (staging_ != nullptr) {
                staged(ACX_DMA_DEVICE_TO_HOST, cursor + off, n,
                       device_address + off);
            } else {
                d2h_per_transfer(cursor + off, static_cast<size_t>(n),
                                 device_address + off);
            }
        });
        account_d2h(bytes);
    }

private:
    uint64_t chunk_limit() const {
        return staging_ != nullptr ? staging_->size_in_bytes
                                   : kMaxSingleTransferBytes;
    }

    // One chunk, already known to fit the staging buffer and the 4 MiB limit.
    // The pinning happened once, at construction.
    void staged(ACX_DMA_DIRECTION direction,
                void* host,
                uint64_t chunk,
                uint64_t chunk_address) {
        {
            if (direction == ACX_DMA_HOST_TO_DEVICE) {
                std::memcpy(staging_->data, host, static_cast<size_t>(chunk));
            }

            // config_xfer before EVERY start, even when the shape is
            // identical to the last transfer.
            //
            // Achronix_DMA.h says a transaction "only needs to be called once
            // to set the initial configuration for a channel" if its shape does
            // not change. Do not believe it. config_xfer programs the channel's
            // SAR, DAR and TRANSFER_SIZE registers, and start_xfer does nothing
            // but ring the doorbell. TRANSFER_SIZE is consumed by the hardware:
            // completion is defined as channel-stopped AND TRANSFER_SIZE == 0
            // (Achronix_DMA.c, dma_update_status_internal; DesignWare PCIe
            // controller section 8.3.4).
            //
            // So a doorbell rung without reconfiguring runs a ZERO-byte
            // transfer and then reads back as ACX_DMA_XFER_COMPLETE. Every
            // transfer after the first would silently move nothing and report
            // success. The SDK's own DMA_example calls config_xfer before every
            // start for exactly this reason.
            //
            // The win here was never skipping this -- it is a handful of
            // register writes. The win is not pinning a host buffer per call.
            check_sdk(
                acx_dma_build_buf_tactn(
                    transaction_, staging_, static_cast<uint8_t>(channel_),
                    direction, chunk_address, /*buffer_offset=*/0, chunk),
                "acx_dma_build_buf_tactn");
            check_sdk(acx_dma_config_xfer(transaction_), "acx_dma_config_xfer");

            check_sdk(acx_dma_start_xfer(transaction_), "acx_dma_start_xfer");
            // Bounded wait + bounded halt, same as the per-transfer path. The
            // first version of this path called the SDK's acx_dma_halt_tactn
            // here, which is the unbounded spin that wait_or_halt exists to
            // avoid -- and this staged path is the DEFAULT, so the bound has
            // to be here or it is nowhere.
            wait_or_halt(transaction_, "acx_dma_wait(staged)");

            if (direction == ACX_DMA_DEVICE_TO_HOST) {
                std::memcpy(host, staging_->data, static_cast<size_t>(chunk));
            }
        }
    }

    // A timeout leaves the channel running; halt it before reporting, or the
    // next transfer configures a busy channel. The SDK's DMA_example does this.
    // Reads one DMA controller register through the DBI, for diagnostics.
    // Returns 0xffffffff if the read itself fails, which is itself a finding:
    // it means the DBI window is not reaching the controller.
    uint32_t dbi(uint64_t address) {
        uint32_t value = 0xffffffffu;
        if (acx_mmio_read_dbi(device_, address, &value) != ACX_SDK_STATUS_OK) {
            return 0xffffffffu;
        }
        return value;
    }

    std::string dma_channel_report(ACX_DMA_CH_OPERATION op, unsigned channel) {
        std::ostringstream out;
        out << std::hex << std::showbase
            << "ctrl1=" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_CH_CONTROL1_OFF, op, channel))
            << " xfer_size=" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_TRANSFER_SIZE_OFF, op, channel))
            << " sar=" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_SAR_HIGH_OFF, op, channel))
            << ":" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_SAR_LOW_OFF, op, channel))
            << " dar=" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_DAR_HIGH_OFF, op, channel))
            << ":" << dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_DAR_LOW_OFF, op, channel))
            << " engine_en=" << dbi(acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ENGINE_EN_OFF, op))
            << " int_status=" << dbi(acx_dma_ctrl_reg_addr(ACX_DMA_CAP_INT_STATUS_OFF, op))
            << " err_status=" << dbi(acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ERR_STATUS_OFF, op))
            << std::dec << std::noshowbase;
        return out.str();
    }

    // A transfer that does not complete must FAIL, not hang.
    //
    // acx_dma_halt_chan() rings the stop doorbell and then spins
    //     while (control.CS != 0x3 && !abort);
    // with no deadline and no iteration cap (Achronix_DMA.c).  If the engine
    // never reaches the halted state -- which is exactly the case when it was
    // never really running, or when the DBI window is not reaching the
    // controller at all and the "status" is whatever happens to be in memory
    // -- that loop never returns.  That is what hung the 2026-09-08 board
    // session: acx_dma_wait honoured its 10 s timeout, and then the cleanup
    // path spun forever with nothing printed.
    //
    // So: bound the halt ourselves, and report the controller state either
    // way.  A stuck DMA should tell you what it is stuck on.
    void wait_or_halt(ACX_DMA_transaction* transaction, const char* what) {
        const ACX_DMA_TRANSFER_STATUS status =
            acx_dma_wait(transaction, timeout_ms_);
        if (status == ACX_DMA_XFER_COMPLETE) {
            return;
        }

        const ACX_DMA_CH_OPERATION op = acx_dma_dir_to_op(transaction->direction);
        const std::string before = dma_channel_report(op, channel_);

        // Bounded stop: ring the doorbell, then poll the channel status for at
        // most halt_timeout_ms_ instead of forever.
        bool halted = false;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(halt_timeout_ms_);
        do {
            acx_mmio_write_dbi(device_,
                               acx_dma_ctrl_reg_addr(ACX_DMA_CAP_DOORBELL_OFF, op),
                               0x80000000u | channel_);
            const uint32_t ctrl =
                dbi(acx_dma_channel_reg_addr(ACX_DMA_CAP_CH_CONTROL1_OFF, op, channel_));
            const uint32_t cs = (ctrl >> 5) & 0x3u;
            if (ctrl == 0xffffffffu || cs == 0x3u || cs == 0x0u) {
                halted = true;
                break;
            }
        } while (std::chrono::steady_clock::now() < deadline);

        std::ostringstream out;
        out << what << " did not complete: status=" << static_cast<int>(status)
            << (halted ? ", channel halted" : ", channel did NOT halt within " +
                                                  std::to_string(halt_timeout_ms_) + " ms")
            << "; DMA regs at timeout [" << before << "]"
            << "; now [" << dma_channel_report(op, channel_) << "]"
            << ". If every field reads 0xffffffff the DBI window is not "
               "reaching the PCIe controller; if they read plausible values "
               "and xfer_size never falls, the engine is not running.";
        throw Error(ErrorKind::Device, out.str());
    }

    // Allocate-per-transfer fallback, kept working as the bring-up control.
    void h2d_per_transfer(const void* source, size_t bytes, uint64_t device_address) {
        ACX_DMA_transaction* raw = nullptr;
        check_sdk(
            acx_dma_start_xfer_h2d(
                const_cast<void*>(source), static_cast<uint32_t>(bytes),
                device_address, channel_, engine_, &raw),
            "acx_dma_start_xfer_h2d");
        ScopedTransaction transaction(raw);
        wait_or_halt(transaction.get(), "acx_dma_wait(h2d)");
    }

    void d2h_per_transfer(void* destination, size_t bytes, uint64_t device_address) {
        ACX_DMA_transaction* raw = nullptr;
        check_sdk(
            acx_dma_start_xfer_d2h(
                static_cast<uint32_t>(bytes), device_address, channel_, engine_, &raw),
            "acx_dma_start_xfer_d2h");
        ScopedTransaction transaction(raw);
        wait_or_halt(transaction.get(), "acx_dma_wait(d2h)");
        std::memcpy(destination, transaction.get()->dma_buffer->data, bytes);
    }

    ACX_DEV_PCIe_device* device_ = nullptr;
    ACX_DMA_engine_context* engine_;
    unsigned channel_;
    uint32_t timeout_ms_;
    // The SDK's own stop path has no deadline; ours does.
    uint32_t halt_timeout_ms_ = 2000;

    ACX_DMA_buffer* staging_ = nullptr;
    ACX_DMA_transaction* transaction_ = nullptr;
};

class SdkBackend : public Backend {
public:
    explicit SdkBackend(const SdkBackendConfig& config) : config_(config) {
        device_ = acx_dev_init_pcie_device_idx(
            static_cast<uint32_t>(config.device_index));
        if (device_ == nullptr || device_->status != ACX_SDK_STATUS_OK) {
            throw Error(ErrorKind::Device,
                        "failed to open PCIe device index " +
                            std::to_string(config.device_index));
        }

        pio_ = (config.bulk_mode == "pio");
        std::memset(&engine_, 0, sizeof(engine_));
        select_dbi_route(config.dbi_route);
        if (!pio_) {
            // Skipped for PIO on purpose: DMA init writes the engine's control
            // registers through the DBI window, and if that window is mapped
            // to memory those writes corrupt whatever they land on.
            acx_dma_build_engine_cntx(&engine_, device_, config.dma_engine);
            acx_dma_init_engine_cntx(&engine_);
            if (engine_.status != ACX_SDK_STATUS_OK) {
                acx_dev_cleanup_pcie_device(device_);
                device_ = nullptr;
                throw Error(ErrorKind::Device, "failed to initialize DMA engine");
            }
        }

        registers_.reset(new SdkRegisters(
            device_, config.bar_index, config.bar_offset, config.bar_length));
        if (pio_) {
            if (config.pio_bar > 5 ||
                device_->bar_handles[config.pio_bar] == nullptr) {
                acx_dev_cleanup_pcie_device(device_);
                device_ = nullptr;
                throw Error(ErrorKind::Config,
                            "PI0_FPGA_PIO_BAR=" + std::to_string(config.pio_bar) +
                                " is not mapped on this device");
            }
            pio_window_ = device_->bar_sizes[config.pio_bar];
            bulk_.reset(new PioBulk(device_, config.pio_bar, config.pio_noc_base,
                                    pio_window_));
        } else {
            bulk_.reset(new SdkBulk(device_, &engine_, config.dma_channel,
                                    config.dma_timeout_ms,
                                    config.dma_staging_bytes));
        }
    }

    // Which window the SDK uses to reach the PCIe controller's DBI registers
    // (DMA engine, ATU, MSI-X). The SDK's default is the COMPRESSED gateway
    // (BAR3, 192 KB, DMA block at +0x20000). Measured on p175bar with
    // pi0_dbi_probe (2026-09-09): this design's Device Manager exposes the
    // FULL 4 MB gateway (CDM +0, ATU +0x300000, DMA +0x310000); at the
    // compressed offsets the SDK reads config-space aliases (ENGINE_EN read
    // back as 0xac101b59) and its DMA init writes into nothing. The BAR4 x16
    // CSR route returns the same registers as the full gateway, so it is the
    // fallback. Selected with PI0_FPGA_DBI_ROUTE; the default is full.
    void select_dbi_route(const std::string& route) {
        ACX_IP which = ACX_IP_DBI_GATEWAY_FULL;
        if (route == "full") {
            acx_dbi_gateway_overide_location(device_, /*compressed=*/0, 3, 0);
        } else if (route == "comp") {
            which = ACX_IP_DBI_GATEWAY_COMP;
        } else if (route == "x16") {
            which = ACX_IP_DBI_X16_INTERFACE;
            acx_dbi_x16_overide_location(device_, 4, 0);
        } else {
            acx_dev_cleanup_pcie_device(device_);
            device_ = nullptr;
            throw Error(ErrorKind::Config,
                        "unknown PI0_FPGA_DBI_ROUTE='" + route +
                            "' (expected full, comp or x16)");
        }
        ACX_IP_block* block = &device_->ip_blocks[which];
        if (!block->initialized) {
            acx_dev_cleanup_pcie_device(device_);
            device_ = nullptr;
            throw Error(ErrorKind::Device,
                        "DBI route '" + route + "' is not usable on this device "
                        "(block not initialized: BAR not mapped, or the ADM has "
                        "not raised the programming-interface ready bit)");
        }
        device_->dbi_interface_block = block;
        block->enabled = 1;
        dbi_route_ = route;

        // Liveness: the CDM's first dword through this route must be the
        // device's own vendor/device id.
        uint32_t id = 0;
        if (acx_mmio_read_dbi(device_, 0x0, &id) != ACX_SDK_STATUS_OK ||
            (id & 0xffffu) != 0x1b59u) {
            std::ostringstream out;
            out << "DBI route '" << route << "' does not reach the PCIe controller: CDM[0]=0x"
                << std::hex << id << " (expected vendor 0x1b59)";
            acx_dev_cleanup_pcie_device(device_);
            device_ = nullptr;
            throw Error(ErrorKind::Device, out.str());
        }
    }

    ~SdkBackend() override {
        // Close the BAR mapping before the device it belongs to.
        registers_.reset();
        bulk_.reset();
        if (device_ != nullptr) {
            acx_dev_cleanup_pcie_device(device_);
            device_ = nullptr;
        }
    }

    RegisterAccess& registers() override { return *registers_; }
    BulkTransfer& bulk() override { return *bulk_; }
    bool is_hardware() const override { return true; }

    std::string description() const override {
        std::ostringstream out;
        out << "Achronix PCIe device " << config_.device_index
            << " BAR" << config_.bar_index
            << " offset=0x" << std::hex << config_.bar_offset
            << " length=0x" << config_.bar_length << std::dec;
        out << " dbi=" << dbi_route_;
        if (pio_) {
            out << " bulk=pio(BAR" << config_.pio_bar << " noc_base=0x"
                << std::hex << config_.pio_noc_base << " window=0x"
                << pio_window_ << std::dec << ")";
        } else if (config_.dma_staging_bytes == 0) {
            out << " dma=allocate-per-transfer";
        } else {
            out << " dma=staged(" << config_.dma_staging_bytes << "B)";
        }
        return out.str();
    }

private:
    SdkBackendConfig config_;
    ACX_DEV_PCIe_device* device_ = nullptr;
    ACX_DMA_engine_context engine_;
    bool pio_ = false;
    std::string dbi_route_;
    uint64_t pio_window_ = 0;
    std::unique_ptr<SdkRegisters> registers_;
    std::unique_ptr<BulkTransfer> bulk_;
};

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value == nullptr || *value == '\0') ? fallback : std::string(value);
}

}  // namespace

std::unique_ptr<Backend> make_sdk_backend(const SdkBackendConfig& config) {
    return std::unique_ptr<Backend>(new SdkBackend(config));
}

std::unique_ptr<Backend> make_backend_from_env(const SdkBackendConfig& defaults) {
    const std::string selected = env_or("PI0_FPGA_BACKEND", "real");

    if (selected == "mock") {
        MockConfig mock;
        const std::string latency = env_or("PI0_FPGA_MOCK_LATENCY_US", "");
        if (!latency.empty()) {
            mock.execute_latency =
                std::chrono::microseconds(std::stoll(latency));
        }
        return make_mock_backend(mock);
    }

    if (selected == "golden") {
        throw Error(
            ErrorKind::Config,
            "PI0_FPGA_BACKEND=golden is not implemented; use 'mock' for a "
            "software model or 'real' for hardware");
    }

    if (selected != "real") {
        throw Error(ErrorKind::Config,
                    "unknown PI0_FPGA_BACKEND='" + selected +
                        "' (expected real, mock or golden)");
    }

    SdkBackendConfig config = defaults;

    // The one knob board bring-up will want without a rebuild: set it to 0 to
    // A/B the reusable staging buffer against the allocate-per-transfer path.
    const std::string staging = env_or("PI0_FPGA_DMA_STAGING_BYTES", "");
    if (!staging.empty()) {
        config.dma_staging_bytes = std::stoull(staging);
    }

    // Bulk transport. "pio" moves payloads through a BAR window instead of the
    // DMA engine; see SdkBackendConfig::bulk_mode.
    config.bulk_mode = env_or("PI0_FPGA_BULK", "dma");
    config.dbi_route = env_or("PI0_FPGA_DBI_ROUTE", config.dbi_route);
    if (config.bulk_mode != "dma" && config.bulk_mode != "pio") {
        throw Error(ErrorKind::Config,
                    "unknown PI0_FPGA_BULK='" + config.bulk_mode +
                        "' (expected dma or pio)");
    }
    const std::string pio_bar = env_or("PI0_FPGA_PIO_BAR", "");
    if (!pio_bar.empty()) {
        config.pio_bar = static_cast<unsigned>(std::stoul(pio_bar, nullptr, 0));
    }
    const std::string pio_base = env_or("PI0_FPGA_PIO_NOC_BASE", "");
    if (!pio_base.empty()) {
        config.pio_noc_base = std::stoull(pio_base, nullptr, 0);
    }

    // Deliberately does NOT read PI0_DEVICE_INDEX / PI0_BAR_INDEX. The board
    // scripts already expand those into the CLI's positional arguments
    // (scripts/run_pi0_int8_board_perf_linux.sh and friends), so reading them
    // here too would let a stale exported variable silently override an
    // explicitly requested device.
    return make_sdk_backend(config);
}

}  // namespace pi0

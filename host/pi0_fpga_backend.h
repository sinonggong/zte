#ifndef PI0_FPGA_BACKEND_H
#define PI0_FPGA_BACKEND_H

// Backend seam for the resident pi0 host engine.
//
// Everything the engine does to the accelerator goes through the two abstract
// interfaces here: 32-bit register access and bulk host<->device transfer.
// Nothing in this header includes an Achronix SDK header, so the entire
// control path -- including the START/DONE handshake, which is the one place
// a mistake corrupts results silently -- can be exercised against a software
// model on a machine with no board in it.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pi0 {

// Typed failures. The engine's callers need to tell a recoverable timeout
// apart from a hardware fault, and matching on message text is not a contract
// anyone can keep.
enum class ErrorKind {
    Config,    // bad argument, missing file, unknown backend
    Device,    // open/close/BAR failure
    Protocol,  // ABI mismatch, or a START the hardware did not accept
    Timeout,   // hardware did not finish inside the deadline
    Fault,     // hardware reported a fault
    Transfer,  // DMA failure
};

const char* to_string(ErrorKind kind);

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}

    ErrorKind kind() const { return kind_; }

private:
    ErrorKind kind_;
};

// 32-bit MMIO register access.
class RegisterAccess {
public:
    virtual ~RegisterAccess() = default;

    virtual void write32(uint64_t offset, uint32_t value) = 0;
    virtual uint32_t read32(uint64_t offset) = 0;

    // Drive one control bit high and back down again, on top of whatever
    // level bits `base` holds.
    //
    // The trailing deassert is mandatory in both control paths, for different
    // reasons. The paged CTRL register edge-detects START/CLEAR/ADVANCE/ISSUE
    // (llama3_8b_runtime_ctrl.sv:96-99), so a bit left high produces no second
    // edge. The generic CTRL register consumes START/CLEAR as levels, and
    // CLEAR held high holds the whole generic subsystem in reset
    // (tc_generic_runtime_subsystem.sv:87).
    void pulse32(uint64_t offset, uint32_t base, uint32_t bit) {
        write32(offset, base);
        write32(offset, base | bit);
        write32(offset, base);
    }
};

// Bulk host<->device movement.
class BulkTransfer {
public:
    virtual ~BulkTransfer() = default;

    virtual void h2d(const void* source, size_t bytes, uint64_t device_address) = 0;
    virtual void d2h(void* destination, size_t bytes, uint64_t device_address) = 0;

    uint64_t h2d_bytes() const { return h2d_bytes_; }
    uint64_t d2h_bytes() const { return d2h_bytes_; }
    uint64_t h2d_transfers() const { return h2d_transfers_; }
    uint64_t d2h_transfers() const { return d2h_transfers_; }

    void reset_counters() {
        h2d_bytes_ = 0;
        d2h_bytes_ = 0;
        h2d_transfers_ = 0;
        d2h_transfers_ = 0;
    }

protected:
    void account_h2d(size_t bytes) {
        h2d_bytes_ += bytes;
        ++h2d_transfers_;
    }
    void account_d2h(size_t bytes) {
        d2h_bytes_ += bytes;
        ++d2h_transfers_;
    }

private:
    uint64_t h2d_bytes_ = 0;
    uint64_t d2h_bytes_ = 0;
    uint64_t h2d_transfers_ = 0;
    uint64_t d2h_transfers_ = 0;
};

// Splits a transfer into staging-buffer-sized pieces, calling
// visitor(host_offset, chunk_bytes) for each.
//
// This lives here, away from the SDK, because it is the part of the staged DMA
// path that can fail *silently*: get the arithmetic wrong and bytes land at the
// wrong device address with every SDK call still returning OK. Keeping it free
// of Achronix types is what lets it be tested on a machine with no board.
template <typename Visitor>
void for_each_dma_chunk(uint64_t total_bytes,
                        uint64_t staging_bytes,
                        Visitor&& visitor) {
    if (staging_bytes == 0) {
        throw Error(ErrorKind::Config, "staging buffer size must be nonzero");
    }
    uint64_t offset = 0;
    while (offset < total_bytes) {
        const uint64_t remaining = total_bytes - offset;
        const uint64_t chunk = remaining < staging_bytes ? remaining : staging_bytes;
        visitor(offset, chunk);
        offset += chunk;
    }
}

// One opened accelerator: registers plus bulk transfer, with a lifetime.
class Backend {
public:
    virtual ~Backend() = default;

    virtual RegisterAccess& registers() = 0;
    virtual BulkTransfer& bulk() = 0;

    // Human-readable identification for logs and health reporting.
    virtual std::string description() const = 0;

    // True when this backend talks to real silicon. The engine refuses to
    // report a mock result as a measurement.
    virtual bool is_hardware() const = 0;
};

}  // namespace pi0

#endif  // PI0_FPGA_BACKEND_H

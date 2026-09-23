#ifndef PI0_FPGA_MOCK_BACKEND_H
#define PI0_FPGA_MOCK_BACKEND_H

// A software model of the accelerator, for a machine with no board in it.
//
// This is deliberately not a stub that returns zeros. It reproduces the one
// hardware behaviour that makes the host's START/DONE handshake load-bearing:
//
//     DONE and FAULT are sticky, and START is *gated* on both being low
//     (tc_generic_runtime_subsystem.sv:115-116). A START issued while either
//     is set is silently swallowed -- no error, no state change -- and the
//     next status read returns the PREVIOUS package's result and counters.
//
// A mock without that property makes a broken handshake look correct, which
// is the failure mode this file exists to prevent. See pi0_fpga_control.h.
//
// The fault-injection knobs exist so the host's error and fallback paths can
// be exercised too, rather than only ever seeing the happy path.

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "pi0_fpga_backend.h"

namespace pi0 {

struct MockConfig {
    // How long a generic command package takes to retire. The default is
    // deliberately non-zero: queue depth and timeout tuning are meaningless
    // against an instantaneous device.
    std::chrono::microseconds execute_latency{500};

    // Model a bitstream whose CLEAR does not actually clear the sticky bits.
    // The host must detect this and refuse to start, rather than issue a
    // START that will be swallowed.
    bool clear_is_ineffective = false;

    // Model hardware that accepts a launch and never reports completion. The
    // host must time out rather than block forever.
    bool never_completes = false;

    // Raise a hardware fault on the given 1-based run. 0 disables.
    uint32_t fault_on_run = 0;
    uint32_t fault_code = 7;

    // Retired-command count reported per run, so a test can tell one run's
    // result from another's.
    uint64_t commands_per_run = 13;
};

// Sparse model of device memory, so an H2D followed by a D2H at the same
// address round-trips and address arithmetic errors show up as data errors.
class MockDeviceMemory {
public:
    void write(uint64_t address, const void* source, size_t bytes);
    void read(uint64_t address, void* destination, size_t bytes) const;

private:
    static constexpr uint64_t kPageBytes = 4096;
    std::unordered_map<uint64_t, std::vector<uint8_t>> pages_;
};

class MockBackend : public Backend {
public:
    explicit MockBackend(const MockConfig& config = MockConfig());
    ~MockBackend() override;

    RegisterAccess& registers() override;
    BulkTransfer& bulk() override;
    std::string description() const override;
    bool is_hardware() const override { return false; }

    // Number of generic launches the model actually accepted. A START that was
    // silently swallowed does not count, which is what lets a test assert that
    // the host's handshake really did land.
    uint64_t accepted_launches() const;

    // Number of START writes the model silently ignored because DONE or FAULT
    // was still set. Any value above zero means the host skipped its CLEAR.
    uint64_t swallowed_starts() const;

    // The package base the model was last armed with, latched at the moment a
    // START was accepted. Lets a test assert the engine launched the package
    // the plan named.
    uint64_t launched_package_base() const;

    MockDeviceMemory& memory();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Backend> make_mock_backend(const MockConfig& config = MockConfig());

}  // namespace pi0

#endif  // PI0_FPGA_MOCK_BACKEND_H

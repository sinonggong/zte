#ifndef PI0_FPGA_SDK_BACKEND_H
#define PI0_FPGA_SDK_BACKEND_H

// The real backend, over the Achronix PCIe/DMA SDK.
//
// This header stays free of Achronix includes on purpose: only
// pi0_fpga_sdk_backend.cpp needs them, so everything else in the host tree
// compiles and tests on a machine with no SDK installed.

#include <cstdint>
#include <memory>
#include <string>

#include "pi0_fpga_backend.h"

namespace pi0 {

struct SdkBackendConfig {
    int device_index = 0;
    unsigned bar_index = 0;
    uint64_t bar_offset = 0;
    uint64_t bar_length = 0x1000;
    unsigned dma_engine = 0;
    unsigned dma_channel = 0;

    // Per-transfer DMA deadline. Never infinite: an unbounded wait on a device
    // that never signals completion is the easiest way to hang a bring-up.
    uint32_t dma_timeout_ms = 10000;

    // Size of the reusable pinned staging buffer, in bytes.
    //
    // acx_dma_start_xfer_h2d/_d2h allocate a pinned buffer and a transaction on
    // every call and free them again afterwards. Pinning is a kernel operation
    // whose cost scales with the size being pinned, so a 3.7 GB model load pays
    // it once per chunk, and every inference pays it twice. The SDK's
    // lower-level path (acx_dma_alloc_buf + acx_dma_build_buf_tactn +
    // acx_dma_config_xfer + acx_dma_start_xfer) exists precisely to hoist that
    // out of the loop, so a transfer costs a memcpy, a few register writes, a
    // doorbell and a wait.
    //
    // Note the header's claim that config_xfer need only be called once per
    // unchanged shape is NOT usable -- see the comment on that call site.
    //
    // Capped at 4 MiB, and that is a hardware limit rather than a preference:
    // the driver allocates physically contiguous memory via
    // dma_alloc_coherent(), and the SDK's own DMA_example enforces the same
    // ceiling ("DMA engine does not support transfers > one VM Page").
    // Larger transfers are split; a larger value here is rejected.
    //
    // Set to 0 to fall back to the allocate-per-transfer calls. That path is
    // kept working on purpose: the staging path has never run against silicon,
    // and the fallback is the A/B control for board bring-up. Overridable with
    // PI0_FPGA_DMA_STAGING_BYTES.
    uint64_t dma_staging_bytes = 4ull * 1024 * 1024;

    // Bulk transport: "dma" (the PCIe DMA engine) or "pio" (32-bit MMIO
    // through a BAR window).
    //
    // PIO exists because the DMA engine is only reachable through the DBI
    // registers, and those are behind a BAR that a given bitstream may not map
    // correctly -- on the 2026-09-08 board session the DBI window landed in
    // memory, so the DMA "initialised", never started, and the first transfer
    // hung. Registers and the GDDR6 BAR window worked, so PIO can carry the
    // package, the arena and the readback and let the datapath be exercised
    // anyway. It is a bring-up path, not a fast one: one 32-bit PCIe
    // transaction per word, ~0.7 us per read and much less per posted write.
    //
    // pio_bar must be a BAR that maps device memory (BAR1 -> GDDR6 at NoC 0 in
    // this design); pio_noc_base is the NoC address that BAR's offset 0 points
    // at, so a plan's absolute device address can be turned into a BAR offset.
    // Overridable with PI0_FPGA_BULK, PI0_FPGA_PIO_BAR, PI0_FPGA_PIO_NOC_BASE.
    std::string bulk_mode = "dma";
    unsigned pio_bar = 1;
    uint64_t pio_noc_base = 0;

    // DBI access route to the PCIe controller's DMA/ATU registers: "full"
    // (4 MB gateway on BAR3, what this design's Device Manager exposes --
    // measured with pi0_dbi_probe on p175bar), "comp" (the SDK's default
    // 192 KB compressed gateway, which on this design reads config-space
    // aliases), or "x16" (the PCIE_1 CSR block on BAR4, ADM-independent).
    // Overridable with PI0_FPGA_DBI_ROUTE. Validated at open by reading the
    // CDM vendor id through the chosen route.
    std::string dbi_route = "full";
};

// Opens the device. Throws pi0::Error on failure.
std::unique_ptr<Backend> make_sdk_backend(const SdkBackendConfig& config);

// Chooses which KIND of backend to build; every other setting comes from
// `defaults`, so a caller's explicit choice always wins over the environment.
//
//   PI0_FPGA_BACKEND = real  (default)  the Achronix SDK, requires a board
//                    | mock             the software model, no board needed
//
//   PI0_FPGA_MOCK_LATENCY_US            mock execution latency, default 500
//
// The device and BAR indices are NOT read here. The board scripts expand
// PI0_DEVICE_INDEX / PI0_BAR_INDEX into the CLI's positional arguments, and
// reading them again would let a stale exported variable silently override an
// explicitly requested device.
//
// `golden` is named in the design notes as a third backend that replays a
// host-computed reference. It is not implemented; asking for it fails loudly
// rather than silently behaving like `mock`.
std::unique_ptr<Backend> make_backend_from_env(const SdkBackendConfig& defaults);

}  // namespace pi0

#endif  // PI0_FPGA_SDK_BACKEND_H

// pi0_dbi_probe -- read-only probe of the PCIe controller's DBI access routes.
//
// Answers, in seconds and without starting a DMA transfer:
//   * which DBI route the SDK picked at device open (compressed gateway on
//     BAR3, full gateway, or the x16 CSR interface) and why;
//   * whether the DMA controller's registers are really where the SDK will
//     write them (the compressed gateway puts the DMA block at BAR3+0x20000,
//     the full 4 MB gateway at BAR3+0x310000 -- p175bar's INFO.txt assumed
//     the latter, the SDK defaults to the former);
//   * what the DMA engine enables, channel status and error registers hold
//     right now, through every route that is reachable.
//
// Nothing here writes to the device unless --write-test is given, and that
// only toggles a DMA channel arbitration-weight register (which the SDK's own
// engine init writes anyway) and restores it.
//
// Usage: pi0_dbi_probe [--device N] [--write-test] [--no-full] [--no-x16]
//
// Built and run against the Achronix SDK v2.1.1; run it through the bundle's
// host/run.sh so the SDK .so files resolve.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "Achronix_DMA.h"
#include "Achronix_IP/Achronix_DBI_interface.h"
#include "Achronix_IP/Achronix_IP_config.h"
#include "Achronix_IP_block.h"
#include "Achronix_MMIO.h"
#include "Achronix_PCIe.h"
#include "Achronix_device.h"
}

namespace {

struct Args {
    unsigned device = 0;
    bool write_test = false;
    bool try_full = true;
    bool try_x16 = true;
};

const char* dbi_status_name(unsigned status) {
    switch (status) {
        case ACX_DBI_INIT_STATUS_NO_INIT: return "NO_INIT";
        case ACX_DBI_INIT_STATUS_OK: return "OK";
        case ACX_DBI_INIT_STATUS_ALLOC_FAIL: return "ALLOC_FAIL";
        case ACX_DBI_INIT_STATUS_NOT_MAPPED: return "NOT_MAPPED";
        case ACX_DBI_INIT_STATUS_NOT_READY: return "NOT_READY(prog-if bit0 clear)";
        case ACX_DBI_INIT_STATUS_FAILED: return "FAILED";
        default: return "?";
    }
}

void print_block(const ACX_IP_block& block) {
    std::printf("  %-18s mapped=%-3s bar_offset=0x%x bar_limit=0x%x initialized=%u enabled=%u status=%s\n",
                block.tag, block.bar_handle ? "yes" : "no", block.bar_offset, block.bar_limit,
                block.initialized, block.enabled, dbi_status_name(block.status));
}

bool bar_read(ACX_DEV_PCIe_device* device, unsigned bar, uint32_t offset, uint32_t* value) {
    return acx_mmio_read_bar_32(device, bar, offset, value) == ACX_SDK_STATUS_OK;
}

void dump_bar_words(ACX_DEV_PCIe_device* device, unsigned bar, uint32_t offset,
                    unsigned count, const char* label) {
    std::printf("  BAR%u+0x%06x %-30s:", bar, offset, label);
    for (unsigned i = 0; i < count; ++i) {
        uint32_t value = 0;
        if (!bar_read(device, bar, offset + 4u * i, &value)) {
            std::printf(" <read-error>");
            break;
        }
        std::printf(" %08x", value);
    }
    std::printf("\n");
}

// The DMA register block as raw BAR3 offsets, for one candidate layout base.
void dump_raw_dma_block(ACX_DEV_PCIe_device* device, uint32_t base, const char* label) {
    if (device->bar_sizes[3] < static_cast<uint64_t>(base) + 0x400u) {
        std::printf("  %s at BAR3+0x%06x: beyond BAR3 (size 0x%" PRIx64 ")\n", label, base,
                    device->bar_sizes[3]);
        return;
    }
    std::printf("  %s at BAR3+0x%06x:\n", label, base);
    dump_bar_words(device, 3, base + 0x0c, 1, "wr ENGINE_EN");
    dump_bar_words(device, 3, base + 0x2c, 1, "rd ENGINE_EN");
    dump_bar_words(device, 3, base + 0x18, 2, "wr ARB_WEIGHT lo/hi");
    dump_bar_words(device, 3, base + 0x38, 2, "rd ARB_WEIGHT lo/hi");
    dump_bar_words(device, 3, base + 0x4c, 1, "wr INT_STATUS");
    dump_bar_words(device, 3, base + 0xa0, 1, "rd INT_STATUS");
    dump_bar_words(device, 3, base + 0x200, 8, "wr ch0 CTRL1..LLP_LO");
    dump_bar_words(device, 3, base + 0x300, 8, "rd ch0 CTRL1..LLP_LO");
}

uint32_t dbi(ACX_DEV_PCIe_device* device, uint64_t address, bool* ok) {
    uint32_t value = 0xffffffffu;
    const ACX_SDK_STATUS status = acx_mmio_read_dbi(device, address, &value);
    if (ok != nullptr) *ok = (status == ACX_SDK_STATUS_OK);
    return status == ACX_SDK_STATUS_OK ? value : 0xffffffffu;
}

void print_dbi(ACX_DEV_PCIe_device* device, const char* name, uint64_t address) {
    bool ok = false;
    const uint32_t value = dbi(device, address, &ok);
    if (ok) {
        std::printf("  %-26s dbi=0x%09" PRIx64 " = %08x\n", name, address, value);
    } else {
        std::printf("  %-26s dbi=0x%09" PRIx64 " = <sdk-error>\n", name, address);
    }
}

void dump_dbi_route(ACX_DEV_PCIe_device* device, const char* route) {
    std::printf("DBI via %s (block %s):\n", route,
                device->dbi_interface_block ? device->dbi_interface_block->tag : "none");
    print_dbi(device, "CDM vendor/device (0x0)", 0x0);
    print_dbi(device, "CDM class/rev (0x8)", 0x8);
    print_dbi(device, "CDM BAR0 (0x10)", 0x10);
    print_dbi(device, "CDM BAR3 (0x1c)", 0x1c);
    print_dbi(device, "wr ENGINE_EN", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ENGINE_EN_OFF, ACX_DMA_WRITE_CH));
    print_dbi(device, "rd ENGINE_EN", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ENGINE_EN_OFF, ACX_DMA_READ_CH));
    print_dbi(device, "wr ARB_WEIGHT_LOW", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_CHANNEL_ARB_WEIGHT_LOW_OFF, ACX_DMA_WRITE_CH));
    print_dbi(device, "rd ARB_WEIGHT_LOW", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_CHANNEL_ARB_WEIGHT_LOW_OFF, ACX_DMA_READ_CH));
    print_dbi(device, "wr INT_STATUS", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_INT_STATUS_OFF, ACX_DMA_WRITE_CH));
    print_dbi(device, "rd INT_STATUS", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_INT_STATUS_OFF, ACX_DMA_READ_CH));
    print_dbi(device, "wr INT_MASK", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_INT_MASK_OFF, ACX_DMA_WRITE_CH));
    print_dbi(device, "rd INT_MASK", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_INT_MASK_OFF, ACX_DMA_READ_CH));
    print_dbi(device, "wr ERR_STATUS", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ERR_STATUS_OFF, ACX_DMA_WRITE_CH));
    print_dbi(device, "rd ERR_STATUS_LOW", acx_dma_ctrl_reg_addr(ACX_DMA_CAP_ERR_STATUS_LOW_OFF, ACX_DMA_READ_CH));
    for (int op = 0; op < 2; ++op) {
        const ACX_DMA_CH_OPERATION ch_op = op == 0 ? ACX_DMA_WRITE_CH : ACX_DMA_READ_CH;
        const char* prefix = op == 0 ? "wr ch0" : "rd ch0";
        char name[48];
        std::snprintf(name, sizeof(name), "%s CTRL1", prefix);
        print_dbi(device, name, acx_dma_channel_reg_addr(ACX_DMA_CAP_CH_CONTROL1_OFF, ch_op, 0));
        std::snprintf(name, sizeof(name), "%s XFER_SIZE", prefix);
        print_dbi(device, name, acx_dma_channel_reg_addr(ACX_DMA_CAP_TRANSFER_SIZE_OFF, ch_op, 0));
        std::snprintf(name, sizeof(name), "%s SAR lo", prefix);
        print_dbi(device, name, acx_dma_channel_reg_addr(ACX_DMA_CAP_SAR_LOW_OFF, ch_op, 0));
        std::snprintf(name, sizeof(name), "%s DAR lo", prefix);
        print_dbi(device, name, acx_dma_channel_reg_addr(ACX_DMA_CAP_DAR_LOW_OFF, ch_op, 0));
    }
}

// Writes the DMA read-channel arbitration weight register twice and reads it
// back, then restores it. A register that holds what was written is a real
// register; one that does not is memory, an alias, or nothing.
void write_test(ACX_DEV_PCIe_device* device, const char* route) {
    const uint64_t address =
        acx_dma_ctrl_reg_addr(ACX_DMA_CAP_CHANNEL_ARB_WEIGHT_LOW_OFF, ACX_DMA_READ_CH);
    bool ok = false;
    const uint32_t original = dbi(device, address, &ok);
    if (!ok) {
        std::printf("WRITE-TEST via %s: cannot read rd ARB_WEIGHT_LOW, skipped\n", route);
        return;
    }
    const uint32_t patterns[2] = {0x00008421u, 0x00001248u};
    bool held = true;
    for (uint32_t pattern : patterns) {
        acx_mmio_write_dbi(device, address, pattern);
        const uint32_t back = dbi(device, address, &ok);
        std::printf("WRITE-TEST via %s: wrote %08x read %08x%s\n", route, pattern, back,
                    (ok && back == pattern) ? "" : "  <-- did not hold");
        held = held && ok && back == pattern;
    }
    acx_mmio_write_dbi(device, address, original);
    std::printf("WRITE-TEST via %s: %s (restored %08x)\n", route,
                held ? "register holds writes" : "register does NOT hold writes", original);
}

bool select_route(ACX_DEV_PCIe_device* device, ACX_IP which, const char* route) {
    ACX_IP_block* block = &device->ip_blocks[which];
    if (!block->initialized) {
        std::printf("route %s: not initialized (%s)\n", route, dbi_status_name(block->status));
        return false;
    }
    device->dbi_interface_block = block;
    block->enabled = 1;
    return true;
}

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--device") == 0 && i + 1 < argc) {
            args.device = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 0));
        } else if (std::strcmp(arg, "--write-test") == 0) {
            args.write_test = true;
        } else if (std::strcmp(arg, "--no-full") == 0) {
            args.try_full = false;
        } else if (std::strcmp(arg, "--no-x16") == 0) {
            args.try_x16 = false;
        } else {
            std::fprintf(stderr, "usage: pi0_dbi_probe [--device N] [--write-test] [--no-full] [--no-x16]\n");
            std::exit(2);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    setvbuf(stdout, nullptr, _IONBF, 0);  // every line reaches the terminal even if we die

    uint32_t major = 0, minor = 0, patch = 0;
    acx_pcie_get_sdk_version(&major, &minor, &patch);
    std::printf("SDK %u.%u.%u\n", major, minor, patch);

    ACX_DEV_PCIe_device* device = acx_dev_init_pcie_device_idx(args.device);
    if (device == nullptr) {
        std::printf("device %u: open failed (driver loaded? /dev/ac7t15xx%u readable?)\n",
                    args.device, args.device);
        return 1;
    }
    std::printf("device %u: status=%d function=%u part=%s\n", args.device,
                static_cast<int>(device->status), device->function_num,
                acx_part_name_to_string(device->part_name));

    ACX_PCIE_device_info info;
    std::memset(&info, 0, sizeof(info));
    if (acx_pcie_get_device_info(device->handle, &info) == ACX_SDK_STATUS_OK) {
        std::printf("  bdf %02x:%02x.%x vendor %04x device %04x rev %02x prog-if %02x (fabric %s)\n",
                    info.bus, info.device, info.function, info.vendor_id, info.device_id,
                    info.revision_id, info.programming_interface,
                    acx_dev_is_fabric_ready(device) == ACX_SDK_STATUS_OK ? "READY" : "NOT READY");
        for (unsigned bar = 0; bar < 6; ++bar) {
            if (info.bar_sizes[bar] != 0) {
                std::printf("  BAR%u size 0x%" PRIx64 " %s\n", bar, info.bar_sizes[bar],
                            device->bar_handles[bar] ? "mapped" : "NOT mapped");
            }
        }
    }

    std::printf("IP blocks after open (SDK picks FULL, then COMP, then X16):\n");
    for (uint32_t i = 0; i < device->num_ip_blocks; ++i) {
        print_block(device->ip_blocks[i]);
    }
    std::printf("  selected: %s\n",
                device->dbi_interface_block ? device->dbi_interface_block->tag : "NONE");

    if (device->bar_sizes[3] != 0 && device->bar_handles[3] != nullptr) {
        std::printf("Raw BAR3 (expect 00101b59 at +0 if the gateway's CDM is reachable):\n");
        dump_bar_words(device, 3, 0x0, 4, "CDM header");
        std::printf("Raw BAR3 alias scan, first word every 256 KiB:\n");
        for (uint64_t offset = 0; offset < device->bar_sizes[3]; offset += 0x40000) {
            uint32_t value = 0;
            const bool ok = bar_read(device, 3, static_cast<uint32_t>(offset), &value);
            std::printf("  +0x%06" PRIx64 ": %s%08x", offset, ok ? "" : "<err> ", value);
            std::printf(((offset / 0x40000) % 4 == 3) ? "\n" : "   ");
        }
        std::printf("\n");
        dump_raw_dma_block(device, 0x20000, "COMPRESSED-layout DMA block (SDK default)");
        dump_bar_words(device, 3, 0x10000, 4, "COMPRESSED-layout ATU block");
        dump_raw_dma_block(device, 0x310000, "FULL-layout DMA block (INFO.txt)");
        if (device->bar_sizes[3] >= 0x300010) {
            dump_bar_words(device, 3, 0x300000, 4, "FULL-layout ATU block");
        }
    }

    if (device->dbi_interface_block != nullptr) {
        dump_dbi_route(device, "SDK default route");
        if (args.write_test) write_test(device, "SDK default route");
    }

    if (args.try_full && device->bar_sizes[3] >= 0x400000) {
        acx_dbi_gateway_overide_location(device, /*compressed=*/0, 3, 0);
        if (select_route(device, ACX_IP_DBI_GATEWAY_FULL, "FULL gateway on BAR3")) {
            dump_dbi_route(device, "FULL gateway on BAR3");
            if (args.write_test) write_test(device, "FULL gateway on BAR3");
        }
    } else if (args.try_full) {
        std::printf("route FULL gateway: BAR3 is smaller than 4 MB, skipped\n");
    }

    if (args.try_x16) {
        if (device->bar_sizes[4] != 0 && device->bar_handles[4] != nullptr) {
            acx_dbi_x16_overide_location(device, 4, 0);
            if (select_route(device, ACX_IP_DBI_X16_INTERFACE, "x16 CSR on BAR4")) {
                dump_bar_words(device, 4, 0x20, 5, "x16 ACCESS/ADDR/WDATA/RDATA/CTRL");
                dump_dbi_route(device, "x16 CSR on BAR4");
                if (args.write_test) write_test(device, "x16 CSR on BAR4");
            }
        } else {
            std::printf("route x16 CSR: BAR4 not mapped, skipped\n");
        }
    }

    acx_dev_cleanup_pcie_device(device);
    return 0;
}

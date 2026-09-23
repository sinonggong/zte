// Inspect and move the PCIe endpoint's inbound address-translation regions (the BAR -> NoC map) through the SDK.
// With the compressed DBI gateway of the node-array bitstreams the SDK's DMA engine cannot be used, but the PIO
// window (BAR1 -> NoC 0..2 MB) can be slid over GDDR6 by rewriting BAR1's ATU region target: that is how a whole
// static image can be loaded without DMA.
//
//   pi0_atu_tool list                         every region: enable, mode, BAR, host base/limit, NoC target
//   pi0_atu_tool set <region> <noc_target>    rewrite the region's target address (hex), read it back
//   pi0_atu_tool probe <region> <noc_target> <bar> <offset>   set, then read one 32-bit word through that BAR
// Env: PI0_FPGA_DBI_ROUTE=comp|full (default comp).  Stop only with SIGINT (holds /dev/ac7t15xx0).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "Achronix_SDK.h"
#include "Achronix_DMA.h"
#include "Achronix_IP/Achronix_IP_config.h"
#include "Achronix_IP/Achronix_DBI_interface.h"

static int fail(const char* what, ACX_SDK_STATUS st) { std::fprintf(stderr, "%s: sdk status %d\n", what, static_cast<int>(st)); return 1; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: pi0_atu_tool list | set <region> <noc_target> | probe <region> <noc_target> <bar> <offset>\n"); return 2; }
    ACX_DEV_PCIe_device* dev = acx_dev_init_pcie_device_idx(0);
    if (!dev || dev->status != ACX_SDK_STATUS_OK) { std::fprintf(stderr, "device open failed\n"); return 1; }
    const char* route = std::getenv("PI0_FPGA_DBI_ROUTE");
    ACX_IP which = (route && std::string(route) == "full") ? ACX_IP_DBI_GATEWAY_FULL : ACX_IP_DBI_GATEWAY_COMP;
    if (which == ACX_IP_DBI_GATEWAY_FULL) acx_dbi_gateway_overide_location(dev, 0, 3, 0);
    ACX_IP_block* blk = &dev->ip_blocks[which];
    if (!blk->initialized) { std::fprintf(stderr, "DBI route not initialized\n"); acx_dev_cleanup_pcie_device(dev); return 1; }
    dev->dbi_interface_block = blk; blk->enabled = 1;
    uint32_t id = 0;
    if (acx_mmio_read_dbi(dev, 0, &id) != ACX_SDK_STATUS_OK || (id & 0xffffu) != 0x1b59u) {
        std::fprintf(stderr, "CDM[0]=0x%08x: the DBI gateway does not answer\n", id); acx_dev_cleanup_pcie_device(dev); return 1;
    }
    const std::string cmd = argv[1];
    int rc = 0;
    if (cmd == "list") {
        ACX_ATU_context ctx; std::memset(&ctx, 0, sizeof(ctx));
        ACX_SDK_STATUS st = acx_atu_get_context(dev, &ctx);
        if (st != ACX_SDK_STATUS_OK) rc = fail("acx_atu_get_context", st);
        else for (int i = 0; i < ACX_NUM_ATU_REGIONS; ++i) {
            const ACX_ATU_region_context& r = ctx.regions[i];
            std::printf("region %d: en=%u match_mode=%u bar=%u base=0x%08x%08x limit=0x%08x%08x target=0x%08x%08x ctrl1=0x%08x ctrl2=0x%08x\n", i,
                        r.iatu_region_ctrl_2_inbound.REGION_EN, r.iatu_region_ctrl_2_inbound.MATCH_MODE, r.iatu_region_ctrl_2_inbound.BAR_NUM,
                        r.iatu_upper_base_addr_inbound, r.iatu_lwr_base_addr_inbound, r.iatu_upper_limit_addr_inbound, r.iatu_lwr_limit_addr_inbound,
                        r.iatu_upper_target_addr_inbound, r.iatu_lwr_target_addr_inbound, r.iatu_region_ctrl_1_inbound.value, r.iatu_region_ctrl_2_inbound.value);
        }
    } else if ((cmd == "set" && argc >= 4) || (cmd == "probe" && argc >= 6)) {
        const int region = std::atoi(argv[2]);
        const uint64_t target = std::strtoull(argv[3], nullptr, 0);
        ACX_ATU_region_context r; std::memset(&r, 0, sizeof(r));
        ACX_SDK_STATUS st = acx_atu_get_region_context(dev, region, &r);
        if (st != ACX_SDK_STATUS_OK) rc = fail("acx_atu_get_region_context", st);
        else {
            std::printf("region %d before: target=0x%08x%08x en=%u bar=%u\n", region, r.iatu_upper_target_addr_inbound, r.iatu_lwr_target_addr_inbound,
                        r.iatu_region_ctrl_2_inbound.REGION_EN, r.iatu_region_ctrl_2_inbound.BAR_NUM);
            r.region_num = region;
            r.iatu_lwr_target_addr_inbound = static_cast<uint32_t>(target & 0xffffffffu);
            r.iatu_upper_target_addr_inbound = static_cast<uint32_t>(target >> 32);
            st = acx_atu_config_region(dev, &r);
            if (st != ACX_SDK_STATUS_OK) rc = fail("acx_atu_config_region", st);
            ACX_ATU_region_context q; std::memset(&q, 0, sizeof(q));
            if (acx_atu_get_region_context(dev, region, &q) == ACX_SDK_STATUS_OK)
                std::printf("region %d after:  target=0x%08x%08x en=%u\n", region, q.iatu_upper_target_addr_inbound, q.iatu_lwr_target_addr_inbound, q.iatu_region_ctrl_2_inbound.REGION_EN);
            if (cmd == "probe" && rc == 0) {
                const unsigned bar = static_cast<unsigned>(std::atoi(argv[4]));
                const uint32_t off = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 0));
                uint32_t v = 0;
                if (acx_mmio_read_bar_32(dev, bar, off, &v) == ACX_SDK_STATUS_OK) std::printf("BAR%u+0x%x = 0x%08x\n", bar, off, v);
                else std::printf("BAR%u read failed\n", bar);
            }
        }
    } else rc = 2;
    uint32_t id2 = 0; acx_mmio_read_dbi(dev, 0, &id2);
    std::printf("gateway after: CDM[0]=0x%08x\n", id2);
    acx_dev_cleanup_pcie_device(dev);
    return rc;
}

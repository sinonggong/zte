// Run a generated pi0 chunk (paper/sw/pi0_chunk_program.py --hex output) on a node-array bitstream.
//
//   pi0_chunk_run selftest                     check that sliding BAR1's ATU window reaches distinct GDDR6 pages
//   pi0_chunk_run run <vec_dir> [--map vector:14,int8:6,uint8:12] [--show N] [--timeout-s T] [--no-load] [--no-scrub]
//                                [--scrub-flags-only] [--check-io-only] [--no-check] [--repeat N]
//                                [--inputs <prefix>] [--dump-actions <file>] [--wrong-list <file>]
//       --inputs: a run list (<prefix>.idx "addr_hex bytes offset" + <prefix>.bin) written before every start --
//       the per-chunk host inputs (paper/sw/pi0_chip_runtime.py); --dump-actions: the IO actions region after
//       every run; --repeat: scrub / inputs / start / poll / check N times on the loaded image
//   pi0_chunk_run serve <vec_dir> [--map ...] [--timeout-s T]    the image is already loaded (a previous run): stay
//       open and run one chunk per "infer <inputs prefix> <actions file>" line on stdin (paper/sw/pi0_chip_runtime.py)
//   Env PI0_HOST_BRIDGE=1: BAR1 is the host->GDDR6 paging bridge (256 MB pages, window beat N_NODE+1), so the chunk
//   may use all of GDDR6; otherwise BAR1 is a fixed window at NoC 0 (compact chunks only).
//       --map: the chip node index of each kind's first node.  The generator numbers its nodes vector, int8,
//       uint8; pi0_chip_top numbers chains first (N_DEEP 32-stage, then 16-stage, the last N_PV uint8) and vector
//       nodes after them.  A chunk generated with --n-stage 16 must land on 16-stage chains.  Default: identity.
//       vec_dir: mem_in.hex (beat index, beat), mem_exp.hex, nodes.txt ("node kind prog_base_hex").
//       1. scrub every expected beat to zero (flags must start below the flag value; outputs so a missing write shows)
//       2. write the image; 3. point every node at its program and start them together (broadcast beat);
//       4. poll the halt vector; 5. read every expected beat back and compare; per-node fabric cycles.
//
// GDDR6 is reached through BAR1 (2 MB, PIO): the ATU region that serves BAR1 is re-targeted per 2 MB page
// (acx_atu_config_region through the DBI gateway), so no DMA engine is needed.  The host register window is BAR0
// (paper/rtl/pi0_chip_ctrl.sv).  Env PI0_FPGA_DBI_ROUTE=comp|full (default comp).  Stop only with SIGINT.
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Achronix_SDK.h"
#include "Achronix_DMA.h"
#include "Achronix_IP/Achronix_IP_config.h"
#include "Achronix_IP/Achronix_DBI_interface.h"

namespace {

std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop = true; }

struct Dev {
    ACX_DEV_PCIe_device* d = nullptr;
    ACX_ATU_region_context bar1_region{};
    bool atu_fixed = false;
    bool bridge = false;              // PI0_HOST_BRIDGE=1: BAR1 is pi0_host_gddr_bridge, paged by window beat N_NODE+1
    unsigned n_node = 0;
    uint64_t window = 0x200000, cur_page = ~0ull;
    uint64_t atu_moves = 0;

    void open() {
        d = acx_dev_init_pcie_device_idx(0);
        if (!d || d->status != ACX_SDK_STATUS_OK) throw std::runtime_error("device open failed");
        const char* route = std::getenv("PI0_FPGA_DBI_ROUTE");
        ACX_IP which = (route && std::string(route) == "full") ? ACX_IP_DBI_GATEWAY_FULL : ACX_IP_DBI_GATEWAY_COMP;
        if (which == ACX_IP_DBI_GATEWAY_FULL) acx_dbi_gateway_overide_location(d, 0, 3, 0);
        ACX_IP_block* blk = &d->ip_blocks[which];
        if (!blk->initialized) throw std::runtime_error("DBI route not initialized");
        d->dbi_interface_block = blk; blk->enabled = 1;
        uint32_t id = 0;
        if (acx_mmio_read_dbi(d, 0, &id) != ACX_SDK_STATUS_OK || (id & 0xffffu) != 0x1b59u)
            throw std::runtime_error("DBI gateway does not answer (reprogram the board)");
        window = d->bar_sizes[1];
        for (unsigned n = 1; n <= 64 && !n_node; ++n) {
            const uint32_t v = rd0(n, 7);
            if ((v & 0xffu) == n && ((v >> 16) & 0xffffu) >= 0x5330u && ((v >> 16) & 0xffffu) <= 0x533fu) n_node = n;
        }
        const char* br = std::getenv("PI0_HOST_BRIDGE");
        if (br && std::string(br) == "1") {
            if (!n_node) throw std::runtime_error("PI0_HOST_BRIDGE=1 but no node-array window behind BAR0");
            bridge = true; cur_page = ~0ull;
            set_page(0);
            std::printf("device: BAR1 = host->GDDR6 bridge, %llu MB pages (window beat %u)\n",
                        static_cast<unsigned long long>(window >> 20), n_node + 1);
            return;
        }
        const char* slide = std::getenv("PI0_ATU_SLIDE");
        if (!slide || std::string(slide) != "1") {
            // fixed window: BAR1 maps NoC [0, window) (pci_express.acxip); the ATU is never touched.  Through the
            // compressed DBI gateway of the S0 bitstreams the ATU reads back as garbage, so sliding is opt-in.
            atu_fixed = true; cur_page = 0;
            std::printf("device: BAR1 window %llu MB, fixed at NoC 0\n", static_cast<unsigned long long>(window >> 20));
            return;
        }
        ACX_ATU_context ctx; std::memset(&ctx, 0, sizeof(ctx));
        if (acx_atu_get_context(d, &ctx) != ACX_SDK_STATUS_OK) throw std::runtime_error("acx_atu_get_context failed");
        int found = -1;
        for (int i = 0; i < ACX_NUM_ATU_REGIONS; ++i)
            if (ctx.regions[i].iatu_region_ctrl_2_inbound.REGION_EN && ctx.regions[i].iatu_region_ctrl_2_inbound.BAR_NUM == 1) { found = i; break; }
        if (found < 0) throw std::runtime_error("no enabled ATU region serves BAR1");
        bar1_region = ctx.regions[found];
        bar1_region.region_num = found;
        cur_page = (static_cast<uint64_t>(bar1_region.iatu_upper_target_addr_inbound) << 32) | bar1_region.iatu_lwr_target_addr_inbound;
        std::printf("device: BAR1 window %llu KB via ATU region %d (target 0x%llx)\n", static_cast<unsigned long long>(window >> 10),
                    found, static_cast<unsigned long long>(cur_page));
    }
    ~Dev() { if (d) { if (!atu_fixed) { try { set_page(0); } catch (...) {} } acx_dev_cleanup_pcie_device(d); } }
    void set_page(uint64_t page) {
        if (page == cur_page) return;
        if (bridge) {
            wr0(n_node + 1, 0, static_cast<uint32_t>(page & 0xffffffffu));
            wr0(n_node + 1, 1, static_cast<uint32_t>((page >> 32) & 0x3ffu));
            const uint64_t back = (static_cast<uint64_t>(rd0(n_node + 1, 1) & 0x3ffu) << 32) | rd0(n_node + 1, 0);
            if (back != page) throw std::runtime_error("bridge page register did not take the page");
            cur_page = page; ++atu_moves;
            return;
        }
        if (atu_fixed) throw std::runtime_error("address beyond the fixed BAR1 window (use a compact chunk, or PI0_ATU_SLIDE=1)");
        bar1_region.iatu_lwr_target_addr_inbound = static_cast<uint32_t>(page & 0xffffffffu);
        bar1_region.iatu_upper_target_addr_inbound = static_cast<uint32_t>(page >> 32);
        if (acx_atu_config_region(d, &bar1_region) != ACX_SDK_STATUS_OK) throw std::runtime_error("acx_atu_config_region failed");
        cur_page = page; ++atu_moves;
    }
    // GDDR6 channel striping of a STRIPE bitstream (window beat N_NODE + 1 bit 64): on for a chunk generated with
    // PI0_CHUNK_MAP=striped, off otherwise; set before the image is loaded (the bridge's addresses depend on it).
    // Returns what the window reads back (bitstreams without the switch read 0).
    bool set_stripe(bool on) {
        if (!bridge) return false;
        wr0(n_node + 1, 2, on ? 1u : 0u);
        return (rd0(n_node + 1, 2) & 1u) != 0;
    }
    void wr_beat(uint64_t addr, const uint8_t* b) {
        set_page(addr & ~(window - 1));
        const uint32_t off = static_cast<uint32_t>(addr & (window - 1));
        for (int i = 0; i < 4; ++i) {
            uint64_t v; std::memcpy(&v, b + 8 * i, 8);
            if (acx_mmio_write_bar_64(d, 1, off + 8 * i, v) != ACX_SDK_STATUS_OK) throw std::runtime_error("BAR1 write failed");
        }
    }
    void wr_u64(uint64_t addr, uint64_t v) {
        set_page(addr & ~(window - 1));
        if (acx_mmio_write_bar_64(d, 1, static_cast<uint32_t>(addr & (window - 1)), v) != ACX_SDK_STATUS_OK)
            throw std::runtime_error("BAR1 write failed");
    }
    uint32_t rd_u32(uint64_t addr) {
        set_page(addr & ~(window - 1));
        uint32_t v = 0;
        if (acx_mmio_read_bar_32(d, 1, static_cast<uint32_t>(addr & (window - 1)), &v) != ACX_SDK_STATUS_OK)
            throw std::runtime_error("BAR1 read failed");
        return v;
    }
    void rd_beat(uint64_t addr, uint8_t* b) {
        set_page(addr & ~(window - 1));
        const uint32_t off = static_cast<uint32_t>(addr & (window - 1));
        for (int i = 0; i < 8; ++i) {
            uint32_t v = 0;
            if (acx_mmio_read_bar_32(d, 1, off + 4 * i, &v) != ACX_SDK_STATUS_OK) throw std::runtime_error("BAR1 read failed");
            std::memcpy(b + 4 * i, &v, 4);
        }
    }
    uint32_t rd0(unsigned beat, unsigned dw) {
        uint32_t v = 0; acx_mmio_read_bar_32(d, 0, beat * 32 + dw * 4, &v); return v;
    }
    void wr0(unsigned beat, unsigned dw, uint32_t v) { acx_mmio_write_bar_32(d, 0, beat * 32 + dw * 4, v); }
};

using Beat = std::array<uint8_t, 32>;

bool parse_beat(const std::string& line, uint64_t& idx, Beat& b) {
    const auto sp = line.find(' ');
    if (sp == std::string::npos || line.size() < sp + 1 + 64) return false;
    idx = std::strtoull(line.substr(0, sp).c_str(), nullptr, 16);
    const char* h = line.c_str() + sp + 1;
    for (int i = 0; i < 32; ++i) {
        char two[3] = {h[2 * (31 - i)], h[2 * (31 - i) + 1], 0};
        b[static_cast<size_t>(i)] = static_cast<uint8_t>(std::strtoul(two, nullptr, 16));
    }
    return true;
}

// every beat of the image ("data") or of the expected set ("exp"): image/<which>.{idx,bin} when the generator
// wrote them (--bin, the whole-chunk format: runs of beats), else mem_in.hex / mem_exp.hex.  cb returns false to stop.
template <class F>
uint64_t for_each_beat(const std::string& dir, const std::string& which, F cb) {
    uint64_t n = 0;
    std::ifstream idx(dir + "/image/" + which + ".idx");
    if (idx) {
        std::ifstream bin(dir + "/image/" + which + ".bin", std::ios::binary);
        if (!bin) throw std::runtime_error("image/" + which + ".bin missing");
        std::string a; uint64_t nbytes, off; Beat b;
        while (idx >> a >> nbytes >> off && !g_stop) {
            const uint64_t addr = std::strtoull(a.c_str(), nullptr, 16);
            bin.seekg(static_cast<std::streamoff>(off));
            for (uint64_t k = 0; k < nbytes; k += 32) {
                bin.read(reinterpret_cast<char*>(b.data()), 32);
                if (!bin) throw std::runtime_error("image/" + which + ".bin short");
                ++n;
                if (!cb(addr + k, b)) return n;
            }
        }
        return n;
    }
    std::ifstream f(dir + (which == "data" ? "/mem_in.hex" : "/mem_exp.hex"));
    if (!f) throw std::runtime_error("no image/" + which + ".idx and no hex file in " + dir);
    std::string line; uint64_t i; Beat b;
    while (std::getline(f, line) && !g_stop)
        if (parse_beat(line, i, b)) { ++n; if (!cb(i * 32, b)) return n; }
    return n;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int selftest(Dev& dev) {
    Beat a{}, b{}, r{};
    for (int i = 0; i < 32; ++i) { a[static_cast<size_t>(i)] = static_cast<uint8_t>(0x10 + i); b[static_cast<size_t>(i)] = static_cast<uint8_t>(0xc0 + i); }
    // fixed window: three far-apart addresses inside it; sliding: pages 0, 1 and 1 GB
    const uint64_t A = 0x3000, B = dev.atu_fixed ? dev.window / 2 + 0x3000 : dev.window + 0x3000,
                   C = dev.atu_fixed ? dev.window - 0x1000 : 0x40000000ull + 0x3000;
    dev.wr_beat(A, a.data()); dev.wr_beat(B, b.data());
    Beat c = a; c[0] = 0x77; dev.wr_beat(C, c.data());
    int fails = 0;
    dev.rd_beat(A, r.data()); if (r != a) { ++fails; std::printf("page 0 read-back wrong\n"); }
    dev.rd_beat(B, r.data()); if (r != b) { ++fails; std::printf("page 1 read-back wrong (window not moved?)\n"); }
    dev.rd_beat(C, r.data()); if (r != c) { ++fails; std::printf("1 GB page read-back wrong\n"); }
    dev.rd_beat(A, r.data()); if (r != a) { ++fails; std::printf("page 0 overwritten by another page (aliasing)\n"); }
    if (dev.bridge) {
        // the bridge reaches all of GDDR6: 9 GB and the last page below 16 GB must hold their own data
        for (uint64_t far : {0x240000000ull + 0x3000, 0x3F0000000ull + 0x5000}) {
            Beat e = b; e[0] = static_cast<uint8_t>(far >> 28); e[31] = 0x5a;
            dev.wr_beat(far, e.data());
            dev.rd_beat(far, r.data()); if (r != e) { ++fails; std::printf("far page 0x%llx read-back wrong\n", static_cast<unsigned long long>(far)); }
        }
        dev.rd_beat(A, r.data()); if (r != a) { ++fails; std::printf("page 0 aliased by a far page\n"); }
        dev.rd_beat(B, r.data()); if (r != b) { ++fails; std::printf("page 1 aliased by a far page\n"); }
        // PIO bandwidth through the bridge: 1 MB of writes, then reads
        std::vector<uint8_t> blk(32, 0xa5);
        const double t0 = now_s();
        for (uint64_t o = 0; o < (1u << 20); o += 32) dev.wr_beat(0x10000000ull + 0x100000 + o, blk.data());
        const double t1 = now_s();
        for (uint64_t o = 0; o < (1u << 20); o += 32) dev.rd_beat(0x10000000ull + 0x100000 + o, r.data());
        const double t2 = now_s();
        std::printf("bridge PIO: write %.1f MB/s, read %.1f MB/s\n", 1.048576 / (t1 - t0), 1.048576 / (t2 - t1));
    }
    std::printf("selftest %s (ATU moves %llu)\n", fails ? "FAIL" : "PASS", static_cast<unsigned long long>(dev.atu_moves));
    return fails ? 1 : 0;
}

// write a distinct beat at every address (in order), then read every one back: an address whose beat was replaced
// by a later write aliases that later address (the GDDR6 map: channel k at k << 33, 2 GB of storage each)
int probe(Dev& dev, const std::vector<uint64_t>& addrs) {
    std::vector<Beat> w(addrs.size());
    for (size_t i = 0; i < addrs.size(); ++i) {
        for (int k = 0; k < 32; ++k) w[i][static_cast<size_t>(k)] = static_cast<uint8_t>(0x11 * (i + 1) + k);
        dev.wr_beat(addrs[i], w[i].data());
    }
    int bad = 0;
    for (size_t i = 0; i < addrs.size(); ++i) {
        Beat r{}; dev.rd_beat(addrs[i], r.data());
        int who = -1;
        for (size_t j = 0; j < addrs.size(); ++j) if (r == w[j]) who = static_cast<int>(j);
        std::printf("  %#014llx: %s\n", static_cast<unsigned long long>(addrs[i]),
                    who == static_cast<int>(i) ? "own data" : who >= 0 ? ("holds the beat written at " + std::to_string(who)).c_str() : "other data");
        if (who != static_cast<int>(i)) ++bad;
    }
    std::printf("probe: %d of %zu addresses do not hold their own beat\n", bad, addrs.size());
    return bad ? 1 : 0;
}

// byte ranges [lo, hi) of the named areas in regions.json (flags, IO) -- a minimal scan, no JSON library
std::vector<std::pair<uint64_t, uint64_t>> area_ranges(const std::string& dir, const std::string& area) {
    std::ifstream f(dir + "/regions.json"); std::stringstream ss; ss << f.rdbuf(); const std::string s = ss.str();
    std::vector<std::pair<uint64_t, uint64_t>> out;
    size_t pos = 0;
    while ((pos = s.find("\"area\": \"" + area + "\"", pos)) != std::string::npos) {
        const size_t obj0 = s.rfind('{', pos), obj1 = s.find('}', pos);
        const std::string obj = s.substr(obj0, obj1 - obj0);
        auto num = [&](const char* key) { const size_t k = obj.find(key); return std::strtoull(obj.c_str() + obj.find(':', k) + 1, nullptr, 10); };
        const uint64_t base = num("\"base\""), size = num("\"size\"");
        out.push_back({base, base + size});
        pos = obj1;
    }
    return out;
}
bool in_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& r, uint64_t a) {
    for (auto& [lo, hi] : r) if (a >= lo && a < hi) return true;
    return false;
}

struct RunOpts {
    int show = 12;
    double timeout_s = 60;
    bool load = true, scrub = true, scrub_flags_only = false, check_io_only = false, check = true, load_only = false;
    int repeat = 1;
    std::string inputs;             // <prefix>.idx / <prefix>.bin: per-chunk host inputs written before every start
    std::string dump_actions;       // write the IO "actions" region's bytes here after every run
    std::string wrong_list;         // run 0: every wrong beat's address, one per line (paper/sw/pi0_chunk_wrong_regions.py)
    std::string timeline;           // run 0: poll every part's flag while the chunk runs, "index first_seen_s" per flag
    std::map<std::string, unsigned> kmap;
};

// write a run list (<prefix>.idx: "addr_hex bytes offset", <prefix>.bin) through BAR1
uint64_t write_runs(Dev& dev, const std::string& prefix) {
    std::ifstream idx(prefix + ".idx"), bin(prefix + ".bin", std::ios::binary);
    if (!idx || !bin) throw std::runtime_error("inputs " + prefix + ".{idx,bin} missing");
    std::string a; uint64_t nbytes, off, n = 0; Beat b;
    while (idx >> a >> nbytes >> off) {
        const uint64_t addr = std::strtoull(a.c_str(), nullptr, 16);
        if (addr % 32 || nbytes % 32) throw std::runtime_error("inputs: runs must be whole beats");
        bin.seekg(static_cast<std::streamoff>(off));
        for (uint64_t k = 0; k < nbytes; k += 32) {
            bin.read(reinterpret_cast<char*>(b.data()), 32);
            dev.wr_beat(addr + k, b.data()); ++n;
        }
    }
    return n;
}

// the first region of an area with this name in regions.json: [base, base + size)
std::pair<uint64_t, uint64_t> named_region(const std::string& dir, const std::string& name) {
    std::ifstream f(dir + "/regions.json"); std::stringstream ss; ss << f.rdbuf(); const std::string s = ss.str();
    const size_t pos = s.find("\"name\": \"" + name + "\"");
    if (pos == std::string::npos) throw std::runtime_error("region " + name + " not in regions.json");
    const size_t obj0 = s.rfind('{', pos), obj1 = s.find('}', pos);
    const std::string obj = s.substr(obj0, obj1 - obj0);
    auto num = [&](const char* key) { const size_t k = obj.find(key); return std::strtoull(obj.c_str() + obj.find(':', k) + 1, nullptr, 10); };
    return {num("\"base\""), num("\"base\"") + num("\"size\"")};
}

// the chunk's GDDR6 map (info.json "gddr6_map"): striped chunks need the bitstream's striping on
bool chunk_striped(const std::string& dir) {
    std::ifstream f(dir + "/info.json"); std::stringstream ss; ss << f.rdbuf();
    return ss.str().find("\"gddr6_map\": \"striped\"") != std::string::npos;
}
void apply_map(Dev& dev, const std::string& dir) {
    const bool want = chunk_striped(dir), got = dev.set_stripe(want);
    std::printf("gddr6 map: %s (striping switch reads %d)\n", want ? "striped" : "per-channel", got ? 1 : 0);
    if (want && !got) std::printf("  WARNING: the bitstream has no striping switch; it must be an always-striped build (s1n)\n");
}

// The soft reset (window beat N_NODE bit 1) resets the whole fabric domain, the host window included: the bridge page
// register returns to 0 and the striping switch to off.  Forget the cached page and set the chunk's map again, or the
// image goes where the nodes will not look (s1s, 2026-09-22: a striped chunk was loaded and run unstriped; small ones
// passed by accident, the whole chunk's addresses above 2 GB aliased).
void soft_reset(Dev& dev, const std::string& dir) {
    dev.wr0(dev.n_node, 0, 2u);
    for (int i = 0; i < 4; ++i) (void)dev.rd0(dev.n_node, 7);          // the top stretches the reset over 256 cycles
    dev.cur_page = ~0ull;
    apply_map(dev, dir);
}

int run(Dev& dev, const std::string& dir, const RunOpts& o) {
    apply_map(dev, dir);
    // nodes: generator index -> chip index by kind
    std::vector<std::pair<unsigned, uint64_t>> nodes;
    { std::ifstream f(dir + "/nodes.txt"); std::string kind; unsigned n; std::string base;
      std::map<std::string, unsigned> ord;
      while (f >> n >> kind >> base) {
          unsigned chip = n;
          auto it = o.kmap.find(kind);
          if (it != o.kmap.end()) chip = it->second + ord[kind];
          ++ord[kind];
          std::printf("  generator node %u (%s) -> chip node %u\n", n, kind.c_str(), chip);
          nodes.push_back({chip, std::strtoull(base.c_str(), nullptr, 16)});
      } }
    if (nodes.empty()) throw std::runtime_error("no nodes.txt");
    unsigned n_node = 0;
    for (unsigned n = 1; n <= 64; ++n) { const uint32_t v = dev.rd0(n, 7); if ((v & 0xffu) == n && ((v >> 16) & 0xffffu) >= 0x5330u && ((v >> 16) & 0xffffu) <= 0x533fu) { n_node = n; break; } }
    if (!n_node) throw std::runtime_error("host window not found behind BAR0");
    std::printf("window: N_NODE=%u id=0x%06x; chunk uses %zu nodes\n", n_node, dev.rd0(n_node, 7) >> 8, nodes.size());
    if (nodes.size() > n_node) throw std::runtime_error("the chunk needs more nodes than the bitstream has");
    Beat zero{};
    const auto flags = area_ranges(dir, "FLAGS"), io = area_ranges(dir, "IO");
    auto load_image = [&]() {
        const double t1 = now_s(); uint64_t n_in = 0;
        double t_last = t1;
        for_each_beat(dir, "data", [&](uint64_t addr, const Beat& b) {
            dev.wr_beat(addr, b.data()); ++n_in;
            if ((n_in & 0xffff) == 0 && now_s() - t_last > 10) {
                t_last = now_s();
                std::printf("  image: %.0f MB, %.1f MB/s\n", n_in * 32.0 / 1e6, n_in * 32.0 / 1e6 / (t_last - t1));
                std::fflush(stdout);
            }
            return true;
        });
        std::printf("image: %llu beats (%.1f MB) in %.1f s (%.1f MB/s), %llu page moves\n", static_cast<unsigned long long>(n_in),
                    n_in * 32.0 / 1e6, now_s() - t1, n_in * 32.0 / 1e6 / std::max(1e-9, now_s() - t1),
                    static_cast<unsigned long long>(dev.atu_moves));
    };
    if (g_stop) return 130;
    // a soft reset of the array first: right after JTAG programming the first run of the half array raised writer /
    // overrun error bits on two 32-stage chains with every beat correct, and never after a soft reset (2026-09-18)
    soft_reset(dev, dir);
    // program bases once; node masks are 64-bit (the full array has 35 nodes: halt vector bits [N_NODE-1:0], error
    // vector bits 128 +, start mask write bits 96 +, all in the summary beat N_NODE)
    uint64_t mask = 0;
    for (auto& [n, base] : nodes) {
        dev.wr0(n, 0, static_cast<uint32_t>(base & 0xffffffffu));
        dev.wr0(n, 1, static_cast<uint32_t>((base >> 32) & 0x3ffu));
        mask |= 1ull << n;
    }
    dev.wr0(n_node, 3, static_cast<uint32_t>(mask));
    if (n_node > 32) dev.wr0(n_node, 4, static_cast<uint32_t>(mask >> 32));
    auto vec64 = [&](unsigned dw) {
        uint64_t v = dev.rd0(n_node, dw);
        if (n_node > 32) v |= static_cast<uint64_t>(dev.rd0(n_node, dw + 1)) << 32;
        return v;
    };
    std::pair<uint64_t, uint64_t> act{0, 0};
    if (!o.dump_actions.empty()) act = named_region(dir, "actions");
    int fails = 0;
    for (int it = 0; it < o.repeat && !g_stop; ++it) {
        const double t0 = now_s();
        uint64_t n_exp = 0;
        if (o.scrub) {
            for_each_beat(dir, "exp", [&](uint64_t addr, const Beat&) {
                const bool is_flag = in_ranges(flags, addr);
                if (o.scrub_flags_only && !is_flag && !in_ranges(io, addr)) return true;
                // node_sync compares only a flag beat's low 32 bits: one 64-bit write clears it
                if (is_flag) dev.wr_u64(addr, 0); else dev.wr_beat(addr, zero.data());
                ++n_exp; return true;
            });
        }
        const double t_scrub = now_s() - t0;
        if (it == 0 && o.load) { load_image(); if (g_stop) return 130; }    // after the scrub: data may share a region with expected beats
        if (o.load_only) { std::printf("loaded (--load-only: not started); striping switch reads %d\n", dev.set_stripe(chunk_striped(dir)) ? 1 : 0); return 0; }
        const double t_in0 = now_s();
        uint64_t n_inp = 0;
        if (!o.inputs.empty()) n_inp = write_runs(dev, o.inputs);
        const double t_inp = now_s() - t_in0;
        (void)vec64(0);                                              // a read: every posted write has landed
        const double ts = now_s();
        dev.wr0(n_node, 0, 1u);
        uint64_t halt = 0, err = 0;
        std::vector<uint64_t> flag_addrs;
        std::vector<double> seen;
        if (it == 0 && !o.timeline.empty()) {
            for_each_beat(dir, "exp", [&](uint64_t a, const Beat&) { if (in_ranges(flags, a)) flag_addrs.push_back(a); return true; });
            seen.assign(flag_addrs.size(), -1.0);
        }
        size_t n_seen = 0;
        for (;;) {
            halt = vec64(0) & mask;
            if (halt == mask || g_stop || now_s() - ts > o.timeout_s) break;
            if (!flag_addrs.empty()) {
                // one scan of the flags not yet seen (they are monotone within a chunk); ~1 us per 32-bit PIO read
                const double t_scan = now_s() - ts;
                for (size_t i = 0; i < flag_addrs.size(); ++i)
                    if (seen[i] < 0 && dev.rd_u32(flag_addrs[i]) != 0) { seen[i] = t_scan; ++n_seen; }
            }
        }
        const double wall = now_s() - ts;
        err = vec64(4) & mask;
        if (!flag_addrs.empty()) {
            std::ofstream ft(o.timeline);
            for (size_t i = 0; i < flag_addrs.size(); ++i) ft << i << ' ' << (seen[i] < 0 ? wall : seen[i]) << '\n';
            std::printf("  timeline: %zu flags, %zu seen while running -> %s\n", flag_addrs.size(), n_seen, o.timeline.c_str());
        }
        std::printf("[run %d] scrub %llu beats %.3f s, inputs %llu beats %.3f s; %s after %.3f s: halt=0x%09llx error=0x%09llx\n",
                    it, static_cast<unsigned long long>(n_exp), t_scrub, static_cast<unsigned long long>(n_inp), t_inp,
                    halt == mask ? "ALL HALTED" : "TIMEOUT", wall, static_cast<unsigned long long>(halt), static_cast<unsigned long long>(err));
        if (it == 0 || halt != mask || err)
            for (auto& [n, base] : nodes)
                std::printf("  node %u: halt=%u error=%u error_bits=0x%02x cycles=%u (%.3f ms at 250 MHz)\n", n, dev.rd0(n, 2) & 1u,
                            (dev.rd0(n, 2) >> 1) & 1u, (dev.rd0(n, 2) >> 8) & 0xffu, dev.rd0(n, 3), dev.rd0(n, 3) / 250e3);
        if (!o.dump_actions.empty()) {
            const double ta = now_s();
            std::ofstream fa(o.dump_actions, std::ios::binary);
            Beat r;
            for (uint64_t a = act.first; a < act.second; a += 32) { dev.rd_beat(a, r.data()); fa.write(reinterpret_cast<const char*>(r.data()), 32); }
            std::printf("  actions: %llu bytes -> %s (%.3f s)\n", static_cast<unsigned long long>(act.second - act.first), o.dump_actions.c_str(), now_s() - ta);
        }
        uint64_t wrong = 0, checked = 0;
        const double tc = now_s();
        std::ofstream fw;
        if (it == 0 && !o.wrong_list.empty()) fw.open(o.wrong_list);
        if (o.check) {
            Beat r;
            for_each_beat(dir, "exp", [&](uint64_t addr, const Beat& b) {
                if (o.check_io_only && !in_ranges(io, addr) && !in_ranges(flags, addr)) return true;
                dev.rd_beat(addr, r.data()); ++checked;
                if (r != b) {
                    if (static_cast<int>(wrong) < o.show) std::printf("  MISMATCH beat %llx\n", static_cast<unsigned long long>(addr >> 5));
                    if (fw.is_open()) fw << std::hex << addr << '\n';
                    ++wrong;
                }
                return true;
            });
        }
        const bool ok = wrong == 0 && halt == mask && err == 0;
        fails += ok ? 0 : 1;
        std::printf("RESULT %s run=%d expected_beats=%llu wrong=%llu (compare %.1f s) wall=%.3f s\n", ok ? "PASS" : "FAIL", it,
                    static_cast<unsigned long long>(checked), static_cast<unsigned long long>(wrong), now_s() - tc, wall);
        std::fflush(stdout);
        if (halt != mask) break;                                     // a node never halted: do not restart on top of it
    }
    return fails ? 1 : 0;
}

// serve: the device stays open and the image loaded; one line per chunk on stdin
//   infer <inputs prefix> <actions file>   -> clear the flags, write the inputs, start, wait, dump the actions
//   quit                                    (or end of stdin: the parent went away) -> clean exit
// replies "OK chip_s=.. flags_s=.. inputs_s=.. actions_s=.. total_s=.." or "ERR <what>" on stdout, flushed.
int serve(Dev& dev, const std::string& dir, const RunOpts& o) {
    apply_map(dev, dir);
    std::vector<std::pair<unsigned, uint64_t>> nodes;
    { std::ifstream f(dir + "/nodes.txt"); std::string kind; unsigned n; std::string base;
      std::map<std::string, unsigned> ord;
      while (f >> n >> kind >> base) {
          unsigned chip = n;
          auto it = o.kmap.find(kind);
          if (it != o.kmap.end()) chip = it->second + ord[kind];
          ++ord[kind];
          nodes.push_back({chip, std::strtoull(base.c_str(), nullptr, 16)});
      } }
    if (nodes.empty()) throw std::runtime_error("no nodes.txt");
    const unsigned n_node = dev.n_node;
    if (!n_node) throw std::runtime_error("host window not found behind BAR0");
    if (nodes.size() > n_node) throw std::runtime_error("the chunk needs more nodes than the bitstream has");
    const auto flags = area_ranges(dir, "FLAGS");
    std::vector<uint64_t> flag_addrs;
    for_each_beat(dir, "exp", [&](uint64_t addr, const Beat&) { if (in_ranges(flags, addr)) flag_addrs.push_back(addr); return true; });
    const auto act = named_region(dir, "actions");
    soft_reset(dev, dir);                                           // (see run())
    uint64_t mask = 0;
    for (auto& [n, base] : nodes) {
        dev.wr0(n, 0, static_cast<uint32_t>(base & 0xffffffffu));
        dev.wr0(n, 1, static_cast<uint32_t>((base >> 32) & 0x3ffu));
        mask |= 1ull << n;
    }
    dev.wr0(n_node, 3, static_cast<uint32_t>(mask));
    if (n_node > 32) dev.wr0(n_node, 4, static_cast<uint32_t>(mask >> 32));
    auto vec64 = [&](unsigned dw) {
        uint64_t v = dev.rd0(n_node, dw);
        if (n_node > 32) v |= static_cast<uint64_t>(dev.rd0(n_node, dw + 1)) << 32;
        return v;
    };
    std::printf("READY nodes=%zu flags=%zu actions=%llu\n", nodes.size(), flag_addrs.size(),
                static_cast<unsigned long long>(act.second - act.first));
    std::fflush(stdout);
    std::string line;
    while (!g_stop && std::getline(std::cin, line)) {
        std::istringstream is(line); std::string cmd, inputs, out;
        is >> cmd >> inputs >> out;
        if (cmd == "quit") break;
        if (cmd != "infer" || out.empty()) { std::printf("ERR bad command\n"); std::fflush(stdout); continue; }
        try {
            const double t0 = now_s();
            for (uint64_t a : flag_addrs) dev.wr_u64(a, 0);
            const double t1 = now_s();
            write_runs(dev, inputs);
            (void)vec64(0);
            const double t2 = now_s();
            dev.wr0(n_node, 0, 1u);
            uint64_t halt = 0;
            for (;;) {
                halt = vec64(0) & mask;
                if (halt == mask || g_stop || now_s() - t2 > o.timeout_s) break;
            }
            const double t3 = now_s();
            const uint64_t err = vec64(4) & mask;
            if (halt != mask || err) {
                std::printf("ERR halt=0x%llx error=0x%llx after %.3f s\n", static_cast<unsigned long long>(halt),
                            static_cast<unsigned long long>(err), t3 - t2);
                std::fflush(stdout);
                if (halt != mask) break;                                 // nodes still running: do not start on top
                continue;
            }
            { std::ofstream fa(out, std::ios::binary); Beat r;
              for (uint64_t a = act.first; a < act.second; a += 32) { dev.rd_beat(a, r.data()); fa.write(reinterpret_cast<const char*>(r.data()), 32); } }
            const double t4 = now_s();
            std::printf("OK chip_s=%.4f flags_s=%.4f inputs_s=%.4f actions_s=%.4f total_s=%.4f\n", t3 - t2, t1 - t0, t2 - t1, t4 - t3, t4 - t0);
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
        }
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);
    if (argc < 2) { std::fprintf(stderr, "usage: pi0_chunk_run selftest | run <vec_dir> [--show N] [--timeout-s T] [--no-load] [--no-scrub]\n"); return 2; }
    try {
        Dev dev; dev.open();
        const std::string cmd = argv[1];
        if (cmd == "selftest") return selftest(dev);
        if (cmd == "probe") {
            std::vector<uint64_t> a;
            for (int i = 2; i < argc; ++i) a.push_back(std::strtoull(argv[i], nullptr, 0));
            return probe(dev, a);
        }
        if (cmd == "fill" && argc >= 4) {          // fill <addr> <beats> [stripe 0|1]: consecutive distinct beats, read back
            const uint64_t a0 = std::strtoull(argv[2], nullptr, 0), nb = std::strtoull(argv[3], nullptr, 0);
            if (argc >= 5) std::printf("striping switch reads %d\n", dev.set_stripe(std::atoi(argv[4]) != 0) ? 1 : 0);
            Beat w, r; uint64_t bad = 0;
            auto pat = [&](uint64_t k) { for (int j = 0; j < 32; ++j) w[static_cast<size_t>(j)] = static_cast<uint8_t>((k * 131 + static_cast<uint64_t>(j) * 7 + (k >> 8)) & 0xff); };
            const bool check_only = argc >= 6 && std::string(argv[5]) == "check";
            if (!check_only) for (uint64_t k = 0; k < nb; ++k) { pat(k); dev.wr_beat(a0 + 32 * k, w.data()); }
            for (uint64_t k = 0; k < nb; ++k) {
                pat(k); dev.rd_beat(a0 + 32 * k, r.data());
                if (r != w) { if (bad < 4) std::printf("  FILL MISMATCH at 0x%llx\n", static_cast<unsigned long long>(a0 + 32 * k)); ++bad; }
            }
            std::printf("FILL %s: %llu beats from 0x%llx, %llu wrong\n", bad ? "FAIL" : "PASS", static_cast<unsigned long long>(nb),
                        static_cast<unsigned long long>(a0), static_cast<unsigned long long>(bad));
            return bad ? 1 : 0;
        }
        if (cmd == "peek" && argc >= 3) {          // print one beat (hex, low byte first) after applying the chunk map
            if (argc >= 4) apply_map(dev, argv[3]);
            Beat r; dev.rd_beat(std::strtoull(argv[2], nullptr, 0), r.data());
            for (int k = 0; k < 32; ++k) std::printf("%02x", r[static_cast<size_t>(k)]);
            std::printf("\n");
            std::printf("page window beat words: %08x %08x %08x (bit 64 = striping)\n", dev.rd0(dev.n_node + 1, 0),
                        dev.rd0(dev.n_node + 1, 1), dev.rd0(dev.n_node + 1, 2));
            return 0;
        }
        if (cmd == "verify" && argc >= 3) {
            // read back the chunk image's beats in [lo, hi) (default: the PROG area) and compare with the image, with
            // the chunk's GDDR6 map applied first: host-side consistency of what the nodes will read
            const std::string dir = argv[2];
            const uint64_t lo = argc > 3 ? std::strtoull(argv[3], nullptr, 0) : 0x08000000ull;
            const uint64_t hi = argc > 4 ? std::strtoull(argv[4], nullptr, 0) : 0x10000000ull;
            apply_map(dev, dir);
            // verify <dir> <lo> <hi> write [wlo whi]: first write the image's beats in [wlo, whi) (default [lo, hi))
            if (argc >= 6 && std::string(argv[5]) == "write") {
                const uint64_t wlo = argc > 6 ? std::strtoull(argv[6], nullptr, 0) : lo, whi = argc > 7 ? std::strtoull(argv[7], nullptr, 0) : hi;
                uint64_t nw = 0;
                // canary: the first 64 image beats of [lo, hi), re-checked whenever the write crosses a 256 MB page
                std::vector<std::pair<uint64_t, Beat>> canary;
                const bool watch = std::getenv("PI0_VERIFY_CANARY") != nullptr;
                uint64_t last_page = ~0ull;
                for_each_beat(dir, "data", [&](uint64_t addr, const Beat& b) {
                    if (addr >= wlo && addr < whi) {
                        if (watch && canary.size() >= 64 && (addr >> 28) != last_page) {
                            Beat r; int bad = 0;
                            for (auto& c : canary) { dev.rd_beat(c.first, r.data()); if (r != c.second) ++bad; }
                            std::printf("  page 0x%llx (write at 0x%llx): canary %d / %zu wrong\n", static_cast<unsigned long long>(addr >> 28 << 28),
                                        static_cast<unsigned long long>(addr), bad, canary.size());
                            std::fflush(stdout);
                        }
                        last_page = addr >> 28;
                        dev.wr_beat(addr, b.data()); ++nw;
                        if (watch && canary.size() < 64 && addr >= lo && addr < hi) canary.push_back({addr, b});
                    }
                    return !g_stop;
                });
                std::printf("wrote %llu beats in [0x%llx, 0x%llx)\n", static_cast<unsigned long long>(nw),
                            static_cast<unsigned long long>(wlo), static_cast<unsigned long long>(whi));
            }
            uint64_t n = 0, bad = 0; Beat r;
            for_each_beat(dir, "data", [&](uint64_t addr, const Beat& b) {
                if (addr < lo || addr >= hi) return true;
                dev.rd_beat(addr, r.data()); ++n;
                if (r != b) { if (bad < 8) std::printf("  VERIFY MISMATCH beat at 0x%llx\n", static_cast<unsigned long long>(addr)); ++bad; }
                return !g_stop;
            });
            std::printf("VERIFY %s: %llu beats in [0x%llx, 0x%llx), %llu wrong\n", bad ? "FAIL" : "PASS",
                        static_cast<unsigned long long>(n), static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi),
                        static_cast<unsigned long long>(bad));
            return bad ? 1 : 0;
        }
        if ((cmd == "run" || cmd == "serve") && argc >= 3) {
            RunOpts o;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--map" && i + 1 < argc) {
                    std::stringstream ss(argv[++i]); std::string item;
                    while (std::getline(ss, item, ',')) {
                        const auto c = item.find(':');
                        if (c != std::string::npos) o.kmap[item.substr(0, c)] = static_cast<unsigned>(std::stoul(item.substr(c + 1)));
                    }
                } else if (a == "--show" && i + 1 < argc) o.show = std::atoi(argv[++i]);
                else if (a == "--timeout-s" && i + 1 < argc) o.timeout_s = std::atof(argv[++i]);
                else if (a == "--repeat" && i + 1 < argc) o.repeat = std::max(1, std::atoi(argv[++i]));
                else if (a == "--inputs" && i + 1 < argc) o.inputs = argv[++i];
                else if (a == "--dump-actions" && i + 1 < argc) o.dump_actions = argv[++i];
                else if (a == "--wrong-list" && i + 1 < argc) o.wrong_list = argv[++i];
                else if (a == "--timeline" && i + 1 < argc) o.timeline = argv[++i];
                else if (a == "--no-load") o.load = false;
                else if (a == "--load-only") o.load_only = true;
                else if (a == "--no-scrub") o.scrub = false;
                else if (a == "--no-check") o.check = false;
                else if (a == "--scrub-flags-only") o.scrub_flags_only = true;     // zero only FLAGS + IO before each start
                else if (a == "--check-io-only") o.check_io_only = true;           // compare only FLAGS + IO (actions)
                else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
            }
            return cmd == "serve" ? serve(dev, argv[2], o) : run(dev, argv[2], o);
        }
        std::fprintf(stderr, "bad arguments\n"); return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pi0_chunk_run: %s\n", e.what());
        return 1;
    }
}

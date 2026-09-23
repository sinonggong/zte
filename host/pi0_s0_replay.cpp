// Silicon replay of the node array's test vectors (S0 bring-up, docs/PI0_E2E_HARDWARE_DELIVERY_20260917.md §7).
//
// The host window (paper/rtl/pi0_chip_ctrl.sv) sits behind PCIe BAR0 (NOC[3][4]); GDDR6 is reached with the
// SDK backend's bulk path (DMA, or PIO through BAR1 with PI0_FPGA_BULK=pio, addresses below 2 MB only).
//
//   pi0_s0_replay identify                       find the window: N_NODE and the ID word
//   pi0_s0_replay reset                          soft reset of every node
//   pi0_s0_replay run <vec_dir> --node N [--repeat K] [--no-load] [--timeout-ms T] [--show M]
//       vec_dir from paper/sw/s0_vectors.py (image.hex, expect.hex, run.json): writes the image, points node N
//       at the program, starts it, waits for halt, reads every expected beat back and compares.  Prints the
//       node's own cycle count (fabric clock, 250 MHz) and the host wall time from start to halt.
//   pi0_s0_replay status --node N                the node's register beat
//
// Every 32-bit register access is one PCIe transaction; the window merges them by byte strobe.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "pi0_fpga_backend.h"
#include "pi0_fpga_sdk_backend.h"

namespace {

struct Beat {
    uint64_t addr;
    std::array<uint8_t, 32> data;
};

std::vector<Beat> load_hex(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<Beat> beats;
    std::string a, d;
    while (in >> a >> d) {
        if (d.size() != 64) { std::fprintf(stderr, "%s: bad beat %s\n", path.c_str(), d.c_str()); std::exit(2); }
        Beat b{};
        b.addr = std::strtoull(a.c_str(), nullptr, 16);
        for (int i = 0; i < 32; ++i) {                       // hex is big-endian: last two digits = byte 0
            const std::string byte = d.substr(2 * (31 - i), 2);
            b.data[static_cast<size_t>(i)] = static_cast<uint8_t>(std::strtoul(byte.c_str(), nullptr, 16));
        }
        beats.push_back(b);
    }
    return beats;                                            // the converter writes them sorted
}

struct Run { uint64_t addr; std::vector<uint8_t> bytes; };

std::vector<Run> runs_of(const std::vector<Beat>& beats) {
    std::vector<Run> runs;
    for (const Beat& b : beats) {
        if (!runs.empty() && runs.back().addr + runs.back().bytes.size() == b.addr) {
            runs.back().bytes.insert(runs.back().bytes.end(), b.data.begin(), b.data.end());
        } else {
            runs.push_back({b.addr, std::vector<uint8_t>(b.data.begin(), b.data.end())});
        }
    }
    return runs;
}

std::string hex_beat(const uint8_t* p) {
    char s[65];
    for (int i = 0; i < 32; ++i) std::snprintf(s + 2 * i, 3, "%02x", p[31 - i]);
    return s;
}

struct Window {
    pi0::RegisterAccess& r;
    unsigned n_node = 0;
    uint32_t id = 0;
    uint32_t rd(unsigned beat, unsigned dword) { return r.read32(static_cast<uint64_t>(beat) * 32 + dword * 4); }
    void wr(unsigned beat, unsigned dword, uint32_t v) { r.write32(static_cast<uint64_t>(beat) * 32 + dword * 4, v); }
    bool identify(unsigned max_nodes = 64) {
        for (unsigned n = 1; n <= max_nodes; ++n) {
            const uint32_t v = rd(n, 7);                     // bits 255:224 = ID_WORD[23:0] << 8 | N_NODE
            if ((v & 0xffu) == n && ((v >> 16) & 0xffffu) == 0x5330u) { n_node = n; id = v >> 8; return true; }
        }
        return false;
    }
};

std::string json_field(const std::string& text, const std::string& key) {
    const auto k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    const auto c = text.find(':', k);
    auto e = text.find_first_of(",}\n", c + 1);
    std::string v = text.substr(c + 1, e - c - 1);
    while (!v.empty() && (v.front() == ' ' || v.front() == '"')) v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '"')) v.pop_back();
    return v;
}

int usage() {
    std::fprintf(stderr, "usage: pi0_s0_replay identify | reset | status --node N | run <vec_dir> --node N [--repeat K] [--no-load] [--timeout-ms T] [--show M]\n");
    return 2;
}

}  // namespace

int run_main(int argc, char** argv);

// never leave main by an uncaught exception while /dev/ac7t15xx0 is open: the driver's release path can freeze the host
int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pi0_s0_replay: %s\n", e.what());
        return 1;
    }
}

int run_main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    std::string vec_dir;
    unsigned node = 0; int repeat = 1; bool load = true; unsigned timeout_ms = 20000; unsigned show = 8;
    // --swap-halves: exchange the two 128-bit halves of every beat on the way in and out.  First silicon
    // (2026-09-17): the vector node saw END in a beat whose low half held w0 and the chain ran its command
    // pairs swapped, i.e. the NoC delivers bytes 16-31 where the GDDR6 model delivers bytes 0-15.
    bool swap_halves = false;
    // --settle-ms N: wait after loading before the start (posted PIO writes to GDDR6 and the start write to the
    // window travel different NoC routes); --check-image: read the image back and compare before starting
    unsigned settle_ms = 0; bool check_image = false;
    // --scan <addr> <bytes>: after each run, read that range back and list every beat that is neither the scrub
    // value nor an expected beat (where did a misplaced write go?)
    uint64_t scan_addr = 0, scan_bytes = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](void) -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--node") node = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--no-load") load = false;
        else if (a == "--timeout-ms") timeout_ms = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--show") show = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--swap-halves") swap_halves = true;
        else if (a == "--settle-ms") settle_ms = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--check-image") check_image = true;
        else if (a == "--scan") { scan_addr = std::strtoull(next().c_str(), nullptr, 0); scan_bytes = std::strtoull(next().c_str(), nullptr, 0); }
        else if (a.rfind("--", 0) == 0) return usage();
        else vec_dir = a;
    }

    pi0::SdkBackendConfig cfg;
    cfg.bar_index = 0;
    cfg.bar_length = 0x4000;                                 // 512 beats of window
    std::unique_ptr<pi0::Backend> backend;
    try {
        backend = pi0::make_backend_from_env(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "backend: %s\n", e.what());
        return 1;
    }
    std::printf("backend: %s\n", backend->description().c_str());
    Window w{backend->registers()};
    if (!w.identify()) {
        std::printf("no node-array window found behind BAR0 (beat[7] of beats 1..64 carries no S0 ID)\n");
        for (unsigned n = 0; n < 4; ++n) std::printf("  beat %u: %08x %08x %08x %08x .. %08x\n", n, w.rd(n, 0), w.rd(n, 1), w.rd(n, 2), w.rd(n, 3), w.rd(n, 7));
        return 1;
    }
    std::printf("window: N_NODE=%u id=0x%06x\n", w.n_node, w.id);
    if (cmd == "identify") return 0;
    if (cmd == "reset") { w.wr(w.n_node, 0, 0x2u); std::printf("soft reset issued\n"); return 0; }
    if (node >= w.n_node) { std::fprintf(stderr, "--node %u out of range (N_NODE=%u)\n", node, w.n_node); return 2; }
    if (cmd == "status") {
        std::printf("node %u: prog_base=0x%03x%08x halt=%u error=%u error_bits=0x%02x cycles=%u\n", node, w.rd(node, 1) & 0x3ffu, w.rd(node, 0),
                    w.rd(node, 2) & 1u, (w.rd(node, 2) >> 1) & 1u, (w.rd(node, 2) >> 8) & 0xffu, w.rd(node, 3));
        std::printf("node %u: dbg=%08x%08x%08x%08x (decode: paper/sw/vu_dbg_decode.py)\n", node, w.rd(node, 7), w.rd(node, 6),
                    w.rd(node, 5), w.rd(node, 4));
        const uint32_t hv = w.rd(w.n_node, 0), ev = w.rd(w.n_node, 4);
        std::printf("all: halt=0x%08x error=0x%08x\n", hv, ev);
        return 0;
    }
    if (cmd != "run" || vec_dir.empty()) return usage();

    std::ifstream jf(vec_dir + "/run.json");
    std::stringstream js; js << jf.rdbuf();
    const uint64_t prog_base = std::strtoull(json_field(js.str(), "prog_base").c_str(), nullptr, 10);
    const std::string kind = json_field(js.str(), "kind");
    std::vector<Beat> image = load_hex(vec_dir + "/image.hex");
    std::vector<Beat> expect = load_hex(vec_dir + "/expect.hex");
    if (swap_halves) {
        for (auto* v : {&image, &expect})
            for (Beat& b : *v) for (int i = 0; i < 16; ++i) std::swap(b.data[static_cast<size_t>(i)], b.data[static_cast<size_t>(i + 16)]);
        std::printf("beat halves swapped on the host side\n");
    }
    const std::vector<Run> image_runs = runs_of(image), expect_runs = runs_of(expect);
    std::printf("vectors: %s kind=%s image %zu beats in %zu runs, expect %zu beats in %zu runs, prog_base=0x%llx\n",
                vec_dir.c_str(), kind.c_str(), image.size(), image_runs.size(), expect.size(), expect_runs.size(),
                static_cast<unsigned long long>(prog_base));
    pi0::BulkTransfer& bulk = backend->bulk();
    if (load) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const Run& r : image_runs) bulk.h2d(r.bytes.data(), r.bytes.size(), r.addr);
        // scrub the expected region so a stale result from an earlier run cannot pass
        for (const Run& r : expect_runs) { std::vector<uint8_t> z(r.bytes.size(), 0xa5u); bulk.h2d(z.data(), z.size(), r.addr); }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("image loaded: %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (check_image) {
            size_t bad = 0; std::vector<uint8_t> rb;
            for (const Run& r : image_runs) {
                rb.assign(r.bytes.size(), 0);
                bulk.d2h(rb.data(), rb.size(), r.addr);
                for (size_t off = 0; off < r.bytes.size(); off += 32)
                    if (std::memcmp(rb.data() + off, r.bytes.data() + off, 32) != 0) ++bad;
            }
            std::printf("image read-back: %zu of %zu beats differ\n", bad, image.size());
        }
    }
    if (scan_bytes) {                                        // the scan window starts scrubbed too
        std::vector<uint8_t> z(scan_bytes, 0xa5u); bulk.h2d(z.data(), z.size(), scan_addr);
    }
    if (settle_ms) std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    int failures = 0;
    for (int rep = 0; rep < repeat; ++rep) {
        w.wr(node, 0, static_cast<uint32_t>(prog_base & 0xffffffffu));
        w.wr(node, 1, static_cast<uint32_t>((prog_base >> 32) & 0x3ffu));
        const auto t0 = std::chrono::steady_clock::now();
        w.wr(node, 2, 1u);                                   // bit 64: start
        uint32_t st = 0; bool halted = false;
        for (;;) {
            st = w.rd(node, 2);
            if (st & 1u) { halted = true; break; }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms > timeout_ms) break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const uint32_t cycles = w.rd(node, 3);
        const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("run %d: %s error=%u error_bits=0x%02x cycles=%u (%.3f ms at 250 MHz) wall=%.3f ms\n", rep, halted ? "halted" : "TIMEOUT",
                    (st >> 1) & 1u, (st >> 8) & 0xffu, cycles, cycles / 250e3, wall_ms);
        if (!halted) { ++failures; break; }
        // compare
        size_t wrong = 0, shown = 0;
        std::vector<uint8_t> buf;
        for (const Run& r : expect_runs) {
            buf.assign(r.bytes.size(), 0);
            bulk.d2h(buf.data(), buf.size(), r.addr);
            for (size_t off = 0; off < r.bytes.size(); off += 32) {
                if (std::memcmp(buf.data() + off, r.bytes.data() + off, 32) != 0) {
                    ++wrong;
                    if (shown < show) {
                        ++shown;
                        std::printf("  MISMATCH @0x%llx\n    got %s\n    exp %s\n", static_cast<unsigned long long>(r.addr + off),
                                    hex_beat(buf.data() + off).c_str(), hex_beat(r.bytes.data() + off).c_str());
                    }
                }
            }
        }
        std::printf("run %d: %s expected_beats=%zu wrong=%zu node_error=%u dbg=%08x%08x%08x%08x\n", rep, (wrong == 0 && !((st >> 1) & 1u)) ? "PASS" : "FAIL",
                    expect.size(), wrong, (st >> 1) & 1u, w.rd(node, 7), w.rd(node, 6), w.rd(node, 5), w.rd(node, 4));
        if (scan_bytes) {
            std::vector<uint8_t> win(scan_bytes, 0);
            bulk.d2h(win.data(), win.size(), scan_addr);
            size_t listed = 0;
            for (uint64_t off = 0; off + 32 <= scan_bytes; off += 32) {
                const uint64_t a = scan_addr + off;
                bool scrub = true; for (int i = 0; i < 32; ++i) if (win[off + i] != 0xa5u) { scrub = false; break; }
                if (scrub) continue;
                bool expected = false;
                for (const Beat& b : expect) if (b.addr == a) { expected = true; break; }
                if (!expected && listed < 16) { ++listed; std::printf("  UNEXPECTED beat @0x%llx: %s\n", static_cast<unsigned long long>(a), hex_beat(win.data() + off).c_str()); }
            }
            std::printf("  scan 0x%llx +%llu: %zu unexpected non-scrub beats listed\n", static_cast<unsigned long long>(scan_addr), static_cast<unsigned long long>(scan_bytes), listed);
        }
        if (wrong != 0 || ((st >> 1) & 1u)) ++failures;
        if (rep + 1 < repeat) {                              // scrub before the next start
            for (const Run& r : expect_runs) { std::vector<uint8_t> z(r.bytes.size(), 0xa5u); bulk.h2d(z.data(), z.size(), r.addr); }
        }
    }
    return failures == 0 ? 0 : 1;
}

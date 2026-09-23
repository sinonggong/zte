#!/usr/bin/env python3
"""Chunk latency of pi0 on the node array, from MEASURED full-size layer simulations.

Each part of the chunk is one layer that ran bit-exact in RTL at its real size (tb_pi0_attn.sv, ideal GDDR6
model +nostall), repeated: 54 SigLIP layer-images (27 layers x 2 cameras), 18 prefix layers (525 tokens), 180
expert layer-steps (18 layers x 10 steps).  A layer is its stage sequence; a stage runs on one node kind
(vector node, int8 chain node, uint8 PV chain node) and its measured fabric cycles are divided over the nodes of
that kind, up to the stage's own parallelism (vector stages split by rows; chain stages by column tiles x row
slices, each row slice costing another read of the weight image, which is ignored here) [P].  Every AXI burst adds
the silicon-inferred transaction term t_txn / D (0.5 us, 8 outstanding) spread over the same nodes.
Not modelled: the patch embedding, projector and action head (small); host PCIe; NoC contention; stages of
different parts running concurrently.

Usage: chunk_latency_calibrated.py [--runs build/paper_pi0_attn_sim_nld] [--nld 4] [--f-vec 250e6] ...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import program_traffic as PT                                              # noqa: E402

REPO = HERE.parents[1]
PARTS = (  # name, vec dir, run log stem, repeats per chunk ({o} = the SigLIP out_proj form)
    ("SigLIP", "vec_vis_L0_i0{o}", "run_L0_s0_vis_i0{o}", 54),
    ("prefix", "vec_lm_L0_tall", "run_L0_s0_lm_tall", 18),
    ("expert", "vec_L0_s0_full", "run_L0_s0_full", 180),
)


def stages_of(runs: Path, vec: str, stem: str, nld: int, nlane: int = 4, n_stage: int = 16) -> list[dict]:
    dtag = ("" if nld == 1 else f"_d{nld}") + ("" if n_stage == 16 else f"_s{n_stage}")
    vec += "" if n_stage == 16 else f"_s{n_stage}"
    logs = sorted(runs.glob(f"{stem}_n{nlane}{dtag}_sb12_*_nostall.log"))
    if not logs:
        raise FileNotFoundError(f"no {stem}_n{nlane}{dtag}_sb12_*_nostall.log in {runs}")
    txt = logs[0].read_text()
    if "RESULT PASS" not in txt:
        raise RuntimeError(f"{logs[0]} did not pass")
    st = [tuple(map(int, m)) for m in re.findall(
        r"STAGE (\d+) node (\d+) fabric_cycles (\d+) rd_beats (\d+) wr_beats (\d+) \(total rd_bursts (\d+) wr_bursts (\d+)\)", txt)]
    prog = PT.decode(runs / vec)
    out, prb, pwb = [], 0, 0
    for i, (idx, node, cyc, rdb, wrb, rbt, wbt) in enumerate(st):
        s = prog[i]
        if node == 0:
            par = 10 ** 9            # vector ops split by rows; every op here has >= 51 rows, more than the nodes
        else:
            par = max(1, s["ops"].get("COLGROUP", 1))
        out.append(dict(node=node, cycles=cyc, bursts=(rbt - prb) + (wbt - pwb), par=par))
        prb, pwb = rbt, wbt
    return out


def chunk_time(parts, n_chain: int, n_pv: int, n_vec: int, f_chain: float, f_vec: float, t_txn: float,
               outstanding: int, k_chain: float = 1.0, k_vec: float = 1.0) -> dict:
    """k_chain / k_vec scale the measured node cycles (what-if: a faster loader, fewer element visits)"""
    res = {}
    total = 0.0
    for name, stages, reps in parts:
        t = 0.0
        for s in stages:
            n = {0: n_vec, 1: n_chain, 2: n_pv}[s["node"]]
            n = max(1, min(n, s["par"]))
            f = f_vec if s["node"] == 0 else f_chain
            k = k_vec if s["node"] == 0 else k_chain
            t += k * s["cycles"] / n / f + k * s["bursts"] * t_txn / (outstanding * n)
        res[name] = t * reps
        total += t * reps
    res["chunk"] = total
    return res


def chunk_time_mixed(parts16, parts32, n16: int, n32: int, n_pv: int, pv_stage: int, n_vec: int, f_chain: float,
                     f_vec: float, t_txn: float, outstanding: int, k_vec: float = 1.0) -> dict:
    """int8 chain nodes of both depths: a chain stage is shared by the two kinds in proportion to their rates, each kind
    running min(n, the stage's column groups at that depth) nodes; PV chains are all pv_stage deep.  Vector stages as in
    chunk_time.  With one kind this is chunk_time exactly."""
    res, total = {}, 0.0
    for (name, st16, reps), (_, st32, _) in zip(parts16, parts32):
        assert len(st16) == len(st32), name
        t = 0.0
        for a, b in zip(st16, st32):
            assert a["node"] == b["node"], name
            if a["node"] == 0:
                n = max(1, min(n_vec, a["par"]))
                t += k_vec * (a["cycles"] / n / f_vec + a["bursts"] * t_txn / (outstanding * n))
                continue
            kinds = ((n_pv, a if pv_stage == 16 else b),) if a["node"] == 2 else ((n16, a), (n32, b))
            rate = 0.0
            for n, sd in kinds:
                if n > 0:
                    rate += min(n, sd["par"]) / (sd["cycles"] / f_chain + sd["bursts"] * t_txn / outstanding)
            t += 1.0 / rate
        res[name] = t * reps
        total += t * reps
    res["chunk"] = total
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="calibration runs (default build/paper_pi0_attn_sim_nld, or build/paper_pi0_attn_sim_s32 with --n-stage 32)")
    ap.add_argument("--n-stage", type=int, choices=(16, 32), default=16,
                    help="chain node depth of the calibration runs (32: RUN with N_STAGE=32, WR_PIPE_EVERY 4, VALID_COPIES 4)")
    ap.add_argument("--nld", type=int, default=4)
    ap.add_argument("--t-txn", type=float, default=0.5e-6)
    ap.add_argument("--outstanding", type=int, default=8)
    ap.add_argument("--what-if", action="store_true", help="sensitivity: transaction cost, outstanding, node cycles")
    ap.add_argument("--search", action="store_true",
                    help="fastest arrays under the RLB-tile and NAP-site limits, 4- or 2-lane vector nodes, 250 / 290 MHz")
    ap.add_argument("--rlb-limit", type=float, default=55.0)
    ap.add_argument("--chain-tiles", type=int, default=0,
                    help="RLB tiles of one chain node (default: 555 at 16 stages, 951 at 32; measured slim result path: 440 / see doc)")
    ap.add_argument("--vec2-tiles", type=int, default=2063, help="RLB tiles of the 2-lane N_LD 2 vector node")
    ap.add_argument("--vec4-tiles", type=int, default=3773, help="RLB tiles of the 4-lane N_LD 4 vector node")
    ap.add_argument("--vec2-mhz", type=float, default=290.0, help="clock of the 2-lane node (routed 298.5 MHz at a 300 MHz target)")
    ap.add_argument("--breakdown", action="store_true", help="--search: split the best arrays into chain / vector / transaction time")
    ap.add_argument("--mixed", nargs=2, metavar=("RUNS16", "RUNS32"),
                    help="search arrays mixing 16- and 32-stage int8 chain nodes (2-lane vector nodes calibrated in both dirs); "
                         "tile sizes from --chain-tiles (16-stage) and --chain32-tiles")
    ap.add_argument("--chain32-tiles", type=int, default=744, help="--mixed: RLB tiles of a 32-stage chain node (slim: 744)")
    ap.add_argument("--vec4-mhz", type=float, default=0.0,
                    help="--mixed: also try 4-lane N_LD 4 vector nodes (--vec4-tiles) at this clock, calibrated on RUNS16's 4-lane runs")
    ap.add_argument("--siglip", choices=["token", "head", "legacy"], default="token",
                    help="SigLIP out_proj form of the calibration run (legacy = the per-head run before the O_PROJ option)")
    a = ap.parse_args()
    if a.mixed:
        o2 = "" if a.siglip == "legacy" else "_" + a.siglip
        p16 = [(nm, stages_of(Path(a.mixed[0]), v.format(o=o2), st.format(o=o2), 2, 2, 16), r) for nm, v, st, r in PARTS]
        p32 = [(nm, stages_of(Path(a.mixed[1]), v.format(o=o2), st.format(o=o2), 2, 2, 32), r) for nm, v, st, r in PARTS]
        t16, t32 = a.chain_tiles or 440, a.chain32_tiles
        # vector stages come from the 16-stage runs (identical in both), chain stages from each depth's runs
        vkinds = [("2-lane", a.vec2_tiles, a.vec2_mhz, p16)]
        if a.vec4_mhz:
            p16_4 = [(nm, stages_of(Path(a.mixed[0]), v.format(o=o2), st.format(o=o2), 4, 4, 16), r) for nm, v, st, r in PARTS]
            vkinds.append(("4-lane", a.vec4_tiles, a.vec4_mhz, p16_4))
        rows = []
        for vname, tv, vmhz, pv16 in vkinds:
          for n16 in range(0, 41, 2):
            for n32 in range(0, 31, 2):
                if n16 + n32 == 0:
                    continue
                for pv_stage in (16, 32):
                    for n_pv in (2, 3, 4, 6):
                        for n_vec in range(2, 21):
                            rlb = n16 * t16 + n32 * t32 + n_pv * (t16 if pv_stage == 16 else t32) + n_vec * tv
                            naps = 2 * (n16 + n32 + n_pv + n_vec) + 2
                            if 100 * rlb / 57600 > a.rlb_limit or naps > 80:
                                continue
                            r = chunk_time_mixed(pv16, p32, n16, n32, n_pv, pv_stage, n_vec, 250e6, vmhz * 1e6,
                                                 a.t_txn, a.outstanding)
                            rows.append((r["chunk"], vname, n16, n32, n_pv, pv_stage, n_vec, rlb, naps, r))
        rows.sort(key=lambda x: x[0])
        print(f"mixed-depth arrays at <= {a.rlb_limit} % RLB, <= 80 NAPs: 16-stage {t16} / 32-stage {t32} tiles; vector "
              + ", ".join(f"{n} {t} tiles at {m:.0f} MHz" for n, t, m, _ in vkinds))
        print(f"{'chunk s':>7} {'vector':>6} {'16st':>4} {'32st':>4} {'PV':>6} {'vec':>4} {'RLB':>6} {'NAPs':>4} {'SigLIP':>6} {'prefix':>6} {'expert':>6}")
        for c, vname, n16, n32, npv, pvs, nv, rlb, naps, r in rows[:10]:
            print(f"{c:7.3f} {vname:>6} {n16:4d} {n32:4d} {npv:3d}x{pvs:<2d} {nv:4d} {100 * rlb / 57600:5.1f}% {naps:4d} "
                  f"{r['SigLIP']:6.3f} {r['prefix']:6.3f} {r['expert']:6.3f}")
        return
    runs = Path(a.runs or REPO / ("build/paper_pi0_attn_sim_nld" if a.n_stage == 16 else "build/paper_pi0_attn_sim_s32"))
    o = "" if a.siglip == "legacy" else "_" + a.siglip
    try:
        parts = [(name, stages_of(runs, vec.format(o=o), stem.format(o=o), a.nld, 4, a.n_stage), reps)
                 for name, vec, stem, reps in PARTS]
    except FileNotFoundError as e:
        if not a.search:
            raise
        print(f"4-lane runs missing ({e}); 2-lane only")
        parts = None
    if parts is None:
        parts = [(name, stages_of(runs, vec.format(o=o), stem.format(o=o), 2, 2, a.n_stage), reps)
                 for name, vec, stem, reps in PARTS]
    print(f"calibration: {runs} N_LD {a.nld} chains {a.n_stage} stages; per layer: " + ", ".join(
        f"{n} chain {sum(s['cycles'] for s in st if s['node']) / 1e6:.1f} M / vector "
        f"{sum(s['cycles'] for s in st if not s['node']) / 1e6:.1f} M cycles" for n, st, _ in parts))
    if a.search:
        # [M] measured: 4-lane N_LD 4 node 3,773 tiles (wide suite 562,619 cycles); 2-lane N_LD 2 node 2,063 tiles
        # (880,860 cycles, i.e. 1.566x the 4-lane node's cycles); 16-stage chain node 555 tiles; 2 NAPs per node
        # (reads and writes separate), 80 NAP sites, 2 for the host window
        # [M] 32-stage chain node (WR_PIPE_EVERY 4, VALID_COPIES 4): 951 tiles, 37 BRAM72K, 37 MLP72, 750 MHz +0.144 ns
        rlb_chain = a.chain_tiles or (555 if a.n_stage == 16 else 951)
        vtypes = {"4-lane": (a.vec4_tiles, 1.0, parts), "2-lane": (a.vec2_tiles, 880860 / 562619, parts)}
        try:                       # the 2-lane node measured on the full-size layers themselves, when those runs exist
            o2 = "" if a.siglip == "legacy" else "_" + a.siglip
            parts2 = [(name, stages_of(runs, vec.format(o=o2), stem.format(o=o2), 2, 2, a.n_stage), reps)
                      for name, vec, stem, reps in PARTS]
            vtypes["2-lane"] = (a.vec2_tiles, 1.0, parts2)
            print("2-lane vector node: calibrated on its own full-size runs")
        except FileNotFoundError:
            print("2-lane vector node: 4-lane runs scaled by the wide-suite ratio 1.566")
        rows = []
        for vname, (vrlb, vk, vparts) in vtypes.items():
            for n_chain in range(8, 41, 2):
                for n_pv in (2, 3, 4, 6, 8):
                    if n_pv > n_chain:
                        continue
                    for n_vec in range(2, 21):
                        rlb = (n_chain + n_pv) * rlb_chain + n_vec * vrlb
                        naps = 2 * (n_chain + n_pv + n_vec) + 2
                        if 100 * rlb / 57600 > a.rlb_limit or naps > 80:
                            continue
                        for f_vec in sorted({250e6, a.vec2_mhz * 1e6}):
                            # [M] the 4-lane node closes 250 MHz only (+0.084 ns); the 2-lane node routes to 298.5 MHz
                            # at a 300 MHz target (-0.018 ns) and 287.6 MHz at 333, so 290 MHz is the realistic clock
                            if vname == "4-lane" and f_vec > 250e6:
                                continue
                            r = chunk_time(vparts, n_chain, n_pv, n_vec, 250e6, f_vec, a.t_txn, a.outstanding, 1.0, vk)
                            rows.append((r["chunk"], vname, n_chain, n_pv, n_vec, f_vec, rlb, naps, r))
        rows.sort(key=lambda x: x[0])
        print(f"fastest arrays at <= {a.rlb_limit} % RLB tiles and <= 80 NAP sites (2 per node), {a.n_stage}-stage chains "
              f"of {rlb_chain} tiles, vector nodes 2-lane {a.vec2_tiles} / 4-lane {a.vec4_tiles}:")
        print(f"{'chunk s':>7} {'vector node':>11} {'int8':>4} {'PV':>3} {'vec':>4} {'MHz':>4} {'RLB':>6} {'NAPs':>4} {'SigLIP':>6} {'prefix':>6} {'expert':>6}")
        for c, vname, nc, npv, nv, fv, rlb, naps, r in rows[:12]:
            print(f"{c:7.3f} {vname:>11} {nc:4d} {npv:3d} {nv:4d} {fv / 1e6:4.0f} {100 * rlb / 57600:5.1f}% {naps:4d} "
                  f"{r['SigLIP']:6.3f} {r['prefix']:6.3f} {r['expert']:6.3f}")
        if a.breakdown:
            for c, vname, nc, npv, nv, fv, rlb, naps, r in rows[:3]:
                vparts = vtypes[vname][2]
                vk = vtypes[vname][1]
                big = 1e30
                t_chain = chunk_time(vparts, nc, npv, nv, 250e6, big, 0.0, a.outstanding, 1.0, 0.0)["chunk"]
                t_vec = chunk_time(vparts, nc, npv, nv, big, fv, 0.0, a.outstanding, 0.0, vk)["chunk"]
                print(f"  {c:.3f} s = chain {t_chain:.3f} + vector {t_vec:.3f} + transactions {c - t_chain - t_vec:.3f}")
        return
    if a.what_if:
        print("what-if at 18 + 6 chains, 5 vector nodes (55.4 % RLB); node cycles scaled, bursts scaled with them")
        print(f"{'t_txn us':>8} {'outstanding':>11} {'chain x':>7} {'vector x':>8} {'chunk s':>7}")
        for t_txn in (0.5e-6, 0.25e-6, 0.1e-6):
            for d in (8, 32):
                for kc, kv in ((1, 1), (0.5, 1), (1, 0.5), (0.5, 0.5), (0.5, 0.35)):
                    r = chunk_time(parts, 18, 6, 5, 250e6, 250e6, t_txn, d, kc, kv)
                    print(f"{t_txn * 1e6:8.2f} {d:11d} {kc:7.2f} {kv:8.2f} {r['chunk']:7.3f}")
        return
    # [M] routed blocks: 16-stage chain node 555 RLB tiles (colpar_chain_node, 750 / 250 MHz); 4-lane vector node at
    # SLOT_BITS 12, ACE 10.5.2: N_LD 1 3,436 / N_LD 4 3,714.  Summing overestimates the placed array by ~5.5 % (c32v4).
    rlb_chain, rlb_vec = 555, (3714 if a.nld == 4 else 3436)
    print(f"{'int8':>5} {'PV':>3} {'vec':>4} {'vec MHz':>7} {'RLB sum':>8} {'SigLIP s':>8} {'prefix s':>8} {'expert s':>8} {'chunk s':>7}")
    for n_chain, n_pv, n_vec in ((24, 8, 4), (16, 4, 5), (12, 4, 6), (9, 3, 7), (18, 6, 5), (24, 8, 6)):
        rlb = (n_chain + n_pv) * rlb_chain + n_vec * rlb_vec
        for f_vec in (250e6, 333e6):
            r = chunk_time(parts, n_chain, n_pv, n_vec, 250e6, f_vec, a.t_txn, a.outstanding)
            print(f"{n_chain:5d} {n_pv:3d} {n_vec:4d} {f_vec / 1e6:7.0f} {100 * rlb / 57600:7.1f}% {r['SigLIP']:8.3f} "
                  f"{r['prefix']:8.3f} {r['expert']:8.3f} {r['chunk']:7.3f}")

if __name__ == "__main__":
    main()

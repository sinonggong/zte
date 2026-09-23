#!/usr/bin/env python3
"""Calibrated chunk time of a generated chunk (pi0_chunk_program.py --out DIR) on its real node assignment.

Every stage of the chunk is a stage of one of the per-layer runs measured bit-exact in RTL (the goldens' stage order
is the builders' order): its fabric cycles and its AXI bursts come from that run's log (tb_pi0_attn.sv STAGE lines,
ideal GDDR6 +nostall, 2-lane N_LD 2 vector node).  The stage was split over the array's nodes (parts.json, with each
part's share of the stage's work), so its time is the slowest part: share x cycles / f_kind + share x bursts x
t_txn / outstanding.  The chunk is the sum over stages (they are sequenced by flags).  Reported beside it: the
same chunk on one node of each kind (no split) and the split's balance.

Usage: pi0_chunk_time.py DIR [--runs16 build/paper_pi0_attn_sim_nld] [--runs32 .../paper_pi0_attn_sim_s32]
                             [--f-array 725e6] [--f-fabric 250e6] [--f-vec 333e6] [--t-txn 0.08e-6] [--outstanding 8]
The chain stages' measured cycles are in fabric cycles at the run's 3:1 array / fabric ratio; --f-fabric scales them
(the array clock enters through the ratio the chain node was measured at).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# layer prefix in parts.json -> (log stem, the measured run's stage count)
LOGS = {"vis.": "run_L0_s0_vis_i0_token_n2_d2{s}_sb12_*_nostall.log",
        "lm.": "run_L0_s0_lm_tall_n2_d2{s}_sb12_*_nostall.log",
        "exp.": "run_L0_s0_full_n2_d2{s}_sb12_*_nostall.log",
        "head": "run_L0_s0_head_n2_d2{s}_sb12_*.log",
        "patch": "run_L0_s0_vision_i0_n2_d2{s}_sb12_*.log",
        "proj": "run_L0_s0_vision_i0_n2_d2{s}_sb12_*.log"}
# builder stage name -> measured stage index of its layer type (the goldens' order); None = not in the run
STAGE_IDX = {
    "vis.": dict(ln1=0, qkv_gemm=1, dequant_quant=2, qk_gemm=3, logits_softmax=4, pv_gemm=5, context=6, o_gemm=7,
                 o_dequant_residual=8, ln2=9, fc1_gemm=10, gelu_quant=11, fc2_gemm=12, fc2_dequant_residual=13),
    "gemma": dict(rms_quant=0, qkv_gemm=1, dequant_rope_quant=2, qk_gemm=3, logits_softmax=4, pv_gemm=5, context=6,
                  o_gemm=7, o_dequant_residual=8, rms2_quant=9, gate_up_gemm=10, geglu_quant=11, down_gemm=12,
                  down_dequant_residual=13),
    "head": dict(xt_quant=0, in_proj_gemm=1, in_proj_dequant_quant=2, mlp_in_gemm=3, mlp_in_dequant_silu_quant=4,
                 mlp_out_gemm=5, mlp_out_dequant=6, final_norm_quant=6, out_proj_gemm=7, out_proj_dequant_euler=8,
                 out_proj_dequant=8, euler=8),
    "patch": dict(patch_quant=0, patch_gemm=1, patch_dequant_pos=2),
    "proj": dict(post_ln_quant=3, proj_gemm=4, proj_dequant=5),
}


def measured(runs: Path, stem: str) -> list[dict]:
    logs = sorted(runs.glob(stem))
    if not logs:
        raise FileNotFoundError(f"no {stem} in {runs}")
    txt = logs[0].read_text()
    if "RESULT PASS" not in txt:
        raise RuntimeError(f"{logs[0]} did not pass")
    st = [tuple(map(int, m)) for m in re.findall(
        r"STAGE (\d+) node (\d+) fabric_cycles (\d+) rd_beats (\d+) wr_beats (\d+) \(total rd_bursts (\d+) wr_bursts (\d+)\)", txt)]
    out, prb, pwb = [], 0, 0
    for idx, node, cyc, rdb, wrb, rbt, wbt in st:
        out.append(dict(node=node, cycles=cyc, bursts=(rbt - prb) + (wbt - pwb), rd_beats=rdb, wr_beats=wrb))
        prb, pwb = rbt, wbt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--runs16", default="/home/sngong/projects/pi0_achronix/build/paper_pi0_attn_sim_nld")
    ap.add_argument("--runs32", default="/home/sngong/projects/pi0_achronix-wt/pi0-throughput/build/paper_pi0_attn_sim_s32")
    ap.add_argument("--f-fabric", type=float, default=250e6)
    ap.add_argument("--f-vec", type=float, default=333.33e6)
    ap.add_argument("--f-array", type=float, default=725e6, help="array clock: chain stages run at min(f_fabric, f_array / 3) (measured at a 3:1 ratio)")
    ap.add_argument("--t-txn", type=float, default=0.08e-6)
    ap.add_argument("--outstanding", type=int, default=8)
    ap.add_argument("--t-wait", type=float, default=0.3e-6, help="latency from a POST to the waiting node seeing it")
    ap.add_argument("--scale", default="", help="what-if: stage name=factor,... on the measured cycles and bursts "
                                                 "(e.g. geglu_quant=0.5 for a fused DEQUANT+DEQUANT+GEGLU pass)")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dir)
    parts = json.loads((d / "parts.json").read_text())
    info = json.loads((d / "info.json").read_text())
    n_stage = info["array"]["n_stage"]
    # mixed arrays (--n-deep): a node's cost counts its cycles whatever its depth, but a 32-stage tile of cost c
    # retires the columns a 16-stage node needs 2c for; stage cycles were measured at depth n_stage, so a part's
    # share is its cost over the stage's work in n_stage-equivalent cost units
    node_depth = info["array"].get("node_depth")
    runs = Path(a.runs16 if n_stage == 16 else a.runs32)
    stag = "" if n_stage == 16 else "_s32"
    cache: dict[str, list[dict]] = {}
    what_if = {k: float(v) for k, v in (kv.split("=") for kv in a.scale.split(",") if kv)}

    def run_of(layer: str):
        for pre, stem in LOGS.items():
            if layer.startswith(pre):
                if pre not in cache:
                    try:
                        cache[pre] = measured(runs, stem.format(s=stag))
                    except (FileNotFoundError, RuntimeError):
                        # the head / vision runs were only made at 32 stages; fall back across depths
                        other = Path(a.runs32 if n_stage == 16 else a.runs16)
                        cache[pre] = measured(other, stem.format(s="_s32" if n_stage == 16 else ""))
                key = pre if pre in STAGE_IDX else "gemma"
                return cache[pre], STAGE_IDX[key]
        raise KeyError(layer)

    per_part = {"SigLIP": 0.0, "vision ends": 0.0, "prefix": 0.0, "expert": 0.0, "head": 0.0}
    by_kind = {"vector": 0.0, "int8": 0.0, "uint8": 0.0, "txn": 0.0}
    single = 0.0
    worst_balance = []
    part_time: dict[tuple[int, int], tuple] = {}             # (stage, node) -> (seconds, waits)
    by_name: dict[tuple[str, str, str], list] = {}         # (group, kind, stage name) -> [seconds, count]
    # sub-stages: pi0_chunk_program.py cuts a vector stage whose parts would race into "<name>#0", "<name>#1", ...;
    # each gets the measured stage's cycles and bursts in proportion to its share of the stage's work
    sub_total: dict[tuple[str, str], int] = {}
    base_name = lambda n: n.split("@")[0].split("#")[0]            # noqa: E731  (row blocks "@r", hazard cuts "#")
    for st in parts:
        if "#" in st["name"] or "@" in st["name"]:
            k = (st["layer"], base_name(st["name"]))
            sub_total[k] = sub_total.get(k, 0) + sum(p["cost"] for p in st["parts"])
    for st in parts:
        total_cost = sum(p["cost"] for p in st["parts"]) or 1
        sub_scale = 1.0
        if "#" in st["name"] or "@" in st["name"]:
            base = base_name(st["name"])
            sub_scale = total_cost / (sub_total[(st["layer"], base)] or 1)
            st = dict(st, name=base)
        if st["name"].startswith("x_exp") or st["layer"].startswith("x_exp"):
            # the suffix-row ADD pass has no measured stage: a vector node moves ~1 element per lane per cycle
            cyc, bursts = total_cost / 2, total_cost * 4 / 512
        else:
            run, idx = run_of(st["layer"])
            i = idx[st["name"]]
            m = run[i]
            cyc, bursts = m["cycles"], m["bursts"]
        if st["layer"].startswith("head") and st["name"] in ("mlp_out_dequant", "final_norm_quant", "out_proj_dequant", "euler"):
            cyc, bursts = cyc * 0.5, bursts * 0.5             # the golden's stage 6 (8) holds both halves
        cyc, bursts = cyc * sub_scale * st.get("fuse", 1.0), bursts * sub_scale * st.get("fuse", 1.0)
        if st["name"] in what_if:
            cyc, bursts = cyc * what_if[st["name"]], bursts * what_if[st["name"]]
        # chain stages were measured at array = 3 x fabric; with a slower array (s1m: 250 / 114.6 = 2.18) the row passes
        # stretch, so the chain runs at the lower of the fabric clock and a third of the array clock (a bound: the
        # loads inside the measured cycles do run on the fabric clock)
        f = a.f_vec if st["kind"] == "vector" else min(a.f_fabric, a.f_array / 3.0)
        fac = [(node_depth[p["node"]] / n_stage if node_depth and st["kind"] != "vector" and node_depth[p["node"]] else 1.0)
               for p in st["parts"]]
        total = sum(p["cost"] * k for p, k in zip(st["parts"], fac)) or 1
        shares = [p["cost"] / total for p in st["parts"]]
        t_parts = [s * cyc / f + s * bursts * a.t_txn / a.outstanding for s in shares]
        for p_, tp in zip(st["parts"], t_parts):
            part_time[(st["stage"], p_["node"])] = (tp, p_.get("waits"))
        t_stage = max(t_parts)
        t_one = cyc / f + bursts * a.t_txn / a.outstanding
        single += t_one
        by_kind[st["kind"]] += max(s * cyc / f for s in shares)
        by_kind["txn"] += max(s * bursts * a.t_txn / a.outstanding for s in shares)
        grp = ("SigLIP" if st["layer"].startswith("vis.") else "prefix" if st["layer"].startswith("lm.") else
               "expert" if st["layer"].startswith("exp.") else "head" if st["layer"].startswith("head") or st["layer"].startswith("x_exp")
               else "vision ends")
        per_part[grp] += t_stage
        e = by_name.setdefault((grp, st["kind"], st["name"]), [0.0, 0])
        e[0] += t_stage
        e[1] += 1
        n = len(st["parts"])
        worst_balance.append((max(shares) * n, st["stage"], st["layer"], st["name"], n))
    chunk = sum(per_part.values())
    # list schedule on the real WAIT sets: a node runs its parts in stage order; a part starts when its node is free
    # and every part it waits for has ended (+ one poll per WAIT); the makespan is the chunk time with overlap
    sched = None
    if all(v[1] is not None for v in part_time.values()):
        free: dict[int, float] = {}
        end: dict[tuple[int, int], float] = {}
        n_w = 0
        for (s_, n_), (tp, waits) in sorted(part_time.items()):
            start = free.get(n_, 0.0)
            for s2, n2 in waits:
                start = max(start, end[(s2, n2)] + a.t_wait)
                n_w += 1
            end[(s_, n_)] = start + tp
            free[n_] = end[(s_, n_)]
        sched = max(end.values()) if end else 0.0
        # every part's scheduled end in flag order (pi0_chunk_program emits one flag per part, stages in order):
        # compare with a silicon timeline (pi0_chunk_run --timeline)
        sched_end = [end[(st["stage"], p_["node"])] for st in parts for p_ in st["parts"]]
    worst_balance.sort(reverse=True)
    top = sorted(((v[0], v[1], g, k, n) for (g, k, n), v in by_name.items()), reverse=True)
    res = dict(schedule_s=sched, sched_end=sched_end if sched is not None else None, top_stages=[dict(group=g, kind=k, name=n, seconds=s, count=c) for s, c, g, k, n in top[:25]],
               chunk_s=chunk, single_node_s=single, per_part=per_part, by_kind=by_kind,
               array=info["array"], sizes=info["chunk"], f_fabric=a.f_fabric, f_vec=a.f_vec, t_txn=a.t_txn,
               outstanding=a.outstanding, worst_balance=worst_balance[:5])
    print(f"chunk {chunk:.3f} s (stage barrier sum)" + (f", {sched:.3f} s scheduled on the WAIT sets ({info['array'].get('sync', 'barrier')})"
                                                         if sched is not None else "") + f" on {info['array']} for {info['chunk']}")
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in per_part.items()))
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in by_kind.items()) + f"; one node per kind: {single:.3f} s")
    print("  top stages: " + "; ".join(f"{g}:{n} {s:.3f} s x{c}" for s, c, g, k, n in top[:8]))
    print("  least balanced stages (max share x parts): " + "; ".join(f"{b:.2f} {l}:{n}({p} parts)" for b, s, l, n, p in worst_balance[:3]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

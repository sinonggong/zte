#!/usr/bin/env python3
"""GDDR6 traffic of lowered node programs, decoded from a vector directory written by pi0_attn_golden.py or
pi0_siglip_golden.py (mem_in.hex + stages.txt), per stage and per node kind -- and, given the testbench log, the
measured beats next to it.

What a program reads and writes follows from its commands alone:
  chain   LOAD      n_regions x n_words x 16 bytes (stage images)
          LOADX     n_regions x n_segs x n_words x 16 bytes
          COLGROUP  T rows x W words x 16 bytes of feeder, and T x 16 P int32 sums written
          program   16 bytes per command (END included)
  vector  per record, every operand slot by shape: E = R x L, R = R, C = L, K = 0 elements, times the slot's format
          width (bf16 2, fp32 4, int32 4, summary 8 bytes; the mask slot reads 2 bytes); writes R x L elements at
          the op's output width (QUANT / SMAX_Q8 1, ROPE_A / EULER / LN_STAT / DEQUANT fp32 4, the rest 2; LN_STAT
          one element per row) plus one 8-byte summary per row; program 16 bytes per word
Checked against tb_pi0_attn.sv's per-stage beat counters: the full-size expert layer (42.11 MB read, 9.26 MB
written) and the prefix layer at 160 tokens (537.02 / 63.97 MB) agree on every stage.

Usage: program_traffic.py VEC_DIR [RUN_LOG] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WIDTH = {0: 2, 1: 4, 2: 4, 3: 8}            # desc [45:44]: bf16, fp32, int32, summary (bytes)
OUT_W = {1: 0, 3: 4, 5: 4, 10: 0, 12: 1, 13: 1, 15: 4}   # RMS_STAT, LN_STAT, ROPE_A, SMAX_SUM, SMAX_Q8, QUANT, EULER
VEC_OPS = {0: "ADD", 1: "RMS_STAT", 2: "RMS_APPLY", 3: "LN_STAT", 4: "LN_APPLY", 5: "ROPE_A", 6: "ROPE_B", 7: "GELU",
           8: "GEGLU", 9: "SILU", 10: "SMAX_SUM", 11: "SMAX_OUT", 12: "SMAX_Q8", 13: "QUANT", 14: "DEQUANT", 15: "EULER"}
CHAIN_OPS = {1: "LOAD", 2: "TILE", 3: "OUT", 4: "FLUSH", 5: "LOADX", 6: "COLGROUP", 7: "WAIT", 8: "POST"}


def load_image(vec: Path) -> dict[int, int]:
    beats = {}
    with open(vec / "mem_in.hex") as f:
        for line in f:
            a, v = line.split()
            beats[int(a, 16)] = int(v, 16)
    return beats


def decode(vec: Path) -> list[dict]:
    beats = load_image(vec)

    def word(addr: int, i: int) -> int:
        return (beats.get((addr >> 5) + i // 2, 0) >> (128 * (i % 2))) & ((1 << 128) - 1)

    stages = []
    for line in open(vec / "stages.txt"):
        node, base = int(line.split()[0]), int(line.split()[1], 16)
        rd = wr = elems = passes = 0
        ops: dict[str, int] = {}
        k = 0
        if node == 0:
            while word(base, k):
                w = [word(base, k + j) for j in range(6)]
                op, R, L = w[0] & 0xF, (w[0] >> 40) & 0xFFFFF, (w[0] >> 60) & 0xFFFF
                descs = [(w[1] >> 42), w[2], w[2] >> 48, w[3], w[3] >> 48, w[4], w[4] >> 48]
                for si, d in enumerate(descs):
                    d &= (1 << 48) - 1
                    shape, fmt = (d >> 42) & 3, (d >> 44) & 3
                    n = {0: R * L, 1: R, 2: L, 3: 0}[shape]
                    rd += n * (2 if si == 6 else WIDTH[fmt])
                ow = (4 if (w[0] >> 6) & 1 else 2) if op == 14 else OUT_W.get(op, 2)
                wr += (R if op == 3 else R * L) * ow + R * 8
                elems += R * L
                ops[VEC_OPS[op]] = ops.get(VEC_OPS[op], 0) + 1
                k += 6
            rd += (k + 1) * 16
        else:
            while True:
                c = word(base, k)
                op = c >> 124
                if op == 0:
                    break
                if op == 1:
                    rd += ((c >> 47) & 31) * ((c >> 52) & 1023) * 16
                elif op == 5:
                    rd += ((c >> 47) & 31) * ((c >> 52) & 1023) * ((c >> 62) & 1023) * 16
                elif op == 6:
                    T, W, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 81) & 1023
                    rd += T * W * 16
                    wr += T * 16 * P * 4
                    passes += T * P
                ops[CHAIN_OPS.get(op, str(op))] = ops.get(CHAIN_OPS.get(op, str(op)), 0) + 1
                k += 1
            rd += (k + 1) * 16
        stages.append(dict(node=node, read=rd, write=wr, commands=k if node else k // 6, vector_elements=elems,
                           row_passes=passes, ops=ops))
    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vec")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--json")
    a = ap.parse_args()
    st = decode(Path(a.vec))
    meas = []
    if a.log:
        meas = [tuple(map(int, m)) for m in
                re.findall(r"STAGE (\d+) node (\d+) fabric_cycles (\d+) rd_beats (\d+) wr_beats (\d+)",
                           Path(a.log).read_text())]
    kind = {0: "vector", 1: "int8 chain", 2: "uint8 chain"}
    print(f"{'st':>2} {'node':>11} {'read MB':>9} {'meas':>9} {'write MB':>9} {'meas':>9}  ops")
    for i, s in enumerate(st):
        m = meas[i] if i < len(meas) else None
        print(f"{i:2d} {kind[s['node']]:>11} {s['read'] / 1e6:9.3f} {m[3] * 32 / 1e6 if m else float('nan'):9.3f} "
              f"{s['write'] / 1e6:9.3f} {m[4] * 32 / 1e6 if m else float('nan'):9.3f}  {s['ops']}")
    tot = {k: sum(s[k] for s in st) for k in ("read", "write", "vector_elements", "row_passes")}
    for nk in (0, 1, 2):
        tot[f"read_{kind[nk]}"] = sum(s["read"] for s in st if s["node"] == nk)
        tot[f"write_{kind[nk]}"] = sum(s["write"] for s in st if s["node"] == nk)
    line = f"total read {tot['read'] / 1e6:.2f} MB, write {tot['write'] / 1e6:.2f} MB"
    if meas:
        line += (f"; measured read {sum(m[3] for m in meas) * 32 / 1e6:.2f} MB, "
                 f"write {sum(m[4] for m in meas) * 32 / 1e6:.2f} MB, fabric cycles {sum(m[2] for m in meas):,}")
    print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(stages=st, total=tot), indent=1))


if __name__ == "__main__":
    main()

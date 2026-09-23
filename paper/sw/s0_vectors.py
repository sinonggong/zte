#!/usr/bin/env python3
"""Turn a node test-vector set into the S0 silicon replay format (host/pi0_s0_replay.cpp):

  image.hex    "<byte address hex> <256-bit beat hex>" per line: everything the host writes into GDDR6 first
  expect.hex   same format: every beat the node must have written when it halts
  run.json     {"prog_base": <int>, "kind": "chain"|"vector", "source": <dir>}

Sources are the goldens' own output directories, so the silicon replays exactly what the Verilator gates ran:
  --chain <dir>   paper/sw/colpar_tile_golden.py --n-stage N (act.memh / act_b.memh / wt.memh / prog.memh /
                  exp_prog.memh / npulses_prog.txt), laid out as tb_colpar_node_prog.sv lays them out:
                  BASE_PROG 0x0800_0000 (two 128-bit commands per beat), BASE_ACT 0x1000_0000 and BASE_ACT2
                  0x1800_0000 (two 144-bit feeder words per beat), BASE_WT 0x2000_0000 (stage s at s * 2^ADDR_BITS
                  words), records at BASE_OUT 0x3000_0000 (N_STAGE int32 per record, beat padded).
  --vector <dir>  paper/sw/vu_node_golden.py (mem_in.hex / mem_exp.hex beat-index lines, prog_base.txt).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

BASE_PROG, BASE_ACT, BASE_ACT2, BASE_WT, BASE_OUT = 0x0800_0000, 0x1000_0000, 0x1800_0000, 0x2000_0000, 0x3000_0000
MASK256 = (1 << 256) - 1


def read_memh(path: Path) -> list[int]:
    out = []
    for line in path.read_text().split("\n"):
        line = line.split("//")[0].strip()
        if line:
            out.append(int(line, 16))
    return out


def raw16(w: int) -> int:
    """a 144-bit UG086 BRAM word {8'h0, b15..b8, 8'h0, b7..b0} -> its 16 raw bytes (the GDDR6 layout the tile
    loader expects: colpar_tile_loader.sv, tb_colpar_node_prog.sv beat_at)"""
    return (((w >> 72) & ((1 << 64) - 1)) << 64) | (w & ((1 << 64) - 1))


def pairs_to_beats(words: list[int], base: int, width: int, beats: dict[int, int]) -> None:
    """two words per 256-bit beat, word 2i in the low half; 144-bit BRAM words go in as their 16 raw bytes"""
    for i in range(0, len(words), 2):
        lo = words[i]
        hi = words[i + 1] if i + 1 < len(words) else 0
        if width == 144:
            lo, hi = raw16(lo), raw16(hi)
        beats[base + 16 * i] = ((hi << 128) | lo) & MASK256


def chain(d: Path, n_stage: int, addr_bits: int):
    nw = 1 << addr_bits
    act, act_b, wt, prog, exp = (read_memh(d / f) for f in ("act.memh", "act_b.memh", "wt.memh", "prog.memh", "exp_prog.memh"))
    npulses = int((d / "npulses_prog.txt").read_text().split()[0])
    assert len(act) == nw and len(act_b) == nw and len(wt) == n_stage * nw, (len(act), len(wt), nw, n_stage)
    image: dict[int, int] = {}
    pairs_to_beats(prog, BASE_PROG, 128, image)
    pairs_to_beats(act, BASE_ACT, 144, image)
    pairs_to_beats(act_b, BASE_ACT2, 144, image)
    pairs_to_beats(wt, BASE_WT, 144, image)
    bpr = (32 * n_stage + 255) // 256
    expect: dict[int, int] = {}
    for k in range(npulses):
        rec = 0
        for s in range(n_stage):
            rec |= (exp[n_stage * k + s] & 0xffff_ffff) << (32 * s)
        for b in range(bpr):
            expect[BASE_OUT + (k * bpr + b) * 32] = (rec >> (256 * b)) & MASK256
    return image, expect, BASE_PROG


def vector(d: Path):
    def load(path: Path) -> dict[int, int]:
        out = {}
        for line in path.read_text().split("\n"):
            if line.strip():
                idx, beat = line.split()
                out[int(idx, 16) * 32] = int(beat, 16)
        return out
    image = load(d / "mem_in.hex")
    expect = load(d / "mem_exp.hex")
    prog_base = int((d / "prog_base.txt").read_text().split()[0], 16)
    return image, expect, prog_base


MASK42 = (1 << 42) - 1
MASK128 = (1 << 128) - 1
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vu_node_golden import SH_K  # the constant-operand shape: its "base" is a value, not an address
except Exception:  # pragma: no cover
    SH_K = 3


def relocate(image: dict[int, int], expect: dict[int, int], prog_base: int, kind: str, new_base: int,
             align: int = 0x1000, window: int = 0x200000):
    """Move every region below new_base + ~2 MB (the PIO window: BAR1 = NoC 0..2 MB) and patch the program's
    address fields the same way.  Regions = clusters of addresses sharing addr >> 26 (the goldens put PROG /
    ACT / ACT2 / WT / OUT / IN 64 MB or more apart); each cluster moves as a block, so alignment inside a
    region is unchanged.  Chain commands: LOAD (1), OUT (3), LOADX (5), COLGROUP (6) carry an address in
    [41:0].  Vector records (6 x 128-bit): w0[117:76] out_base, w1[41:0] sum_base, w1[89:42] x, w2/w3/w4
    two 48-bit operand descriptors each ([41:0] address unless shape == SH_K)."""
    addrs = sorted(set(image) | set(expect))
    clusters: dict[int, list[int]] = {}
    for a in addrs:
        clusters.setdefault(a >> 26, []).append(a)
    table = []                       # (lo, hi_exclusive, new_lo)
    p = new_base
    for key in sorted(clusters):
        lo, hi = min(clusters[key]), max(clusters[key]) + 32
        table.append((lo, hi, p))
        p += ((hi - lo) + align - 1) // align * align
    if p > window:
        raise SystemExit(f"relocated image ends at {p:#x}, beyond the {window:#x} PIO window")

    def f(a: int) -> int:
        for lo, hi, nlo in table:
            if lo <= a < hi:
                return nlo + (a - lo)
        return a

    def fdesc(d: int) -> int:        # 48-bit operand descriptor
        if ((d >> 42) & 3) == SH_K:
            return d
        return (d & ~MASK42) | f(d & MASK42)

    prog_lo, prog_hi = next((lo, hi) for lo, hi, _ in table if lo <= prog_base < hi)
    words = []
    for a in range(prog_lo, prog_hi, 32):
        b = image.get(a, 0)
        words += [b & MASK128, b >> 128]
    if kind == "chain":
        for i, w in enumerate(words):
            if (w >> 124) in (1, 3, 5, 6):
                words[i] = (w & ~MASK42) | f(w & MASK42)
    else:
        for i, w in enumerate(words):
            k = i % 6
            if k == 0 and (w >> 124) == 0xA:
                words[i] = (w & ~(MASK42 << 76)) | (f((w >> 76) & MASK42) << 76)
            elif k == 1:                                  # w1 = sum_base | (x << 42): the x descriptor starts at bit 42
                words[i] = (w & ~((1 << 90) - 1)) | f(w & MASK42) | (fdesc((w >> 42) & 0xFFFF_FFFF_FFFF) << 42)
            elif k in (2, 3, 4):
                d0, d1 = w & 0xFFFF_FFFF_FFFF, (w >> 48) & 0xFFFF_FFFF_FFFF
                words[i] = (w & ~((1 << 96) - 1)) | fdesc(d0) | (fdesc(d1) << 48)
    new_image = {f(a): v for a, v in image.items() if not (prog_lo <= a < prog_hi)}
    for j in range(0, len(words), 2):
        new_image[f(prog_lo + 16 * j)] = (words[j + 1] << 128) | words[j]
    new_expect = {f(a): v for a, v in expect.items()}
    assert len(new_image) == len(image) and len(new_expect) == len(expect)
    return new_image, new_expect, f(prog_base), table


def write_tb_format(out: Path, image: dict[int, int], expect: dict[int, int], prog_base: int) -> None:
    """the vu_node testbench's own files, so a relocated vector set can be re-simulated"""
    with (out / "mem_in.hex").open("w") as fh:
        for a in sorted(image):
            fh.write(f"{a >> 5:x} {image[a]:064x}\n")
    with (out / "mem_exp.hex").open("w") as fh:
        for a in sorted(expect):
            fh.write(f"{a >> 5:x} {expect[a]:064x}\n")
    (out / "prog_base.txt").write_text(f"{prog_base:x}\n")


def write_hex(path: Path, beats: dict[int, int]) -> None:
    with path.open("w") as f:
        for a in sorted(beats):
            f.write(f"{a:x} {beats[a]:064x}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chain", type=Path)
    g.add_argument("--vector", type=Path)
    ap.add_argument("--n-stage", type=int, default=16)
    ap.add_argument("--addr-bits", type=int, default=9)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=lambda s: int(s, 0), default=0x200000, help="BAR1 PIO window bytes (2 MB; 512 MB with the big BAR1)")
    ap.add_argument("--relocate", type=lambda s: int(s, 0), default=None,
                    help="move every region below this base (e.g. 0x10000) for the 2 MB PIO window; patches the program")
    a = ap.parse_args()
    if a.chain:
        image, expect, prog_base = chain(a.chain, a.n_stage, a.addr_bits)
        kind, src = "chain", a.chain
    else:
        image, expect, prog_base = vector(a.vector)
        kind, src = "vector", a.vector
    table = []
    if a.relocate is not None:
        image, expect, prog_base, table = relocate(image, expect, prog_base, kind, a.relocate, window=a.window)
    a.out.mkdir(parents=True, exist_ok=True)
    if kind == "vector":
        write_tb_format(a.out, image, expect, prog_base)
    for lo, hi, nlo in table:
        print(f"  region {lo:#011x}-{hi:#011x} ({hi - lo:7d} B) -> {nlo:#08x}")
    write_hex(a.out / "image.hex", image)
    write_hex(a.out / "expect.hex", expect)
    (a.out / "run.json").write_text(json.dumps({"prog_base": prog_base, "kind": kind, "source": str(src),
                                                "image_beats": len(image), "expect_beats": len(expect)}, indent=1) + "\n")
    print(f"{kind}: image {len(image)} beats ({32 * len(image)} B), expect {len(expect)} beats, prog_base {prog_base:#x} -> {a.out}")


if __name__ == "__main__":
    main()

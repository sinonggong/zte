#!/usr/bin/env python3
"""D2: the whole-chunk GDDR6 map for the pi0 node array (docs/PI0_E2E_HARDWARE_DELIVERY_20260917.md §4 D2).

One `ChunkMem` holds every region of a chunk in the VP815's GDDR6, in five areas.  The GDDR6 is NOT a flat space:
16 channels (8 controllers x 2), channel k at NoC address k << 33, each an 8 GB window with 2 GB of storage
(Clamshell-x8, 16 Gb parts: 32 GB in all); an address above 2 GB inside a window aliases the one 2 GB below it
(measured on silicon 2026-09-18, pi0_chunk_run probe).  So every area is a list of windows inside channels, a region
never crosses a window, and STATIC / SCRATCH regions go round-robin over all 16 channels so concurrent nodes read
from different channels (one channel is ~32 GB/s; the chunk moves ~44 GB):

    PROG     node programs (chain commands 128-bit, vector records 6 x 128-bit): written once
    STATIC   INT8 weight stage images per column tile, fp32 scale / bias / gain vectors, RoPE cos / sin, the
             SigLIP position table, masks: written once
    IO       per-chunk inputs the host writes (patch codes, prompt embeddings, state token, noise) and the
             outputs it reads (the 50 x 32 fp32 actions)
    FLAGS    one 32-byte beat per (stage, node part): WAIT / POST words, monotone within a chunk
    SCRATCH  activation, summary, K/V and int32 sum regions between stages (rewritten every chunk)

Every region is 32-byte aligned (an AXI beat, the unit every node reads and writes); the allocator keeps them in
address order and refuses any overlap, so `regions()` is a proof of the map.  A region carries the image bytes the
host writes before the chunk (`data`) and, for regions a node writes, the bytes it must hold afterwards (`exp`),
beat by beat, which is what the whole-chunk testbench checks.

The put / expect API mirrors paper/sw/pi0_attn_golden.py's Build so the layer builders (pi0_chunk_layers.py) can
be transcriptions of the verified per-layer goldens.  The lowering constraints those goldens obey are re-checked
here: operand regions start on a beat, LOADX rows on even words, stage images P * W <= 512 words, LOAD / LOADX
<= 31 targets, column groups <= 1023 rows, rows <= 2^SLOT_BITS.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BEAT = 32
N_CH, CH_SHIFT, CH_CAP = 16, 33, 1 << 31               # GDDR6 channels, window k << 33, 2 GB of storage each
AREAS = {                       # name: [(base, limit), ...] windows, each inside one channel's 2 GB
    "PROG":    [(0x0_0800_0000, 0x0_1000_0000)],          # 128 MB, channel 0
    "FLAGS":   [(0x0_1000_0000, 0x0_1100_0000)],          # 16 MB, channel 0
    "IO":      [(0x0_1100_0000, 0x0_2000_0000)],          # 240 MB, channel 0
    "STATIC":  [((c << CH_SHIFT) + 0x2000_0000, (c << CH_SHIFT) + 0x4000_0000) for c in range(N_CH)],   # 16 x 512 MB
    "SCRATCH": [((c << CH_SHIFT) + 0x4000_0000, (c << CH_SHIFT) + 0x8000_0000) for c in range(N_CH)],   # 16 x 1 GB
}

# A compact map for silicon runs of cut-down chunks through a PIO BAR window (the node-array bitstreams have no
# usable DMA yet): every area inside the first 512 MB of GDDR6, which BAR1 maps when it is 512 MB.  Selected with
# PI0_CHUNK_COMPACT=1 (pi0_chunk_program.py --compact); the full chunk does not fit and keeps the map above.
AREAS_COMPACT = {             # sized for the tiny chunk (1 SigLIP + 1 prefix + 1 expert layer: 154.5 + 302.4 MB)
    "PROG":    [(0x0_0010_0000, 0x0_0040_0000)],          # 3 MB
    "FLAGS":   [(0x0_0040_0000, 0x0_0050_0000)],          # 1 MB
    "IO":      [(0x0_0050_0000, 0x0_0100_0000)],          # 11 MB
    "STATIC":  [(0x0_0100_0000, 0x0_0B00_0000)],          # 160 MB
    "SCRATCH": [(0x0_0B00_0000, 0x0_2000_0000)],          # 336 MB
}
for _ws in list(AREAS.values()) + list(AREAS_COMPACT.values()):
    for _b, _l in _ws:            # every window inside one channel's storage
        assert (_b >> CH_SHIFT) == ((_l - 1) >> CH_SHIFT) and (_l - 1) - ((_b >> CH_SHIFT) << CH_SHIFT) < CH_CAP, hex(_b)
# The map for bitstreams built with GDDR6 channel striping (pi0_chip_top STRIPE = 12, paper/rtl/axi_stripe.sv): the
# nodes and the host see one flat LOGICAL 32 GB space, which the hardware stripes at 4 KB over the 16 channels, so any
# region -- an activation matrix 12 chain nodes read at once -- is served by every channel.  Selected with
# PI0_CHUNK_MAP=striped; such a chunk does not run on an unstriped bitstream, and vice versa.
AREAS_STRIPED = {
    "PROG":    [(0x0_0800_0000, 0x0_1000_0000)],          # 128 MB
    "FLAGS":   [(0x0_1000_0000, 0x0_1100_0000)],          # 16 MB
    "IO":      [(0x0_1100_0000, 0x0_2000_0000)],          # 240 MB
    "STATIC":  [(0x0_2000_0000, 0x2_2000_0000)],          # 8 GB
    "SCRATCH": [(0x2_2000_0000, 0x8_0000_0000)],          # 23.5 GB (logical 32 GB = 16 x 2 GB)
}
import os as _os
MAP_KIND = _os.environ.get("PI0_CHUNK_MAP", "compact" if _os.environ.get("PI0_CHUNK_COMPACT") == "1" else "channels")
if MAP_KIND == "compact":
    AREAS.clear()
    AREAS.update(AREAS_COMPACT)
elif MAP_KIND == "striped":
    AREAS.clear()
    AREAS.update(AREAS_STRIPED)
else:
    assert MAP_KIND == "channels", f"PI0_CHUNK_MAP={MAP_KIND}: channels | compact | striped"
DT = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}


@dataclass
class Region:
    name: str
    base: int
    size: int                   # bytes, a multiple of BEAT
    area: str
    data: np.ndarray | None = None      # what the host writes (None: nothing)
    exp: np.ndarray | None = None       # what a node must have written (None: nothing)
    exp_beats: np.ndarray | None = None  # bool per beat of exp
    data_beats: np.ndarray | None = None
    tag: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.base + self.size

    def _arr(self, which: str, store: bool = True) -> np.ndarray | None:
        a = getattr(self, which)
        if getattr(self, which + "_beats") is None:
            setattr(self, which + "_beats", np.zeros(self.size // BEAT, bool))
        if a is None and store:
            a = np.zeros(self.size, np.uint8)
            setattr(self, which, a)
        return a


def to_bytes(values, width: int) -> bytes:
    """values (ints or an integer numpy array) as little-endian `width`-bit elements"""
    if width in DT:
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
            return values.reshape(-1).astype(DT[width]).tobytes()      # integer casts wrap: two's complement
        arr = np.array([int(v) & ((1 << width) - 1) for v in values], np.uint64)
        return arr.astype(DT[width]).tobytes()
    nb = width // 8
    return b"".join((int(v) & ((1 << width) - 1)).to_bytes(nb, "little") for v in values)


class ChunkMem:
    def __init__(self, store_data: bool = True, store_exp: bool = True, exp_areas: set | None = None):
        """store_data / store_exp False: keep the map and the written-beat bitmaps but not the bytes (a whole chunk's
        expected regions are ~8 GB; sizing, traffic and timing need only the map).  exp_areas: store expected bytes
        only for regions in these areas (e.g. {"IO", "FLAGS"}: the actions and every part's flag -- what a silicon
        run of a whole chunk compares)"""
        self.regions: list[Region] = []           # in address order
        self._bases: list[int] = []
        self.next = {k: [w[0] for w in v] for k, v in AREAS.items()}      # per window
        self.rr = {k: 0 for k in AREAS}                                      # round-robin start per area
        self.store = {"data": store_data, "exp": store_exp}
        self.exp_areas = set(exp_areas) if exp_areas else None

    # ------------------------------------------------------------------ allocation (the overlap proof)
    def alloc(self, name: str, size: int, area: str, align: int = BEAT, **tag) -> Region:
        """the next window of the area (round-robin over its channels) with room for the region"""
        wins = AREAS[area]
        size = max(BEAT, -(-size // BEAT) * BEAT)
        n = len(wins)
        for k in range(n):
            wi = (self.rr[area] + k) % n
            base = -(-self.next[area][wi] // align) * align
            if base + size <= wins[wi][1]:
                r = Region(name, base, size, area, tag=tag)
                self._insert(r)
                self.next[area][wi] = base + size
                self.rr[area] = (wi + 1) % n
                return r
        raise MemoryError(f"area {area} overflows at {name} ({size} bytes; windows {[hex(b) for b, _ in wins]})")

    def _insert(self, r: Region) -> None:
        i = bisect.bisect_left(self._bases, r.base)
        if i > 0 and self.regions[i - 1].end > r.base:
            raise ValueError(f"overlap: {self.regions[i - 1].name} [{self.regions[i - 1].base:#x}, {self.regions[i - 1].end:#x}) and {r.name} @ {r.base:#x}")
        if i < len(self.regions) and r.end > self.regions[i].base:
            raise ValueError(f"overlap: {r.name} [{r.base:#x}, {r.end:#x}) and {self.regions[i].name} @ {self.regions[i].base:#x}")
        self.regions.insert(i, r)
        self._bases.insert(i, r.base)

    def region_of(self, addr: int) -> Region:
        i = bisect.bisect_right(self._bases, addr) - 1
        if i < 0 or addr >= self.regions[i].end:
            raise KeyError(f"address {addr:#x} is in no region")
        return self.regions[i]

    # ------------------------------------------------------------------ contents
    def _write(self, which: str, addr: int, values, width: int) -> int:
        assert addr % BEAT == 0, f"{addr:#x} is not beat aligned"
        r = self.region_of(addr)
        off = addr - r.base
        store = self.store[which] and (which != "exp" or self.exp_areas is None or r.area in self.exp_areas)
        if store:
            raw = to_bytes(values, width)
            nbytes = len(raw)
        else:
            n = values.size if isinstance(values, np.ndarray) else len(values)
            nbytes = n * width // 8
        if off + nbytes > r.size:
            raise ValueError(f"{nbytes} bytes at {addr:#x} overflow region {r.name} [{r.base:#x}, {r.end:#x})")
        arr = r._arr(which, store)
        if arr is not None:
            arr[off:off + nbytes] = np.frombuffer(raw, np.uint8)
        nb = -(-nbytes // BEAT)
        getattr(r, which + "_beats")[off // BEAT: off // BEAT + nb] = True
        return nbytes

    def put(self, addr: int, values, width: int) -> int:
        """image bytes (what the host writes)"""
        return self._write("data", addr, values, width)

    def expect(self, addr: int, values, width: int) -> int:
        """bytes a node must have written by the end of the chunk"""
        return self._write("exp", addr, values, width)

    # ------------------------------------------------------------------ outputs
    def iter_beats(self, which: str):
        for r in self.regions:
            arr = getattr(r, which)
            if arr is None:
                continue
            flags = getattr(r, which + "_beats")
            for b in np.nonzero(flags)[0]:
                yield r.base + BEAT * int(b), arr[BEAT * b: BEAT * (b + 1)]

    def write_hex(self, path: Path, which: str) -> int:
        """tb_pi0_attn.sv format: beat index (address >> 5) and the 256-bit beat, one per line"""
        n = 0
        with open(path, "w") as f:
            for addr, beat in self.iter_beats(which):
                f.write(f"{addr >> 5:x} {int.from_bytes(beat.tobytes(), 'little'):064x}\n")
                n += 1
        return n

    def write_host_image(self, out: Path) -> dict:
        """the host write list: one binary file per region with image data, plus host_writes.json"""
        out.mkdir(parents=True, exist_ok=True)
        lst = []
        total = 0
        for i, r in enumerate(self.regions):
            if r.data is None:
                continue
            fn = f"r{i:05d}_{r.area.lower()}.bin"
            r.data.tofile(out / fn)
            lst.append(dict(name=r.name, addr=r.base, size=r.size, area=r.area, file=fn))
            total += r.size
        (out / "host_writes.json").write_text(json.dumps(dict(total_bytes=total, regions=lst), indent=1))
        return dict(total_bytes=total, files=len(lst))

    def write_image_bin(self, out: Path, which: str = "data") -> dict:
        """the host loader's format (host/pi0_chunk_run.cpp --image-bin): <which>.bin holds every run of written
        beats back to back, <which>.idx one line per run: address (hex), bytes, offset into the .bin (both decimal).
        Only beats marked written go out, so untouched scratch costs nothing."""
        out.mkdir(parents=True, exist_ok=True)
        n_runs = total = 0
        with open(out / f"{which}.bin", "wb") as fb, open(out / f"{which}.idx", "w") as fi:
            for r in self.regions:
                arr = getattr(r, which)
                if arr is None:
                    continue
                bits = getattr(r, which + "_beats").astype(np.int8)
                edges = np.flatnonzero(np.diff(np.concatenate([[0], bits, [0]])))
                for b0, b1 in zip(edges[0::2], edges[1::2]):
                    chunk = arr[BEAT * b0: BEAT * b1]
                    fi.write(f"{r.base + BEAT * int(b0):x} {len(chunk)} {total}\n")
                    fb.write(chunk.tobytes())
                    total += len(chunk)
                    n_runs += 1
        return dict(bytes=total, runs=n_runs)

    def summary(self) -> dict:
        per = {}
        for r in self.regions:
            d = per.setdefault(r.area, dict(regions=0, bytes=0, data_bytes=0, exp_beats=0))
            d["regions"] += 1
            d["bytes"] += r.size
            if r.data_beats is not None:
                d["data_bytes"] += int(r.data_beats.sum()) * BEAT
            if r.exp_beats is not None:
                d["exp_beats"] += int(r.exp_beats.sum())
        for k, wins in AREAS.items():
            if k in per:
                used = sum(nx - b for nx, (b, _) in zip(self.next[k], wins))
                per[k]["used"] = used
                per[k]["fill"] = used / sum(lim - b for b, lim in wins)
                per[k]["channels"] = sorted({b >> CH_SHIFT for (b, _), nx in zip(wins, self.next[k]) if nx > b})
        return per

    def check(self) -> None:
        """the overlap proof, re-run over the final table"""
        prev = None
        for r in self.regions:
            assert r.base % BEAT == 0 and r.size % BEAT == 0
            if prev is not None:
                assert prev.end <= r.base, (prev.name, r.name)
            assert any(b <= r.base and r.end <= lim for b, lim in AREAS[r.area]), r.name
            prev = r


# ---------------------------------------------------------------- the lowering constraints, re-checked
SLOT_BITS_MAX = 12


def check_chain_program(words: list[int], n_stage: int) -> dict:
    """decode a chain program and assert the command constraints the RTL relies on"""
    n = dict(LOAD=0, LOADX=0, COLGROUP=0, OUT=0, FLUSH=0, WAIT=0, POST=0)
    for c in words:
        op = c >> 124
        if op == 0:
            break
        base = c & ((1 << 42) - 1)
        if op == 1:
            nwords, nreg, first = (c >> 52) & 1023, (c >> 47) & 31, (c >> 42) & 31
            assert base % 32 == 0 and 1 <= nreg <= 31 and first + nreg <= n_stage + 1 and 1 <= nwords <= 512, hex(c)
            n["LOAD"] += 1
        elif op == 5:
            nwords, nreg, first, nsegs = (c >> 52) & 1023, (c >> 47) & 31, (c >> 42) & 31, (c >> 62) & 1023
            assert base % 32 == 0 and 1 <= nreg <= 31 and first + nreg <= n_stage + 1 and nsegs * nwords <= 512, hex(c)
            n["LOADX"] += 1
        elif op == 6:
            T, W, M, P = (c >> 42) & 1023, (c >> 61) & 1023, (c >> 71) & 1023, (c >> 81) & 1023
            G = (c >> 91) & 31
            assert base % 32 == 0 and 0 < T < 1024 and M * W <= 512 and P * W <= 512, hex(c)
            assert G >= 1 and W + G >= n_stage, f"row protocol: W {W} + G {G} < {n_stage} stages ({c:#x})"
            n["COLGROUP"] += 1
        elif op == 3:
            assert base % 32 == 0, hex(c)
            n["OUT"] += 1
        elif op == 4:
            n["FLUSH"] += 1
        elif op == 7:
            n["WAIT"] += 1
        elif op == 8:
            n["POST"] += 1
        else:
            raise AssertionError(f"unknown chain opcode {op}")
    return n


def check_vector_program(words: list[int], slot_bits: int = SLOT_BITS_MAX) -> dict:
    n = dict(ops=0, WAIT=0, POST=0, widest=0)
    k = 0
    while k + 5 < len(words) and words[k]:
        w0 = words[k]
        tag = w0 >> 124
        if tag == 0xB:
            n["WAIT"] += 1
        elif tag == 0xC:
            n["POST"] += 1
        else:
            assert tag == 0xA, hex(w0)
            L = (w0 >> 60) & 0xFFFF
            assert L <= (1 << slot_bits), f"vector op {L} wide > 2^{slot_bits}"
            n["widest"] = max(n["widest"], L)
            n["ops"] += 1
            for d in ((words[k + 1] >> 42), words[k + 2], words[k + 2] >> 48, words[k + 3], words[k + 3] >> 48,
                      words[k + 4], words[k + 4] >> 48):
                d &= (1 << 48) - 1
                if (d >> 42) & 3 != 3:                                     # not a constant
                    assert (d & ((1 << 42) - 1)) % 32 == 0, f"operand {d:#x} not beat aligned"
        k += 6
    return n

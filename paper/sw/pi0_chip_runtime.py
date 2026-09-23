#!/usr/bin/env python3
"""Host runtime of the whole-pi0 node array: per-chunk inputs in, actions out.

A generated chunk (pi0_chunk_program.py --out DIR --bin) is loaded into GDDR6 once (host/pi0_chunk_run.cpp run DIR).
Between chunks only the host inputs change, and they sit in named regions of DIR/regions.json:

  siglip_in (x2, IO)      SigLIP layer-0 input per camera: patch embedding + position table, bf16 codes (256 x 1152)
  siglip_in.amax (x2, IO) per-row |max| code of it (16-bit)
  x_lm rows 512.. (SCRATCH) the prompt's language embeddings as they enter prefix layer 0, bf16 codes (n_lang x 2048)
  x_lm.sum rows 512..     their summaries: row |max| code << 48 (64-bit)
  x_exp row 0 (x10, SCRATCH) the state token (state_proj output, bf16 codes, 1024) at the head of every Euler step
  noise (IO)              the initial x_t, fp32 (50 x 32)
  actions (IO)            the chunk's output, fp32 (50 x 32), written by the chip

encode() turns float inputs into those bytes exactly the way the generator wrote them for the captured frame, so a
chunk run on the capture's own inputs reproduces the generator's expected actions bit for bit; write_runs() emits the
run list pi0_chunk_run --inputs reads; infer() runs one chunk on a loaded image and returns the actions.

  python pi0_chip_runtime.py check DIR                 encode the capture's inputs, compare with DIR's image bytes
  python pi0_chip_runtime.py infer DIR --map ...       one chunk on the loaded image from the capture's inputs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vector_unit_ref as R          # noqa: E402

BEAT = 32
RUNNER = Path(os.environ.get("PI0_CHUNK_RUN", str(HERE.parents[1] / "build/host/pi0_chunk_run")))


class ChunkIO:
    """the host-input and output addresses of one generated chunk"""

    def __init__(self, chunk_dir: str | Path):
        self.dir = Path(chunk_dir)
        regs = json.loads((self.dir / "regions.json").read_text())
        by: dict[str, list[dict]] = {}
        for r in regs:
            by.setdefault(r["name"], []).append(r)
        for v in by.values():
            v.sort(key=lambda r: r["base"])
        self.siglip_in = [r["base"] for r in by["siglip_in"]]
        self.siglip_amax = [r["base"] for r in by["siglip_in.amax"]]
        self.noise = by["noise"][0]["base"]
        self.actions = (by["actions"][0]["base"], by["actions"][0]["size"])
        self.x_lm, self.x_lm_sum = by["x_lm"][0]["base"], by["x_lm.sum"][0]["base"]
        self.x_exp = [r["base"] for r in by["x_exp"]]
        info = json.loads((self.dir / "info.json").read_text())
        self.steps = info["chunk"]["steps"]
        assert len(self.x_exp) == self.steps, (len(self.x_exp), self.steps)
        self.n_img = 512                      # image tokens ahead of the language rows (2 cameras x 256)
        self.d_lm = 2048
        self.n_lang = by["x_lm"][0]["size"] // (self.d_lm * 2) - self.n_img      # prompt tokens the chunk was built for

    def encode(self, vis_in: np.ndarray, lang: np.ndarray, state: np.ndarray, noise: np.ndarray) -> list[tuple[int, bytes]]:
        """float inputs -> [(addr, bytes)]: vis_in (2, 256, 1152), lang (n_lang, 2048), state (1024,), noise (50, 32)"""
        runs = []
        for cam in range(len(self.siglip_in)):
            xc = R.bf16_codes(np.asarray(vis_in[cam], np.float64))
            runs.append((self.siglip_in[cam], xc.astype("<u2").tobytes()))
            runs.append((self.siglip_amax[cam], np.asarray(R.row_amax_code(xc)).astype("<u2").tobytes()))
        lc = R.bf16_codes(np.asarray(lang, np.float64))
        runs.append((self.x_lm + self.n_img * self.d_lm * 2, lc.astype("<u2").tobytes()))
        sums = np.array([int(v) << 48 for v in R.row_amax_code(lc)], np.uint64)
        runs.append((self.x_lm_sum + self.n_img * 8, sums.astype("<u8").tobytes()))
        sc = R.bf16_codes(np.asarray(state, np.float64).reshape(1, -1))
        for base in self.x_exp:
            runs.append((base, sc.astype("<u2").tobytes()))
        runs.append((self.noise, np.asarray(noise, np.float32).view(np.uint32).astype("<u4").tobytes()))
        return runs

    @staticmethod
    def from_capture(z) -> tuple:
        return (z["vis.L0.layer_in"], z["lm.L0.layer_in"][512:], z["ex.L0.layer_in"][0][0], z["ex.noise"].astype(np.float32))

    @staticmethod
    def write_runs(runs: list[tuple[int, bytes]], prefix: str | Path) -> None:
        """pi0_chunk_run --inputs format; a run that does not fill its last beat is padded by reading nothing: the
        tail keeps the image's bytes only if they are zero, so every input region here ends on a beat boundary"""
        prefix = str(prefix)
        off = 0
        with open(prefix + ".bin", "wb") as fb, open(prefix + ".idx", "w") as fi:
            for addr, raw in runs:
                assert addr % BEAT == 0
                pad = (-len(raw)) % BEAT
                raw = raw + b"\0" * pad
                fi.write(f"{addr:x} {len(raw)} {off}\n")
                fb.write(raw)
                off += len(raw)

    def image_bytes(self, addr: int, n: int) -> bytes | None:
        """the generator's image bytes at [addr, addr + n) (image/data.{idx,bin}), None where not written"""
        out = bytearray(n)
        seen = 0
        with open(self.dir / "image/data.idx") as fi, open(self.dir / "image/data.bin", "rb") as fb:
            for line in fi:
                a, nb, off = line.split()
                a, nb, off = int(a, 16), int(nb), int(off)
                lo, hi = max(a, addr), min(a + nb, addr + n)
                if lo < hi:
                    fb.seek(off + lo - a)
                    out[lo - addr:hi - addr] = fb.read(hi - lo)
                    seen += hi - lo
        return bytes(out) if seen == n else None

    def check(self, runs) -> int:
        bad = 0
        for addr, raw in runs:
            img = self.image_bytes(addr, len(raw))
            if img != raw:
                bad += 1
                print(f"  input at {addr:#x} ({len(raw)} bytes) differs from the image" + (" (not all in the image)" if img is None else ""))
        return bad


class ChipServer:
    """`pi0_chunk_run serve`: the device stays open between chunks (no process start, SDK open or region scan per
    chunk).  The child is only ever asked to quit (or sees its stdin close): a process holding /dev/ac7t15xx0 must
    not be killed (acxpcie release double-unlock)."""

    def __init__(self, io: ChunkIO, kmap: str, timeout_s: float = 60):
        import atexit
        cmd = [str(RUNNER), "serve", str(io.dir), "--timeout-s", str(timeout_s)] + (["--map", kmap] if kmap else [])
        env = dict(os.environ, PI0_HOST_BRIDGE=os.environ.get("PI0_HOST_BRIDGE", "1"),
                   PI0_FPGA_DBI_ROUTE=os.environ.get("PI0_FPGA_DBI_ROUTE", "comp"))
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.banner = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("pi0_chunk_run serve exited: " + " | ".join(self.banner))
            self.banner.append(line.strip())
            if line.startswith("READY"):
                break
        atexit.register(self.close)

    def infer(self, prefix: Path, act: Path) -> dict:
        self.p.stdin.write(f"infer {prefix} {act}\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline().strip()
        if not line.startswith("OK"):
            raise RuntimeError(f"chip chunk failed: {line or 'runner exited'}")
        return {k: float(v) for k, v in (kv.split("=") for kv in line.split()[1:])}

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write("quit\n")
                self.p.stdin.flush()
                self.p.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            self.p.wait()


_SERVERS: dict = {}


def infer(io: ChunkIO, inputs: tuple, kmap: str, workdir: Path, check: bool = False, timeout_s: float = 60) -> tuple[np.ndarray, dict]:
    """one chunk on the loaded image.  Default: through a persistent `serve` child (PI0_CHIP_SERVE=0 or check=True:
    one `pi0_chunk_run run` process per chunk, which can also compare flags + actions with the generator's)"""
    if os.environ.get("PI0_CHIP_SERVE", "1") == "1" and not check:
        workdir.mkdir(parents=True, exist_ok=True)
        prefix, act = workdir / "inputs", workdir / "actions.bin"
        t0 = time.time()
        io.write_runs(io.encode(*inputs), prefix)
        t_enc = time.time() - t0
        key = (str(io.dir), kmap)
        if key not in _SERVERS:
            _SERVERS[key] = ChipServer(io, kmap, timeout_s)
        st = _SERVERS[key].infer(prefix, act)
        a = np.frombuffer(act.read_bytes()[:50 * 32 * 4], np.float32).reshape(50, 32).copy()
        return a, dict(st, encode_s=t_enc, wall_s=time.time() - t0)
    return infer_once(io, inputs, kmap, workdir, check, timeout_s)


def infer_once(io: ChunkIO, inputs: tuple, kmap: str, workdir: Path, check: bool = False, timeout_s: float = 60) -> tuple[np.ndarray, dict]:
    """one chunk on the loaded image: write the run list, start, wait, read the actions (50 x 32 fp32)"""
    workdir.mkdir(parents=True, exist_ok=True)
    prefix, act = workdir / "inputs", workdir / "actions.bin"
    t0 = time.time()
    io.write_runs(io.encode(*inputs), prefix)
    cmd = [str(RUNNER), "run", str(io.dir), "--no-load", "--scrub-flags-only", "--inputs", str(prefix),
           "--dump-actions", str(act), "--timeout-s", str(timeout_s)]
    if kmap:
        cmd += ["--map", kmap]
    cmd += ["--check-io-only"] if check else ["--no-check"]
    env = dict(os.environ, PI0_HOST_BRIDGE=os.environ.get("PI0_HOST_BRIDGE", "1"),
               PI0_FPGA_DBI_ROUTE=os.environ.get("PI0_FPGA_DBI_ROUTE", "comp"))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    stats = dict(rc=p.returncode, wall_s=wall, log=p.stdout[-4000:] + p.stderr[-2000:])
    for line in p.stdout.splitlines():
        if line.startswith("[run 0]"):
            stats["run_line"] = line
            try:
                stats["chip_s"] = float(line.split(" after ")[1].split(" s")[0])
            except (IndexError, ValueError):
                pass
    if p.returncode != 0 or not act.exists():
        raise RuntimeError(f"pi0_chunk_run failed (rc {p.returncode}):\n{stats['log']}")
    a = np.frombuffer(act.read_bytes()[:50 * 32 * 4], np.float32).reshape(50, 32).copy()
    return a, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "infer"])
    ap.add_argument("dir")
    ap.add_argument("--capture", default=None, help="activation capture (default: pi0_layer_lower.CAPTURE)")
    ap.add_argument("--map", default="", help="pi0_chunk_run --map (generator kind -> first chip node)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    import pi0_layer_lower as LL
    z = np.load(a.capture or LL.CAPTURE)
    io = ChunkIO(a.dir)
    inputs = io.from_capture(z)
    if a.cmd == "check":
        bad = io.check(io.encode(*inputs))
        print(f"inputs vs image: {'IDENTICAL' if not bad else f'{bad} regions differ'}")
        sys.exit(1 if bad else 0)
    exp = np.load(io.dir / "actions_chip.npy")
    ref = z["ex.actions_fp32"].astype(np.float64)
    work = Path(a.work or io.dir / "runtime")
    for k in range(a.repeat):
        act, st = infer(io, inputs, a.map, work, check=True)
        same = np.array_equal(act.view(np.uint32), exp.astype(np.float32).view(np.uint32))
        rel = np.linalg.norm(act - ref) / np.linalg.norm(ref)
        print(f"chunk {k}: chip {st.get('chip_s', float('nan')):.3f} s, wall {st['wall_s']:.3f} s; actions "
              f"{'== generator (bit-exact)' if same else '!= generator'}; vs fp32 model rel {rel:.4f}")


if __name__ == "__main__":
    main()

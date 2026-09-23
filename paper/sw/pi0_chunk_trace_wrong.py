#!/usr/bin/env python3
"""Data-flow trace of a silicon run's wrong beats: the stages, in execution order, whose writes hold wrong beats, and
which of them read only correct data (the root causes).

  pi0_chunk_trace_wrong.py <stage_io.json (PI0_CHUNK_STAGE_IO=1)> <wrong list (pi0_chunk_run --wrong-list)>
"""
import bisect
import json
import sys

sio = json.load(open(sys.argv[1]))
wrong = sorted(int(l, 16) for l in open(sys.argv[2]) if l.strip())


def count(ranges):
    n = 0
    for lo, hi in ranges:
        n += bisect.bisect_left(wrong, hi) - bisect.bisect_left(wrong, lo)
    return n


roots, shown = [], 0
for st in sio:
    wo, wi = count(st["writes"]), count(st["reads"])
    if wo:
        tag = "ROOT" if wi == 0 else "    "
        if wi == 0:
            roots.append(st["name"])
        if shown < 40:
            print(f"{tag} stage {st['stage']:4d} {st['kind']:6s} {st['name']:40s} wrong out {wo:8d}  wrong in {wi:8d}")
            shown += 1
print("roots:", roots[:20])

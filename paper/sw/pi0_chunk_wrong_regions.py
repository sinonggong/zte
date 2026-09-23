#!/usr/bin/env python3
"""Which regions of a chunk hold the wrong beats a silicon run found, in allocation order (~ stage order).

  pi0_chunk_wrong_regions.py <chunk dir> <wrong list from pi0_chunk_run --wrong-list>
"""
import bisect
import collections
import json
import sys

d, wl = sys.argv[1], sys.argv[2]
alloc = json.load(open(f"{d}/regions.json"))                 # allocation order ~ build (stage) order
for k, r in enumerate(alloc):
    r["k"] = k
regs = sorted(alloc, key=lambda r: r["base"])
bases = [r["base"] for r in regs]
cnt, first = collections.Counter(), {}
for line in open(wl):
    a = int(line, 16)
    i = bisect.bisect_right(bases, a) - 1
    key = regs[i]["k"] if i >= 0 and a < regs[i]["base"] + regs[i]["size"] else -1
    cnt[key] += 1
    first.setdefault(key, a)
print(f"{sum(cnt.values())} wrong beats in {len(cnt)} regions (allocation order)")
for key in sorted(cnt)[:40]:
    r = alloc[key] if key >= 0 else {"name": "?", "size": 0, "area": "?"}
    print(f"#{key:<5d} {cnt[key]:8d} / {r['size'] // 32:<8d} {r['name']:40s} {r['area']:8s} @{first[key]:#x}")

#!/usr/bin/env python3
"""Decode vu_node_ml's 128-bit debug snapshot (o_dbg; host window read beat n [255:128]; tb 'DBG' lines)."""
import sys
FIELDS = [("op", 4), ("pad_w", 6), ("ow_log2", 3), ("L_len[7:0]", 8), ("epos", 8), ("off_end", 8), ("obits0", 8),
          ("obits1", 8), ("nb0", 16), ("nb1", 16), ("part", 2), ("rows_l0", 10), ("rows_l1", 10), ("zero", 1),
          ("ecarry_nz", 1), ("n_ops", 8)]


def decode(v: int) -> dict:
    out, pos = {}, sum(w for _, w in FIELDS)
    for name, w in FIELDS:
        pos -= w
        out[name] = (v >> pos) & ((1 << w) - 1)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(decode(int(arg, 16)))

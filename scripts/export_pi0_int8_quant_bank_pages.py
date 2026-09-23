#!/usr/bin/env python3
"""Export calibrated pi0 INT8 requantization metadata in RTL page format.

The output scale may be calibrated per tensor (one scalar per GEMM region, the
default and the only behaviour before 2026-08-17) or per output channel.  The
quant bank already carries a multiplier, right shift, bias and zero point for
every channel, so per-channel folding costs nothing in hardware -- but it is
only an improvement when the calibration saw enough tokens for the per-channel
amax to be sampled.  See MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS below: a
per-channel output scale calibrated on too few tokens is REFUSED, not warned
about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any


WEIGHT_SCHEMA = "pi0-openpi-int8-transformer-arena-v1"
CALIBRATION_SCHEMA = "pi0-int8-gemm-calibration-v1"
OUTPUT_SCHEMA = "pi0-int8-quant-bank-pages-v2"
QUANT_CHANNELS = 64
QUANT_TENSOR_CORES = 64
QUANT_BASE = 0x7000
PACKED_RECORD_BYTES = 16
PAGE_BYTES = QUANT_CHANNELS * PACKED_RECORD_BYTES
BIAS_MIN = -(1 << 47)
BIAS_MAX = (1 << 47) - 1
MULTIPLIER_MAX = (1 << 16) - 1
SHIFT_MAX = (1 << 6) - 1
OUTPUT_SCALE_MODES = ("per_tensor", "per_output_channel")

# Why a per-output-channel output scale needs a calibration token floor.
#
# A per-tensor output scale pools T*N samples into one amax, so it is saturated
# long before the calibration set is.  A per-output-channel amax is the maximum
# of only the T tokens that were calibrated, so a fresh token exceeds it with
# probability ~1/T and is then clamped.  That 1/T model is not a guess; the
# measured clamp rate sits on it across a factor of sixteen in T
# (scripts/check_pi0_int8_numerics_gate.py, real action expert, 18 chained
# layers, per_tensor activation scale, calibration draws independent of the
# measured draw):
#
#   tokens   1/T     measured clamp   rms_rel per_tensor -> per_output_channel
#       51   1.96%          1.8364%   0.19304 -> 0.30707    59.1% WORSE
#      102   0.98%          0.8799%   0.20760 -> 0.23600    13.7% worse
#      153   0.65%          0.5943%   0.22365 -> 0.21825     2.4% better
#      204   0.49%          0.4590%   0.25570 -> 0.22881    10.5% better
#      306   0.33%          0.3104%   0.25017 -> 0.20855    16.6% better
#      408   0.25%          0.2326%   0.26598 -> 0.21224    20.2% better
#      612   0.16%          0.1582%   0.26726 -> 0.21442    19.8% better
#      816   0.12%          0.1195%   0.27141 -> 0.22375    17.6% better
#     1224   0.08%          0.0790%   0.31764 -> 0.21704    31.7% better
#
# The crossover is between 102 and 153 tokens and the benefit only settles
# above ~300.  1224 tokens, the first point measured above the floor below,
# clamps 0.079% and wins by 31.7%.
#
# The floor is deliberately NOT put at the crossover: every draw above is
# i.i.d. with the measured draw, the mildest distribution shift there is, and
# since the clamp rate is ~1/T a fatter real-world tail moves the crossover
# up.  1024 tokens is the first power of two at or above the
# 816-token point that was measured to win outright -- about 7x the measured
# crossover, ~0.1% predicted clamping, and only 20 action chunks of 51 tokens
# to collect.  There is deliberately no flag to lower it.  A thin per-channel
# calibration does not fail loudly at runtime; it silently returns worse
# actions than the per-tensor default, so it has to be unreachable rather than
# documented.
#
# WHAT THIS FLOOR DOES NOT GUARD ON ITS OWN, and what does.  The floor keys on
# the declared output_scale FIELD, not on whether per-column requantisation is
# happening, so it never saw the same variation folded into the weight_scale
# sidecar instead.  Declaring output_scale: 1.0 and pre-multiplying the sidecar
# used to be accepted with no calibration_tokens at all, and emitted pages that
# were sha256-IDENTICAL to the per-column export the floor refuses.
#
# That path is now closed at the source of the numbers rather than at the field.
# verify_weight_store_binding() requires every weight entry to carry the weight
# store's own sha256 for BOTH the INT8 payload and the scale sidecar, and
# refuses unless the bytes on disk hash to them; scale_min/scale_max are checked
# against the array too.  A hand-edited sidecar no longer reaches the packer,
# and neither does a genuine sidecar paired with different weights.
#
# What this exporter cannot settle on its own.  The hashes live in the weight
# manifest, so a coordinated edit -- doctor the sidecar AND restamp
# scale_sha256 -- passes every check reachable from here.  It is not silent:
# the manifest is hashed into sources.weight_manifest_sha256 of every
# quant.json, so a doctored export stops being byte-identical to the honest one.
# And it is now decidable, just not here: each weight entry carries
# source_sha256, the digest of the fp32 tensor it was quantised from, which
# cannot be reproduced without the checkpoint.  Run
# `export_pi0_openpi_int8_transformer_weights.py --verify <manifest>
# --checkpoint <ckpt>` to re-quantise every tensor and confirm the shipped
# payloads and sidecars are the ones that checkpoint implies.  This exporter
# passes source_sha256 through to weight_store_binding rather than checking it,
# because it never sees the checkpoint.
#
# Two limits remain, stated rather than fixed: calibration_tokens is
# self-declared and unverifiable (nothing in this repo writes the calibration
# manifest), and a mid-manifest refusal leaves already-written payloads on disk
# with no quant.json, so a reused output directory can hold stale ones.
WEIGHT_STORE_BINDING_RULE = (
    "every weight entry must declare sha256 for its INT8 payload and scale_sha256 "
    "for its scale sidecar, and the bytes on disk must hash to them"
)
MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS = 1024
PER_OUTPUT_CHANNEL_TOKEN_RULE = (
    "a per_output_channel output scale requires the calibration to declare "
    f"calibration_tokens >= {MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS}"
)
PER_OUTPUT_CHANNEL_TOKEN_EVIDENCE: dict[str, Any] = {
    "source": "scripts/check_pi0_int8_numerics_gate.py",
    "model": "pi0 action expert, 18 chained layers, per_tensor activation scale, "
             "amax calibration on draws independent of the measured draw",
    "exceedance_model":
        "a per-channel amax over T tokens is exceeded by a fresh token with "
        "probability ~1/T; the measured clamp rate tracks 1/T from T=51 to T=816",
    "measured": [
        {
            "calibration_tokens": 51,
            "rms_rel_per_tensor_output_scale": 0.19304487646476956,
            "rms_rel_per_output_channel_output_scale": 0.3070655966265378,
            "per_output_channel_saturation_rate": 0.01836444574936456,
        },
        {
            "calibration_tokens": 102,
            "rms_rel_per_tensor_output_scale": 0.20760,
            "rms_rel_per_output_channel_output_scale": 0.23600,
            "per_output_channel_saturation_rate": 0.008799,
        },
        {
            "calibration_tokens": 153,
            "rms_rel_per_tensor_output_scale": 0.22365,
            "rms_rel_per_output_channel_output_scale": 0.21825,
            "per_output_channel_saturation_rate": 0.005943,
        },
        {
            "calibration_tokens": 204,
            "rms_rel_per_tensor_output_scale": 0.25570,
            "rms_rel_per_output_channel_output_scale": 0.22881,
            "per_output_channel_saturation_rate": 0.004590,
        },
        {
            "calibration_tokens": 306,
            "rms_rel_per_tensor_output_scale": 0.25017,
            "rms_rel_per_output_channel_output_scale": 0.20855,
            "per_output_channel_saturation_rate": 0.003104,
        },
        {
            "calibration_tokens": 408,
            "rms_rel_per_tensor_output_scale": 0.26598,
            "rms_rel_per_output_channel_output_scale": 0.21224,
            "per_output_channel_saturation_rate": 0.002326,
        },
        {
            "calibration_tokens": 612,
            "rms_rel_per_tensor_output_scale": 0.26726,
            "rms_rel_per_output_channel_output_scale": 0.21442,
            "per_output_channel_saturation_rate": 0.001582,
        },
        {
            "calibration_tokens": 816,
            "rms_rel_per_tensor_output_scale": 0.27140646730797685,
            "rms_rel_per_output_channel_output_scale": 0.22375379840816909,
            "per_output_channel_saturation_rate": 0.001194728831926289,
        },
        {
            "calibration_tokens": 1224,
            "rms_rel_per_tensor_output_scale": 0.31764,
            "rms_rel_per_output_channel_output_scale": 0.21704,
            "per_output_channel_saturation_rate": 0.000790,
        },
    ],
    "crossover_tokens_between": [102, 153],
    "floor": MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS,
    "floor_rationale":
        "the first power of two at or above the 816-token point measured to win "
        "outright; the crossover itself is not used as the floor because these "
        "draws are i.i.d. with the measured draw and real data has a fatter tail",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_key(entry: dict[str, Any]) -> tuple[str, int, str]:
    return str(entry["expert"]), int(entry["layer"]), str(entry["slot"])


def keyed_entries(value: object, label: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} entries must be a list of objects")
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw in value:
        entry = raw
        key = entry_key(entry)
        if key in result:
            raise ValueError(f"duplicate {label} entry: {key}")
        result[key] = entry
    return result


def resolve_payload(
    *,
    manifest_path: Path,
    direct_value: object,
    relative_value: object,
    payload_root: Path | None,
    label: str,
) -> Path:
    candidates: list[Path] = []
    if isinstance(direct_value, str) and direct_value:
        direct = Path(direct_value)
        candidates.append(direct if direct.is_absolute() else manifest_path.parent / direct)
    if isinstance(relative_value, str) and relative_value:
        relative = Path(relative_value)
        if payload_root is not None:
            candidates.append(payload_root / relative)
        candidates.append(manifest_path.parent / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates) or "<no path supplied>"
    raise FileNotFoundError(f"{label} payload is missing; tried {rendered}")


class OutputScale:
    """The calibrated output scale of one GEMM region, per tensor or per channel.

    ``per_tensor`` keeps a Python float, exactly as this exporter has always
    done, so a scalar calibration re-exports byte-identical pages.  Per-channel
    values are held as float32 -- the same width as the weight-scale sidecars --
    so the exporter and the checker fold bit-identical numbers whether the
    calibration supplied an inline array or a sidecar payload.
    """

    __slots__ = ("mode", "scalar", "values", "source_kind", "source_path")

    def __init__(
        self,
        *,
        mode: str,
        scalar: float | None,
        values: Any,
        source_kind: str,
        source_path: Path | None,
    ) -> None:
        self.mode = mode
        self.scalar = scalar
        self.values = values
        self.source_kind = source_kind
        self.source_path = source_path

    @property
    def per_output_channel(self) -> bool:
        return self.mode == "per_output_channel"

    def value(self, channel: int) -> float:
        if self.values is None:
            assert self.scalar is not None
            return self.scalar
        return float(self.values[channel])

    def minimum(self) -> float:
        return self.scalar if self.values is None else float(self.values.min())

    def maximum(self) -> float:
        return self.scalar if self.values is None else float(self.values.max())


def resolve_output_scale(
    *,
    key: tuple[str, int, str],
    calibration_entry: dict[str, Any],
    calibration_path: Path,
    calibration_root: Path | None,
    channels: int,
) -> OutputScale:
    """Read ``output_scale`` as either a scalar or a per-output-channel array.

    Accepted shapes, in the order they are discriminated:

    * ``output_scale: <number>``                     -> per_tensor (the default)
    * ``output_scale: [<number>, ...]``              -> per_output_channel, inline
    * ``output_scale_path`` / ``output_scale_relative_path`` pointing at a
      little-endian float32 payload of ``channels`` values, with ``output_scale``
      absent or null -> per_output_channel, sidecar

    ``output_scale_mode`` is optional.  When present it must agree with the
    shape actually supplied, so a manifest cannot claim per-channel resolution
    while handing over one scalar.
    """
    import numpy as np

    declared_mode = calibration_entry.get("output_scale_mode")
    if declared_mode is not None and declared_mode not in OUTPUT_SCALE_MODES:
        raise ValueError(
            f"{key} unsupported output_scale_mode {declared_mode!r}; "
            f"expected one of {list(OUTPUT_SCALE_MODES)}"
        )
    raw = calibration_entry.get("output_scale")
    if isinstance(raw, bool):
        raise ValueError(f"{key} output_scale must be a number or an array of numbers")
    sidecar_declared = any(
        isinstance(calibration_entry.get(name), str) and calibration_entry.get(name)
        for name in ("output_scale_path", "output_scale_relative_path")
    )

    if isinstance(raw, list) or sidecar_declared:
        mode = "per_output_channel"
    elif isinstance(raw, (int, float)):
        mode = "per_tensor"
    else:
        raise ValueError(
            f"{key} output_scale must be a number, an array of numbers, or a "
            "float32 sidecar payload"
        )
    if declared_mode is not None and declared_mode != mode:
        raise ValueError(
            f"{key} declares output_scale_mode={declared_mode!r} but supplies a "
            f"{mode} output scale"
        )

    if mode == "per_tensor":
        scalar = float(raw)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{key} output_scale must be finite and positive")
        return OutputScale(
            mode=mode,
            scalar=scalar,
            values=None,
            source_kind="scalar",
            source_path=None,
        )

    if isinstance(raw, list) and sidecar_declared:
        raise ValueError(
            f"{key} supplies both an inline output_scale array and a sidecar payload"
        )
    if isinstance(raw, list):
        if not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw
        ):
            raise ValueError(f"{key} inline per-channel output_scale must contain numbers only")
        values = np.asarray(raw, dtype=np.float32)
        source_kind = "inline_array"
        source_path = None
    else:
        if raw is not None:
            raise ValueError(
                f"{key} output_scale must be null when a per-channel sidecar is supplied"
            )
        source_path = resolve_payload(
            manifest_path=calibration_path,
            direct_value=calibration_entry.get("output_scale_path"),
            relative_value=calibration_entry.get("output_scale_relative_path"),
            payload_root=calibration_root,
            label=f"{key} output scale",
        )
        values = np.fromfile(source_path, dtype="<f4")
        source_kind = "float32_sidecar"
    if values.size != channels:
        raise ValueError(
            f"{key} per-channel output_scale count mismatch: expected {channels}, "
            f"got {values.size}"
        )
    if not bool(np.all(np.isfinite(values))) or not bool(np.all(values > 0.0)):
        raise ValueError(f"{key} per-channel output scales must be finite and positive")
    return OutputScale(
        mode=mode,
        scalar=None,
        values=values,
        source_kind=source_kind,
        source_path=source_path,
    )


def calibration_token_count(
    *,
    key: tuple[str, int, str],
    calibration: dict[str, Any],
    calibration_entry: dict[str, Any],
) -> tuple[int | None, str | None]:
    """Return the declared calibration token count and where it was declared.

    ``calibration_tokens`` may sit on the manifest (the usual case: one
    calibration pass covers every region) or on an individual entry, which
    overrides the manifest.  It is optional for a per-tensor output scale so
    that manifests written before this field existed keep loading unchanged.
    """
    for scope, holder in (("entry", calibration_entry), ("manifest", calibration)):
        raw = holder.get("calibration_tokens")
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"{key} calibration_tokens must be an integer token count, got {raw!r}"
            )
        if raw <= 0:
            raise ValueError(f"{key} calibration_tokens must be positive, got {raw}")
        return raw, scope
    return None, None


def require_per_output_channel_calibration(
    *,
    key: tuple[str, int, str],
    tokens: int | None,
) -> None:
    """Refuse a per-output-channel output scale calibrated on too little data.

    This is the whole point of the token bookkeeping.  Per-channel scales taken
    from a thin calibration are not merely unproven, they are measurably worse
    than the per-tensor default they replace, and nothing downstream reports it:
    the clamped channels just return worse actions.  See
    MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS for the measurement.
    """
    if tokens is None:
        raise ValueError(
            f"{key} refuses a per_output_channel output scale: the calibration does "
            "not declare calibration_tokens, so the per-channel amax cannot be shown "
            f"to be sampled. {PER_OUTPUT_CHANNEL_TOKEN_RULE}."
        )
    if tokens < MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS:
        raise ValueError(
            f"{key} refuses a per_output_channel output scale calibrated on {tokens} "
            f"tokens: {PER_OUTPUT_CHANNEL_TOKEN_RULE}. A per-channel amax over T "
            "tokens is exceeded by a fresh token with probability ~1/T; measured on "
            "the real action expert, 51 tokens clamped 1.84% of elements and made "
            "the layer output 59% worse than the per-tensor scale it replaced, while "
            "816 tokens clamped 0.12% and made it 17.6% better. Re-run the "
            "calibration over more tokens; there is no flag to lower this floor."
        )


def verify_weight_store_binding(
    *,
    key: tuple[str, int, str],
    weight_entry: dict[str, Any],
    weight_manifest_path: Path,
    weight_payload_root: Path | None,
    scale_path: Path,
    scale_values: Any,
) -> dict[str, Any]:
    """Bind the weight-scale sidecar to the weight store that produced it.

    The token floor above guards a FIELD; this guards the NUMBERS.  Without it
    the exporter read whatever float32 array the scale path happened to point
    at, so per-column output variation could be folded into the weight scales
    and shipped as a per-tensor export, with no calibration token count and
    sha256-identical pages.  Here every weight entry must carry the sha256 the
    weight store wrote for its INT8 payload and for its scale sidecar, and the
    bytes on disk must hash to them.

    The payload is checked as well as the sidecar because the pairing is what
    carries meaning: a scale sidecar is only correct for the weights it was
    derived from, and `scale_formula` in the weight manifest cannot be replayed
    from the INT8 store (post-quantisation every column has amax 127 by
    construction, so the magnitude is unrecoverable).  Hashes are the strongest
    check available on this side of the fp32 checkpoint.
    """
    import numpy as np

    declared_scale_sha = weight_entry.get("scale_sha256")
    if not isinstance(declared_scale_sha, str) or not declared_scale_sha:
        raise ValueError(
            f"{key} weight entry does not declare scale_sha256, so the weight scales "
            "cannot be tied to the weight store that produced them. Re-export the "
            "weight manifest with export_pi0_openpi_int8_transformer_weights.py; "
            "there is no flag to skip this."
        )
    actual_scale_sha = sha256_file(scale_path)
    if actual_scale_sha != declared_scale_sha:
        raise ValueError(
            f"{key} weight scale sidecar does not match the weight store: "
            f"{scale_path} hashes to {actual_scale_sha}, but the weight manifest "
            f"declares {declared_scale_sha}. Per-column variation folded into the "
            "weight scales is refused here, not warned about."
        )

    declared_payload_sha = weight_entry.get("sha256")
    if not isinstance(declared_payload_sha, str) or not declared_payload_sha:
        raise ValueError(
            f"{key} weight entry does not declare sha256 for its INT8 payload, so the "
            "scale sidecar cannot be shown to belong to these weights"
        )
    payload_path = resolve_payload(
        manifest_path=weight_manifest_path,
        direct_value=weight_entry.get("path"),
        relative_value=weight_entry.get("payload_relative_path"),
        payload_root=weight_payload_root,
        label=f"{key} INT8 weight",
    )
    actual_payload_sha = sha256_file(payload_path)
    if actual_payload_sha != declared_payload_sha:
        raise ValueError(
            f"{key} INT8 weight payload does not match the weight store: "
            f"{payload_path} hashes to {actual_payload_sha}, but the weight manifest "
            f"declares {declared_payload_sha}"
        )
    payload_bytes = payload_path.stat().st_size
    declared_bytes = weight_entry.get("nbytes")
    if isinstance(declared_bytes, int) and payload_bytes != declared_bytes:
        raise ValueError(
            f"{key} INT8 weight payload is {payload_bytes} bytes, but the weight "
            f"manifest declares nbytes={declared_bytes}"
        )
    matrix_shape = weight_entry.get("matrix_shape")
    if isinstance(matrix_shape, list) and len(matrix_shape) == 2:
        expected_bytes = int(matrix_shape[0]) * int(matrix_shape[1])
        if payload_bytes != expected_bytes:
            raise ValueError(
                f"{key} INT8 weight payload is {payload_bytes} bytes, but "
                f"matrix_shape {matrix_shape} needs {expected_bytes}"
            )

    extrema_checked = False
    for field, observed in (
        ("scale_min", float(np.min(scale_values))),
        ("scale_max", float(np.max(scale_values))),
    ):
        declared = weight_entry.get(field)
        if declared is None:
            continue
        extrema_checked = True
        if float(declared) != observed:
            raise ValueError(
                f"{key} weight scale {field} is {observed!r} on disk but the weight "
                f"manifest declares {float(declared)!r}"
            )

    # Passed through, not verified: reproducing a source digest needs the
    # checkpoint, which this exporter never sees.  Recording it keeps the chain
    # traceable, so an auditor can run
    # `export_pi0_openpi_int8_transformer_weights.py --verify` against the
    # checkpoint and settle whether these scales are the ones it implies.
    source_sha = weight_entry.get("source_sha256")

    return {
        "verified": True,
        "source_sha256": source_sha if isinstance(source_sha, str) else None,
        "auditable_against_checkpoint": isinstance(source_sha, str) and bool(source_sha),
        "weight_payload_path": str(payload_path),
        "weight_payload_sha256": actual_payload_sha,
        "weight_payload_bytes": payload_bytes,
        "scale_sha256_matches_weight_store": True,
        "scale_extrema_matched": extrema_checked,
    }


def encode_effective_scale(scale: float, max_relative_error: float) -> tuple[int, int, float, float]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"effective scale must be finite and positive, got {scale!r}")
    if scale > MULTIPLIER_MAX:
        raise ValueError(f"effective scale exceeds hardware maximum {MULTIPLIER_MAX}: {scale}")
    minimum = math.ldexp(1.0, -SHIFT_MAX)
    if scale < minimum:
        raise ValueError(f"effective scale is below hardware minimum 2^-{SHIFT_MAX}: {scale}")

    shift = min(
        SHIFT_MAX,
        max(0, int(math.floor(math.log2(MULTIPLIER_MAX / scale)))),
    )
    multiplier = int(round(math.ldexp(scale, shift)))
    while multiplier > MULTIPLIER_MAX and shift > 0:
        shift -= 1
        multiplier = int(round(math.ldexp(scale, shift)))
    if multiplier < 1 or multiplier > MULTIPLIER_MAX:
        raise ValueError(f"effective scale cannot be encoded: {scale}")

    # Canonicalize exact powers of two without changing the represented value.
    while shift > 0 and (multiplier & 1) == 0:
        multiplier >>= 1
        shift -= 1

    decoded = math.ldexp(float(multiplier), -shift)
    relative_error = abs(decoded - scale) / scale
    if relative_error > max_relative_error:
        raise ValueError(
            "effective scale encoding error exceeds limit: "
            f"scale={scale} decoded={decoded} relative_error={relative_error} "
            f"limit={max_relative_error}"
        )
    return multiplier, shift, decoded, relative_error


def pack_quant_entry(
    *,
    bias: int,
    multiplier: int,
    right_shift: int,
    zero_point: int,
    valid: bool,
) -> bytes:
    if not BIAS_MIN <= bias <= BIAS_MAX:
        raise ValueError(f"bias is outside signed 48-bit range: {bias}")
    if not 0 <= multiplier <= MULTIPLIER_MAX:
        raise ValueError(f"multiplier is outside unsigned 16-bit range: {multiplier}")
    if not 0 <= right_shift <= SHIFT_MAX:
        raise ValueError(f"right shift is outside unsigned 6-bit range: {right_shift}")
    if not -256 <= zero_point <= 255:
        raise ValueError(f"zero point is outside signed 9-bit range: {zero_point}")
    packed = (
        (bias & ((1 << 48) - 1))
        | (multiplier << 48)
        | (right_shift << 64)
        | ((zero_point & 0x1FF) << 70)
        | (int(valid) << 79)
    )
    return struct.pack(
        "<IIII",
        packed & 0xFFFFFFFF,
        (packed >> 32) & 0xFFFFFFFF,
        (packed >> 64) & 0xFFFF,
        0,
    )


def output_bias_values(
    *,
    calibration: dict[str, Any],
    calibration_path: Path,
    calibration_root: Path | None,
    channels: int,
) -> tuple[Any, Path | None]:
    import numpy as np

    bias_mode = calibration.get("bias_mode", "none")
    if bias_mode == "none":
        return np.zeros(channels, dtype=np.float64), None
    if bias_mode != "float32_per_output_channel":
        raise ValueError(f"unsupported bias_mode: {bias_mode!r}")
    path = resolve_payload(
        manifest_path=calibration_path,
        direct_value=calibration.get("bias_path"),
        relative_value=calibration.get("bias_relative_path"),
        payload_root=calibration_root,
        label="bias",
    )
    values = np.fromfile(path, dtype="<f4")
    if values.size != channels:
        raise ValueError(f"bias value count mismatch: expected {channels}, got {values.size}")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"bias values must be finite: {path}")
    return values.astype(np.float64), path


def export_quant_pages(
    *,
    weight_manifest_path: Path,
    calibration_path: Path,
    out_dir: Path,
    out_json: Path,
    out_plan: Path,
    out_markdown: Path,
    weight_payload_root: Path | None,
    calibration_root: Path | None,
    allow_debug_calibration: bool,
    max_relative_error: float,
) -> dict[str, Any]:
    import numpy as np

    weights = load_json(weight_manifest_path)
    calibration = load_json(calibration_path)
    if weights.get("schema") != WEIGHT_SCHEMA:
        raise ValueError("unexpected transformer weight manifest schema")
    if weights.get("status") != "payloads_exported":
        raise ValueError("quant pages require exported weight scale sidecars")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected INT8 calibration manifest schema")
    calibration_status = calibration.get("status")
    if calibration_status not in {"calibrated", "synthetic_debug"}:
        raise ValueError("calibration status must be calibrated or synthetic_debug")
    if calibration_status == "calibrated" and calibration.get("real_calibration") is not True:
        raise ValueError("calibrated input must declare real_calibration=true")
    if calibration_status == "synthetic_debug" and not allow_debug_calibration:
        raise ValueError("synthetic_debug calibration requires --allow-debug-calibration")
    if not math.isfinite(max_relative_error) or not 0.0 < max_relative_error < 1.0:
        raise ValueError("max_relative_error must be finite and in (0, 1)")

    weight_entries = keyed_entries(weights.get("entries"), "weight")
    calibration_entries = keyed_entries(calibration.get("entries"), "calibration")
    if set(weight_entries) != set(calibration_entries):
        missing = sorted(set(weight_entries) - set(calibration_entries))
        extra = sorted(set(calibration_entries) - set(weight_entries))
        raise ValueError(f"calibration identity mismatch: missing={missing[:8]} extra={extra[:8]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_entries: list[dict[str, Any]] = []
    plan_lines = [
        "# Auto-generated pi0 INT8 quant-page declarations.",
        "# Declarations are not an execution schedule; tile lowering must emit run_quant_page.",
    ]
    next_page = 0
    per_output_channel_entries = 0
    per_tensor_entries = 0
    weight_store_verified = 0
    weight_store_payload_bytes = 0
    total_valid_channels = 0
    total_padded_channels = 0
    error_sum = 0.0
    error_max = 0.0
    bias_min = 0
    bias_max = 0

    for key, weight in weight_entries.items():
        cal = calibration_entries[key]
        activation_scale = float(cal.get("activation_scale", 0.0))
        activation_zero_point = int(cal.get("activation_zero_point", 0))
        output_zero_point = int(cal.get("output_zero_point", 0))
        if not math.isfinite(activation_scale) or activation_scale <= 0.0:
            raise ValueError(f"{key} activation_scale must be finite and positive")
        if activation_zero_point != 0:
            raise ValueError(f"{key} activation_zero_point must be zero for the current MAC")
        if not -128 <= output_zero_point <= 127:
            raise ValueError(f"{key} output_zero_point must be in int8 range")

        channels = int(weight["scale_values"])
        matrix_shape = weight.get("matrix_shape")
        if not isinstance(matrix_shape, list) or len(matrix_shape) != 2:
            raise ValueError(f"{key} weight matrix_shape must contain K and N")
        if channels != int(matrix_shape[1]):
            raise ValueError(f"{key} scale count does not match matrix N")
        scale_path = resolve_payload(
            manifest_path=weight_manifest_path,
            direct_value=weight.get("scale_path"),
            relative_value=weight.get("scale_relative_path"),
            payload_root=weight_payload_root,
            label=f"{key} weight scale",
        )
        weight_scales = np.fromfile(scale_path, dtype="<f4")
        if weight_scales.size != channels:
            raise ValueError(f"{key} weight scale count mismatch")
        if not bool(np.all(np.isfinite(weight_scales))) or not bool(np.all(weight_scales > 0.0)):
            raise ValueError(f"{key} weight scales must be finite and positive")
        weight_store_binding = verify_weight_store_binding(
            key=key,
            weight_entry=weight,
            weight_manifest_path=weight_manifest_path,
            weight_payload_root=weight_payload_root,
            scale_path=scale_path,
            scale_values=weight_scales,
        )
        output_scale = resolve_output_scale(
            key=key,
            calibration_entry=cal,
            calibration_path=calibration_path,
            calibration_root=calibration_root,
            channels=channels,
        )
        tokens, tokens_scope = calibration_token_count(
            key=key,
            calibration=calibration,
            calibration_entry=cal,
        )
        if output_scale.per_output_channel:
            require_per_output_channel_calibration(key=key, tokens=tokens)
        real_bias, bias_path = output_bias_values(
            calibration=cal,
            calibration_path=calibration_path,
            calibration_root=calibration_root,
            channels=channels,
        )

        page_count = (channels + QUANT_CHANNELS - 1) // QUANT_CHANNELS
        padded_channels = page_count * QUANT_CHANNELS
        relative_path = Path("payloads") / key[0] / f"layer_{key[1]:02d}" / f"{key[2]}.qmeta128.bin"
        payload_path = out_dir / relative_path
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload = bytearray()
        local_error_sum = 0.0
        local_error_max = 0.0
        local_bias_min = 0
        local_bias_max = 0

        for channel in range(channels):
            weight_scale = float(weight_scales[channel])
            effective_scale = activation_scale * weight_scale / output_scale.value(channel)
            multiplier, right_shift, _, relative_error = encode_effective_scale(
                effective_scale,
                max_relative_error,
            )
            bias_denominator = activation_scale * weight_scale
            bias_acc = int(np.rint(float(real_bias[channel]) / bias_denominator))
            if not BIAS_MIN <= bias_acc <= BIAS_MAX:
                raise ValueError(f"{key} channel {channel} bias exceeds signed 48-bit range")
            payload.extend(
                pack_quant_entry(
                    bias=bias_acc,
                    multiplier=multiplier,
                    right_shift=right_shift,
                    zero_point=output_zero_point,
                    valid=True,
                )
            )
            local_error_sum += relative_error
            local_error_max = max(local_error_max, relative_error)
            local_bias_min = min(local_bias_min, bias_acc)
            local_bias_max = max(local_bias_max, bias_acc)

        payload.extend(
            pack_quant_entry(
                bias=0,
                multiplier=0,
                right_shift=0,
                zero_point=0,
                valid=False,
            )
            * (padded_channels - channels)
        )
        payload_path.write_bytes(payload)
        if payload_path.stat().st_size != page_count * PAGE_BYTES:
            raise RuntimeError(f"unexpected quant payload size: {payload_path}")

        # A per-channel output scale is written out beside the pages in exactly
        # the float32 form that was folded, whatever shape the calibration used.
        # The checker then has one thing to re-read and one thing to hash.
        output_scale_relative: Path | None = None
        output_scale_path: Path | None = None
        if output_scale.per_output_channel:
            output_scale_relative = relative_path.with_name(f"{key[2]}.oscale.f32.bin")
            output_scale_path = out_dir / output_scale_relative
            output_scale.values.astype(np.float32).tofile(output_scale_path)
            if output_scale_path.stat().st_size != channels * 4:
                raise RuntimeError(f"unexpected output scale payload size: {output_scale_path}")

        first_page = next_page
        for local_page in range(page_count):
            valid_channels = min(QUANT_CHANNELS, channels - local_page * QUANT_CHANNELS)
            tc_id = local_page % QUANT_TENSOR_CORES
            plan_lines.append(
                f"quant_page,{next_page},{tc_id},{payload_path.resolve()},"
                f"{local_page * PAGE_BYTES},{valid_channels}"
            )
            next_page += 1
        output_entries.append(
            {
                "expert": key[0],
                "layer": key[1],
                "slot": key[2],
                "weight_sel": int(weight["weight_sel"]),
                "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
                "output_channels": channels,
                "activation_scale": activation_scale,
                "activation_zero_point": activation_zero_point,
                "output_scale_mode": output_scale.mode,
                "output_scale": output_scale.scalar,
                "output_scale_min": output_scale.minimum(),
                "output_scale_max": output_scale.maximum(),
                "output_scale_path":
                    str(output_scale_path.resolve()) if output_scale_path is not None else None,
                "output_scale_relative_path":
                    output_scale_relative.as_posix() if output_scale_relative is not None else None,
                "output_scale_sha256":
                    sha256_file(output_scale_path) if output_scale_path is not None else None,
                "output_scale_source": {
                    "kind": output_scale.source_kind,
                    "path": str(output_scale.source_path) if output_scale.source_path else None,
                    "sha256":
                        sha256_file(output_scale.source_path)
                        if output_scale.source_path is not None
                        else None,
                },
                "calibration_tokens": tokens,
                "calibration_tokens_scope": tokens_scope,
                "output_zero_point": output_zero_point,
                "bias_mode": cal.get("bias_mode", "none"),
                "bias_path": str(bias_path) if bias_path is not None else None,
                "bias_sha256": sha256_file(bias_path) if bias_path is not None else None,
                "first_page_index": first_page,
                "page_count": page_count,
                "output_wave_count":
                    (page_count + QUANT_TENSOR_CORES - 1) // QUANT_TENSOR_CORES,
                "tc_id_rule": "local_page_index % 64",
                "last_page_valid_channels": channels - (page_count - 1) * QUANT_CHANNELS,
                "payload_relative_path": relative_path.as_posix(),
                "path": str(payload_path.resolve()),
                "bytes": payload_path.stat().st_size,
                "sha256": sha256_file(payload_path),
                "weight_scale_path": str(scale_path),
                "weight_scale_sha256": sha256_file(scale_path),
                "weight_store_binding": weight_store_binding,
                "max_relative_scale_error": local_error_max,
                "mean_relative_scale_error": local_error_sum / channels,
                "bias_acc_min": local_bias_min,
                "bias_acc_max": local_bias_max,
            }
        )
        weight_store_verified += 1
        weight_store_payload_bytes += int(weight_store_binding["weight_payload_bytes"])
        if output_scale.per_output_channel:
            per_output_channel_entries += 1
        else:
            per_tensor_entries += 1
        total_valid_channels += channels
        total_padded_channels += padded_channels - channels
        error_sum += local_error_sum
        error_max = max(error_max, local_error_max)
        bias_min = min(bias_min, local_bias_min)
        bias_max = max(bias_max, local_bias_max)

    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "format_version": 2,
        "schema": OUTPUT_SCHEMA,
        "status": "calibrated_pages_exported"
        if calibration_status == "calibrated"
        else "synthetic_debug_pages_exported",
        "sources": {
            "weight_manifest": str(weight_manifest_path.resolve()),
            "weight_manifest_sha256": sha256_file(weight_manifest_path),
            "calibration_manifest": str(calibration_path.resolve()),
            "calibration_manifest_sha256": sha256_file(calibration_path),
        },
        "weight_store": {
            "binding_verified_entries": weight_store_verified,
            "binding_rule": WEIGHT_STORE_BINDING_RULE,
            "weight_payload_bytes_hashed": weight_store_payload_bytes,
        },
        "calibration": {
            "declared_tokens": calibration.get("calibration_tokens"),
            "output_scale_modes": list(OUTPUT_SCALE_MODES),
            "default_output_scale_mode": "per_tensor",
            "per_tensor_output_scale_entries": per_tensor_entries,
            "per_output_channel_output_scale_entries": per_output_channel_entries,
            "min_tokens_for_per_output_channel_output_scale":
                MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS,
            "refusal_rule": PER_OUTPUT_CHANNEL_TOKEN_RULE,
            "refusal_evidence": PER_OUTPUT_CHANNEL_TOKEN_EVIDENCE,
        },
        "hardware_abi": {
            "channels_per_page": QUANT_CHANNELS,
            "tensor_cores": QUANT_TENSOR_CORES,
            "tc_mem_address_first": f"0x{QUANT_BASE:08x}",
            "tc_mem_address_last": f"0x{QUANT_BASE + QUANT_TENSOR_CORES * QUANT_CHANNELS - 1:08x}",
            "tc_mem_address_formula": "0x7000 | (tc_id << 6) | channel",
            "tc_id_range": [0, QUANT_TENSOR_CORES - 1],
            "record_bytes": PACKED_RECORD_BYTES,
            "page_bytes": PAGE_BYTES,
            "entry_bits": {
                "bias": [47, 0],
                "multiplier": [63, 48],
                "right_shift": [69, 64],
                "zero_point": [78, 70],
                "valid": [79, 79],
                "reserved": [127, 80],
            },
            "effective_scale_formula": "multiplier / 2**right_shift",
            "max_relative_scale_error": max_relative_error,
            "requant_formula": "sat_int8(rne((acc+bias)*effective_scale)+zero_point)",
            "runtime_plan_declaration":
                "quant_page,page_index,tc_id,path,byte_offset,valid_channels",
            "runtime_plan_operation": "run_quant_page,page_index",
        },
        "summary": {
            "gemm_calibration_entries": len(output_entries),
            "quant_pages": next_page,
            "valid_channels": total_valid_channels,
            "padded_channels": total_padded_channels,
            "payload_bytes": (total_valid_channels + total_padded_channels) * PACKED_RECORD_BYTES,
            "max_relative_scale_error": error_max,
            "mean_relative_scale_error": error_sum / total_valid_channels,
            "bias_acc_min": bias_min,
            "bias_acc_max": bias_max,
        },
        "plan_fragment": str(out_plan.resolve()),
        "plan_fragment_sha256": sha256_file(out_plan),
        "entries": output_entries,
        "claims": {
            "quant_metadata_numerically_encoded": True,
            "rtl_entry_layout_matched": True,
            "runtime_page_loader_abi_available": True,
            "gemm_tile_run_sequence_bound": False,
            "high_level_gemm_tile_lowering_complete": False,
            "board_numerical_result_proven": False,
        },
        "next_binding": (
            "Lower each high-level INT8 GEMM into fixed 32x64 hardware tiles, then emit "
            "run_quant_page immediately before the UOP page that launches each output tile."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# pi0 INT8 Quant-Bank Pages",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"- GEMM calibration entries: `{len(output_entries)}`",
        f"- Quant pages / valid channels: `{next_page}` / `{total_valid_channels}`",
        f"- Packed payload bytes: `{manifest['summary']['payload_bytes']}`",
        f"- Maximum relative scale error: `{error_max:.9g}`",
        f"- Accumulator-domain bias range: `{bias_min} .. {bias_max}`",
        f"- Output scale: `{per_tensor_entries}` per_tensor, "
        f"`{per_output_channel_entries}` per_output_channel",
        f"- Declared calibration tokens: `{calibration.get('calibration_tokens')}`",
        "",
        "The output scale is per tensor unless the calibration supplies a "
        "per-output-channel array, and a per-channel array is refused below "
        f"`{MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS}` calibration tokens: at 51 "
        "tokens the per-channel amax is undersampled and per-column requantisation "
        "measures 59% worse than the per-tensor default, at 816 tokens it measures "
        "17.6% better.",
        "",
        "The payload ABI matches `pi0_int8_quant_bank.sv`: each 64-channel page",
        "contains 64 little-endian 128-bit records, with bits 127:80 reserved as zero.",
        "Each declaration targets one physical tensor core; address bits 11:6 carry",
        "`tc_id`, so distinct 64-channel N shards never share requant metadata.",
        "The plan fragment only declares pages. It deliberately does not claim that the",
        "current high-level GEMM schedule has been lowered and bound to output tiles.",
    ]
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return manifest


def make_self_test_inputs(
    root: Path,
    *,
    output_scale_mode: str = "per_tensor",
    output_scale_source: str = "inline",
    calibration_tokens: int | None = None,
    declare_output_scale_mode: bool = True,
) -> tuple[Path, Path]:
    """Build a two-entry weight/calibration pair for the self tests.

    The default arguments reproduce the pre-per-channel manifest byte for byte,
    which is what keeps the old self-test constants meaningful.  With
    ``output_scale_mode="per_output_channel"`` the same scalars are handed over
    as constant per-channel arrays (inline or as a float32 sidecar), so the
    exported pages must come out identical -- a per-channel scale that happens
    to be constant is the scalar case.
    """
    import numpy as np

    if output_scale_mode not in OUTPUT_SCALE_MODES:
        raise ValueError(f"unsupported output_scale_mode: {output_scale_mode!r}")
    if output_scale_source not in {"inline", "sidecar"}:
        raise ValueError(f"unsupported output_scale_source: {output_scale_source!r}")

    weight_entries: list[dict[str, Any]] = []
    calibration_entries: list[dict[str, Any]] = []
    for layer, slot, channels, activation_scale, output_scale in [
        (0, "q", 70, 0.25, 0.5),
        (0, "k", 64, 0.125, 0.25),
    ]:
        scale_relative = Path("scales") / f"{slot}.scale.f32.bin"
        scale_path = root / scale_relative
        scale_path.parent.mkdir(parents=True, exist_ok=True)
        scales = np.linspace(0.25, 1.0, channels, dtype=np.float32)
        scales.tofile(scale_path)
        # A real INT8 payload, because the exporter now refuses a scale sidecar
        # that is not tied by hash to the weights it was derived from.  Column n
        # reaches amax 127 the way quantize_per_output_channel() leaves it.
        weight_relative = Path("payloads") / f"{slot}.int8.bin"
        weight_path = root / weight_relative
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.zeros((32, channels), dtype=np.int8)
        matrix[0, :] = 127
        matrix[1:, :] = np.arange(-31, 0, dtype=np.int8)[:, None]
        matrix.tofile(weight_path)
        weight_entries.append(
            {
                "expert": "prefix",
                "layer": layer,
                "slot": slot,
                "weight_sel": 2 if slot == "q" else 3,
                "matrix_shape": [32, channels],
                "nbytes": 32 * channels,
                "scale_values": channels,
                "payload_relative_path": weight_relative.as_posix(),
                "path": str(weight_path.resolve()),
                "sha256": sha256_file(weight_path),
                "scale_relative_path": scale_relative.as_posix(),
                "scale_path": str(scale_path.resolve()),
                "scale_sha256": sha256_file(scale_path),
                "scale_min": float(np.min(scales)),
                "scale_max": float(np.max(scales)),
            }
        )
        calibration_entry: dict[str, Any] = {
            "expert": "prefix",
            "layer": layer,
            "slot": slot,
            "activation_scale": activation_scale,
            "activation_zero_point": 0,
            "output_scale": output_scale,
            "output_zero_point": 0,
            "bias_mode": "none",
        }
        if output_scale_mode == "per_output_channel":
            if declare_output_scale_mode:
                calibration_entry["output_scale_mode"] = "per_output_channel"
            if output_scale_source == "inline":
                calibration_entry["output_scale"] = [float(output_scale)] * channels
            else:
                oscale_relative = Path("oscales") / f"{slot}.oscale.f32.bin"
                oscale_path = root / oscale_relative
                oscale_path.parent.mkdir(parents=True, exist_ok=True)
                np.full(channels, output_scale, dtype=np.float32).tofile(oscale_path)
                calibration_entry.pop("output_scale")
                calibration_entry["output_scale_relative_path"] = oscale_relative.as_posix()
        calibration_entries.append(calibration_entry)
    weight_manifest = root / "weights.json"
    weight_manifest.write_text(
        json.dumps(
            {
                "schema": WEIGHT_SCHEMA,
                "status": "payloads_exported",
                "entries": weight_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    calibration_document: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "status": "synthetic_debug",
        "entries": calibration_entries,
    }
    if calibration_tokens is not None:
        calibration_document["calibration_tokens"] = calibration_tokens
    elif output_scale_mode == "per_output_channel":
        calibration_document["calibration_tokens"] = (
            2 * MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS
        )
    calibration_manifest = root / "calibration.json"
    calibration_manifest.write_text(
        json.dumps(calibration_document, indent=2) + "\n",
        encoding="utf-8",
    )
    return weight_manifest, calibration_manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pi0_int8_quant_pages_") as temp:
        root = Path(temp)
        weights, calibration = make_self_test_inputs(root)
        out_dir = root / "out"
        manifest = export_quant_pages(
            weight_manifest_path=weights,
            calibration_path=calibration,
            out_dir=out_dir,
            out_json=out_dir / "quant.json",
            out_plan=out_dir / "quant.plan",
            out_markdown=out_dir / "quant.md",
            weight_payload_root=None,
            calibration_root=None,
            allow_debug_calibration=True,
            max_relative_error=1e-4,
        )
        assert manifest["summary"]["quant_pages"] == 3
        assert manifest["summary"]["valid_channels"] == 134
        assert manifest["summary"]["padded_channels"] == 58
        first_payload = Path(manifest["entries"][0]["path"])
        words = struct.unpack("<IIII", first_payload.read_bytes()[:16])
        assert words == (0, 0x00010000, 0x00008003, 0)
        assert encode_effective_scale(1.0, 1e-4)[:2] == (1, 0)
        try:
            encode_effective_scale(math.ldexp(1.0, -64), 1e-4)
        except ValueError as exc:
            assert "below hardware minimum" in str(exc)
        else:
            raise AssertionError("unrepresentable scale was accepted")
        assert all(entry["output_scale_mode"] == "per_tensor" for entry in manifest["entries"])
        assert manifest["calibration"]["per_output_channel_output_scale_entries"] == 0
        assert manifest["calibration"]["declared_tokens"] is None

        def export_case(
            name: str,
            **fixture: Any,
        ) -> dict[str, Any]:
            case_root = root / name
            case_root.mkdir(parents=True, exist_ok=True)
            case_weights, case_calibration = make_self_test_inputs(case_root, **fixture)
            case_out = case_root / "out"
            return export_quant_pages(
                weight_manifest_path=case_weights,
                calibration_path=case_calibration,
                out_dir=case_out,
                out_json=case_out / "quant.json",
                out_plan=case_out / "quant.plan",
                out_markdown=case_out / "quant.md",
                weight_payload_root=None,
                calibration_root=None,
                allow_debug_calibration=True,
                max_relative_error=1e-4,
            )

        # A constant per-channel output scale IS the scalar case, so the pages
        # must come out byte identical however the calibration spelled it.
        baseline_sha = [entry["sha256"] for entry in manifest["entries"]]
        for source in ("inline", "sidecar"):
            per_channel = export_case(
                f"per_channel_{source}",
                output_scale_mode="per_output_channel",
                output_scale_source=source,
            )
            assert [entry["sha256"] for entry in per_channel["entries"]] == baseline_sha
            assert per_channel["calibration"]["per_output_channel_output_scale_entries"] == 2
            assert per_channel["calibration"]["per_tensor_output_scale_entries"] == 0
            assert (
                per_channel["calibration"]["min_tokens_for_per_output_channel_output_scale"]
                == MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS
            )
            for entry in per_channel["entries"]:
                assert entry["output_scale_mode"] == "per_output_channel"
                assert entry["output_scale"] is None
                assert entry["calibration_tokens"] == 2 * MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS
                assert entry["calibration_tokens_scope"] == "manifest"
                assert entry["output_scale_source"]["kind"] == (
                    "inline_array" if source == "inline" else "float32_sidecar"
                )
                folded = Path(entry["output_scale_path"])
                assert folded.is_file()
                assert folded.stat().st_size == 4 * entry["output_channels"]

        # The token floor is a refusal with no flag past it.
        def refuse(name: str, tokens: int | None, fragment: str) -> None:
            case_root = root / name
            case_root.mkdir(parents=True, exist_ok=True)
            case_weights, case_calibration = make_self_test_inputs(
                case_root,
                output_scale_mode="per_output_channel",
            )
            document = json.loads(case_calibration.read_text(encoding="utf-8"))
            if tokens is None:
                document.pop("calibration_tokens", None)
            else:
                document["calibration_tokens"] = tokens
            case_calibration.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            case_out = case_root / "out"
            try:
                export_quant_pages(
                    weight_manifest_path=case_weights,
                    calibration_path=case_calibration,
                    out_dir=case_out,
                    out_json=case_out / "quant.json",
                    out_plan=case_out / "quant.plan",
                    out_markdown=case_out / "quant.md",
                    weight_payload_root=None,
                    calibration_root=None,
                    allow_debug_calibration=True,
                    max_relative_error=1e-4,
                )
            except ValueError as exc:
                assert fragment in str(exc), (fragment, str(exc))
            else:
                raise AssertionError(f"thin per-channel calibration was accepted: {name}")

        refuse("undeclared_tokens", None, "does not declare calibration_tokens")
        refuse("thin_tokens_51", 51, "calibrated on 51 tokens")
        refuse(
            "thin_tokens_816",
            816,
            f"calibration_tokens >= {MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS}",
        )
        refuse(
            "thin_tokens_boundary",
            MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS - 1,
            "refuses a per_output_channel output scale",
        )
        at_floor = export_case(
            "tokens_at_floor",
            output_scale_mode="per_output_channel",
            calibration_tokens=MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS,
        )
        assert [entry["sha256"] for entry in at_floor["entries"]] == baseline_sha

        # The weight-store binding, which is what stops the token floor from
        # being walked around.  Each case doctors exactly one thing about an
        # otherwise honest per-TENSOR export -- no token count is involved --
        # and every one of them has to refuse.
        def refuse_binding(name: str, mutate: Any, fragment: str) -> None:
            case_root = root / name
            case_root.mkdir(parents=True, exist_ok=True)
            case_weights, case_calibration = make_self_test_inputs(case_root)
            mutate(case_root, case_weights)
            case_out = case_root / "out"
            try:
                export_quant_pages(
                    weight_manifest_path=case_weights,
                    calibration_path=case_calibration,
                    out_dir=case_out,
                    out_json=case_out / "quant.json",
                    out_plan=case_out / "quant.plan",
                    out_markdown=case_out / "quant.md",
                    weight_payload_root=None,
                    calibration_root=None,
                    allow_debug_calibration=True,
                    max_relative_error=1e-4,
                )
            except (ValueError, FileNotFoundError) as exc:
                assert fragment in str(exc), (name, fragment, str(exc))
            else:
                raise AssertionError(f"unbound weight scales were accepted: {name}")

        def edit_weight_manifest(path: Path, mutate: Any) -> None:
            document = json.loads(path.read_text(encoding="utf-8"))
            for entry in document["entries"]:
                mutate(entry)
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

        def fold_into_weight_scales(case_root: Path, case_weights: Path) -> None:
            """The documented bypass, verbatim: per-column variation moved out of
            the refused output_scale field and into the weight scales, leaving
            output_scale a scalar and the calibration token count absent."""
            import numpy as np

            for slot in ("q", "k"):
                scale_path = case_root / "scales" / f"{slot}.scale.f32.bin"
                scales = np.fromfile(scale_path, dtype="<f4")
                per_column = np.linspace(0.9, 1.1, scales.size, dtype=np.float32)
                (scales * per_column).astype("<f4").tofile(scale_path)

        refuse_binding(
            "folded_per_column_into_weight_scales",
            fold_into_weight_scales,
            "does not match the weight store",
        )
        refuse_binding(
            "missing_scale_sha256",
            lambda case_root, case_weights: edit_weight_manifest(
                case_weights, lambda entry: entry.pop("scale_sha256", None)
            ),
            "does not declare scale_sha256",
        )
        refuse_binding(
            "missing_payload_sha256",
            lambda case_root, case_weights: edit_weight_manifest(
                case_weights, lambda entry: entry.pop("sha256", None)
            ),
            "does not declare sha256 for its INT8 payload",
        )
        refuse_binding(
            "swapped_weight_payload",
            lambda case_root, case_weights: (
                case_root / "payloads" / "q.int8.bin"
            ).write_bytes(
                bytes(1 for _ in range((case_root / "payloads" / "q.int8.bin").stat().st_size))
            ),
            "INT8 weight payload does not match the weight store",
        )
        refuse_binding(
            "absent_weight_payload",
            lambda case_root, case_weights: [
                (case_root / "payloads" / f"{slot}.int8.bin").unlink() for slot in ("q", "k")
            ]
            and edit_weight_manifest(
                case_weights, lambda entry: entry.__setitem__("path", None)
            ),
            "INT8 weight payload is missing",
        )
        refuse_binding(
            "restamped_scale_extrema",
            lambda case_root, case_weights: edit_weight_manifest(
                case_weights, lambda entry: entry.__setitem__("scale_max", 2.0)
            ),
            "weight manifest declares",
        )

        # The honest export still records what it verified.
        assert manifest["weight_store"]["binding_verified_entries"] == 2
        assert manifest["weight_store"]["weight_payload_bytes_hashed"] == 32 * (70 + 64)
        assert all(
            entry["weight_store_binding"]["verified"]
            and entry["weight_store_binding"]["scale_extrema_matched"]
            for entry in manifest["entries"]
        )

        # The floor gates per-channel scales only; a per-tensor export never
        # needed a token count and still does not.
        thin_per_tensor = export_case("thin_per_tensor", calibration_tokens=51)
        assert [entry["sha256"] for entry in thin_per_tensor["entries"]] == baseline_sha
        assert thin_per_tensor["calibration"]["declared_tokens"] == 51

        print(
            json.dumps(
                {
                    "status": "ok",
                    "self_test": "pi0_int8_quant_bank_pages",
                    "summary": manifest["summary"],
                    "per_output_channel_output_scale": {
                        "constant_array_matches_scalar": True,
                        "min_calibration_tokens": MIN_PER_OUTPUT_CHANNEL_CALIBRATION_TOKENS,
                        "refused": ["undeclared", "51", "816", "1023"],
                    },
                    "weight_store_binding": {
                        "rule": WEIGHT_STORE_BINDING_RULE,
                        "refused": [
                            "folded_per_column_into_weight_scales",
                            "missing_scale_sha256",
                            "missing_payload_sha256",
                            "swapped_weight_payload",
                            "absent_weight_payload",
                            "restamped_scale_extrema",
                        ],
                    },
                },
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-plan", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--weight-payload-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--allow-debug-calibration", action="store_true")
    parser.add_argument("--max-relative-error", type=float, default=1e-4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    required = {
        "--weight-manifest": args.weight_manifest,
        "--calibration": args.calibration,
        "--out-dir": args.out_dir,
        "--out-json": args.out_json,
        "--out-plan": args.out_plan,
        "--out-md": args.out_md,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"required arguments missing: {', '.join(missing)}")
    manifest = export_quant_pages(
        weight_manifest_path=args.weight_manifest.expanduser().resolve(),
        calibration_path=args.calibration.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        out_json=args.out_json.expanduser().resolve(),
        out_plan=args.out_plan.expanduser().resolve(),
        out_markdown=args.out_md.expanduser().resolve(),
        weight_payload_root=args.weight_payload_root.expanduser().resolve()
        if args.weight_payload_root
        else None,
        calibration_root=args.calibration_root.expanduser().resolve()
        if args.calibration_root
        else None,
        allow_debug_calibration=args.allow_debug_calibration,
        max_relative_error=args.max_relative_error,
    )
    print(json.dumps({"status": manifest["status"], "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()

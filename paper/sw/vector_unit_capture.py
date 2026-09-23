#!/usr/bin/env python3
"""Capture real pi0 activations at every vector-unit op site, for paper/sw/vector_unit_ref.py.

Runs ONE real frame through the fp32 LeRobot model on the CPU -- SigLIP on the two cameras, the
compact PaliGemma prefix, and the 10-step torch expert on the captured prefix KV -- with forward
hooks that record the float inputs of each non-GEMM op (the ops of the fabric vector unit):

  SigLIP  layers {0,13,26}: LayerNorm 1/2 in (all rows) + gamma/beta, fc1 out (GELU in, 96 rows),
          layer input + attention output (residual add), post-attention q/k (softmax logits),
          patch-embedding out + position table (pos-emb add), post_layernorm in
  LM      layers {0,9,17}: RMSNorm 1/2 in (all 525 rows) + gains, q/k before RoPE + cos/sin,
          q/k after RoPE (logits), gate/up out (GeGLU in, 48 rows), o_proj in (attention context),
          layer input + o_proj out (residual add)
  expert  layers {0,9,17} at Euler steps 0 and 9: the same set with the 867-key masked softmax,
          action_time_mlp_in out (SiLU in) and action_out_proj out (v_t) at all 10 steps, final norm in

Output: build/paper_vector_unit/acts_<episode>_f<NN>.npz (float32, plus a JSON manifest of shapes).

Memory: the checkpoint is mmap-shared with any other process that maps it (page cache); run as
  systemd-run --user --scope -q -p MemoryHigh=6G nice -n 10 \\
      ~/lerobot/.venv/bin/python paper/sw/vector_unit_capture.py --threads 6
"""
from __future__ import annotations

import sys

if "--threads" not in sys.argv:
    sys.argv += ["--threads", "6"]

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prefix_w8a8_eval as P  # noqa: E402  (patches the eager attention functions at import; mode "fp")
from transformers.models.gemma import modeling_gemma as MG  # noqa: E402
from transformers.models.siglip import modeling_siglip as MS  # noqa: E402

VIS_LAYERS = (0, 13, 26)
LM_LAYERS = (0, 9, 17)
EX_LAYERS = (0, 9, 17)
EX_STEPS = (0, 9)
S: dict = {}
STATE = {"step": -1, "stage": ""}


def _n(*_):
    """Hook bodies must return None: a forward hook's non-None return value replaces the output."""
    return None


def put(key, t, rows=None):
    x = t.detach().float()
    if rows is not None:
        x = x[..., rows, :]
    S.setdefault(key, []).append(x.cpu().numpy().astype(np.float32).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--frame", default="demo1_ep20:2")
    ap.add_argument("--out", default=str(P.REPO / "build" / "paper_vector_unit"))
    a = ap.parse_args()
    ep, f = a.frame.split(":")
    fr = (ep, int(f))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    P.G.cache_dir = out_dir / "cache"
    t0 = time.time()
    P.G.policy, _, _ = P.load_policy(P.CKPT)
    P.G.pwe = pwe = P.G.policy.model.paligemma_with_expert
    P.log(f"policy loaded in {time.time() - t0:.0f} s")

    hooks = []
    vm = pwe.paligemma.model.vision_tower.vision_model
    for c in {id(l.self_attn.config): l.self_attn.config for l in vm.encoder.layers}.values():
        c._attn_implementation = "eager"
    attn_names = {}

    # ---------------- SigLIP ----------------
    hooks.append(vm.embeddings.patch_embedding.register_forward_hook(
        lambda m, i, o: put("vis.patch_out", o.flatten(2).transpose(1, 2)[0])))
    S["vis.pos_table"] = [vm.embeddings.position_embedding.weight.detach().float().numpy().copy()]
    for li in VIS_LAYERS:
        layer = vm.encoder.layers[li]
        attn_names[id(layer.self_attn)] = f"vis.L{li}"
        for nm in ("layer_norm1", "layer_norm2"):
            ln = getattr(layer, nm)
            S[f"vis.L{li}.{nm}.gamma"] = [ln.weight.detach().float().numpy().copy()]
            S[f"vis.L{li}.{nm}.beta"] = [ln.bias.detach().float().numpy().copy()]
            hooks.append(ln.register_forward_hook(lambda m, i, o, k=f"vis.L{li}.{nm}.in": put(k, i[0][0])))
        rows96 = torch.linspace(0, 255, 96).long()
        hooks.append(layer.mlp.fc1.register_forward_hook(
            lambda m, i, o, k=f"vis.L{li}.fc1_out": put(k, o[0], rows96)))
        hooks.append(layer.self_attn.register_forward_hook(
            lambda m, i, o, k=f"vis.L{li}.attn_out": put(k, o[0][0])))
        hooks.append(layer.register_forward_pre_hook(
            lambda m, args, kw, k=f"vis.L{li}.layer_in": put(k, (args[0] if args else kw["hidden_states"])[0]),
            with_kwargs=True))
    S["vis.post_ln.gamma"] = [vm.post_layernorm.weight.detach().float().numpy().copy()]
    S["vis.post_ln.beta"] = [vm.post_layernorm.bias.detach().float().numpy().copy()]
    hooks.append(vm.post_layernorm.register_forward_hook(lambda m, i, o: put("vis.post_ln.in", i[0][0])))

    # ---------------- LM prefix + expert ----------------
    lm_rows48 = torch.linspace(0, 524, 48).long()
    for li in LM_LAYERS:
        pl = pwe.joint_layers[li].paligemma_layer
        attn_names[id(pl.self_attn)] = f"lm.L{li}"
        for nm in ("input_layernorm", "post_attention_layernorm"):
            n = getattr(pl, nm)
            S[f"lm.L{li}.{nm}.gain"] = [n.weight.detach().float().numpy().copy()]
            hooks.append(n.register_forward_hook(lambda m, i, o, k=f"lm.L{li}.{nm}.in": put(k, i[0][0])))
        hooks.append(pl.self_attn.q_proj.register_forward_hook(lambda m, i, o, k=f"lm.L{li}.q_pre": put(k, o[0])))
        hooks.append(pl.self_attn.k_proj.register_forward_hook(lambda m, i, o, k=f"lm.L{li}.k_pre": put(k, o[0])))
        hooks.append(pl.self_attn.o_proj.register_forward_hook(
            lambda m, i, o, k=f"lm.L{li}": _n(put(k + ".o_in", i[0][0]), put(k + ".o_out", o[0]))))
        hooks.append(pl.mlp.gate_proj.register_forward_hook(
            lambda m, i, o, k=f"lm.L{li}.gate_out": put(k, o[0], lm_rows48)))
        hooks.append(pl.mlp.up_proj.register_forward_hook(
            lambda m, i, o, k=f"lm.L{li}.up_out": put(k, o[0], lm_rows48)))
        hooks.append(pl.register_forward_pre_hook(
            lambda m, args, kw, k=f"lm.L{li}.layer_in": put(k, (args[0] if args else kw["hidden_states"])[0]),
            with_kwargs=True))

    def ex_on():
        return STATE["stage"] == "expert" and STATE["step"] in EX_STEPS

    for li in EX_LAYERS:
        xl = pwe.joint_layers[li].expert_layer
        attn_names[id(xl.self_attn)] = f"ex.L{li}"
        for nm in ("input_layernorm", "post_attention_layernorm"):
            n = getattr(xl, nm)
            S[f"ex.L{li}.{nm}.gain"] = [n.weight.detach().float().numpy().copy()]
            hooks.append(n.register_forward_hook(
                lambda m, i, o, k=f"ex.L{li}.{nm}.in": put(k, i[0][0]) if ex_on() else None))
        for proj, key in (("q_proj", "q_pre"), ("k_proj", "k_pre")):
            hooks.append(getattr(xl.self_attn, proj).register_forward_hook(
                lambda m, i, o, k=f"ex.L{li}.{key}": put(k, o[0]) if ex_on() else None))
        hooks.append(xl.self_attn.o_proj.register_forward_hook(
            lambda m, i, o, k=f"ex.L{li}": _n(put(k + ".o_in", i[0][0]), put(k + ".o_out", o[0])) if ex_on() else None))
        hooks.append(xl.mlp.gate_proj.register_forward_hook(
            lambda m, i, o, k=f"ex.L{li}.gate_out": put(k, o[0]) if ex_on() else None))
        hooks.append(xl.mlp.up_proj.register_forward_hook(
            lambda m, i, o, k=f"ex.L{li}.up_out": put(k, o[0]) if ex_on() else None))
        hooks.append(xl.register_forward_pre_hook(
            lambda m, args, kw, k=f"ex.L{li}.layer_in": put(k, (args[0] if args else kw["hidden_states"])[0])
            if ex_on() else None, with_kwargs=True))
    ge = pwe.gemma_expert.model
    S["ex.final_norm.gain"] = [ge.norm.weight.detach().float().numpy().copy()]
    hooks.append(ge.norm.register_forward_hook(
        lambda m, i, o: put("ex.final_norm.in", i[0][0]) if ex_on() else None))
    model = P.G.policy.model

    def step_hook(m, i, o):
        STATE["step"] += 1

    hooks.append(model.action_in_proj.register_forward_hook(step_hook))
    hooks.append(model.action_time_mlp_in.register_forward_hook(
        lambda m, i, o: put("ex.silu_in", o[0]) if STATE["stage"] == "expert" else None))
    hooks.append(model.action_out_proj.register_forward_hook(
        lambda m, i, o: put("ex.v_t", o[0]) if STATE["stage"] == "expert" else None))
    # rotary tables (cos, sin) as the Gemma modules see them
    for nm, rot in (("lm", pwe.paligemma.model.language_model.rotary_emb), ("ex", ge.rotary_emb)):
        hooks.append(rot.register_forward_hook(
            lambda m, i, o, k=nm: _n(put(k + ".rope_cos", o[0][0]), put(k + ".rope_sin", o[1][0]))
            if (k == "lm" and STATE["stage"] == "prefix") or (k == "ex" and ex_on()) else None))

    # attention: record post-RoPE q, k, the additive mask and the scaling
    inner_g, inner_s = MG.eager_attention_forward, MS.eager_attention_forward

    def wrap(inner):
        def f(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
            name = attn_names.get(id(module))
            if name is not None and (not name.startswith("ex.") or ex_on()):
                put(name + ".q", query[0])
                put(name + ".k", key[0])
                if attention_mask is not None:
                    S.setdefault(name + ".mask", []).append((attention_mask[0, 0] == 0).cpu().numpy().copy())
                S[name + ".scaling"] = [np.float32(scaling)]
            return inner(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        return f

    MG.eager_attention_forward = wrap(inner_g)
    MS.eager_attention_forward = wrap(inner_s)
    pwe.paligemma.model.language_model.config._attn_implementation = "eager"
    ge.config._attn_implementation = "eager"

    fin, cap = P.get_inputs(fr), P.get_capture(fr)
    STATE["stage"] = "prefix"
    t0 = time.time()
    out = P.prefix_compact(fin)
    P.log(f"prefix {time.time() - t0:.1f} s")
    k_rel = max(P.rel(out["K"][li], cap["K"][li]) for li in range(out["K"].shape[0]))
    STATE["stage"], STATE["step"] = "expert", -1
    t0 = time.time()
    actions = P.run_expert(cap["K"], cap["V"], cap["valid"], cap)
    P.log(f"expert {time.time() - t0:.1f} s, K rel vs capture max {k_rel:.2e}, "
          f"actions vs capture max abs {np.abs(actions - cap['actions']).max():.2e}")
    for h in hooks:
        h.remove()
    S["ex.noise"] = [cap["noise"].astype(np.float32)]
    S["ex.actions_fp32"] = [actions.astype(np.float32)]
    arrays, manifest = {}, {}
    for k, v in S.items():
        arr = np.stack(v) if len(v) > 1 else np.asarray(v[0])
        arrays[k] = arr
        manifest[k] = [list(arr.shape), str(arr.dtype)]
    tag = f"{ep}_f{int(f):02d}"
    np.savez(out_dir / f"acts_{tag}.npz", **arrays)
    meta = dict(frame=a.frame, prefix_K_rel_vs_capture=k_rel,
                actions_max_abs_vs_capture=float(np.abs(actions - cap["actions"]).max()),
                vis_layers=VIS_LAYERS, lm_layers=LM_LAYERS, ex_layers=EX_LAYERS, ex_steps=EX_STEPS,
                manifest=manifest)
    (out_dir / f"acts_{tag}.json").write_text(json.dumps(meta, indent=1))
    P.log(f"wrote {out_dir}/acts_{tag}.npz ({sum(x.nbytes for x in arrays.values()) / 1e6:.0f} MB, {len(arrays)} arrays)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate tools/golden/blocks.npz — numpy reference of src/model/blocks.{h,cpp},
itself ported from Ovi/ovi/modules/model.py (+ ovi_fusion_engine.py dims).

Inputs/weights drawn fp32 from one default_rng(42) stream; reference math in
float64 on those exact fp32 values; outputs stored fp32.
np.savez on purpose (STORED zip entries only — npz.cpp limit)."""
import os

import numpy as np

rng = np.random.default_rng(42)

DIM, HEADS, FFN = 3072, 24, 14336  # video.json / audio.json
HD = DIM // HEADS  # 128
EPS = 1e-6
L, LC = 64, 32  # video tokens (grid 4x4x4) / context tokens
T, H, W = 4, 4, 4
AUDIO_SCALE = 0.19676  # audio.json temporal_rope_scaling_factor

out = {}


def lin(in_, out_dim):
    return (rng.standard_normal((in_, out_dim)) * 0.02).astype(np.float32)


def bias(n):
    return (rng.standard_normal(n) * 0.02).astype(np.float32)


def nrmw(n):
    return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)


# ---- reference ops (f64) ----
def rmsnorm(x, w):
    x = x.astype(np.float64)
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w.astype(np.float64)


def layernorm(x, w=None, b=None):
    x = x.astype(np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + EPS)
    if w is not None:
        y = y * w.astype(np.float64) + b.astype(np.float64)
    return y


def rope_params(max_len, real_dims, theta=10000.0, scaling=1.0):
    # model.py:38-45
    j = np.arange(0, real_dims, 2, dtype=np.float64)
    freqs = scaling * (1.0 / np.power(theta, j / real_dims))
    return np.outer(np.arange(max_len, dtype=np.float64), freqs)


def rope_angles_3d(t, h, w, head_dim=HD):
    # model.py:681-686 + 77-92: t axis gets d-4*(d//6) real dims, h/w get 2*(d//6)
    d = head_dim
    t_real, s_real = d - 4 * (d // 6), 2 * (d // 6)
    ft = rope_params(t, t_real)
    fs = rope_params(max(h, w), s_real)
    ang = np.concatenate([np.broadcast_to(ft[:, None, None, :], (t, h, w, t_real // 2)),
                          np.broadcast_to(fs[None, :h, None, :], (t, h, w, s_real // 2)),
                          np.broadcast_to(fs[None, None, :w, :], (t, h, w, s_real // 2))],
                         axis=-1).reshape(t * h * w, -1)
    return ang


def apply_rope(x, ang):
    # model.py:72-100 (3d full rotate; 1d rotates first ang.shape[1] pairs)
    Lx, C = x.shape
    xc = x.astype(np.float64).reshape(Lx, HEADS, HD // 2, 2)
    cplx = xc[..., 0] + 1j * xc[..., 1]  # [L, H, 64]
    a = ang.astype(np.float64)
    nrot = a.shape[1]
    cplx[:, :, :nrot] *= (np.cos(a) + 1j * np.sin(a))[:, None, :]
    return np.stack([cplx.real, cplx.imag], -1).reshape(Lx, C)


def sdpa(q, k, v):
    Lq, C = q.shape
    Lk = k.shape[0]
    hd = C // HEADS
    qh = q.astype(np.float64).reshape(Lq, HEADS, hd).transpose(1, 0, 2)
    kh = k.astype(np.float64).reshape(Lk, HEADS, hd).transpose(1, 0, 2)
    vh = v.astype(np.float64).reshape(Lk, HEADS, hd).transpose(1, 0, 2)
    s = qh @ kh.transpose(0, 2, 1) * hd ** -0.5
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return (p @ vh).transpose(1, 0, 2).reshape(Lq, C)


def linf(x, w, b):
    return x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)


# ---- component: rope (video 3D grid 4x4x4, audio 1D scaled) ----
rope_x = rng.standard_normal((L, DIM)).astype(np.float32)
v_ang = rope_angles_3d(T, H, W)
a_ang = rope_params(L, HD - 4 * (HD // 6), scaling=AUDIO_SCALE)  # model.py:679
out["rope_x"] = rope_x
out["rope_v_ang"] = v_ang.astype(np.float32)
out["rope_a_ang"] = a_ang.astype(np.float32)
out["rope_v_out"] = apply_rope(rope_x, v_ang).astype(np.float32)
out["rope_a_out"] = apply_rope(rope_x.copy(), a_ang).astype(np.float32)

# ---- weights (shared by component tests and the composite block) ----
sa = dict(wqkv=lin(DIM, 3 * DIM), bqkv=bias(3 * DIM),
          nq=nrmw(DIM), nk=nrmw(DIM), wo=lin(DIM, DIM), bo=bias(DIM))
ca = dict(wq=lin(DIM, DIM), bq=bias(DIM), wk=lin(DIM, DIM), bk=bias(DIM),
          wv=lin(DIM, DIM), bv=bias(DIM), wo=lin(DIM, DIM), bo=bias(DIM),
          nq=nrmw(DIM), nk=nrmw(DIM))
ffn = dict(w1=lin(DIM, FFN), b1=bias(FFN), w2=lin(FFN, DIM), b2=bias(DIM))
for pfx, d in (("sa_", sa), ("ca_", ca), ("ffn_", ffn)):
    for k, v in d.items():
        out[pfx + k] = v
out["ln3_w"] = nrmw(DIM)
out["ln3_b"] = bias(DIM)
out["mod_bias"] = (rng.standard_normal((6, DIM)) / DIM ** 0.5).astype(np.float32)  # model.py:371

# ---- component: self_attn (model.py:198-265) ----
sa_x = rng.standard_normal((L, DIM)).astype(np.float32)
q = rmsnorm(linf(sa_x, sa["wqkv"][:, :DIM], sa["bqkv"][:DIM]), sa["nq"])
k = rmsnorm(linf(sa_x, sa["wqkv"][:, DIM:2 * DIM], sa["bqkv"][DIM:2 * DIM]), sa["nk"])
v = linf(sa_x, sa["wqkv"][:, 2 * DIM:], sa["bqkv"][2 * DIM:])
q = apply_rope(q, v_ang)
k = apply_rope(k, v_ang)
out["sa_x"] = sa_x
out["sa_out"] = (linf(sdpa(q, k, v), sa["wo"], sa["bo"])).astype(np.float32)

# ---- component: cross_attn (model.py:268-294, no rope) ----
ca_x = rng.standard_normal((L, DIM)).astype(np.float32)
ca_ctx = rng.standard_normal((LC, DIM)).astype(np.float32)
cq = rmsnorm(linf(ca_x, ca["wq"], ca["bq"]), ca["nq"])
ck = rmsnorm(linf(ca_ctx, ca["wk"], ca["bk"]), ca["nk"])
cv = linf(ca_ctx, ca["wv"], ca["bv"])
out["ca_x"] = ca_x
out["ca_ctx"] = ca_ctx
out["ca_out"] = (linf(sdpa(cq, ck, cv), ca["wo"], ca["bo"])).astype(np.float32)

# ---- component: ffn (gelu tanh) ----
ffn_x = rng.standard_normal((L, DIM)).astype(np.float32)
h = linf(ffn_x, ffn["w1"], ffn["b1"])
h = 0.5 * h * (1.0 + np.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
out["ffn_x"] = ffn_x
out["ffn_out"] = linf(h, ffn["w2"], ffn["b2"]).astype(np.float32)

# ---- component: modulation apply (ffn-branch convention, model.py:464-467) ----
mod_x = rng.standard_normal((L, DIM)).astype(np.float32)
mod_e = rng.standard_normal((L, 6, DIM)).astype(np.float32)
m = layernorm(mod_x) * (1 + mod_e[:, 4].astype(np.float64)) + mod_e[:, 3].astype(np.float64)
out["mod_x"] = mod_x
out["mod_e"] = mod_e
out["mod_out"] = (mod_x.astype(np.float64) + m * mod_e[:, 5].astype(np.float64)).astype(np.float32)

# ---- composite transformer block (model.py:430-471) ----
blk_x = rng.standard_normal((L, DIM)).astype(np.float32)
blk_e = rng.standard_normal((L, 6, DIM)).astype(np.float32)
x64 = blk_x.astype(np.float64)
e = (blk_e.astype(np.float64) + out["mod_bias"].astype(np.float64)[None])  # ModulationAdd
h = layernorm(x64) * (1 + e[:, 1]) + e[:, 0]
q = rmsnorm(linf(h, sa["wqkv"][:, :DIM], sa["bqkv"][:DIM]), sa["nq"])
k = rmsnorm(linf(h, sa["wqkv"][:, DIM:2 * DIM], sa["bqkv"][DIM:2 * DIM]), sa["nk"])
v = linf(h, sa["wqkv"][:, 2 * DIM:], sa["bqkv"][2 * DIM:])
q = apply_rope(q, v_ang)
k = apply_rope(k, v_ang)
x64 = x64 + linf(sdpa(q, k, v), sa["wo"], sa["bo"]) * e[:, 2]
cq = rmsnorm(linf(layernorm(x64, out["ln3_w"], out["ln3_b"]), ca["wq"], ca["bq"]), ca["nq"])
ck = rmsnorm(linf(ca_ctx, ca["wk"], ca["bk"]), ca["nk"])
cv = linf(ca_ctx, ca["wv"], ca["bv"])
x64 = x64 + linf(sdpa(cq, ck, cv), ca["wo"], ca["bo"])
h = layernorm(x64) * (1 + e[:, 4]) + e[:, 3]
h = linf(h, ffn["w1"], ffn["b1"])
h = 0.5 * h * (1.0 + np.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
x64 = x64 + linf(h, ffn["w2"], ffn["b2"]) * e[:, 5]
out["blk_x"] = blk_x
out["blk_e"] = blk_e
out["blk_out"] = x64.astype(np.float32)

os.makedirs("tools/golden", exist_ok=True)
np.savez("tools/golden/blocks.npz", **out)
print("wrote tools/golden/blocks.npz:", len(out), "arrays")

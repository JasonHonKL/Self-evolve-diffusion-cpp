#!/usr/bin/env python3
"""Generate tools/golden/ovit_golden.npz — numpy reference of the scaled-down
OviDit dual-backbone fusion forward (src/model/ovi_dit.{h,cpp}), itself ported
from Ovi/ovi/modules/{fusion,model}.py.

Reference math float64 on the exact fp32 weights (gen_blocks.py conventions).
Weights are stored under checkpoint-style key names (torch layout) so the C++
test exercises the same load() name-mapping used for the real safetensors."""
import numpy as np

rng = np.random.default_rng(7)

# OviDitCfg::test() — dims /8, 2+2 layers, fusion at layer 1
DIM, HEADS, FFN, FREQD, TEXTD, TEXTLEN = 384, 3, 1792, 32, 512, 64
NL = 2
FUSION = {1}
VIN, VOUT, AIN, AOUT = 48, 48, 20, 20
PT, PH, PW = 1, 2, 2
ASC = 0.19676  # audio.json temporal_rope_scaling_factor
HID = 256 * ((2 * 4 * DIM // 3 + 255) // 256)  # ConvMLP hidden, model.py:144-145
HD = DIM // HEADS
EPS = 1e-6

F, H, W = 4, 8, 8
Hp, Wp = H // PH, W // PW
LA = 48
LC = 40
T = 0.37

out = {}


def save(n, a):
    out[n] = np.ascontiguousarray(a, dtype=np.float32)


def lin(o, i):
    return (rng.standard_normal((o, i)) * 0.02).astype(np.float32)


def bias(n):
    return (rng.standard_normal(n) * 0.02).astype(np.float32)


def nrmw(n):
    return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)


def convw(o, i):
    return (rng.standard_normal((o, i, 7)) * 0.02).astype(np.float32)


# ---- weights (checkpoint naming, fusion.py:40-52 injections on FUSION) ----
def gen_backbone(pfx, in_dim, out_dim, audio):
    if audio:
        save(pfx + "patch_embedding.0.weight", convw(DIM, in_dim))
        save(pfx + "patch_embedding.0.bias", bias(DIM))
        save(pfx + "patch_embedding.2.w1.weight", convw(HID, DIM))
        save(pfx + "patch_embedding.2.w2.weight", convw(DIM, HID))
        save(pfx + "patch_embedding.2.w3.weight", convw(HID, DIM))
    else:
        save(pfx + "patch_embedding.weight",
             (rng.standard_normal((DIM, in_dim, PT, PH, PW)) * 0.05).astype(np.float32))
        save(pfx + "patch_embedding.bias", bias(DIM))
    save(pfx + "text_embedding.0.weight", lin(DIM, TEXTD))
    save(pfx + "text_embedding.0.bias", bias(DIM))
    save(pfx + "text_embedding.2.weight", lin(DIM, DIM))
    save(pfx + "text_embedding.2.bias", bias(DIM))
    save(pfx + "time_embedding.0.weight", lin(DIM, FREQD))
    save(pfx + "time_embedding.0.bias", bias(DIM))
    save(pfx + "time_embedding.2.weight", lin(DIM, DIM))
    save(pfx + "time_embedding.2.bias", bias(DIM))
    save(pfx + "time_projection.1.weight", lin(6 * DIM, DIM))
    save(pfx + "time_projection.1.bias", bias(6 * DIM))
    pout = (1 if audio else PT * PH * PW) * out_dim
    save(pfx + "head.head.weight", lin(pout, DIM))
    save(pfx + "head.head.bias", bias(pout))
    save(pfx + "head.modulation", (rng.standard_normal((1, 2, DIM)) / DIM ** 0.5).astype(np.float32))
    for i in range(NL):
        B = f"{pfx}blocks.{i}."
        for w in "qkov":
            save(B + "self_attn." + w + ".weight", lin(DIM, DIM))
            save(B + "self_attn." + w + ".bias", bias(DIM))
            save(B + "cross_attn." + w + ".weight", lin(DIM, DIM))
            save(B + "cross_attn." + w + ".bias", bias(DIM))
        save(B + "self_attn.norm_q.weight", nrmw(DIM))
        save(B + "self_attn.norm_k.weight", nrmw(DIM))
        save(B + "cross_attn.norm_q.weight", nrmw(DIM))
        save(B + "cross_attn.norm_k.weight", nrmw(DIM))
        save(B + "norm3.weight", nrmw(DIM))
        save(B + "norm3.bias", bias(DIM))
        save(B + "ffn.0.weight", lin(FFN, DIM))
        save(B + "ffn.0.bias", bias(FFN))
        save(B + "ffn.2.weight", lin(DIM, FFN))
        save(B + "ffn.2.bias", bias(DIM))
        save(B + "modulation.modulation", (rng.standard_normal((1, 6, DIM)) / DIM ** 0.5).astype(np.float32))
        if i in FUSION:
            save(B + "cross_attn.k_fusion.weight", lin(DIM, DIM))
            save(B + "cross_attn.k_fusion.bias", bias(DIM))
            save(B + "cross_attn.v_fusion.weight", lin(DIM, DIM))
            save(B + "cross_attn.v_fusion.bias", bias(DIM))
            save(B + "cross_attn.pre_attn_norm_fusion.weight", nrmw(DIM))
            save(B + "cross_attn.pre_attn_norm_fusion.bias", bias(DIM))
            save(B + "cross_attn.norm_k_fusion.weight", nrmw(DIM))


gen_backbone("video_model.", VIN, VOUT, audio=False)
gen_backbone("audio_model.", AIN, AOUT, audio=True)

in_video = rng.standard_normal((VIN, F, H, W)).astype(np.float32)
in_audio = rng.standard_normal((LA, AIN)).astype(np.float32)
in_vctx = rng.standard_normal((LC, TEXTD)).astype(np.float32)
in_actx = rng.standard_normal((LC, TEXTD)).astype(np.float32)
save("in_video", in_video)
save("in_audio", in_audio)
save("in_vctx", in_vctx)
save("in_actx", in_actx)
save("t", np.array([T], dtype=np.float32))


# ---- reference ops (float64 on fp32 weights, gen_blocks.py math) ----
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


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_params(max_len, real_dims, theta=10000.0, scaling=1.0):
    j = np.arange(0, real_dims, 2, dtype=np.float64)
    freqs = scaling * (1.0 / np.power(theta, j / real_dims))
    return np.outer(np.arange(max_len, dtype=np.float64), freqs)


def rope_angles_3d(t, h, w):
    t_real, s_real = HD - 4 * (HD // 6), 2 * (HD // 6)
    ft = rope_params(t, t_real)
    fs = rope_params(max(h, w), s_real)
    return np.concatenate(
        [np.broadcast_to(ft[:, None, None, :], (t, h, w, t_real // 2)),
         np.broadcast_to(fs[None, :h, None, :], (t, h, w, s_real // 2)),
         np.broadcast_to(fs[None, None, :w, :], (t, h, w, s_real // 2))],
        axis=-1).reshape(t * h * w, -1)


def apply_rope(x, ang):
    Lx, C = x.shape
    xc = x.astype(np.float64).reshape(Lx, HEADS, HD // 2, 2)
    cplx = xc[..., 0] + 1j * xc[..., 1]
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
    return x.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)


def conv1d(x, w, b=None):
    # ChannelLastConv1d, k=7 p=3 (model.py:110-116)
    L = x.shape[0]
    K = w.shape[2]
    P = K // 2
    xp = np.zeros((L + 2 * P, w.shape[1]))
    xp[P:P + L] = x.astype(np.float64)
    y = np.zeros((L, w.shape[0]))
    if b is not None:
        y += b.astype(np.float64)
    for j in range(K):
        y += xp[j:j + L] @ w[:, :, j].astype(np.float64).T
    return y


gt = lambda n: out[n]  # noqa: E731

# ---- embeddings ----
def embed_video(x):
    w = gt("video_model.patch_embedding.weight").astype(np.float64)
    b = gt("video_model.patch_embedding.bias").astype(np.float64)
    C = x.shape[0]
    xr = x.astype(np.float64).reshape(C, F, Hp, PH, Wp, PW)
    xr = xr.transpose(1, 2, 4, 0, 3, 5).reshape(F * Hp * Wp, C * PT * PH * PW)
    return xr @ w.reshape(DIM, -1).T + b


def embed_audio(x):
    p = "audio_model.patch_embedding."
    h = conv1d(x, gt(p + "0.weight"), gt(p + "0.bias"))
    h = silu(h)
    a = silu(conv1d(h, gt(p + "2.w1.weight"))) * conv1d(h, gt(p + "2.w3.weight"))
    return conv1d(a, gt(p + "2.w2.weight"))


def embed_text(pfx, ctx):
    e = gelu(linf(ctx, gt(pfx + "text_embedding.0.weight"), gt(pfx + "text_embedding.0.bias")))
    e = linf(e, gt(pfx + "text_embedding.2.weight"), gt(pfx + "text_embedding.2.bias"))
    full = np.zeros((TEXTLEN, DIM))
    full[: ctx.shape[0]] = e  # zero-pad, padded rows attended (model.py:784-789)
    return full


def time_rows(pfx):
    half = FREQD // 2
    f = 10000.0 ** (-np.arange(half) / half)
    pos = np.concatenate([np.cos(T * f), np.sin(T * f)])  # model.py:24-34
    e = silu(linf(pos[None], gt(pfx + "time_embedding.0.weight"), gt(pfx + "time_embedding.0.bias")))[0]
    e = linf(e[None], gt(pfx + "time_embedding.2.weight"), gt(pfx + "time_embedding.2.bias"))[0]
    # time_projection = SiLU + Linear (model.py:633)
    e6 = linf(silu(e)[None], gt(pfx + "time_projection.1.weight"), gt(pfx + "time_projection.1.bias"))[0]
    return e, e6.reshape(6, DIM)


# ---- block pieces ----
def self_attn(B, x, ang):
    q = rmsnorm(linf(x, gt(B + ".self_attn.q.weight"), gt(B + ".self_attn.q.bias")), gt(B + ".self_attn.norm_q.weight"))
    k = rmsnorm(linf(x, gt(B + ".self_attn.k.weight"), gt(B + ".self_attn.k.bias")), gt(B + ".self_attn.norm_k.weight"))
    v = linf(x, gt(B + ".self_attn.v.weight"), gt(B + ".self_attn.v.bias"))
    return linf(sdpa(apply_rope(q, ang), apply_rope(k, ang), v),
                gt(B + ".self_attn.o.weight"), gt(B + ".self_attn.o.bias"))


def block_self(B, x, ang, e):
    h = layernorm(x) * (1 + e[1]) + e[0]
    return x + self_attn(B, h, ang) * e[2]


def ffn_part(B, x, e):
    h = layernorm(x) * (1 + e[4]) + e[3]
    h = gelu(linf(h, gt(B + ".ffn.0.weight"), gt(B + ".ffn.0.bias")))
    return x + linf(h, gt(B + ".ffn.2.weight"), gt(B + ".ffn.2.bias")) * e[5]


def cross_text(B, x, ctx):
    q = rmsnorm(linf(x, gt(B + ".cross_attn.q.weight"), gt(B + ".cross_attn.q.bias")), gt(B + ".cross_attn.norm_q.weight"))
    k = rmsnorm(linf(ctx, gt(B + ".cross_attn.k.weight"), gt(B + ".cross_attn.k.bias")), gt(B + ".cross_attn.norm_k.weight"))
    v = linf(ctx, gt(B + ".cross_attn.v.weight"), gt(B + ".cross_attn.v.bias"))
    return linf(sdpa(q, k, v), gt(B + ".cross_attn.o.weight"), gt(B + ".cross_attn.o.bias"))


def fusion_cross_ffn(B, x, ctx, e, target, ang_src, ang_tgt):
    # fusion.py:72-159: text attn + target attn with roped q/k_t
    q = rmsnorm(linf(x, gt(B + ".cross_attn.q.weight"), gt(B + ".cross_attn.q.bias")), gt(B + ".cross_attn.norm_q.weight"))
    k = rmsnorm(linf(ctx, gt(B + ".cross_attn.k.weight"), gt(B + ".cross_attn.k.bias")), gt(B + ".cross_attn.norm_k.weight"))
    v = linf(ctx, gt(B + ".cross_attn.v.weight"), gt(B + ".cross_attn.v.bias"))
    xt = sdpa(q, k, v)  # q NOT roped here (fusion.py:103)
    tn = layernorm(target, gt(B + ".cross_attn.pre_attn_norm_fusion.weight"),
                   gt(B + ".cross_attn.pre_attn_norm_fusion.bias"))
    kt = rmsnorm(linf(tn, gt(B + ".cross_attn.k_fusion.weight"), gt(B + ".cross_attn.k_fusion.bias")),
                 gt(B + ".cross_attn.norm_k_fusion.weight"))
    vt = linf(tn, gt(B + ".cross_attn.v_fusion.weight"), gt(B + ".cross_attn.v_fusion.bias"))
    xt = xt + sdpa(apply_rope(q, ang_src), apply_rope(kt, ang_tgt), vt)
    x = x + linf(xt, gt(B + ".cross_attn.o.weight"), gt(B + ".cross_attn.o.bias"))
    return ffn_part(B, x, e)


def block_vanilla(B, x, ctx, ang, e):
    x = block_self(B, x, ang, e)
    x = x + cross_text(B, layernorm(x, gt(B + ".norm3.weight"), gt(B + ".norm3.bias")), ctx)
    return ffn_part(B, x, e)


# ---- full forward ----
v_ang = rope_angles_3d(F, Hp, Wp)
a_ang = rope_params(LA, HD - 4 * (HD // 6), scaling=ASC)

xv = embed_video(in_video)
xa = embed_audio(in_audio)
vctx = embed_text("video_model.", in_vctx)
actx = embed_text("audio_model.", in_actx)
ve, ve6 = time_rows("video_model.")
ae, ae6 = time_rows("audio_model.")

for i in range(NL):
    e_v = ve6 + gt(f"video_model.blocks.{i}.modulation.modulation").astype(np.float64)[0]
    e_a = ae6 + gt(f"audio_model.blocks.{i}.modulation.modulation").astype(np.float64)[0]
    if i in FUSION:
        # fusion.py:161-240
        xa = block_self(f"audio_model.blocks.{i}", xa, a_ang, e_a)
        xv = block_self(f"video_model.blocks.{i}", xv, v_ang, e_v)
        og_audio = xa.copy()
        xa = fusion_cross_ffn(f"audio_model.blocks.{i}", xa, actx, e_a, xv, a_ang, v_ang)
        xv = fusion_cross_ffn(f"video_model.blocks.{i}", xv, vctx, e_v, og_audio, v_ang, a_ang)
    else:
        xa = block_vanilla(f"audio_model.blocks.{i}", xa, actx, a_ang, e_a)
        xv = block_vanilla(f"video_model.blocks.{i}", xv, vctx, v_ang, e_v)


def head_out(pfx, x, e):
    hm = gt(pfx + "head.modulation").astype(np.float64)[0]  # [2, DIM]
    h = layernorm(x) * (1 + hm[1] + e) + (hm[0] + e)
    return linf(h, gt(pfx + "head.head.weight"), gt(pfx + "head.head.bias"))


hv = head_out("video_model.", xv, ve)  # [Lv, PT*PH*PW*VOUT]
u = hv.reshape(F, Hp, Wp, PT, PH, PW, VOUT)
save("out_video", np.einsum("fhwpqrc->cfphqwr", u).reshape(VOUT, F, H, W))
save("out_audio", head_out("audio_model.", xa, ae))  # [LA, AOUT]

np.savez("tools/golden/ovit_golden.npz", **out)
print("wrote tools/golden/ovit_golden.npz:", len(out), "arrays")

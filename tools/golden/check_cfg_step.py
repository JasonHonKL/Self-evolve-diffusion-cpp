#!/usr/bin/env python3
"""CFG step cross-check for src/pipeline/denoise.cpp.

Recomputes the FIRST denoise step's CFG-combined model_out in numpy
(float64 on the fp32 golden weights, machinery copied from gen_ovit.py,
parameterized by inputs/t/slg) from the x_t dumped by the C++ loop
(env OVI_DENOISE_DUMP) and asserts rel-err < 1e-3 against the C++ dump.

Usage: check_cfg_step.py <dump_prefix> <video_cfg> <audio_cfg> <slg_layer>

Neg-ctx convention shared with tests/test_denoise.cpp: pos ctx is
(in_vctx, in_actx), neg ctx is the swap (in_actx, in_vctx) — distinct
per-branch contexts, exactly the CFG shape the engine exercises.
"""
import sys

import numpy as np

# OviDitCfg::test() constants (gen_ovit.py)
DIM, HEADS, FFN, FREQD, TEXTD, TEXTLEN = 384, 3, 1792, 32, 512, 64
NL = 2
FUSION = {1}
VIN, AIN = 48, 20
PT, PH, PW = 1, 2, 2
ASC = 0.19676
HD = DIM // HEADS
EPS = 1e-6

g = np.load("tools/golden/ovit_golden.npz")
gt = lambda n: g[n]  # noqa: E731


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


def forward(xv_in, xa_in, T, in_vctx, in_actx, slg):
    """gen_ovit.py full forward, parameterized (slg: skip block i==slg>0)."""
    C, F, H, W = xv_in.shape
    Hp, Wp = H // PH, W // PW

    def embed_video(x):
        w = gt("video_model.patch_embedding.weight").astype(np.float64)
        b = gt("video_model.patch_embedding.bias").astype(np.float64)
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
        full[: ctx.shape[0]] = e
        return full

    def time_rows(pfx):
        half = FREQD // 2
        f = 10000.0 ** (-np.arange(half) / half)
        pos = np.concatenate([np.cos(T * f), np.sin(T * f)])
        e = silu(linf(pos[None], gt(pfx + "time_embedding.0.weight"), gt(pfx + "time_embedding.0.bias")))[0]
        e = linf(e[None], gt(pfx + "time_embedding.2.weight"), gt(pfx + "time_embedding.2.bias"))[0]
        e6 = linf(silu(e)[None], gt(pfx + "time_projection.1.weight"), gt(pfx + "time_projection.1.bias"))[0]
        return e, e6.reshape(6, DIM)

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
        q = rmsnorm(linf(x, gt(B + ".cross_attn.q.weight"), gt(B + ".cross_attn.q.bias")), gt(B + ".cross_attn.norm_q.weight"))
        k = rmsnorm(linf(ctx, gt(B + ".cross_attn.k.weight"), gt(B + ".cross_attn.k.bias")), gt(B + ".cross_attn.norm_k.weight"))
        v = linf(ctx, gt(B + ".cross_attn.v.weight"), gt(B + ".cross_attn.v.bias"))
        xt = sdpa(q, k, v)
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

    v_ang = rope_angles_3d(F, Hp, Wp)
    a_ang = rope_params(xa_in.shape[0], HD - 4 * (HD // 6), scaling=ASC)

    xv = embed_video(xv_in)
    xa = embed_audio(xa_in)
    vctx = embed_text("video_model.", in_vctx)
    actx = embed_text("audio_model.", in_actx)
    ve, ve6 = time_rows("video_model.")
    ae, ae6 = time_rows("audio_model.")

    for i in range(NL):
        if slg > 0 and i == slg:  # fusion.py:290
            continue
        e_v = ve6 + gt(f"video_model.blocks.{i}.modulation.modulation").astype(np.float64)[0]
        e_a = ae6 + gt(f"audio_model.blocks.{i}.modulation.modulation").astype(np.float64)[0]
        if i in FUSION:
            xa = block_self(f"audio_model.blocks.{i}", xa, a_ang, e_a)
            xv = block_self(f"video_model.blocks.{i}", xv, v_ang, e_v)
            og_audio = xa.copy()
            xa = fusion_cross_ffn(f"audio_model.blocks.{i}", xa, actx, e_a, xv, a_ang, v_ang)
            xv = fusion_cross_ffn(f"video_model.blocks.{i}", xv, vctx, e_v, og_audio, v_ang, a_ang)
        else:
            xa = block_vanilla(f"audio_model.blocks.{i}", xa, actx, a_ang, e_a)
            xv = block_vanilla(f"video_model.blocks.{i}", xv, vctx, v_ang, e_v)

    def head_out(pfx, x, e):
        hm = gt(pfx + "head.modulation").astype(np.float64)[0]
        h = layernorm(x) * (1 + hm[1] + e) + (hm[0] + e)
        return linf(h, gt(pfx + "head.head.weight"), gt(pfx + "head.head.bias"))

    hv = head_out("video_model.", xv, ve)
    u = hv.reshape(F, Hp, Wp, PT, PH, PW, gt("video_model.head.head.weight").shape[0] // (PT * PH * PW))
    out_v = np.einsum("fhwpqrc->cfphqwr", u).reshape(-1, F, H, W)
    return out_v, head_out("audio_model.", xa, ae)


def relerr(a, b):
    return np.sqrt(((a - b) ** 2).sum()) / (np.sqrt((b ** 2).sum()) + 1e-12)


def main():
    prefix, vcfg, acfg, slg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    xv = np.load(f"{prefix}_s0_xv.npy").astype(np.float64)
    xa = np.load(f"{prefix}_s0_xa.npy").astype(np.float64)
    t = float(np.load(f"{prefix}_s0_t.npy")[0])

    pv, pa = forward(xv, xa, t, gt("in_vctx"), gt("in_actx"), slg=-1)  # cond: no SLG
    nv, na = forward(xv, xa, t, gt("in_actx"), gt("in_vctx"), slg=slg)  # uncond: SLG

    gv = nv + vcfg * (pv - nv)
    ga = na + acfg * (pa - na)

    ev = relerr(gv, np.load(f"{prefix}_s0_pv.npy").astype(np.float64))
    ea = relerr(ga, np.load(f"{prefix}_s0_pa.npy").astype(np.float64))
    print(f"cfg step0 video rel-err {ev:.3e}  audio rel-err {ea:.3e}  (tol 1e-3)")
    if not (ev < 1e-3 and ea < 1e-3):
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()

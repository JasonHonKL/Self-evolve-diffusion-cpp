#!/usr/bin/env python3
"""Generate tests/golden/basic.npz — golden parity inputs+expected for the sd core ops.
Deterministic (default_rng). np.savez on purpose: STORED zip entries only (npz.cpp limit)."""
import os

import numpy as np

rng = np.random.default_rng(5)


def lround(v):
    # std::lround semantics: round-half-away-from-zero, exact on f32 inputs via f64
    v = v.astype(np.float64)
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5))


out = {}

# rmsnorm [64,128] eps=1e-6 (op is in-place; keep input copy)
x = rng.uniform(-1, 1, (64, 128)).astype(np.float32)
w = rng.uniform(0.5, 1.5, 128).astype(np.float32)
ss = (x.astype(np.float64) ** 2).sum(axis=1, keepdims=True)
inv = 1.0 / np.sqrt(ss / 128 + 1e-6)
out["rms_x"], out["rms_w"], out["rms_out"] = x, w, (x * inv * w).astype(np.float32)

# silu_inplace [256] (op is in-place; keep input copy)
s = rng.uniform(-3, 3, 256).astype(np.float32)
d = s.astype(np.float64)
out["silu_x"], out["silu_out"] = s, (d / (1.0 + np.exp(-d))).astype(np.float32)

# matmul [17,31]@[31,13]
A = rng.uniform(-1, 1, (17, 31)).astype(np.float32)
B = rng.uniform(-1, 1, (31, 13)).astype(np.float32)
out["mm_a"], out["mm_b"], out["mm_out"] = A, B, (A.astype(np.float64) @ B).astype(np.float32)

# bf16 roundtrip [1000] — bit-exact RNE replication of tensor.cpp to_bf16
xb = rng.uniform(-1, 1, 1000).astype(np.float32)
u = xb.view(np.uint32)
ub = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
b16 = (ub >> 16).astype(np.uint16)
out["bf16_x"] = xb
out["bf16_rt"] = (b16.astype(np.uint32) << np.uint32(16)).view(np.float32)

# quantize_rowwise [64,256] — exact float32 max-abs/127 + lround replication of tensor.cpp
W = rng.uniform(-1, 1, (64, 256)).astype(np.float32)
sc = np.abs(W).max(axis=1) / np.float32(127.0)
sc = np.where(sc == np.float32(0), np.float32(1), sc).astype(np.float32)
out["q_w"] = W
out["q_wq"] = lround(W / sc[:, None]).astype(np.int8)
out["q_scale"] = sc

# qgemm [64,256]@[128,256]^T — contract: close to fp32 matmul (cosine > 0.999)
qa = rng.standard_normal((64, 256)).astype(np.float32)
qb = rng.integers(-127, 128, (128, 256)).astype(np.int8)
qs = rng.uniform(0.001, 0.01, 128).astype(np.float32)
acc = qa @ (qb.T.astype(np.float32) * qs[None, :])
out["qg_a"], out["qg_bq"], out["qg_bs"] = qa, qb, qs
out["qg_out"] = (qa @ (qb.T.astype(np.float32) * qs[None, :])).astype(np.float32)

os.makedirs("tests/golden", exist_ok=True)
np.savez("tests/golden/basic.npz", **out)
print("wrote tests/golden/basic.npz:", len(out), "arrays")

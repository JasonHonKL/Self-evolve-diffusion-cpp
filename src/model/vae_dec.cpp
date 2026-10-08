// Wan2.2-TI2V-5B causal 3D VAE decoder. See vae_dec.h for the arch summary.
// All math cites Ovi/ovi/modules/vae2_2.py (line refs in comments).
// Internal activation layout is [T,C,H,W] (time-major, batchless): hist
// frames and chunk concat are contiguous; the public API uses torch order
// ([C,T,H,W] latent in, [3,T,H,W] rgb out).
#include "model/vae_dec.h"
#include "model/blocks.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sd {
namespace vae {
namespace {

// Wan2_2_VAE scale (vae2_2.py:904-1012); decode does z*std + mean (its
// scale[1] is 1/std) before conv2.
constexpr float kMean[48] = {
    -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f,
    0.1557f,  -0.1382f, 0.0542f,  0.2813f,  0.0891f,  0.1570f, -0.0098f,
    0.0375f,  -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f,
    -0.2108f, -0.2158f, 0.2502f,  -0.2055f, -0.0322f, 0.1109f, 0.1567f,
    -0.0729f, 0.0899f,  -0.2799f, -0.1230f, -0.0313f, -0.1649f, 0.0117f,
    0.0723f,  -0.2839f, -0.2083f, -0.0520f, 0.3748f,  0.0152f,  0.1957f,
    0.1433f,  -0.2944f, 0.3573f,  -0.0548f, -0.1681f, -0.0667f};
constexpr float kStd[48] = {
    0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
    0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
    0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
    0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
    0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
    0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};

constexpr int64_t kTileElems = 64ll << 20;  // im2col tile budget: 256 MB f32

using Conv = VaeDecoder::Conv;
using Hist = VaeDecoder::Hist;

struct D4 {
  int64_t T, C, H, W;
};
D4 d4(const Tensor& t) {
  return {(int64_t)t.dim(0), (int64_t)t.dim(1), (int64_t)t.dim(2),
          (int64_t)t.dim(3)};
}

Tensor clone4(const Tensor& x) {
  Tensor o(x.shape, x.dtype);
  memcpy(o.data, x.data, x.nbytes);
  return o;
}

void add_(Tensor& a, const Tensor& b) {
  int64_t n = a.numel();
  float* p = a.ptr<float>();
  const float* q = b.ptr<float>();
  parallel_for(0, n, 65536, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) p[i] += q[i];
  });
}

// dst = frame t of x ([C,H,W]); t < 0 -> zeros.
void copy_frame(Tensor& dst, const Tensor& x, int64_t t) {
  auto [T, C, H, W] = d4(x);
  if (t < 0) {
    dst = Tensor({C, H, W}, DType::F32);
    return;
  }
  dst = Tensor({C, H, W}, DType::F32);
  memcpy(dst.data, x.ptr<float>() + t * C * H * W, (size_t)C * H * W * 4);
}

// hist = last 2 frames of the virtual sequence [old f0, old f1, x0..x_{T-1}]
// — this is exactly the reference cache invariant (feat_cache update with the
// <2-frame fixup, vae2_2.py:121-141 / 217-229).
void hist_update(Hist& hh, const Tensor& x) {
  auto [T, C, H, W] = d4(x);
  if (T >= 2) {
    copy_frame(hh.f[0], x, T - 2);
    copy_frame(hh.f[1], x, T - 1);
  } else {
    Tensor keep = std::move(hh.f[1]);
    hh.f[0] = std::move(keep);
    copy_frame(hh.f[1], x, 0);
  }
}

// ---- causal conv3d: im2col + sd::matmul, row-tiled ----
// ponytail: naive im2col (27x activation blowup) + 256 MB row tiles; a packed
// direct conv3d would cut the fill traffic if decode becomes the bottleneck.
// P: [T + kt-1, C, H + 2*(kd==3), W + 2*(kd==3)]; output frame t reads
// P[t+dt], dt<kt, so the kt-1 history rows sit at P[0..kt-2].
void conv_padded(const Tensor& P, int64_t T, int64_t H, int64_t W,
                 const Conv& cv, Tensor& out) {
  const int64_t C = cv.Cin, K = cv.w.rows(), N = cv.Cout;
  const int64_t kt = cv.kt, kd = cv.kd;
  const int64_t Hp = P.dim(2), Wp = P.dim(3);
  const int64_t R = T * H * W;
  out = Tensor({T, N, H, W}, DType::F32);
  if (R == 0) return;
  const int64_t TR = std::min(R, std::max<int64_t>(64, kTileElems / K));
  const float* Pp = P.ptr<float>();
  float* op = out.ptr<float>();
  Tensor A({TR, K}, DType::F32);  // reused across tiles
  for (int64_t r0 = 0; r0 < R; r0 += TR) {
    const int64_t tr = std::min(TR, R - r0);
    float* ap = A.ptr<float>();
    parallel_for(0, tr, 8, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; r++) {
        const int64_t pos = r0 + r, t = pos / (H * W), hw = pos % (H * W);
        const int64_t h = hw / W, w = hw % W;
        float* arow = ap + r * K;
        int64_t k = 0;
        for (int64_t c = 0; c < C; c++)
          for (int64_t dt = 0; dt < kt; dt++)
            for (int64_t dh = 0; dh < kd; dh++) {
              const float* src =
                  Pp + (((t + dt) * C + c) * Hp + (h + dh)) * Wp + w;
              for (int64_t dw = 0; dw < kd; dw++) arow[k++] = src[dw];
            }
      }
    });
    Tensor Cm;
    matmul(A, cv.w, Cm, cv.b.ptr<float>());  // [TR, N] (+bias)
    const float* cm = Cm.ptr<float>();
    parallel_for(0, tr, 8, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; r++) {
        const int64_t pos = r0 + r, t = pos / (H * W), hw = pos % (H * W);
        const int64_t h = hw / W, w = hw % W;
        float* orow = op + t * (N * H * W) + h * W + w;
        const float* cr = cm + r * N;
        for (int64_t o = 0; o < N; o++) orow[o * H * W] = cr[o];
      }
    });
  }
}

// Zero-padded copy of x: kt==3 -> prepend hist frames h0,h1 (null = zeros);
// kd==3 -> 1-pixel spatial border. (CausalConv3d._padding, vae2_2.py:24-32.)
Tensor pad_input(const Tensor& x, const Tensor* h0, const Tensor* h1,
                 int64_t kt, int64_t kd) {
  auto [T, C, H, W] = d4(x);
  const int64_t sp = (kd == 3) ? 1 : 0;
  const int64_t HT = T + (kt == 3 ? 2 : 0);
  const int64_t H2 = H + 2 * sp, W2 = W + 2 * sp;
  Tensor P({HT, C, H2, W2}, DType::F32);  // zero-initialized
  float* Pp = P.ptr<float>();
  auto copy_frame_in = [&](int64_t ti, const float* fp) {
    for (int64_t c = 0; c < C; c++)
      for (int64_t h = 0; h < H; h++)
        memcpy(Pp + (ti * C + c) * H2 * W2 + (h + sp) * W2 + sp,
               fp + (c * H + h) * W, (size_t)W * 4);
  };
  for (int64_t ti = 0; ti < HT; ti++) {
    if (kt == 3 && ti < 2) {
      const Tensor* fr = ti == 0 ? h0 : h1;
      if (fr) copy_frame_in(ti, fr->ptr<float>());  // else stays zero
    } else {
      copy_frame_in(ti,
                    x.ptr<float>() + (ti - (kt == 3 ? 2 : 0)) * C * H * W);
    }
  }
  return P;
}

// Causal k3 conv with persistent 2-frame history (feat_cache protocol).
void causal_conv(const Tensor& x, const Conv& cv, Hist& hh, Tensor& out) {
  auto [T, C, H, W] = d4(x);
  if (!hh.started) {
    hh.f[0] = Tensor({C, H, W}, DType::F32);  // zeros = causal pad
    hh.f[1] = Tensor({C, H, W}, DType::F32);
    hh.started = true;
  }
  Tensor P = pad_input(x, &hh.f[0], &hh.f[1], cv.kt, cv.kd);
  conv_padded(P, T, H, W, cv, out);
  hist_update(hh, x);
}

// RMS_norm (vae2_2.py:45-60): per (t,h,w) position,
// x / max(||x||_2 over C, 1e-12) * sqrt(C) * gamma. images=True/False are
// the same math here (normalize over the channel dim).
void rmsnorm_channel(Tensor& x, const Tensor& gamma) {
  auto [T, C, H, W] = d4(x);
  const float* g = gamma.ptr<float>();
  const float sc = std::sqrt((float)C);
  const int64_t HW = H * W;
  float* p = x.ptr<float>();
  parallel_for(0, T * HW, 64, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      const int64_t t = i / HW, hw = i % HW;
      float* v = p + t * C * HW + hw;  // (t,h,w); channel stride HW
      float ss = 0;
      for (int64_t c = 0; c < C; c++) ss += v[c * HW] * v[c * HW];
      const float inv = sc / std::max(std::sqrt(ss), 1e-12f);
      for (int64_t c = 0; c < C; c++) v[c * HW] = v[c * HW] * inv * g[c];
    }
  });
}

// ResidualBlock (vae2_2.py:193-235): h = shortcut(x) (1x1x1 if in!=out, else
// identity); x = norm+silu+conv, norm+silu+conv; return x + h.
void resblock(Tensor& x, Conv* cv, const Conv* sc, Hist* hh) {
  Tensor hsc;
  if (sc) {
    conv_padded(x, x.dim(0), x.dim(2), x.dim(3), *sc, hsc);  // 1x1x1
  } else {
    hsc = clone4(x);
  }
  rmsnorm_channel(x, cv[0].gamma);
  silu_inplace(x);
  Tensor y;
  causal_conv(x, cv[0], hh[0], y);
  x = Tensor();  // free before the next allocation
  rmsnorm_channel(y, cv[1].gamma);
  silu_inplace(y);
  Tensor z;
  causal_conv(y, cv[1], hh[1], z);
  add_(z, hsc);
  x = std::move(z);
}

// AttentionBlock (vae2_2.py:238-277): single head over HW tokens per frame.
// ponytail: frames looped serially; blk::sdpa is O(HW^2*C) — fine at latent
// resolutions (<=32x32 here), batch/flash-tile it for production sizes.
void attn_block(Tensor& x, const decltype(VaeDecoder::attn)& at) {
  auto [T, C, H, W] = d4(x);
  Tensor idn = clone4(x);
  rmsnorm_channel(x, at.gamma);
  Tensor y({T, C, H, W}, DType::F32);
  const int64_t L = H * W;
  Tensor rows({L, C}, DType::F32), q({L, C}, DType::F32), k({L, C}, DType::F32),
      v({L, C}, DType::F32);
  for (int64_t t = 0; t < T; t++) {
    const float* fp = x.ptr<float>() + t * C * L;
    float* rp = rows.ptr<float>();
    for (int64_t c = 0; c < C; c++)
      for (int64_t i = 0; i < L; i++) rp[i * C + c] = fp[c * L + i];
    Tensor qkv, o, o2;
    matmul(rows, at.qkv_w, qkv, at.qkv_b.ptr<float>());  // [L, 3C]
    const float* qp = qkv.ptr<float>();
    float* qd = q.ptr<float>();
    float* kd = k.ptr<float>();
    float* vd = v.ptr<float>();
    for (int64_t i = 0; i < L; i++) {
      memcpy(qd + i * C, qp + i * 3 * C, (size_t)C * 4);
      memcpy(kd + i * C, qp + i * 3 * C + C, (size_t)C * 4);
      memcpy(vd + i * C, qp + i * 3 * C + 2 * C, (size_t)C * 4);
    }
    blk::sdpa(q, k, v, 1, o);  // scale C^-0.5 = torch SDPA default
    matmul(o, at.proj_w, o2, at.proj_b.ptr<float>());
    float* yp = y.ptr<float>() + t * C * L;
    const float* op = o2.ptr<float>();
    for (int64_t c = 0; c < C; c++)
      for (int64_t i = 0; i < L; i++) yp[c * L + i] = op[i * C + c];
  }
  add_(y, idn);
  x = std::move(y);
}

// nearest-exact 2x (Upsample, vae2_2.py:62-68): out[..., 2h+a, 2w+b] = x[h,w].
Tensor nearest2x(const Tensor& x) {
  auto [T, C, H, W] = d4(x);
  Tensor o({T, C, 2 * H, 2 * W}, DType::F32);
  const float* xp = x.ptr<float>();
  float* op = o.ptr<float>();
  parallel_for(0, T * C, 1, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      const float* src = xp + i * H * W;
      float* dst = op + i * 4 * H * W;
      for (int64_t h = 0; h < H; h++) {
        float* d0 = dst + (2 * h) * 2 * W;
        float* d1 = d0 + 2 * W;
        for (int64_t w = 0; w < W; w++) {
          const float val = src[h * W + w];
          d0[2 * w] = d0[2 * w + 1] = d1[2 * w] = d1[2 * w + 1] = val;
        }
      }
    }
  });
  return o;
}

// interleave doubling (vae2_2.py:148-151): r [Tr,2C,H,W] -> [2Tr,C,H,W],
// out[2t] = r[t][:C], out[2t+1] = r[t][C:]. Optional frame 0 prepended.
Tensor interleave(const Tensor& r, const Tensor* frame0) {
  auto [Tr, C2, H, W] = d4(r);
  const int64_t C = C2 / 2;
  const int64_t T2 = 2 * Tr + (frame0 ? 1 : 0);
  Tensor o({T2, C, H, W}, DType::F32);
  if (frame0) memcpy(o.data, frame0->data, (size_t)C * H * W * 4);
  float* op = o.ptr<float>() + (frame0 ? C * H * W : 0);
  const float* rp = r.ptr<float>();
  parallel_for(0, Tr, 1, [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; t++) {
      memcpy(op + (2 * t) * C * H * W, rp + t * C2 * H * W,
             (size_t)C * H * W * 4);
      memcpy(op + (2 * t + 1) * C * H * W, rp + (t * C2 + C) * H * W,
             (size_t)C * H * W * 4);
    }
  });
  return o;
}

// DupUp3D shortcut (vae2_2.py:370-412): channel repeat_interleave + nearest
// upsample in time/space; first_chunk drops the first ft-1 frames
// (x[:, :, factor_t-1:], matching the anchor frame's undoubled pass).
Tensor dupup(const Tensor& x, int64_t out_c, int64_t ft, bool first_chunk) {
  auto [T, C, H, W] = d4(x);
  const int64_t repeats = out_c * ft * 4 / C;
  const int64_t drop = first_chunk ? ft - 1 : 0;
  Tensor o({T * ft - drop, out_c, 2 * H, 2 * W}, DType::F32);
  const float* xp = x.ptr<float>();
  float* op = o.ptr<float>();
  const int64_t W2 = 2 * W;
  parallel_for(0, T, 1, [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; t++) {
      for (int64_t c = 0; c < C; c++) {
        const float* src = xp + (t * C + c) * H * W;
        for (int64_t j = 0; j < repeats; j++) {
          const int64_t n = c * repeats + j;
          const int64_t co = n / (ft * 4), rem = n % (ft * 4);
          const int64_t ot = t * ft + rem / 4 - drop;
          if (ot < 0) continue;  // dropped anchor-pad frame, never emitted
          const int64_t ho = (rem % 4) / 2, wo = rem % 2;
          for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
              op[(((ot * out_c + co) * 2 * H + 2 * h + ho) * W2) + 2 * w +
                  wo] = src[h * W + w];
        }
      }
    }
  });
  return o;
}

// Resample spatial half (both upsample2d/3d): nearest 2x + Conv2d 3x3 pad 1.
void spatial_resample(Tensor& x, const Conv& cv) {
  Tensor up = nearest2x(x);
  auto [T, C, H, W] = d4(up);
  Tensor P = pad_input(up, nullptr, nullptr, 1, cv.kd);
  Tensor o;
  conv_padded(P, T, H, W, cv, o);
  x = std::move(o);
}

// Resample(mode='upsample3d') (vae2_2.py:112-151): the time_conv site.
// Anchor chunk (first ever): frame 0 passes through UNDOUBLED and never
// enters time_conv ('Rep' semantics); frames 1.. see [0,0,f1..] context;
// later chunks see the 2-frame hist. Then nearest2x + Conv2d 3x3.
void resample_up3d(Tensor& x, const Conv& tc, const Conv& sc, Hist& hh) {
  auto [T, C, H, W] = d4(x);
  Tensor dbl;
  if (!hh.started) {  // anchor chunk
    hh.started = true;
    hh.f[0] = Tensor({C, H, W}, DType::F32);  // hist = last 2 of [0,0,rest]
    hh.f[1] = Tensor({C, H, W}, DType::F32);
    if (T == 1) {
      dbl = Tensor({1, C, H, W}, DType::F32);
      memcpy(dbl.data, x.data, (size_t)C * H * W * 4);
    } else {
      Tensor rest({T - 1, C, H, W}, DType::F32);
      memcpy(rest.data, x.ptr<float>() + C * H * W,
             (size_t)(T - 1) * C * H * W * 4);
      Tensor P = pad_input(rest, nullptr, nullptr, tc.kt, tc.kd);
      Tensor rout;
      conv_padded(P, T - 1, H, W, tc, rout);
      Tensor f0({1, C, H, W}, DType::F32);
      memcpy(f0.data, x.data, (size_t)C * H * W * 4);
      dbl = interleave(rout, &f0);
      if (T >= 3) copy_frame(hh.f[0], rest, T - 3);
      copy_frame(hh.f[1], rest, T - 2);
    }
  } else {
    Tensor P = pad_input(x, &hh.f[0], &hh.f[1], tc.kt, tc.kd);
    Tensor rout;
    conv_padded(P, T, H, W, tc, rout);
    dbl = interleave(rout, nullptr);
    hist_update(hh, x);
  }
  spatial_resample(dbl, sc);
  x = std::move(dbl);
}

// unpatchify(2) (vae2_2.py:299-313): x12 [T,12,H,W] -> rgb [3,T,2H,2W],
// rgb[c][t][2h+q][2w+r] = x12[t][4c + 2r + q][h][w].
Tensor unpatchify(const Tensor& x12) {
  auto [T, C12, H, W] = d4(x12);
  Tensor o({3, T, 2 * H, 2 * W}, DType::F32);
  const float* xp = x12.ptr<float>();
  float* op = o.ptr<float>();
  const int64_t W2 = 2 * W;
  parallel_for(0, T, 1, [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; t++)
      for (int64_t c = 0; c < 3; c++)
        for (int64_t r = 0; r < 2; r++)
          for (int64_t q = 0; q < 2; q++) {
            const float* src = xp + (t * 12 + 4 * c + 2 * r + q) * H * W;
            float* dst = op + ((c * T + t) * 2 * H + q) * W2 + r;
            for (int64_t h = 0; h < H; h++)
              for (int64_t w = 0; w < W; w++)
                dst[(2 * h) * W2 + 2 * w] = src[h * W + w];
          }
  });
  return o;
}

}  // namespace

VaeDecoder::VaeDecoder(const std::string& path) : st(load_safetensors(path)) {
  auto get = [&](const std::string& k) -> const Tensor& {
    auto it = st.tensors.find(k);
    if (it == st.tensors.end())
      throw std::runtime_error("vae_dec: missing tensor " + k);
    return it->second;
  };
  // torch [Cout,Cin,(kt,)kd,kd] -> w [Cin*kt*kd*kd, Cout] (+bias, +gamma).
  auto load = [&](Conv& cv, const std::string& wk, const std::string& bk,
                  const std::string& gk = "") {
    const Tensor& w = get(wk);
    to_f32(w, cv.w);  // temp f32, then permute in place layout
    Tensor wf = std::move(cv.w);
    cv.Cout = wf.dim(0);
    cv.Cin = wf.dim(1);
    if (wf.shape.size() == 5) {
      cv.kt = (int)wf.dim(2);
      cv.kd = (int)wf.dim(3);
      if (wf.dim(3) != wf.dim(4)) throw std::runtime_error("vae_dec: " + wk);
    } else if (wf.shape.size() == 4) {
      cv.kt = 1;
      cv.kd = (int)wf.dim(2);
      if (wf.dim(2) != wf.dim(3)) throw std::runtime_error("vae_dec: " + wk);
    } else {
      throw std::runtime_error("vae_dec: bad rank " + wk);
    }
    const int64_t K = cv.Cin * cv.kt * cv.kd * cv.kd;
    cv.w = Tensor({K, cv.Cout}, DType::F32);
    const float* wp = wf.ptr<float>();
    float* dp = cv.w.ptr<float>();
    parallel_for(0, K, 128, [&](int64_t b, int64_t e) {
      for (int64_t k = b; k < e; k++)
        for (int64_t o = 0; o < cv.Cout; o++)
          dp[k * cv.Cout + o] = wp[o * K + k];
    });
    to_f32(get(bk), cv.b);
    if (!gk.empty()) {
      to_f32(get(gk), cv.gamma);
      if (cv.gamma.numel() != cv.Cin)
        throw std::runtime_error("vae_dec: gamma mismatch " + gk);
    }
  };
  auto expect = [&](bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("vae_dec: arch ") + what);
  };

  load(conv2, "conv2.weight", "conv2.bias");
  load(conv1, "dec.conv1.weight", "dec.conv1.bias");
  expect(conv1.Cin == 48 && conv1.Cout == 1024 && conv1.kt == 3, "conv1");
  load(mid[0], "dec.middle.0.residual.2.weight", "dec.middle.0.residual.2.bias",
       "dec.middle.0.residual.0.gamma");
  load(mid[1], "dec.middle.0.residual.6.weight", "dec.middle.0.residual.6.bias",
       "dec.middle.0.residual.3.gamma");
  load(mid[2], "dec.middle.2.residual.2.weight", "dec.middle.2.residual.2.bias",
       "dec.middle.2.residual.0.gamma");
  load(mid[3], "dec.middle.2.residual.6.weight", "dec.middle.2.residual.6.bias",
       "dec.middle.2.residual.3.gamma");
  {
    Conv t;
    load(t, "dec.middle.1.to_qkv.weight", "dec.middle.1.to_qkv.bias");
    expect(t.Cin == 1024 && t.Cout == 3072, "qkv");
    attn.qkv_w = std::move(t.w);
    attn.qkv_b = std::move(t.b);
    load(t, "dec.middle.1.proj.weight", "dec.middle.1.proj.bias");
    expect(t.Cin == 1024 && t.Cout == 1024, "proj");
    attn.proj_w = std::move(t.w);
    attn.proj_b = std::move(t.b);
    to_f32(get("dec.middle.1.norm.gamma"), attn.gamma);
  }
  for (int s = 0; s < 4; s++)
    for (int r = 0; r < 3; r++) {
      const std::string b = "dec.upsamples." + std::to_string(s) +
                            ".upsamples." + std::to_string(r) + ".residual.";
      load(up[s][r][0], b + "2.weight", b + "2.bias", b + "0.gamma");
      load(up[s][r][1], b + "6.weight", b + "6.bias", b + "3.gamma");
    }
  load(up_sc[0], "dec.upsamples.2.upsamples.0.shortcut.weight",
       "dec.upsamples.2.upsamples.0.shortcut.bias");
  load(up_sc[1], "dec.upsamples.3.upsamples.0.shortcut.weight",
       "dec.upsamples.3.upsamples.0.shortcut.bias");
  expect(up_sc[0].Cin == 1024 && up_sc[0].Cout == 512, "sc0");
  expect(up_sc[1].Cin == 512 && up_sc[1].Cout == 256, "sc1");
  for (int s = 0; s < 2; s++) {
    load(tconv[s], "dec.upsamples." + std::to_string(s) +
                       ".upsamples.3.time_conv.weight",
         "dec.upsamples." + std::to_string(s) +
             ".upsamples.3.time_conv.bias");
    expect(tconv[s].Cin == 1024 && tconv[s].Cout == 2048 &&
               tconv[s].kt == 3 && tconv[s].kd == 1,
           "tconv");
  }
  for (int s = 0; s < 3; s++) {
    load(sconv[s], "dec.upsamples." + std::to_string(s) +
                       ".upsamples.3.resample.1.weight",
         "dec.upsamples." + std::to_string(s) +
             ".upsamples.3.resample.1.bias");
    expect(sconv[s].kt == 1 && sconv[s].kd == 3, "sconv");
  }
  load(head, "dec.head.2.weight", "dec.head.2.bias", "dec.head.0.gamma");
  expect(head.Cin == 256 && head.Cout == 12, "head");

  mean_ = Tensor({48}, DType::F32);
  std_ = Tensor({48}, DType::F32);
  for (int i = 0; i < 48; i++) {
    mean_.ptr<float>()[i] = kMean[i];
    std_.ptr<float>()[i] = kStd[i];
  }
}

void VaeDecoder::decode_chunk(const Tensor& z, Tensor& out, Cache& cache) {
  const bool first = !cache.anchor_done;
  if (z.shape.size() != 4 || z.dim(0) != z_dim || z.dtype != DType::F32)
    throw std::runtime_error("vae_dec: z must be F32 [48,T,H,W]");
  const int64_t T = z.dim(1), H = z.dim(2), W = z.dim(3);
  // z*std + mean into time-major [T,48,H,W] (vae2_2.py:812-820).
  Tensor zs({T, z_dim, H, W}, DType::F32);
  {
    const float* zp = z.ptr<float>();
    float* dp = zs.ptr<float>();
    const float* mn = mean_.ptr<float>();
    const float* sd = std_.ptr<float>();
    const int64_t HW = H * W;
    parallel_for(0, T * HW, 512, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; i++) {
        const int64_t t = i / HW, hw = i % HW;
        for (int64_t c = 0; c < z_dim; c++)
          dp[(t * z_dim + c) * HW + hw] =
              zp[(c * T + t) * HW + hw] * sd[c] + mn[c];
      }
    });
  }
  Tensor x;
  conv_padded(zs, T, H, W, conv2, x);  // 1x1x1

  int slot = 0;
  Tensor y;
  causal_conv(x, conv1, cache.h[slot++], y);  // [T,1024,H,W]
  resblock(y, &mid[0], nullptr, &cache.h[slot]);
  slot += 2;
  attn_block(y, attn);
  resblock(y, &mid[2], nullptr, &cache.h[slot]);
  slot += 2;

  static constexpr int64_t kNext[4] = {1024, 1024, 512, 256};
  for (int s = 0; s < 4; s++) {
    Tensor sc_up;
    if (s < 3)  // Up_ResidualBlock shortcut from the stage input
      sc_up = dupup(y, kNext[s], s < 2 ? 2 : 1, first);
    for (int r = 0; r < 3; r++) {
      const Conv* sc = (r == 0 && s >= 2) ? &up_sc[s - 2] : nullptr;
      resblock(y, &up[s][r][0], sc, &cache.h[slot]);
      slot += 2;
    }
    if (s < 2)
      resample_up3d(y, tconv[s], sconv[s], cache.h[slot++]);
    else if (s == 2)
      spatial_resample(y, sconv[2]);
    if (s < 3) add_(y, sc_up);
  }
  rmsnorm_channel(y, head.gamma);
  silu_inplace(y);
  Tensor o12;
  causal_conv(y, head, cache.h[slot++], o12);
  if (slot != 32) throw std::runtime_error("vae_dec: slot accounting");
  out = unpatchify(o12);
  cache.anchor_done = true;
}

void VaeDecoder::decode(const Tensor& z, Tensor& out) {
  if (z.shape.size() != 4 || z.dim(0) != z_dim || z.dtype != DType::F32)
    throw std::runtime_error("vae_dec: z must be F32 [48,F,H,W]");
  const int64_t F = z.dim(1), H = z.dim(2), W = z.dim(3);
  Cache cache;
  // Anchor frame, then everything else in one chunk (the reference streams
  // per-frame; any chunking is equivalent by the cache invariant).
  auto slice = [&](int64_t t0, int64_t t1) {
    Tensor c({z_dim, t1 - t0, H, W}, DType::F32);
    for (int64_t ch = 0; ch < z_dim; ch++)
      memcpy(c.ptr<float>() + ch * (t1 - t0) * H * W,
             z.ptr<float>() + ch * F * H * W + t0 * H * W,
             (size_t)(t1 - t0) * H * W * 4);
    return c;
  };
  Tensor z0 = slice(0, 1), o0;
  decode_chunk(z0, o0, cache);
  if (F == 1) {
    out = std::move(o0);
    return;
  }
  Tensor zr = slice(1, F), o1;
  decode_chunk(zr, o1, cache);
  const int64_t T0 = o0.dim(1), T1 = o1.dim(1), Ho = o0.dim(2), Wo = o0.dim(3);
  out = Tensor({3, T0 + T1, Ho, Wo}, DType::F32);
  for (int64_t c = 0; c < 3; c++) {
    memcpy(out.ptr<float>() + (c * (T0 + T1)) * Ho * Wo,
           o0.ptr<float>() + c * T0 * Ho * Wo, (size_t)T0 * Ho * Wo * 4);
    memcpy(out.ptr<float>() + (c * (T0 + T1) + T0) * Ho * Wo,
           o1.ptr<float>() + c * T1 * Ho * Wo, (size_t)T1 * Ho * Wo * 4);
  }
}

}  // namespace vae
}  // namespace sd

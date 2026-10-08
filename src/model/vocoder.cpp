// Implementation of voc.h (issue #13). Layouts: activations are channels-first
// [C, T] f32 contiguous. Convs run as GEMM: out = W_view[N, C*k] @ P[C*k, T_tile]
// where P rows are (c, kk) tap slices of x — torch weight layout used as-is.
#include "model/vocoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

namespace voc {

using sd::DType;
using sd::Tensor;

// optional debug hook (parity hunting): name + channels-first [C,T] snapshot
static std::function<void(const std::string&, const Tensor&)> g_trace;
void set_trace(const std::function<void(const std::string&, const Tensor&)>& fn) { g_trace = fn; }

// ---- BigVGAN config (Ovi/ovi/modules/mmaudio/ext/bigvgan/bigvgan_vocoder.yml)
static const int64_t kUpRates[6] = {4, 4, 2, 2, 2, 2};
static const int64_t kUpKernels[6] = {8, 8, 4, 4, 4, 4};
static const int64_t kRbKernels[3] = {3, 7, 11};  // one resblock per kernel
static const int64_t kNumMels = 80;

static Tensor view2d(const Tensor& t, int64_t rows, int64_t cols) {
  Tensor v;
  v.shape = {rows, cols};
  v.dtype = t.dtype;
  v.data = t.data;
  v.nbytes = t.nbytes;
  v.owning = false;
  return v;
}

static void check_w(const Tensor& w, int64_t N, int64_t C, int64_t k, const char* who) {
  if (w.dtype != DType::F32 || (int64_t)w.dim(0) != N || (int64_t)w.dim(1) != C ||
      (int64_t)w.dim(2) != k) {
    fprintf(stderr, "%s: weight shape mismatch: got [%lld,%lld,%lld] want N=%lld C=%lld k=%lld\n",
            who, (long long)w.dim(0), (long long)w.dim(1), (long long)w.dim(2),
            (long long)N, (long long)C, (long long)k);
    abort();
  }
}

// ---- generic stride-1 conv1d, same-length output (symmetric pad = dil*(k-1)/2 uses
// pad passed explicitly). x [C,T] -> out [N, T + 2*pad - dil*(k-1)].
static Tensor conv1d(const Tensor& x, const Tensor& w, const float* bias,
                     int64_t k, int64_t dil, int64_t pad) {
  const int64_t C = x.dim(0), T = x.dim(1), N = w.dim(0);
  check_w(w, N, C, k, "conv1d");
  const int64_t Tout = T + 2 * pad - dil * (k - 1);
  if (Tout <= 0) throw std::runtime_error("conv1d: non-positive output length");
  Tensor out({N, Tout}, DType::F32);
  const Tensor w2 = view2d(w, N, C * k);
  const int64_t K = C * k;
  const float* xp = x.ptr<float>();
  float* op = out.ptr<float>();
  const int64_t tile = 1024;
  for (int64_t t0 = 0; t0 < Tout; t0 += tile) {
    const int64_t Tt = std::min(tile, Tout - t0);
    Tensor P({K, Tt}, DType::F32);
    float* pp = P.ptr<float>();
    sd::parallel_for(0, K, 128, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; r++) {
        const int64_t c = r / k, kk = r % k;
        const int64_t src0 = t0 + kk * dil - pad;  // x index of local i = 0
        float* dst = pp + r * Tt;
        const int64_t i0 = std::max<int64_t>(0, -src0);
        const int64_t i1 = std::min(Tt, T - src0);
        for (int64_t i = 0; i < i0; i++) dst[i] = 0.f;
        if (i1 > i0) memcpy(dst + i0, xp + c * T + src0 + i0, (i1 - i0) * sizeof(float));
        for (int64_t i = i1; i < Tt; i++) dst[i] = 0.f;
      }
    });
    Tensor Y;
    sd::matmul(w2, P, Y);  // bias added by caller (matmul bias is per-column)
    const float* yp = Y.ptr<float>();
    for (int64_t n = 0; n < N; n++) memcpy(op + n * Tout + t0, yp + n * Tt, (size_t)Tt * 4);
  }
  if (bias) {  // conv bias is per output channel = per ROW here
    sd::parallel_for(0, N, 8, [&](int64_t b, int64_t e) {
      for (int64_t n = b; n < e; n++) {
        float* r = op + n * Tout;
        const float v = bias[n];
        for (int64_t t = 0; t < Tout; t++) r[t] += v;
      }
    });
  }
  return out;
}

// ---- ConvTranspose1d (torch semantics: out len (T-1)*u + k - 2*pad).
// Direct tap-column build: P[(c*k+kk)][j] = x[c][(j+pad-kk)/u] when divisible.
static Tensor conv_transpose1d(const Tensor& x, const Tensor& w, const float* bias,
                               int64_t k, int64_t u, int64_t pad) {
  const int64_t C = x.dim(0), T = x.dim(1), N = w.dim(0);
  check_w(w, N, C, k, "conv_transpose1d");
  const int64_t Tout = (T - 1) * u + k - 2 * pad;
  if (Tout <= 0) throw std::runtime_error("conv_transpose1d: non-positive output length");
  Tensor out({N, Tout}, DType::F32);
  const Tensor w2 = view2d(w, N, C * k);
  const int64_t K = C * k;
  const float* xp = x.ptr<float>();
  float* op = out.ptr<float>();
  const int64_t tile = 1024;
  for (int64_t t0 = 0; t0 < Tout; t0 += tile) {
    const int64_t Tt = std::min(tile, Tout - t0);
    Tensor P({K, Tt}, DType::F32);
    memset(P.data, 0, (size_t)K * Tt * 4);
    float* pp = P.ptr<float>();
    sd::parallel_for(0, K, 128, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; r++) {
        const int64_t c = r / k, kk = r % k;
        // j = i*u + kk - pad  ->  i = (j + pad - kk)/u
        for (int64_t j = 0; j < Tt; j++) {
          const int64_t num = j + t0 + pad - kk;
          if (num < 0 || num % u != 0) continue;
          const int64_t i = num / u;
          if (i < T) pp[r * Tt + j] = xp[c * T + i];
        }
      }
    });
    Tensor Y;
    sd::matmul(w2, P, Y);  // bias added by caller (matmul bias is per-column)
    const float* yp = Y.ptr<float>();
    for (int64_t n = 0; n < N; n++) memcpy(op + n * Tout + t0, yp + n * Tt, (size_t)Tt * 4);
  }
  if (bias) {  // conv bias is per output channel = per ROW here
    sd::parallel_for(0, N, 8, [&](int64_t b, int64_t e) {
      for (int64_t n = b; n < e; n++) {
        float* r = op + n * Tout;
        const float v = bias[n];
        for (int64_t t = 0; t < Tout; t++) r[t] += v;
      }
    });
  }
  return out;
}

// ---- alias-free Activation1d (kaiser up/down, ratio 2, kernel 12 — module defaults)
// up: y[s] = 2 * sum_kk f[kk] * x[clamp((s+15-kk)/2 - 5, 0, T-1)], kk ≡ s+15 (mod 2)
static Tensor kaiser_up(const Tensor& x, const float* f) {
  const int64_t C = x.dim(0), T = x.dim(1), T2 = 2 * T - 1;
  Tensor y({C, T2}, DType::F32);
  const float* xp = x.ptr<float>();
  float* yp = y.ptr<float>();
  sd::parallel_for(0, C, 4, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++) {
      const float* xr = xp + c * T;
      float* yr = yp + c * T2;
      for (int64_t s = 0; s < T2; s++) {
        float acc = 0.f;
        for (int64_t kk = ((s + 15) & 1); kk < 12; kk += 2) {
          const int64_t j = (s + 15 - kk) >> 1;             // xp index, always in [0, T+10)
          const int64_t src = std::min(std::max<int64_t>(j - 5, 0), T - 1);
          acc += f[kk] * xr[src];
        }
        yr[s] = 2.f * acc;
      }
    }
  });
  return y;
}

// down: out[t] = sum_kk f[kk] * x[clamp(2t+kk-5, 0, T2-1)]
static Tensor kaiser_down(const Tensor& x, const float* f) {
  const int64_t C = x.dim(0), T2 = x.dim(1), T = T2 / 2 + 1;
  Tensor y({C, T}, DType::F32);
  const float* xp = x.ptr<float>();
  float* yp = y.ptr<float>();
  sd::parallel_for(0, C, 4, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++) {
      const float* xr = xp + c * T2;
      float* yr = yp + c * T;
      for (int64_t t = 0; t < T; t++) {
        float acc = 0.f;
        for (int64_t kk = 0; kk < 12; kk++) {
          const int64_t src = std::min(std::max<int64_t>(2 * t + kk - 5, 0), T2 - 1);
          acc += f[kk] * xr[src];
        }
        yr[t] = acc;
      }
    }
  });
  return y;
}

// y = y + sin(a*y)^2 / (b + 1e-9), a/b per channel (already exp'd)
static void snake_beta_inplace(Tensor& x, const float* alpha, const float* beta) {
  const int64_t C = x.dim(0), T = x.dim(1);
  float* xp = x.ptr<float>();
  sd::parallel_for(0, C, 2, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++) {
      const float a = alpha[c], ib = 1.f / (beta[c] + 1e-9f);
      float* r = xp + c * T;
      for (int64_t t = 0; t < T; t++) {
        const float v = r[t];
        const float s = std::sin(v * a);
        r[t] = v + s * s * ib;
      }
    }
  });
}

// x [C,T] -> [C, 2T-1] -> snake -> [C, T]
static Tensor act1d(const Tensor& x, const float* alpha, const float* beta,
                    const float* upf, const float* dof) {
  Tensor y = kaiser_up(x, upf);
  snake_beta_inplace(y, alpha, beta);
  return kaiser_down(y, dof);
}

// ---- VAE (EDM2-style) ops ----
static void pixel_norm_inplace(Tensor& x, float eps = 1e-4f) {
  const int64_t C = x.dim(0), T = x.dim(1);
  float* xp = x.ptr<float>();
  const float inv_sqrt_c = std::sqrt(1.f / (float)C);
  sd::parallel_for(0, T, 16, [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; t++) {
      float ss = 0.f;
      for (int64_t c = 0; c < C; c++) ss += xp[c * T + t] * xp[c * T + t];
      const float denom = eps + std::sqrt(ss) * inv_sqrt_c;
      const float inv = 1.f / denom;
      for (int64_t c = 0; c < C; c++) xp[c * T + t] *= inv;
    }
  });
}

static void mp_silu_inplace(Tensor& x) {
  const int64_t n = x.numel();
  float* p = x.ptr<float>();
  sd::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      const float v = p[i];
      p[i] = v / (1.f + std::exp(-v)) * (1.f / 0.596f);
    }
  });
}

// a = (a + t*(b-a)) / sqrt((1-t)^2 + t^2), t = 0.3
static void mp_sum_inplace(Tensor& a, const Tensor& b) {
  const int64_t n = a.numel();
  float* ap = a.ptr<float>();
  const float* bp = b.ptr<float>();
  const float s = 1.f / std::sqrt(0.58f);
  sd::parallel_for(0, n, 8192, [&](int64_t beg, int64_t e) {
    for (int64_t i = beg; i < e; i++) ap[i] = (0.7f * ap[i] + 0.3f * bp[i]) * s;
  });
}

static void clamp_inplace(Tensor& x, float lim) {
  const int64_t n = x.numel();
  float* p = x.ptr<float>();
  sd::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) p[i] = std::min(std::max(p[i], -lim), lim);
  });
}

static Tensor upsample_nearest2(const Tensor& x) {
  const int64_t C = x.dim(0), T = x.dim(1);
  Tensor y({C, 2 * T}, DType::F32);
  const float* xp = x.ptr<float>();
  float* yp = y.ptr<float>();
  sd::parallel_for(0, C, 8, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++)
      for (int64_t t = 0; t < T; t++) {
        yp[c * 2 * T + 2 * t] = xp[c * T + t];
        yp[c * 2 * T + 2 * t + 1] = xp[c * T + t];
      }
  });
  return y;
}

static Tensor transpose_ct(const Tensor& x) {  // [C, T] -> [T, C]
  const int64_t C = x.dim(0), T = x.dim(1);
  Tensor y({T, C}, DType::F32);
  const float* xp = x.ptr<float>();
  float* yp = y.ptr<float>();
  sd::parallel_for(0, C, 8, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++)
      for (int64_t t = 0; t < T; t++) yp[t * C + c] = xp[c * T + t];
  });
  return y;
}

static Tensor transpose_tc(const Tensor& x) {  // [T, C] -> [C, T]
  return transpose_ct(x);                      // inverse of itself
}

// ---- Vocoder ----

const Tensor& Vocoder::W(const std::string& key) const {
  auto it = st.tensors.find(key);
  if (it == st.tensors.end()) throw std::runtime_error("vocoder: missing tensor " + key);
  return it->second;
}

Vocoder::Vocoder(const std::string& path) : st(sd::load_safetensors(path)) {
  dec_gain_ = W("dec_gain").ptr<float>()[0] + 1.f;
  W("g.conv_pre.weight");
  W("dec.conv_in.weight");
  // ConvTranspose1d stores weight as [in, out, k]; permute to [out, in, k]
  // so conv_transpose1d + GEMM can use it as a plain [N, C*k] matrix.
  for (int i = 0; i < 6; i++) {
    const Tensor& w = W("g.ups." + std::to_string(i) + ".0.weight");
    const int64_t Ci = w.dim(0), N = w.dim(1), k = w.dim(2);
    Tensor t({N, Ci, k}, DType::F32);
    const float* src = w.ptr<float>();
    float* dst = t.ptr<float>();
    sd::parallel_for(0, Ci, 8, [&](int64_t b, int64_t e) {
      for (int64_t c = b; c < e; c++)
        for (int64_t n = 0; n < N; n++)
          for (int64_t kk = 0; kk < k; kk++)
            dst[(n * Ci + c) * k + kk] = src[(c * N + n) * k + kk];
    });
    ups_w_.emplace_back(std::move(t));
  }
}

static Tensor clone_tensor(const Tensor& x) {
  Tensor y(x.shape, x.dtype);
  memcpy(y.data, x.data, x.nbytes);
  return y;
}

// ResnetBlock1D. Keys prefix e.g. "dec.mid.block_1".
// one AMPBlock (resblock index ib, kernel kj, dilations 1/3/5). Exposed for tests.
Tensor amp_block(const sd::SafetensorsFile& st, const Tensor& x_in, int64_t ib, int64_t kj) {
  Tensor x = clone_tensor(x_in);
  const std::string rb = "g.resblocks." + std::to_string(ib);
  for (int m = 0; m < 3; m++) {
    const int64_t dil = 1 + 2 * m;
    const std::string am = rb + ".activations." + std::to_string(2 * m);
    const std::string am2 = rb + ".activations." + std::to_string(2 * m + 1);
    Tensor xt = act1d(x, st.tensors.at(am + ".act.alpha_exp").ptr<float>(),
                      st.tensors.at(am + ".act.beta_exp").ptr<float>(),
                      st.tensors.at(am + ".upsample.filter").ptr<float>(),
                      st.tensors.at(am + ".downsample.lowpass.filter").ptr<float>());
    xt = conv1d(xt, st.tensors.at(rb + ".convs1." + std::to_string(m) + ".weight"),
                st.tensors.at(rb + ".convs1." + std::to_string(m) + ".bias").ptr<float>(), kj, dil, dil * (kj - 1) / 2);
    xt = act1d(xt, st.tensors.at(am2 + ".act.alpha_exp").ptr<float>(),
               st.tensors.at(am2 + ".act.beta_exp").ptr<float>(),
               st.tensors.at(am2 + ".upsample.filter").ptr<float>(),
               st.tensors.at(am2 + ".downsample.lowpass.filter").ptr<float>());
    xt = conv1d(xt, st.tensors.at(rb + ".convs2." + std::to_string(m) + ".weight"),
                st.tensors.at(rb + ".convs2." + std::to_string(m) + ".bias").ptr<float>(), kj, 1, (kj - 1) / 2);
    float* xp = xt.ptr<float>();
    const float* op = x.ptr<float>();
    const int64_t n = xt.numel();
    sd::parallel_for(0, n, 8192, [&](int64_t b2, int64_t e2) {
      for (int64_t q = b2; q < e2; q++) xp[q] += op[q];
    });
    x = std::move(xt);
  }
  return x;
}

// ResnetBlock1D. Keys prefix e.g. "dec.mid.block_1".
static Tensor vae_resblock(const sd::SafetensorsFile& st,
                           const Tensor& x, const std::string& p) {
  auto get = [&](const std::string& k) -> const Tensor& { return st.tensors.at(k); };
  Tensor xn = clone_tensor(x);
  pixel_norm_inplace(xn);
  Tensor h = clone_tensor(xn);
  mp_silu_inplace(h);
  h = conv1d(h, get(p + ".conv1.weight"), nullptr, 3, 1, 1);
  mp_silu_inplace(h);
  h = conv1d(h, get(p + ".conv2.weight"), nullptr, 3, 1, 1);
  Tensor xs;
  auto nin = st.tensors.find(p + ".nin_shortcut.weight");
  if (nin != st.tensors.end()) {
    xs = conv1d(xn, nin->second, nullptr, 1, 1, 0);
  } else {
    xs = std::move(xn);
  }
  mp_sum_inplace(xs, h);
  return xs;
}

Tensor vae_attn(const sd::SafetensorsFile& st, const Tensor& x,
                       const std::string& p) {
  auto get = [&](const std::string& k) -> const Tensor& { return st.tensors.at(k); };
  const int64_t C = x.dim(0), T = x.dim(1);
  Tensor y = conv1d(x, get(p + ".qkv.weight"), nullptr, 1, 1, 0);  // [3C, T], interleaved q/k/v
  const float* yp = y.ptr<float>();
  Tensor Q({T, C}, DType::F32), K({C, T}, DType::F32), V({T, C}, DType::F32);
  float* qp = Q.ptr<float>();
  float* kp = K.ptr<float>();
  float* vp = V.ptr<float>();
  const float inv_sqrt_c = std::sqrt(1.f / (float)C);
  const float scale = inv_sqrt_c;  // SDPA scale = 1/sqrt(head_dim=C)
  sd::parallel_for(0, T, 8, [&](int64_t b, int64_t e) {
    std::vector<float> buf(3 * C);  // per-thread scratch
    for (int64_t t = b; t < e; t++) {
      for (int64_t c = 0; c < C; c++)
        for (int w = 0; w < 3; w++) buf[c * 3 + w] = yp[(c * 3 + w) * T + t];
      // normalize each of q,k,v over C
      for (int w = 0; w < 3; w++) {
        float ss = 0.f;
        for (int64_t c = 0; c < C; c++) {
          const float v = buf[c * 3 + w];
          ss += v * v;
        }
        const float inv = 1.f / (1e-4f + std::sqrt(ss) * inv_sqrt_c);
        for (int64_t c = 0; c < C; c++) buf[c * 3 + w] *= inv;
      }
      for (int64_t c = 0; c < C; c++) {
        qp[t * C + c] = buf[c * 3 + 0];
        kp[c * T + t] = buf[c * 3 + 1];
        vp[t * C + c] = buf[c * 3 + 2];
      }
    }
  });
  Tensor S;
  sd::matmul(Q, K, S);  // [T, T] scores
  {
    float* sp = S.ptr<float>();
    sd::parallel_for(0, T, 8, [&](int64_t b, int64_t e) {
      for (int64_t t = b; t < e; t++) {
        float* r = sp + t * T;
        float mx = -1e30f;
        for (int64_t i = 0; i < T; i++) {
          r[i] *= scale;
          mx = std::max(mx, r[i]);
        }
        float sum = 0.f;
        for (int64_t i = 0; i < T; i++) {
          r[i] = std::exp(r[i] - mx);
          sum += r[i];
        }
        const float inv = 1.f / sum;
        for (int64_t i = 0; i < T; i++) r[i] *= inv;
      }
    });
  }
  Tensor O;  // [T, C]
  sd::matmul(S, V, O);
  Tensor oc = transpose_tc(O);  // [C, T]
  Tensor proj = conv1d(oc, get(p + ".proj_out.weight"), nullptr, 1, 1, 0);
  Tensor res = clone_tensor(x);  // mp_sum(x, h, t=0.3): identity is `a`
  mp_sum_inplace(res, proj);
  return res;
}

void Vocoder::decode_latent(const Tensor& z, Tensor& mel) const {
  if (z.dim(0) != 20) throw std::runtime_error("decode_latent: expected z [20, T]");
  Tensor x = conv1d(z, W("dec.conv_in.weight"), nullptr, 3, 1, 1);  // [1536, L]
  if (g_trace) g_trace("dec.conv_in", x);
  x = vae_resblock(st, x, "dec.mid.block_1");
  if (g_trace) g_trace("dec.mid.b1", x);
  x = vae_attn(st, x, "dec.mid.attn_1");
  if (g_trace) g_trace("dec.mid.attn", x);
  x = vae_resblock(st, x, "dec.mid.block_2");
  clamp_inplace(x, 256.f);
  if (g_trace) g_trace("dec.mid.b2", x);
  const int levels[3] = {2, 1, 0};
  for (int li = 0; li < 3; li++) {
    const int lvl = levels[li];
    for (int b = 0; b < 3; b++) {
      x = vae_resblock(st, x, "dec.up." + std::to_string(lvl) + ".block." + std::to_string(b));
      clamp_inplace(x, 256.f);
    }
    if (g_trace) g_trace("dec.up" + std::to_string(lvl), x);
    if (lvl == 1) {
      x = upsample_nearest2(x);
      x = conv1d(x, W("dec.up.1.upsample.conv.weight"), nullptr, 3, 1, 1);
      if (g_trace) g_trace("dec.up1.up", x);
    }
  }
  if (g_trace) g_trace("dec.pre_out", x);
  mp_silu_inplace(x);
  x = conv1d(x, W("dec.conv_out.weight"), nullptr, 3, 1, 1);  // [80, 2L]
  {
    const int64_t n = x.numel();
    float* p = x.ptr<float>();
    sd::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; i++) p[i] *= dec_gain_;
    });
  }
  if (g_trace) g_trace("dec.dec_out", x);
  const float* mean = W("data_mean").ptr<float>();
  const float* stdv = W("data_std").ptr<float>();
  const int64_t C = x.dim(0), T = x.dim(1);
  float* xp = x.ptr<float>();
  sd::parallel_for(0, C, 1, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; c++)
      for (int64_t t = 0; t < T; t++) xp[c * T + t] = xp[c * T + t] * stdv[c] + mean[c];
  });
  mel = std::move(x);
}

void Vocoder::vocode(const Tensor& mel, Tensor& wav) const {
  if (mel.dim(0) != kNumMels) throw std::runtime_error("vocode: expected mel [80, T]");
  Tensor x = conv1d(mel, W("g.conv_pre.weight"), W("g.conv_pre.bias").ptr<float>(), 7, 1, 3);
  if (g_trace) g_trace("g.conv_pre", x);
  for (int i = 0; i < 6; i++) {
    const int64_t u = kUpRates[i], k = kUpKernels[i];
    x = conv_transpose1d(x, ups_w_[i],
                         W("g.ups." + std::to_string(i) + ".0.bias").ptr<float>(), k, u, (k - u) / 2);
    if (g_trace) g_trace("g.upT" + std::to_string(i), x);
    Tensor acc;
    for (int j = 0; j < 3; j++) {
      Tensor xb = amp_block(st, x, i * 3 + j, kRbKernels[j]);
      if (j == 0) acc = std::move(xb);
      else {
        float* ap = acc.ptr<float>();
        const float* bp = xb.ptr<float>();
        const int64_t n = acc.numel();
        sd::parallel_for(0, n, 8192, [&](int64_t b2, int64_t e2) {
          for (int64_t q = b2; q < e2; q++) ap[q] += bp[q];
        });
      }
    }
    float* ap = acc.ptr<float>();
    const int64_t n = acc.numel();
    sd::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
      for (int64_t q = b; q < e; q++) ap[q] *= (1.f / 3.f);
    });
    x = std::move(acc);
    if (g_trace) g_trace("g.stage" + std::to_string(i), x);
  }
  x = act1d(x, W("g.activation_post.act.alpha_exp").ptr<float>(),
            W("g.activation_post.act.beta_exp").ptr<float>(),
            W("g.activation_post.upsample.filter").ptr<float>(),
            W("g.activation_post.downsample.lowpass.filter").ptr<float>());
  x = conv1d(x, W("g.conv_post.weight"), W("g.conv_post.bias").ptr<float>(), 7, 1, 3);  // [1, T*256]
  {
    const int64_t n = x.numel();
    float* p = x.ptr<float>();
    sd::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; i++) p[i] = std::tanh(p[i]);
    });
  }
  wav = std::move(x);
}

void Vocoder::latent_to_wav(const Tensor& z, Tensor& wav) const {
  Tensor mel;
  decode_latent(z, mel);
  vocode(mel, wav);
}

}  // namespace voc

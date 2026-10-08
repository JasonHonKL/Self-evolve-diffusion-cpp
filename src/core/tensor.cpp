// Minimal correct implementations; agent (issue #2) upgrades performance.
#include "core/tensor.h"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

namespace sd {

static size_t dtype_size(DType d) {
  switch (d) { case DType::F32: return 4; case DType::BF16: return 2; case DType::I8: return 1; }
  return 4;
}

Tensor::Tensor(std::vector<int64_t> s, DType d) : shape(std::move(s)), dtype(d) {
  int64_t n = 1; for (auto x : shape) n *= x;
  nbytes = (size_t)n * dtype_size(dtype);
  data = aligned_alloc(64, ((nbytes + 63) / 64) * 64);
  memset(data, 0, nbytes);
}
Tensor::~Tensor() { if (owning && data) free(data); }
Tensor::Tensor(Tensor&& o) noexcept { *this = std::move(o); }
Tensor& Tensor::operator=(Tensor&& o) noexcept {
  if (this != &o) { if (owning && data) free(data); shape=std::move(o.shape); dtype=o.dtype;
    data=o.data; nbytes=o.nbytes; owning=o.owning; o.data=nullptr; o.nbytes=0; }
  return *this;
}
int64_t Tensor::numel() const { int64_t n=1; for (auto x: shape) n*=x; return n; }
void Tensor::reshape(std::vector<int64_t> s) {
  int64_t n=1; for (auto x: s) n*=x;
  if (n != numel()) abort();
  shape = std::move(s);
}

static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1; u += 0x7fff + lsb;   // round-to-nearest-even
  return (uint16_t)(u >> 16);
}
static float bf16_bits_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

void to_f32(const Tensor& bf16, Tensor& out) {
  out = Tensor(bf16.shape, DType::F32);
  auto* s = bf16.ptr<uint16_t>(); auto* d = out.ptr<float>();
  int64_t n = bf16.numel();
  for (int64_t i = 0; i < n; i++) d[i] = bf16_bits_to_f32(s[i]);
}
void to_bf16(const Tensor& f32, Tensor& out) {
  out = Tensor(f32.shape, DType::BF16);
  auto* s = f32.ptr<float>(); auto* d = out.ptr<uint16_t>();
  int64_t n = f32.numel();
  for (int64_t i = 0; i < n; i++) d[i] = f32_to_bf16_bits(s[i]);
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C, const float* bias) {
  // naive row-major reference; upgraded by issue #2
  Tensor Af, Bf;
  const Tensor* a = &A; const Tensor* b = &B;
  if (A.dtype == DType::BF16) { to_f32(A, Af); a = &Af; }
  if (B.dtype == DType::BF16) { to_f32(B, Bf); b = &Bf; }
  int64_t M = a->rows(), K = a->cols(), N = b->cols();
  C = Tensor({M, N}, DType::F32);
  const float* ap = a->ptr<float>(); const float* bp = b->ptr<float>(); float* cp = C.ptr<float>();
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < N; j++) {
      float s = bias ? bias[j] : 0.f;
      for (int64_t k = 0; k < K; k++) s += ap[i*K+k] * bp[k*N+j];
      cp[i*N+j] = s;
    }
}

void quantize_rowwise(const Tensor& W, Tensor& Wq, Tensor& scale) {
  int64_t N = W.rows(), K = W.cols();
  Wq = Tensor({N, K}, DType::I8); scale = Tensor({N}, DType::F32);
  const float* w = W.ptr<float>(); auto* q = Wq.ptr<int8_t>(); float* s = scale.ptr<float>();
  parallel_for(0, N, 8, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      float mx = 0; for (int64_t k = 0; k < K; k++) mx = std::max(mx, std::abs(w[i*K+k]));
      float sc = mx / 127.f; if (sc == 0) sc = 1;
      s[i] = sc;
      for (int64_t k = 0; k < K; k++) {
        float v = w[i*K+k] / sc;
        q[i*K+k] = (int8_t)std::lround(v);
      }
    }
  });
}

void qgemm(const Tensor& A, const Tensor& Bq, const Tensor& bs, Tensor& out) {
  // naive reference; AVX2 fast path by issue #3
  int64_t M = A.rows(), K = A.cols(), N = Bq.rows();
  out = Tensor({M, N}, DType::F32);
  const float* a = A.ptr<float>(); const int8_t* b = Bq.ptr<int8_t>();
  const float* s = bs.ptr<float>(); float* o = out.ptr<float>();
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < N; j++) {
      int32_t acc = 0;
      for (int64_t k = 0; k < K; k++) acc += (int32_t)std::lround(a[i*K+k]) * b[j*K+k];
      o[i*N+j] = acc * s[j];
    }
}

// ---- threading ----
static int g_nthreads = std::thread::hardware_concurrency();
int nthreads() { return g_nthreads; }
void set_nthreads(int n) { g_nthreads = n > 0 ? n : 1; }

void parallel_for(int64_t begin, int64_t end, int64_t grain,
                  const std::function<void(int64_t, int64_t)>& fn) {
  int64_t total = end - begin;
  if (total <= 0) return;
  int nt = g_nthreads;
  if (nt <= 1 || total <= grain) { fn(begin, end); return; }
  int64_t nchunks = (total + grain - 1) / grain;
  if (nchunks < nt) nt = (int)nchunks;
  std::vector<std::thread> th;
  std::atomic<int64_t> next{0};
  for (int t = 1; t < nt; t++)
    th.emplace_back([&] {
      for (;;) { int64_t c = next.fetch_add(1); int64_t b = begin + c*grain;
        if (b >= end) break; int64_t e = std::min(end, b+grain); fn(b, e); }
    });
  for (;;) { int64_t c = next.fetch_add(1); int64_t b = begin + c*grain;
    if (b >= end) break; int64_t e = std::min(end, b+grain); fn(b, e); }
  for (auto& x : th) x.join();
}

void silu_inplace(Tensor& x) {
  int64_t n = x.numel(); float* p = x.ptr<float>();
  parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) p[i] = p[i] / (1.f + std::exp(-p[i]));
  });
}

void rmsnorm(Tensor& x, const Tensor& weight, float eps) {
  int64_t N = x.cols(), M = x.rows();
  const float* w = weight.ptr<float>(); float* p = x.ptr<float>();
  parallel_for(0, M, 1, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      float* r = p + i*N; float ss = 0;
      for (int64_t k = 0; k < N; k++) ss += r[k]*r[k];
      float inv = 1.f / std::sqrt(ss / N + eps);
      for (int64_t k = 0; k < N; k++) r[k] = r[k]*inv*w[k];
    }
  });
}

void randu(Tensor& x, uint64_t seed) {
  int64_t n = x.numel(); float* p = x.ptr<float>();
  uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ull;
  for (int64_t i = 0; i < n; i++) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    p[i] = ((int32_t)(s >> 40) % 20001 - 10000) / 10000.f;
  }
}

}  // namespace sd

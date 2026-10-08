// sd::matmul: AVX2 6x16 register-tiled GEMM + OpenBLAS dlopen dispatch (issue #2).
#include "core/tensor.h"
#include <algorithm>
#include <cstdlib>
#ifdef SD_OPENBLAS
#include <dlfcn.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#define SD_GEMM_SIMD 1
#include <immintrin.h>
#endif

namespace sd {
namespace {

constexpr int64_t kBlasMinWork = 64ll << 20;  // M*N*K above this -> OpenBLAS

using SGemmFn = void (*)(int, int, int, int, int, int, float, const float*, int,
                         const float*, int, float, float*, int);

SGemmFn blas_sgemm() {
#ifdef SD_OPENBLAS
  static SGemmFn fn = [] {
    void* h = dlopen(SD_OPENBLAS, RTLD_NOW | RTLD_LOCAL);
    return h ? reinterpret_cast<SGemmFn>(dlsym(h, "cblas_sgemm")) : nullptr;
  }();
  return fn;
#else
  return nullptr;
#endif
}

void add_bias(float* C, int64_t M, int64_t N, const float* bias) {
  parallel_for(0, M, 32, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      float* r = C + i * N;
      for (int64_t j = 0; j < N; j++) r[j] += bias[j];
    }
  });
}

#if SD_GEMM_SIMD

// C[i..i+R)[j0..j0+16) += A rows x packed B panel bpp = bp + jb*K*16 ([K][16], zero-padded cols)
template <int R>
void row_group(const float* ap, const float* bp, float* cp, int64_t K, int64_t N,
               int64_t nb, int64_t i) {
  for (int64_t jb = 0; jb < nb; jb++) {
    const int64_t j0 = jb * 16;
    const int nc = (int)std::min<int64_t>(16, N - j0);
    alignas(32) int m0[8], m1[8];
    for (int c = 0; c < 8; c++) {
      m0[c] = c < nc ? -1 : 0;
      m1[c] = c + 8 < nc ? -1 : 0;
    }
    const __m256i mask0 = _mm256_load_si256((const __m256i*)m0);
    const __m256i mask1 = _mm256_load_si256((const __m256i*)m1);
    const float* bpp = bp + jb * K * 16;
    __m256 acc[R][2];
    for (int p = 0; p < R; p++) acc[p][0] = acc[p][1] = _mm256_setzero_ps();
    for (int64_t k = 0; k < K; k++) {
      const __m256 b0 = _mm256_loadu_ps(bpp + k * 16);
      const __m256 b1 = _mm256_loadu_ps(bpp + k * 16 + 8);
      for (int p = 0; p < R; p++) {
        const __m256 av = _mm256_broadcast_ss(ap + (i + p) * K + k);
        acc[p][0] = _mm256_fmadd_ps(av, b0, acc[p][0]);
        acc[p][1] = _mm256_fmadd_ps(av, b1, acc[p][1]);
      }
    }
    for (int p = 0; p < R; p++) {
      float* dst = cp + (i + p) * N + j0;
      _mm256_maskstore_ps(dst, mask0, acc[p][0]);
      _mm256_maskstore_ps(dst + 8, mask1, acc[p][1]);
    }
  }
}

void gemm_simd(const float* ap, const float* bp, float* cp, int64_t M, int64_t K,
               int64_t N, int64_t nb) {
  const int64_t ngroups = (M + 5) / 6;
  parallel_for(0, ngroups, 2, [&](int64_t b, int64_t e) {
    for (int64_t g = b; g < e; g++) {
      const int64_t i = g * 6;
      const int r = (int)std::min<int64_t>(6, M - i);
      switch (r) {
        case 6: row_group<6>(ap, bp, cp, K, N, nb, i); break;
        case 5: row_group<5>(ap, bp, cp, K, N, nb, i); break;
        case 4: row_group<4>(ap, bp, cp, K, N, nb, i); break;
        case 3: row_group<3>(ap, bp, cp, K, N, nb, i); break;
        case 2: row_group<2>(ap, bp, cp, K, N, nb, i); break;
        default: row_group<1>(ap, bp, cp, K, N, nb, i); break;
      }
    }
  });
}

#else  // scalar fallback

void gemm_simd(const float* ap, const float* bp, float* cp, int64_t M, int64_t K,
               int64_t N, int64_t /*nb*/) {
  parallel_for(0, M, 8, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++)
      for (int64_t j = 0; j < N; j++) {
        float s = 0;
        for (int64_t k = 0; k < K; k++) s += ap[i * K + k] * bp[k * N + j];
        cp[i * N + j] = s;
      }
  });
}

#endif

}  // namespace

void matmul(const Tensor& A, const Tensor& B, Tensor& C, const float* bias) {
  Tensor Af, Bf;
  const Tensor* a = &A;
  const Tensor* b = &B;
  if (A.dtype == DType::BF16) { to_f32(A, Af); a = &Af; }
  if (B.dtype == DType::BF16) { to_f32(B, Bf); b = &Bf; }
  const int64_t M = a->rows(), K = a->cols(), N = b->cols();
  C = Tensor({M, N}, DType::F32);
  if (M == 0 || N == 0 || K == 0) {
    if (bias && M && N) add_bias(C.ptr<float>(), M, N, bias);
    return;
  }
  const float* ap = a->ptr<float>();
  const float* bp = b->ptr<float>();
  float* cp = C.ptr<float>();

  SGemmFn sg = blas_sgemm();
  if (sg && M * N * K > kBlasMinWork) {
    sg(101, 111, 111, (int)M, (int)N, (int)K, 1.f, ap, (int)K, bp, (int)N, 0.f,
       cp, (int)N);
    if (bias) add_bias(cp, M, N, bias);
    return;
  }

#if SD_GEMM_SIMD
  const int64_t nb = (N + 15) / 16;
  float* pack = (float*)aligned_alloc(64, ((size_t)nb * K * 16 * 4 + 63) / 64 * 64);
  parallel_for(0, nb, 4, [&](int64_t b0, int64_t e0) {
    for (int64_t jb = b0; jb < e0; jb++) {
      const int64_t j0 = jb * 16;
      const int nc = (int)std::min<int64_t>(16, N - j0);
      float* dst = pack + jb * K * 16;
      for (int64_t k = 0; k < K; k++) {
        const float* src = bp + k * N + j0;
        for (int c = 0; c < nc; c++) dst[k * 16 + c] = src[c];
        for (int c = nc; c < 16; c++) dst[k * 16 + c] = 0.f;
      }
    }
  });
  gemm_simd(ap, pack, cp, M, K, N, nb);
  free(pack);
#else
  gemm_simd(ap, bp, cp, M, K, N, 0);
#endif
  if (bias) add_bias(cp, M, N, bias);
}

}  // namespace sd

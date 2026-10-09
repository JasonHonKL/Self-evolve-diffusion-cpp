// sd::qgemm: dynamic per-row int8 quantized GEMM (issue #3).
#include "core/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sd {
namespace {

#if defined(__AVX2__)
inline int32_t hsum_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

// ponytail: s32 accum overflows past K~133k (127*127*K > 2^31); fine for LLM K<=16k
inline int32_t dot_s8s8(const int8_t* a, const int8_t* b, int64_t K) {
  __m256i acc = _mm256_setzero_si256();
  int64_t k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256i a16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + k)));
    __m256i b16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + k)));
    __m256i a16h = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + k + 16)));
    __m256i b16h = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + k + 16)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, b16));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16h, b16h));
  }
  int32_t s = hsum_epi32(acc);
  for (; k < K; k++) s += (int32_t)a[k] * b[k];
  return s;
}

inline int32_t dot_s8s16(const int8_t* a, const int16_t* b, int64_t K) {
  __m256i acc = _mm256_setzero_si256();
  int64_t k = 0;
  for (; k + 16 <= K; k += 16) {
    __m256i a16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + k)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, _mm256_loadu_si256((const __m256i*)(b + k))));
  }
  int32_t s = hsum_epi32(acc);
  for (; k < K; k++) s += (int32_t)a[k] * b[k];
  return s;
}
#else
inline int32_t dot_s8s8(const int8_t* a, const int8_t* b, int64_t K) {
  int32_t s = 0;
  for (int64_t k = 0; k < K; k++) s += (int32_t)a[k] * b[k];
  return s;
}
inline int32_t dot_s8s16(const int8_t* a, const int16_t* b, int64_t K) {
  int32_t s = 0;
  for (int64_t k = 0; k < K; k++) s += (int32_t)a[k] * b[k];
  return s;
}
#endif

inline int round_to_i8(float x) {
#if defined(__AVX2__)
  int v = _mm_cvtss_si32(_mm_set_ss(x));
#else
  int v = (int)std::lrintf(x);
#endif
  return std::max(-127, std::min(127, v));
}

}  // namespace

void qgemm(const Tensor& A, const Tensor& Bq, const Tensor& bs, Tensor& out) {
  const int64_t M = A.rows(), K = A.cols(), N = Bq.rows();
  out = Tensor({M, N}, DType::F32);
  if (M == 0 || N == 0 || K == 0) return;
  const float* ap = A.ptr<float>();
  const int8_t* bqp = Bq.ptr<int8_t>();
  const float* bsp = bs.ptr<float>();
  float* op = out.ptr<float>();

  int8_t* aq = (int8_t*)aligned_alloc(64, ((size_t)M * K + 63) / 64 * 64);
  float* as = (float*)aligned_alloc(64, ((size_t)M * 4 + 63) / 64 * 64);
  parallel_for(0, M, 8, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      const float* r = ap + i * K;
      float mx = 0;
      for (int64_t k = 0; k < K; k++) mx = std::max(mx, std::fabs(r[k]));
      float sc = mx / 127.f;
      if (sc == 0.f) sc = 1.f;
      as[i] = sc;
      const float inv = 1.f / sc;
      int8_t* q = aq + i * K;
      for (int64_t k = 0; k < K; k++) q[k] = (int8_t)round_to_i8(r[k] * inv);
    }
  });

  // #17: widen B to s16 only when the M rows amortize it (M >= 2048), and
  // then per COLUMN-CHUNK task-local scratch — a full N*K panel would stream
  // 3x weight bytes per call (read i8 + write s16 + read s16) and at model
  // scale (11 GB of i8 per forward) that dominated wall time. Task = chunk of
  // B rows: scratch stays L2-resident across the M loop (same cache behavior
  // as the old full-widen, bounded memory).
#ifndef SD_QGEMM_CHUNK
#define SD_QGEMM_CHUNK 64
#endif
  if (M >= 2048) {
    const int64_t ch = std::min<int64_t>(SD_QGEMM_CHUNK, N);
    parallel_for(0, N, ch, [&](int64_t jb, int64_t je) {
      const int64_t nc = je - jb;
      int16_t* bq16 =
          (int16_t*)aligned_alloc(64, ((size_t)nc * K * 2 + 63) / 64 * 64);
      for (int64_t j = jb; j < je; j++) {
        const int8_t* src = bqp + j * K;
        int16_t* dst = bq16 + (j - jb) * K;
        for (int64_t k = 0; k < K; k++) dst[k] = (int16_t)src[k];
      }
      for (int64_t i = 0; i < M; i++) {
        const int8_t* ar = aq + i * K;
        const float asc = as[i];
        float* orow = op + i * N;
        for (int64_t j = jb; j < je; j++)
          orow[j] = (float)dot_s8s16(ar, bq16 + (j - jb) * K, K) *
                    (asc * bsp[j]);
      }
      free(bq16);
    });
  } else {
    parallel_for(0, N, 64, [&](int64_t jb, int64_t je) {
      for (int64_t i = 0; i < M; i++) {
        const int8_t* ar = aq + i * K;
        const float asc = as[i];
        float* orow = op + i * N;
        for (int64_t j = jb; j < je; j++)
          orow[j] = (float)dot_s8s8(ar, bqp + j * K, K) * (asc * bsp[j]);
      }
    });
  }

  free(aq);
  free(as);
}

}  // namespace sd

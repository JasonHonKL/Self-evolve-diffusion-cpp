// correctness + --bench for issues #2/#3 (matmul, qgemm).
#include "core/tensor.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

using namespace sd;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; }                  \
    else { printf("ok  : %s\n", msg); }                                    \
  } while (0)

static void naive_matmul(const Tensor& A, const Tensor& B, Tensor& C, const float* bias = nullptr) {
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

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double best_of(int reps, const std::function<void()>& f) {
  f();  // warmup
  double best = 1e30;
  for (int i = 0; i < reps; i++) {
    double t0 = now_s();
    f();
    best = std::min(best, now_s() - t0);
  }
  return best;
}

static double max_abs_err(const float* a, const float* b, int64_t n) {
  double m = 0;
  for (int64_t i = 0; i < n; i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

static double cosine(const float* a, const float* b, int64_t n) {
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
  return dot / std::sqrt(na * nb);
}

static void test_matmul() {
  struct SH { int64_t M, K, N; bool bf16; bool bias; };
  const SH shapes[] = {
    {17, 31, 13, false, false}, {3, 7, 5, false, false}, {64, 64, 64, false, true},
    {128, 77, 256, false, false}, {33, 65, 47, true, false}, {1, 64, 32, false, false},
    {512, 512, 512, false, false},
  };
  for (const auto& s : shapes) {
    Tensor A({s.M, s.K}, DType::F32), B({s.K, s.N}, DType::F32);
    randu(A, 100 + s.M), randu(B, 200 + s.K);
    Tensor Ab, Ab16;
    const Tensor* Mp = &A;   // what matmul sees (bf16 when testing that path)
    const Tensor* Rp = &A;   // what naive sees (f32 roundtrip of same values)
    if (s.bf16) { to_bf16(A, Ab); to_f32(Ab, Ab16); Mp = &Ab; Rp = &Ab16; }
    std::vector<float> bias(s.N);
    if (s.bias) for (int64_t j = 0; j < s.N; j++) bias[j] = (float)(j % 7) * 0.25f;
    Tensor C, R;
    matmul(*Mp, B, C, s.bias ? bias.data() : nullptr);
    naive_matmul(*Rp, B, R, s.bias ? bias.data() : nullptr);
    double refmax = 1e-30;
    for (int64_t i = 0; i < s.M * s.N; i++) refmax = std::max(refmax, (double)std::fabs(R.ptr<float>()[i]));
    double rel = max_abs_err(C.ptr<float>(), R.ptr<float>(), s.M * s.N) / refmax;
    char msg[128];
    snprintf(msg, sizeof msg, "matmul %lldx%lld @ %lldx%lld%s%s rel_err=%.2e",
             (long long)s.M, (long long)s.K, (long long)s.K, (long long)s.N,
             s.bf16 ? " bf16" : "", s.bias ? " +bias" : "", rel);
    CHECK(rel < 1e-3, msg);
  }
}

static void test_qgemm() {
  struct SH { int64_t M, K, N; };
  const SH shapes[] = {{17, 31, 13}, {64, 1024, 512}, {3072, 5120, 5120}};
  for (const auto& s : shapes) {
    Tensor A({s.M, s.K}, DType::F32), W({s.N, s.K}, DType::F32);
    randu(A, 7 + s.M); randu(W, 13 + s.N);
    Tensor Wq, sc;
    quantize_rowwise(W, Wq, sc);
    Tensor B({s.K, s.N}, DType::F32);  // B = dequant(W)^T
    {
      const int8_t* q = Wq.ptr<int8_t>(); const float* sp = sc.ptr<float>(); float* bp = B.ptr<float>();
      parallel_for(0, s.N, 64, [&](int64_t b, int64_t e) {
        for (int64_t j = b; j < e; j++)
          for (int64_t k = 0; k < s.K; k++) bp[k * s.N + j] = (float)q[j * s.K + k] * sp[j];
      });
    }
    Tensor O, Ref;
    qgemm(A, Wq, sc, O);
    matmul(A, B, Ref);
    double cos = cosine(O.ptr<float>(), Ref.ptr<float>(), s.M * s.N);
    char msg[128];
    snprintf(msg, sizeof msg, "qgemm %lldx%lld @ %lldx%lld int8 cosine=%.6f",
             (long long)s.M, (long long)s.K, (long long)s.N, (long long)s.K, cos);
    CHECK(cos > 0.999, msg);
  }
}

static void test_quantize_rowwise() {
  Tensor W({37, 513}, DType::F32); randu(W, 42);
  Tensor Wq, sc;
  quantize_rowwise(W, Wq, sc);
  const float* w = W.ptr<float>(); const int8_t* q = Wq.ptr<int8_t>(); const float* s = sc.ptr<float>();
  double maxerr = 0;
  for (int64_t i = 0; i < 37; i++)
    for (int64_t k = 0; k < 513; k++)
      maxerr = std::max(maxerr, (double)std::fabs(w[i*513+k] - (float)q[i*513+k] * s[i]) / s[i]);
  char msg[96];
  snprintf(msg, sizeof msg, "quantize_rowwise dequant err %.3f x scale", maxerr);
  CHECK(maxerr < 1.0, msg);
}

static void test_bf16_roundtrip() {
  Tensor X({129, 64}, DType::F32); randu(X, 5);
  Tensor Bf, Back;
  to_bf16(X, Bf); to_f32(Bf, Back);
  double err = max_abs_err(X.ptr<float>(), Back.ptr<float>(), X.numel());
  char msg[96];
  snprintf(msg, sizeof msg, "bf16 roundtrip max_abs_err %.5f", err);
  CHECK(err < 0.01, msg);
}

static void bench_one(int64_t M, int64_t K, int64_t N) {
  Tensor A({M, K}, DType::F32), B({K, N}, DType::F32), C;
  randu(A, 1); randu(B, 2);
  double t = best_of(M * N * K > (64ll << 20) ? 3 : 5, [&] { matmul(A, B, C); });
#ifdef SD_OPENBLAS
  const char* path = M * N * K > (64ll << 20) ? "openblas" : "custom  ";
#else
  const char* path = "custom  ";
#endif
  printf("matmul %5lldx%5lld x %5lldx%5lld [%s] %8.2f ms  %8.1f GFLOPS\n",
         (long long)M, (long long)K, (long long)K, (long long)N, path, t * 1e3,
         2.0 * M * N * K / t / 1e9);
}

static void bench() {
  printf("== bench (threads=%d) ==\n", nthreads());
  bench_one(1024, 1024, 1024);
  bench_one(3072, 5120, 5120);
  bench_one(192, 512, 512);
  bench_one(240, 1024, 256);

  const int64_t M = 3072, K = 5120, N = 5120;
  Tensor A({M, K}, DType::F32), W({N, K}, DType::F32);
  randu(A, 3); randu(W, 4);
  Tensor Wq, sc;
  quantize_rowwise(W, Wq, sc);
  Tensor O;
  double t = best_of(3, [&] { qgemm(A, Wq, sc, O); });
  printf("qgemm  %5lldx%5lld @ %5lldx%5lld int8      %8.2f ms  %8.1f TOPS\n",
         (long long)M, (long long)K, (long long)N, (long long)K, t * 1e3,
         2.0 * M * N * K / t / 1e9);
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(); return 0; }
  test_matmul();
  test_qgemm();
  test_quantize_rowwise();
  test_bf16_roundtrip();
  printf(g_fail ? "TEST_CORE FAILED (%d)\n" : "test_core OK\n", g_fail);
  return g_fail ? 1 : 0;
}

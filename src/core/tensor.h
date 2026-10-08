// Self-evolve-diffusion-cpp: core tensor API contract.
// Implemented by src/core/*.cpp (issue #2, #3). Everyone codes against THIS header.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sd {

enum class DType : uint8_t { F32 = 0, BF16 = 1, I8 = 2 };

// Row-major, CPU, owning or view (non-owning) tensor.
struct Tensor {
  std::vector<int64_t> shape;   // e.g. {B, M, K}
  DType dtype = DType::F32;
  void* data = nullptr;         // always allocated 64-byte aligned when owned
  size_t nbytes = 0;
  bool owning = true;

  Tensor() = default;
  Tensor(std::vector<int64_t> s, DType d);
  Tensor(const Tensor&) = delete;             // no accidental copies
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& o) noexcept;
  Tensor& operator=(Tensor&& o) noexcept;
  ~Tensor();

  int64_t numel() const;
  int64_t rows() const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
  int64_t cols() const { return shape.size() >= 1 ? shape[shape.size()-1] : 1; }
  size_t dim(int i) const { return (size_t)shape[i]; }
  void reshape(std::vector<int64_t> s);       // same numel required
  template <typename T> T* ptr() { return reinterpret_cast<T*>(data); }
  template <typename T> const T* ptr() const { return reinterpret_cast<const T*>(data); }
};

// ---- elementwise / layout (implemented in core) ----
void to_f32(const Tensor& bf16, Tensor& out);           // bf16 -> f32
void to_bf16(const Tensor& f32, Tensor& out);           // f32 -> bf16 (round-to-nearest-even)

// ---- GEMM: C[M,N] = A[M,K] @ B[K,N] (+ bias) ----
// A: F32 or BF16 (converted internally), B: F32, row-major, no transposes.
void matmul(const Tensor& A, const Tensor& B, Tensor& C, const float* bias = nullptr);

// ---- int8 weight-stationary qGEMM (issue #3) ----
// A: F32 activations [M,K]; Bq: I8 [N,K] (per-output-channel int8 weights);
// bs: F32 [N] scale per row of B (weight_scale); out: F32 [M,N].
// out = (A @ (Bq^T * bs)) computed as s32 accum then rescaled.
void qgemm(const Tensor& A, const Tensor& Bq, const Tensor& bs, Tensor& out);

// quantize fp32 [N,K] weights rowwise -> I8 [N,K] + scale [N] (max-abs / 127)
void quantize_rowwise(const Tensor& W, Tensor& Wq, Tensor& scale);

// ---- threading (issue #2) ----
void parallel_for(int64_t begin, int64_t end, int64_t grain,
                  const std::function<void(int64_t, int64_t)>& fn);
int nthreads();                       // hardware concurrency, cached
void set_nthreads(int n);

// ---- misc numerics used across blocks ----
void silu_inplace(Tensor& x);          // x * sigmoid(x), elementwise F32/BF16
void rmsnorm(Tensor& x, const Tensor& weight, float eps); // over last dim

// small deterministic RNG for tests (xorshift), fills f32 in [-1,1)
void randu(Tensor& x, uint64_t seed);

}  // namespace sd

// UMT5-XXL encoder forward pass. See t5enc.h for architecture notes.
#include "model/t5enc.h"
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sd {
namespace t5 {

using Clk = std::chrono::steady_clock;

static Tensor dup_f32(const Tensor& t) {  // small tensors only (norms, pos)
  Tensor o;
  if (t.dtype == DType::F32) {
    o = Tensor(t.shape, DType::F32);
    memcpy(o.data, t.data, t.nbytes);
  } else {
    to_f32(t, o);
  }
  return o;
}

T5Encoder::T5Encoder(const std::string& path) : st(load_safetensors(path)) {
  auto need = [&](const std::string& k) -> const Tensor& {
    auto it = st.tensors.find(k);
    if (it == st.tensors.end()) throw std::runtime_error("missing tensor " + k);
    return it->second;
  };
  norm1_w.resize(num_layers);
  norm2_w.resize(num_layers);
  pos_w.resize(num_layers);
  for (int i = 0; i < num_layers; i++) {
    norm1_w[i] = dup_f32(need("blocks." + std::to_string(i) + ".norm1.weight"));
    norm2_w[i] = dup_f32(need("blocks." + std::to_string(i) + ".norm2.weight"));
    pos_w[i] = dup_f32(need("blocks." + std::to_string(i) +
                            ".pos_embedding.embedding.weight"));
  }
  final_norm_w = dup_f32(need("norm.weight"));
  const Tensor& tok = need("token_embedding.weight");
  if (tok.dim(0) != vocab || tok.dim(1) != dim)
    throw std::runtime_error("bad token_embedding shape");
}

// Verbatim port of T5RelativeEmbedding._relative_position_bucket
// (bidirectional: num_buckets=32 -> 16 effective, max_exact=8, max_dist=128).
// Only called for p >= max_exact, so log(0) never happens (torch computes it
// unconditionally but discards it via where(); C++ must not — (long)-inf is UB).
static int rel_bucket(int rel) {
  const int nb = t5::T5Encoder::num_buckets / 2, max_exact = nb / 2;
  int b = (rel > 0) ? nb : 0;
  int p = std::abs(rel);
  int bucket;
  if (p < max_exact) {
    bucket = p;
  } else {
    float v = max_exact + std::log((float)p / max_exact) /
                             std::log(128.0f / max_exact) * (nb - max_exact);
    bucket = (int)std::min((long)v, (long)(nb - 1));  // trunc, torch .long()
  }
  return bucket + b;
}

static void gelu_tanh_inplace(Tensor& x) {
  float* p = x.ptr<float>();
  parallel_for(0, x.numel(), 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      float v = p[i];
      p[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
  });
}

static void add_inplace(Tensor& x, const Tensor& y) {
  float* a = x.ptr<float>();
  const float* b = y.ptr<float>();
  parallel_for(0, x.numel(), 4096, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) a[i] += b[i];
  });
}

void T5Encoder::build_f32_cache() const {
  // one-time bf16 -> f32 of every big weight (issue #17). token_embedding
  // excluded: 4.2 GB for a gather that touches only L rows. Each converted
  // bf16 region gets MADV_DONTNEED so peak RSS stays ~cache size, not
  // cache+mmap (18.5GB + 7.3GB would OOM this 23GB box).
  Clk::time_point t0 = Clk::now();
  int64_t n = 0;
  for (const auto& kv : st.tensors) {
    const Tensor& t = kv.second;
    if (t.dtype != DType::BF16 || kv.first == "token_embedding.weight") continue;
    Tensor o(t.shape, DType::F32);
    const uint16_t* s = t.ptr<uint16_t>();
    float* d = o.ptr<float>();
    // parallel (to_f32 in core is scalar; 4.6B elems would take ~minutes)
    parallel_for(0, t.numel(), 1 << 20, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; i++) {
        uint32_t u = (uint32_t)s[i] << 16;
        std::memcpy(&d[i], &u, 4);
      }
    });
    n += t.numel();
    w32_.emplace(kv.first, std::move(o));
    // drop the now-shadowed bf16 file pages (clean MAP_PRIVATE: re-faults
    // from disk if ever touched again — they aren't, except token gather)
    uintptr_t lo = (uintptr_t)t.data & ~uintptr_t(4095);
    uintptr_t hi = ((uintptr_t)t.data + t.nbytes + 4095) & ~uintptr_t(4095);
    madvise((void*)lo, hi - lo, MADV_DONTNEED);
  }
  std::printf("[t5] f32 weight cache built: %d tensors, %.2f GB, %.1f s\n",
              (int)w32_.size(), (double)n * 4 / 1e9,
              std::chrono::duration<double>(Clk::now() - t0).count());
}

void T5Encoder::free_weights() {
  // ponytail: whole-object kill switch, not fine-grained — pipeline sequences
  // T5 strictly BEFORE the DiT loads; encoder is unusable after this call.
  w32_.clear();
  st = SafetensorsFile();  // releases the mmap + all bf16 views
}

void T5Encoder::forward(const Tensor& ids, const Tensor& mask, Tensor& out) const {
  const int64_t L = ids.numel();
  const float* mf = mask.ptr<float>();
  auto W = [&](const std::string& k) -> const Tensor& {
    if (f32_cache) {
      if (w32_.empty()) build_f32_cache();
      auto it = w32_.find(k);
      if (it != w32_.end()) return it->second;
    }
    return st.tensors.at(k);
  };

  // token embedding gather (bf16 rows -> f32, row-wise to_f32)
  Tensor x({L, dim}, DType::F32);
  {
    const Tensor& tok = W("token_embedding.weight");
    const float* idf = ids.ptr<float>();
    for (int64_t i = 0; i < L; i++) {
      Tensor row({1, dim}, DType::F32);
      Tensor view({1, dim}, DType::BF16);
      view.owning = false;
      view.data = const_cast<void*>((void*)(tok.ptr<uint16_t>() + (int64_t)(int)idf[i] * dim));
      to_f32(view, row);
      memcpy(x.ptr<float>() + i * dim, row.data, (size_t)dim * 4);
    }
  }

  Tensor q, k, v, ctx, y, h, a, g;
  std::vector<float> logits(L), posb((int64_t)num_heads * L * L);

  for (int layer = 0; layer < num_layers; layer++) {
    const std::string p = "blocks." + std::to_string(layer) + ".";

    // ---- self-attention (no qk scaling, per-layer rel pos bias, pad mask) ----
    h = Tensor({L, dim}, DType::F32);
    memcpy(h.data, x.data, x.nbytes);
    rmsnorm(h, norm1_w[layer], 1e-6f);
    matmul(h, W(p + "attn.q.weight"), q);
    matmul(h, W(p + "attn.k.weight"), k);
    matmul(h, W(p + "attn.v.weight"), v);

    {  // pos bias [heads, L, L] = emb[bucket(j-i)][head]
      const float* ew = pos_w[layer].ptr<float>();
      for (int64_t i = 0; i < L; i++)
        for (int64_t j = 0; j < L; j++) {
          int bkt = rel_bucket((int)(j - i));
          for (int n = 0; n < num_heads; n++)
            posb[((int64_t)n * L + i) * L + j] = ew[bkt * num_heads + n];
        }
    }

    ctx = Tensor({L, dim}, DType::F32);
    const float* qp = q.ptr<float>();
    const float* kp = k.ptr<float>();
    const float* vp = v.ptr<float>();
    float* cp = ctx.ptr<float>();
    parallel_for(0, L, 1, [&](int64_t b, int64_t e) {
      std::vector<float> row(L);
      for (int64_t i = b; i < e; i++)
        for (int n = 0; n < num_heads; n++) {
          const float* qi = qp + i * dim + n * head_dim;
          float denom = 0, maxl = -std::numeric_limits<float>::infinity();
          for (int64_t j = 0; j < L; j++) {
            const float* kj = kp + j * dim + n * head_dim;
            float s = 0;
            for (int c = 0; c < head_dim; c++) s += qi[c] * kj[c];
            if (mf[j] == 0) s = -std::numeric_limits<float>::infinity();
            else s += posb[((int64_t)n * L + i) * L + j];
            row[j] = s;
            maxl = std::max(maxl, s);
          }
          for (int64_t j = 0; j < L; j++) {
            row[j] = std::exp(row[j] - maxl);
            denom += row[j];
          }
          float inv = 1.0f / denom;
          float* co = cp + i * dim + n * head_dim;
          for (int c = 0; c < head_dim; c++) co[c] = 0;
          for (int64_t j = 0; j < L; j++) {
            float wgt = row[j] * inv;
            const float* vj = vp + j * dim + n * head_dim;
            for (int c = 0; c < head_dim; c++) co[c] += wgt * vj[c];
          }
        }
    });
    matmul(ctx, W(p + "attn.o.weight"), y);
    add_inplace(x, y);

    // ---- gated-GELU FFN: fc2(fc1(h) * gelu(gate(h))) ----
    memcpy(h.data, x.data, x.nbytes);
    rmsnorm(h, norm2_w[layer], 1e-6f);
    matmul(h, W(p + "ffn.fc1.weight"), a);         // [L, dim_ffn]
    matmul(h, W(p + "ffn.gate.0.weight"), g);      // [L, dim_ffn]
    gelu_tanh_inplace(g);
    {
      float* ap = a.ptr<float>();
      const float* gp = g.ptr<float>();
      parallel_for(0, a.numel(), 4096, [&](int64_t s, int64_t e2) {
        for (int64_t i = s; i < e2; i++) ap[i] *= gp[i];
      });
    }
    matmul(a, W(p + "ffn.fc2.weight"), y);         // [L, dim]
    add_inplace(x, y);
  }

  rmsnorm(x, final_norm_w, 1e-6f);
  out = std::move(x);
}

}  // namespace t5
}  // namespace sd

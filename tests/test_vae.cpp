// Wan2.2-TI2V-5B VAE decoder parity test (issue #12).
// Run from repo root:
//   python3 tools/convert_vae.py && python3 tools/golden/gen_vae_golden.py
//   ./build/tests/test_vae
// Pass: cosine > 0.99 vs the bf16 torch golden (fp32 C++ from bf16 weights —
// exact parity impossible), chunked == whole within 1e-4.
#include "model/vae_dec.h"
#include "test_support/npz.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace sd;

static double cosine(const float* a, const float* b, int64_t n) {
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; i++) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

static double maxabs(const float* a, const float* b, int64_t n) {
  double m = 0;
  for (int64_t i = 0; i < n; i++)
    m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

// concat chunk outputs along dim 1 ([3,T,H,W] each)
static Tensor cat_t(const Tensor& a, const Tensor& b) {
  const int64_t T0 = a.dim(1), T1 = b.dim(1), H = a.dim(2), W = a.dim(3);
  Tensor o({3, T0 + T1, H, W}, DType::F32);
  for (int64_t c = 0; c < 3; c++) {
    memcpy(o.ptr<float>() + c * (T0 + T1) * H * W, a.ptr<float>() + c * T0 * H * W,
           (size_t)T0 * H * W * 4);
    memcpy(o.ptr<float>() + (c * (T0 + T1) + T0) * H * W,
           b.ptr<float>() + c * T1 * H * W, (size_t)T1 * H * W * 4);
  }
  return o;
}

static Tensor slice_z(const Tensor& z, int64_t t0, int64_t t1) {
  const int64_t F = z.dim(1), H = z.dim(2), W = z.dim(3);
  Tensor c({48, t1 - t0, H, W}, DType::F32);
  for (int64_t ch = 0; ch < 48; ch++)
    memcpy(c.ptr<float>() + ch * (t1 - t0) * H * W,
           z.ptr<float>() + ch * F * H * W + t0 * H * W,
           (size_t)(t1 - t0) * H * W * 4);
  return c;
}

int main() {
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/vae_golden.npz");
  } catch (const std::exception& e) {
    printf("load_npz: %s\n", e.what());
    return 1;
  }
  vae::VaeDecoder dec("ckpts/vae_dec.safetensors");

  // ---- 1. whole decode vs torch bf16 golden ----
  const Tensor& lat = g.at("latent");  // [48,2,8,8]
  const Tensor& ref = g.at("out");     // [3,5,128,128]
  Tensor out;
  dec.decode(lat, out);
  bool shape_ok = out.shape == ref.shape;
  printf("golden: latent [%lld,%lld,%lld,%lld] -> out [%lld,%lld,%lld,%lld] "
         "(expect [3,5,128,128])\n",
         (long long)lat.dim(0), (long long)lat.dim(1), (long long)lat.dim(2),
         (long long)lat.dim(3), (long long)out.dim(0), (long long)out.dim(1),
         (long long)out.dim(2), (long long)out.dim(3));
  if (!shape_ok) {
    printf("VAE TEST FAILED (shape)\n");
    return 1;
  }
  double cos_w = cosine(out.ptr<float>(), ref.ptr<float>(), out.numel());
  double mx_w = maxabs(out.ptr<float>(), ref.ptr<float>(), out.numel());
  printf("whole  vs golden: cosine=%.6f max_abs=%.4f %s\n", cos_w, mx_w,
         cos_w > 0.99 ? "PASS" : "FAIL");

  // ---- 2. chunked (1,1) vs golden — reference's own chunking ----
  {
    vae::VaeDecoder::Cache c;
    Tensor a, b;
    dec.decode_chunk(slice_z(lat, 0, 1), a, c);
    dec.decode_chunk(slice_z(lat, 1, 2), b, c);
    Tensor cc = cat_t(a, b);
    double cos_c = cosine(cc.ptr<float>(), ref.ptr<float>(), ref.numel());
    printf("chunks(1,1) vs golden: cosine=%.6f %s\n", cos_c,
           cos_c > 0.99 ? "PASS" : "FAIL");
    if (cos_c <= 0.99) {
      printf("VAE TEST FAILED\n");
      return 1;
    }
  }

  // ---- 3. chunked == whole identity on a random [48,8,8,8] ----
  {
    Tensor z({48, 8, 8, 8}, DType::F32);
    randu(z, 7);
    Tensor w;
    dec.decode(z, w);  // internal chunks: [0], [1..7]
    vae::VaeDecoder::Cache c;
    Tensor a, b, d;
    dec.decode_chunk(slice_z(z, 0, 1), a, c);  // anchor
    dec.decode_chunk(slice_z(z, 1, 2), b, c);  // 1-frame chunk (cache fixup)
    dec.decode_chunk(slice_z(z, 2, 8), d, c);  // 6-frame chunk
    Tensor cc = cat_t(cat_t(a, b), d);
    double mx = maxabs(w.ptr<float>(), cc.ptr<float>(), w.numel());
    printf("whole vs chunks(1,1,6): T=%lld max_abs=%.2e %s\n",
           (long long)w.dim(1), mx, mx < 1e-4 ? "PASS" : "FAIL");
    if (mx >= 1e-4 || w.dim(1) != 29) {
      printf("VAE TEST FAILED\n");
      return 1;
    }
  }

  if (cos_w <= 0.99) {
    printf("VAE TEST FAILED\n");
    return 1;
  }

  // ---- 4. timing: [48,8,32,32] whole decode ----
  {
    Tensor z({48, 8, 32, 32}, DType::F32);
    randu(z, 42);
    auto t0 = std::chrono::steady_clock::now();
    Tensor o;
    dec.decode(z, o);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("decode [48,8,32,32] -> [3,%lld,%lld,%lld] in %.1f s\n",
           (long long)o.dim(1), (long long)o.dim(2), (long long)o.dim(3), s);
  }

  printf("VAE TEST PASSED\n");
  return 0;
}

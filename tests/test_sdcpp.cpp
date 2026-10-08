// sdcpp tests: pack/load roundtrip, error cases, real-checkpoint header + slice (issue #15)
#include "core/tensor.h"
#include "serde/safetensors.h"
#include "serde/sdcpp.h"

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sd;

#define CHECK(x)                                                                    \
  do {                                                                              \
    if (!(x)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; }     \
  } while (0)

static const char* ST = "/tmp/opencode/sdcpp_small.st";
static const char* OUT = "/tmp/opencode/sdcpp_small.sdcpp";

static std::string repo_root() {
  std::string f = __FILE__;
  size_t p = f.rfind("/tests/");
  return p == std::string::npos ? "." : f.substr(0, p);
}

static int run(const std::string& cmd) {
  printf("-- %s\n", cmd.c_str());
  int rc = system(cmd.c_str());
  CHECK(rc == 0);
  return 0;
}

static void expect_throw(const char* what, const std::string& path) {
  try {
    std::shared_ptr<void> o;
    load_sdcpp(path, o);
  } catch (const std::exception&) {
    return;
  }
  printf("FAIL: expected throw: %s (%s)\n", what, path.c_str());
  exit(1);
}

// (1) mixed small file: [256,512] bf16 (quantized), [17] bf16 (<65536 -> raw f32), [5,7] f32
static int test_roundtrip() {
  Tensor big({256, 512}, DType::F32);
  randu(big, 7);
  Tensor bigbf;
  to_bf16(big, bigbf);
  Tensor ref;
  to_f32(bigbf, ref);  // exactly what the packer sees (bf16 -> f32)

  Tensor tiny({17}, DType::F32);
  randu(tiny, 9);
  Tensor tinybf;
  to_bf16(tiny, tinybf);
  Tensor tinyref;
  to_f32(tinybf, tinyref);

  Tensor fs({5, 7}, DType::F32);
  randu(fs, 11);
  std::vector<uint8_t> fs_bytes((uint8_t*)fs.data, (uint8_t*)fs.data + fs.nbytes);

  std::unordered_map<std::string, Tensor> m;
  m.emplace("big", std::move(bigbf));
  m.emplace("tiny", std::move(tinybf));
  m.emplace("f32t", std::move(fs));
  write_safetensors(ST, m);

  auto t0 = std::chrono::steady_clock::now();
  if (int rc = run("python3 " + repo_root() + "/tools/quantize/pack_sdcpp.py " + ST + " " + OUT))
    return rc;
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
  printf("small pack time: %.0f ms\n", ms);

  std::shared_ptr<void> owner;
  auto ts = load_sdcpp(OUT, owner);
  CHECK(owner != nullptr);
  CHECK(ts.size() == 4);  // big + big.scale + tiny + f32t

  const Tensor& q = ts.at("big");
  CHECK(q.dtype == DType::I8 && !q.owning);
  CHECK(q.shape == std::vector<int64_t>({256, 512}) && q.nbytes == 256u * 512);
  const Tensor& sc = ts.at("big.scale");
  CHECK(sc.dtype == DType::F32 && sc.shape == std::vector<int64_t>({256}));

  // dequant q*scale[row], compare vs bf16->f32 reference
  const float* rp = ref.ptr<float>();
  const int8_t* qp = q.ptr<int8_t>();
  const float* sp = sc.ptr<float>();
  double dot = 0, nq = 0, nr = 0;
  for (int64_t j = 0; j < 256; j++) {
    double amax = 0;
    for (int64_t k = 0; k < 512; k++) amax = std::max(amax, (double)std::abs(rp[j * 512 + k]));
    double bound = amax / 127.0 + 1e-5;
    for (int64_t k = 0; k < 512; k++) {
      double d = qp[j * 512 + k] * (double)sp[j];
      double r = rp[j * 512 + k];
      CHECK(std::abs(d - r) <= bound);  // max quant err = amax/127
      dot += d * r; nq += d * d; nr += r * r;
    }
  }
  double cos = dot / std::sqrt(nq * nr);
  printf("big: cosine=%.6f\n", cos);
  CHECK(cos > 0.999);

  const Tensor& t2 = ts.at("tiny");  // preserved: raw f32
  CHECK(t2.dtype == DType::F32 && t2.shape == std::vector<int64_t>({17}));
  CHECK(t2.nbytes == 68u && std::memcmp(t2.data, tinyref.data, 68) == 0);

  const Tensor& t3 = ts.at("f32t");
  CHECK(t3.dtype == DType::F32 && t3.shape == std::vector<int64_t>({5, 7}));
  CHECK(t3.nbytes == 140u && std::memcmp(t3.data, fs_bytes.data(), 140) == 0);
  return 0;
}

// (2) error cases: bad magic, truncation, missing file
static int test_errors() {
  std::ifstream in(OUT, std::ios::binary);
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  CHECK(all.size() > 64);
  { std::ofstream o("/tmp/opencode/sdcpp_trunc.sdcpp", std::ios::binary);
    o.write(all.data(), (std::streamsize)all.size() - 64); }
  { std::string bad = all; bad[0] = 'X';
    std::ofstream o("/tmp/opencode/sdcpp_badmagic.sdcpp", std::ios::binary);
    o.write(bad.data(), (std::streamsize)bad.size()); }
  expect_throw("truncated", "/tmp/opencode/sdcpp_trunc.sdcpp");
  expect_throw("bad magic", "/tmp/opencode/sdcpp_badmagic.sdcpp");
  expect_throw("missing file", "/tmp/opencode/sdcpp_nope.sdcpp");
  return 0;
}

static float bf16_elem(const void* base, int64_t i) {
  uint16_t u;
  std::memcpy(&u, (const char*)base + i * 2, 2);
  uint32_t b = (uint32_t)u << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

// python cross-check: packer's stored scales == amax(row)/127 of the original
static const char* PYVERIFY = R"PY(
import json, mmap, struct, sys
import numpy as np
orig_path, sdcpp_path, name, ncheck = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(orig_path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
e = hdr[name]
beg = e["data_offsets"][0]
last = e["shape"][-1]
with open(orig_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    x = ((np.frombuffer(mm, np.uint16, ncheck * last, 8 + n + beg).astype(np.uint32)) << 16
        ).view(np.float32).reshape(ncheck, last)
    ref = (np.abs(x).max(axis=1) / 127.0).astype(np.float32)
    mm.close()
with open(sdcpp_path, "rb") as f:  # stream-parse the header (file can be GBs)
    assert f.read(4) == b"SDCP"
    ver, cnt = struct.unpack("<II", f.read(8))
    assert ver == 1
    found = None
    for _ in range(cnt):
        (ln,) = struct.unpack("<I", f.read(4))
        nm = f.read(ln).decode()
        code = struct.unpack("<B", f.read(1))[0]
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dq" % nd, f.read(8 * nd))
        (off,) = struct.unpack("<Q", f.read(8))
        if nm == name:
            found = (code, dims, off, f.tell())
    code, dims, off, blob_base = found
    numel = int(np.prod(dims)) if dims else 1
    rows = numel // dims[-1]
    with open(sdcpp_path, "rb") as g:
        mm = mmap.mmap(g.fileno(), 0, access=mmap.ACCESS_READ)
        scales = np.frombuffer(mm, np.float32, rows, blob_base + off + numel)
        ok = np.array_equal(scales[:ncheck], ref)
        print("python verify: scales", "match" if ok else "MISMATCH", name)
        sys.exit(0 if ok else 1)
)PY";

// (3) real checkpoint (only if download finished): header listing, pack 3 largest, slice roundtrip
static int test_real() {
  std::string ck = repo_root() + "/ckpts/Ovi/model.safetensors";
  struct stat st;
  if (stat(ck.c_str(), &st) != 0 || st.st_size < 23200000000ll) {
    printf("skip real-checkpoint part (size=%lld)\n",
           stat(ck.c_str(), &st) == 0 ? (long long)st.st_size : -1LL);
    return 0;
  }
  SafetensorsFile f = load_safetensors(ck);  // header-only validation; no data pages touched
  printf("Ovi header: %zu tensors\n", f.tensors.size());
  int64_t total = 0;
  for (const auto& kv : f.tensors) total += (int64_t)kv.second.nbytes;
  for (const auto& kv : f.tensors) {
    const Tensor& t = kv.second;
    printf("  %-60s %-18s %12zu bytes\n", kv.first.c_str(),
           (std::to_string(t.shape[0]) +
            (t.shape.size() > 1
                 ? " x " + std::to_string(t.shape[t.shape.size() - 1]) +
                       (t.shape.size() > 2 ? " x ..." : "")
                 : ""))
               .c_str(),
           t.nbytes);
  }
  printf("total: %.2f GB across %zu tensors\n", total / 1e9, f.tensors.size());

  std::vector<const std::pair<const std::string, Tensor>*> top;
  for (const auto& kv : f.tensors) top.push_back(&kv);
  std::stable_sort(top.begin(), top.end(), [](auto* a, auto* b) {
    return a->second.nbytes > b->second.nbytes;
  });
  std::string only = top[0]->first;
  for (int i = 1; i < 3 && i < (int)top.size(); i++) only += "," + top[i]->first;
  printf("packing 3 largest via --only %s\n", only.c_str());
  if (int rc = run("python3 " + repo_root() + "/tools/quantize/pack_sdcpp.py " + ck +
                   " /tmp/opencode/test_pack.sdcpp --only " + only))
    return rc;

  std::shared_ptr<void> own;
  auto ts = load_sdcpp("/tmp/opencode/test_pack.sdcpp", own);
  CHECK(ts.size() == 6);  // 3 quantized + 3 .scale
  for (int i = 0; i < 3; i++) {
    const std::string& nm = top[i]->first;
    CHECK(ts.count(nm) && ts.count(nm + ".scale"));
    CHECK(ts.at(nm).dtype == DType::I8);
  }

  // roundtrip-verify one 4KB slice of the largest tensor vs original bf16->f32
  const std::string& nm = top[0]->first;
  const Tensor& o = f.tensors.at(nm);
  const Tensor& q = ts.at(nm);
  const Tensor& sc = ts.at(nm + ".scale");
  int64_t last = q.cols();
  int64_t e0 = 2 * last;                       // row-aligned start
  int64_t E = std::min<int64_t>(4096, q.numel() - e0);
  int64_t r0 = e0 / last, r1 = (e0 + E - 1) / last;
  std::vector<double> amax(r1 - r0 + 1, 0.0);
  for (int64_t j = r0; j <= r1; j++)
    for (int64_t k = 0; k < last; k++)
      amax[j - r0] = std::max(amax[j - r0], (double)std::abs(bf16_elem(o.data, j * last + k)));
  const int8_t* qp = q.ptr<int8_t>();
  const float* sp = sc.ptr<float>();
  double maxrel = 0;
  for (int64_t i = 0; i < E; i++) {
    int64_t e = e0 + i, j = e / last;
    double d = qp[e] * (double)sp[j];
    double r = bf16_elem(o.data, e);
    double bound = amax[j - r0] / 127.0 + 1e-6 * amax[j - r0] + 1e-30;
    CHECK(std::abs(d - r) <= bound);
    maxrel = std::max(maxrel, std::abs(d - r) / (r != 0 ? std::abs(r) : 1));
  }
  printf("slice roundtrip %s: %lld elems, max rel err %.2e\n", nm.c_str(), (long long)E, maxrel);

  { std::ofstream py("/tmp/opencode/sdcpp_verify.py"); py << PYVERIFY; }
  if (int rc = run("python3 /tmp/opencode/sdcpp_verify.py " + ck +
                   " /tmp/opencode/test_pack.sdcpp " + nm + " 16"))
    return rc;
  unlink("/tmp/opencode/test_pack.sdcpp");
  unlink("/tmp/opencode/sdcpp_verify.py");
  return 0;
}

int main() {
  if (int r = test_roundtrip()) return r;
  if (int r = test_errors()) return r;
  if (int r = test_real()) return r;
  for (const char* p : {ST, OUT, "/tmp/opencode/sdcpp_trunc.sdcpp",
                        "/tmp/opencode/sdcpp_badmagic.sdcpp"})
    unlink(p);
  printf("test_sdcpp OK\n");
  return 0;
}

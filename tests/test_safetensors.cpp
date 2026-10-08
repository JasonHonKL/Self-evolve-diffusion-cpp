// safetensors tests: round-trip, python cross-check, error cases, big-file offsets (issue #4)
#include "core/tensor.h"
#include "serde/safetensors.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

using namespace sd;

#define CHECK(x)                                                                    \
  do {                                                                              \
    if (!(x)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; }     \
  } while (0)

static void expect_throw(const char* what, const std::string& path) {
  try {
    load_safetensors(path);
  } catch (const std::exception&) {
    return;
  }
  printf("FAIL: expected throw: %s (%s)\n", what, path.c_str());
  exit(1);
}

static void put_file(const char* path, const void* p, size_t n) {
  std::ofstream o(path, std::ios::binary);
  o.write((const char*)p, (std::streamsize)n);
}

// (1) round-trip: F32 odd shape {3,5,7}, BF16 {4,6}, I8 {13} — byte-compare
static int test_roundtrip() {
  std::unordered_map<std::string, Tensor> m;
  std::vector<uint8_t> ea, eb, ec;

  Tensor a({3, 5, 7}, DType::F32);
  for (int64_t i = 0; i < a.numel(); i++) a.ptr<float>()[i] = (float)i * 0.25f - 11.f;
  ea.assign((uint8_t*)a.data, (uint8_t*)a.data + a.nbytes);

  Tensor b({4, 6}, DType::BF16);
  for (size_t i = 0; i < b.nbytes; i++) ((uint8_t*)b.data)[i] = (uint8_t)(i * 31 + 5);
  eb.assign((uint8_t*)b.data, (uint8_t*)b.data + b.nbytes);

  Tensor c({13}, DType::I8);
  for (int64_t i = 0; i < c.numel(); i++) c.ptr<int8_t>()[i] = (int8_t)(i * 3 - 20);
  ec.assign((uint8_t*)c.data, (uint8_t*)c.data + c.nbytes);

  m.emplace("a", std::move(a));
  m.emplace("b", std::move(b));
  m.emplace("c", std::move(c));
  write_safetensors("/tmp/sd_st_roundtrip.st", m);

  SafetensorsFile f = load_safetensors("/tmp/sd_st_roundtrip.st");
  CHECK(f.tensors.size() == 3);
  CHECK(f.mapping && f.path == "/tmp/sd_st_roundtrip.st");

  const Tensor& a2 = f.tensors.at("a");
  CHECK(a2.dtype == DType::F32 && !a2.owning);
  CHECK(a2.shape == std::vector<int64_t>({3, 5, 7}));
  CHECK(a2.nbytes == 420u && memcmp(a2.data, ea.data(), 420) == 0);

  const Tensor& b2 = f.tensors.at("b");
  CHECK(b2.dtype == DType::BF16 && !b2.owning);
  CHECK(b2.shape == std::vector<int64_t>({4, 6}));
  CHECK(b2.nbytes == 48u && memcmp(b2.data, eb.data(), 48) == 0);

  const Tensor& c2 = f.tensors.at("c");
  CHECK(c2.dtype == DType::I8 && !c2.owning);
  CHECK(c2.shape == std::vector<int64_t>({13}));
  CHECK(c2.nbytes == 13u && memcmp(c2.data, ec.data(), 13) == 0);
  return 0;
}

// (2) python-generated file (F16/I32/U8 + __metadata__) — verify dtypes, shapes, bytes
static const char* PYSCRIPT = R"PY(
import json, struct, sys
f16 = [1.0, -1.0, 2.0, 0.5, -0.5, 0.0]
i32 = [0, 1, -1, 2147483647, -2147483648, 7]
u8 = [0, 1, 127, 128, 255, 42]
parts, hdr = b"", {"__metadata__": {"repo": "sd-test"}}
for name, code, dt, vals in [("f16", "e", "F16", f16), ("i32", "i", "I32", i32), ("u8", "B", "U8", u8)]:
    blob = struct.pack("<%d%s" % (len(vals), code), *vals)
    hdr[name] = {"dtype": dt, "shape": [len(vals)], "data_offsets": [len(parts), len(parts) + len(blob)]}
    parts += blob
js = json.dumps(hdr).encode()
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(js)) + js + parts)
# bad file: data starts at 4 -> non-contiguous offsets
bad = {"x": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]}}
js = json.dumps(bad).encode()
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<Q", len(js)) + js + b"\0" * 8)
)PY";

static int test_python_file() {
  { std::ofstream("/tmp/sd_st_gen.py") << PYSCRIPT; }
  std::string cmd = "python3 /tmp/sd_st_gen.py /tmp/sd_st_python.st /tmp/sd_st_gap.st";
  int rc = system(cmd.c_str());
  if (rc != 0) { printf("FAIL: python3 generation failed (rc=%d)\n", rc); return 1; }

  SafetensorsFile f = load_safetensors("/tmp/sd_st_python.st");
  CHECK(f.tensors.size() == 3);  // __metadata__ skipped
  CHECK(f.tensors.count("__metadata__") == 0);

  const Tensor& h = f.tensors.at("f16");
  CHECK(h.dtype == DType::BF16 && h.shape == std::vector<int64_t>({6}) && h.nbytes == 12u);
  static const uint16_t f16_bits[6] = {0x3C00, 0xBC00, 0x4000, 0x3800, 0xB800, 0x0000};
  CHECK(memcmp(h.data, f16_bits, 12) == 0);

  const Tensor& q = f.tensors.at("i32");
  CHECK(q.dtype == DType::F32 && q.shape == std::vector<int64_t>({6}) && q.nbytes == 24u);
  static const int32_t i32_vals[6] = {0, 1, -1, 2147483647, -2147483647 - 1, 7};
  CHECK(memcmp(q.data, i32_vals, 24) == 0);

  const Tensor& u = f.tensors.at("u8");
  CHECK(u.dtype == DType::I8 && u.shape == std::vector<int64_t>({6}) && u.nbytes == 6u);
  static const uint8_t u8_vals[6] = {0, 1, 127, 128, 255, 42};
  CHECK(memcmp(u.data, u8_vals, 6) == 0);

  expect_throw("non-contiguous offsets", "/tmp/sd_st_gap.st");
  return 0;
}

// (3) error cases: truncation and bad header length must throw
static int test_errors() {
  put_file("/tmp/sd_st_err0.st", "", 0);  // empty
  expect_throw("empty file", "/tmp/sd_st_err0.st");
  put_file("/tmp/sd_st_err1.st", "abcd", 4);  // shorter than the 8-byte prefix
  expect_throw("short file", "/tmp/sd_st_err1.st");
  uint8_t huge[10] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, '{', '}'};
  put_file("/tmp/sd_st_err2.st", huge, sizeof huge);  // absurd header length
  expect_throw("bad header length", "/tmp/sd_st_err2.st");
  uint8_t badjson[11] = {3, 0, 0, 0, 0, 0, 0, 0, '{', 'n', 'o'};
  put_file("/tmp/sd_st_err3.st", badjson, sizeof badjson);
  expect_throw("bad json", "/tmp/sd_st_err3.st");

  std::ifstream in("/tmp/sd_st_roundtrip.st", std::ios::binary);
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  CHECK(all.size() > 16);
  put_file("/tmp/sd_st_err4.st", all.data(), all.size() - 8);  // data region cut
  expect_throw("truncated data", "/tmp/sd_st_err4.st");

  expect_throw("missing file", "/tmp/sd_st_does_not_exist.st");
  return 0;
}

// (4) 300MB F32 tensor — offset math sanity via random spot checks
static int test_big() {
  const int64_t n = 300ll * 1024 * 1024 / 4;
  Tensor big({n}, DType::F32);
  float* p = big.ptr<float>();
  auto val = [](int64_t i) { return (float)(i % 1000003) / 8.f - 500.f; };
  for (int64_t i = 0; i < n; i++) p[i] = val(i);
  std::unordered_map<std::string, Tensor> m;
  m.emplace("big", std::move(big));
  write_safetensors("/tmp/sd_st_big.st", m);

  SafetensorsFile f = load_safetensors("/tmp/sd_st_big.st");
  const Tensor& t = f.tensors.at("big");
  CHECK(t.dtype == DType::F32 && !t.owning);
  CHECK(t.shape == std::vector<int64_t>({n}) && t.nbytes == (size_t)n * 4);
  const float* q = t.ptr<float>();
  CHECK(q[0] == val(0) && q[n - 1] == val(n - 1));  // first/last exercise begin/end offsets
  uint64_t x = 88172645463325252ull;
  for (int k = 0; k < 1000; k++) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    int64_t i = (int64_t)(x >> 33) % n;
    CHECK(q[i] == val(i));
  }
  return 0;
}

int main() {
  if (int r = test_roundtrip()) return r;
  if (int r = test_python_file()) return r;
  if (int r = test_errors()) return r;
  if (int r = test_big()) return r;
  for (const char* p : {"/tmp/sd_st_roundtrip.st", "/tmp/sd_st_python.st", "/tmp/sd_st_gap.st",
                        "/tmp/sd_st_big.st", "/tmp/sd_st_gen.py", "/tmp/sd_st_err0.st",
                        "/tmp/sd_st_err1.st", "/tmp/sd_st_err2.st", "/tmp/sd_st_err3.st",
                        "/tmp/sd_st_err4.st"})
    unlink(p);
  printf("test_safetensors OK\n");
  return 0;
}

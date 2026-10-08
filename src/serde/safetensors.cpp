// safetensors mmap loader + writer (issue #4). POSIX only, no deps.
#include "serde/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sd {
namespace {

// ---- minimal JSON: objects, strings, arrays of ints (all a safetensors header has) ----
struct Json {
  enum Type { OBJ, ARR, STR };
  Type type = STR;
  std::string str;
  std::vector<int64_t> arr;
  std::vector<std::pair<std::string, Json>> obj;
};

struct Parser {
  const char* p;
  const char* end;
  [[noreturn]] void fail(const char* m) const {
    throw std::runtime_error(std::string("safetensors: bad header JSON (") + m + ")");
  }
  void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
  char peek() { ws(); if (p >= end) fail("eof"); return *p; }
  void expect(char c) { if (peek() != c) fail("syntax"); ++p; }
  std::string str() {
    expect('"');
    std::string s;
    for (;;) {
      if (p >= end) fail("eof");
      char c = *p++;
      if (c == '"') return s;
      if (c != '\\') { s += c; continue; }
      if (p >= end) fail("eof");
      char e = *p++;
      switch (e) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u':  // \uXXXX dropped: tensor names / metadata are ASCII in practice
          if (end - p < 4) fail("escape");
          p += 4;
          break;
        default: fail("escape");
      }
    }
  }
  int64_t integer() {
    ws();
    bool neg = (p < end && *p == '-');
    if (neg) ++p;
    if (p >= end || *p < '0' || *p > '9') fail("int");
    int64_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return neg ? -v : v;
  }
  Json value() {
    Json j;
    char c = peek();
    if (c == '"') { j.type = Json::STR; j.str = str(); return j; }
    if (c == '[') {
      j.type = Json::ARR; ++p;
      if (peek() == ']') { ++p; return j; }
      for (;;) {
        j.arr.push_back(integer());
        char d = peek();
        if (d == ',') { ++p; continue; }
        if (d == ']') { ++p; break; }
        fail("array");
      }
      return j;
    }
    if (c == '{') {
      j.type = Json::OBJ; ++p;
      if (peek() == '}') { ++p; return j; }
      for (;;) {
        std::string k = str();
        expect(':');
        j.obj.emplace_back(std::move(k), value());
        char d = peek();
        if (d == ',') { ++p; continue; }
        if (d == '}') { ++p; break; }
        fail("object");
      }
      return j;
    }
    fail("value");
  }
};

const Json* find(const Json& o, const char* k) {
  for (const auto& kv : o.obj)
    if (kv.first == k) return &kv.second;
  return nullptr;
}

// ponytail: sd::DType has no F16/I32/I64/U8/BOOL/F8 — file dtypes map to a
// same-width sd::DType (I64 off); views carry raw bytes + true nbytes from the
// file width. Extend DType when real compute on those dtypes is needed.
bool dtype_info(const std::string& s, DType& dt, size_t& es) {
  if (s == "F32") { dt = DType::F32; es = 4; }
  else if (s == "I32") { dt = DType::F32; es = 4; }
  else if (s == "I64") { dt = DType::F32; es = 8; }
  else if (s == "BF16" || s == "F16") { dt = DType::BF16; es = 2; }
  else if (s == "I8" || s == "U8" || s == "BOOL" || s == "F8_E4M3" || s == "F8_E5M2") {
    dt = DType::I8; es = 1;
  } else return false;
  return true;
}

struct Entry {
  int64_t begin, end;
  std::string name;
  std::vector<int64_t> shape;
  DType dt;
  size_t bytes;
};

void write_all(int fd, const void* p, size_t n) {
  const char* q = (const char*)p;
  while (n) {
    ssize_t w = write(fd, q, n);
    if (w <= 0) throw std::runtime_error("safetensors: write failed");
    q += w;
    n -= (size_t)w;
  }
}

std::string escape_name(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '"' || c == '\\') { r += '\\'; r += c; }
    else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); r += b; }
    else r += c;
  }
  return r;
}

}  // namespace

SafetensorsFile load_safetensors(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 10) {
    close(fd);
    throw std::runtime_error("safetensors: truncated file " + path);
  }
  size_t size = (size_t)st.st_size;
  void* base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed " + path);

  SafetensorsFile out;
  out.mapping = std::shared_ptr<void>(base, [size](void* b) { munmap(b, size); });
  out.path = path;

  const auto* b = (const uint8_t*)base;
  uint64_t n = 0;
  for (int i = 0; i < 8; i++) n |= (uint64_t)b[i] << (8 * i);
  if (n < 2 || n > 100000000ull || 8 + n > size)
    throw std::runtime_error("safetensors: bad header length in " + path);

  Parser ps{(const char*)b + 8, (const char*)b + 8 + n};
  Json root = ps.value();
  ps.ws();
  if (ps.p != ps.end) ps.fail("trailing bytes");
  if (root.type != Json::OBJ) ps.fail("root not an object");

  const uint64_t data_len = size - 8 - n;
  std::vector<Entry> es;
  for (const auto& kv : root.obj) {
    if (kv.first == "__metadata__") continue;
    const Json& o = kv.second;
    if (o.type != Json::OBJ) throw std::runtime_error("safetensors: bad tensor entry " + kv.first);
    const Json* dts = find(o, "dtype");
    const Json* sh = find(o, "shape");
    const Json* off = find(o, "data_offsets");
    if (!dts || dts->type != Json::STR || !sh || sh->type != Json::ARR || !off ||
        off->type != Json::ARR || off->arr.size() != 2)
      throw std::runtime_error("safetensors: bad tensor fields for " + kv.first);
    DType d;
    size_t elsz;
    if (!dtype_info(dts->str, d, elsz))
      throw std::runtime_error("safetensors: unsupported dtype " + dts->str);
    int64_t numel = 1;
    for (int64_t dim : sh->arr) {
      if (dim < 0) throw std::runtime_error("safetensors: negative dim for " + kv.first);
      if (dim && numel > (int64_t{1} << 60) / dim)
        throw std::runtime_error("safetensors: shape overflow for " + kv.first);
      numel *= dim;
    }
    if (off->arr[0] < 0 || off->arr[1] < off->arr[0])
      throw std::runtime_error("safetensors: bad offsets for " + kv.first);
    if ((uint64_t)off->arr[1] > data_len)
      throw std::runtime_error("safetensors: offsets out of bounds for " + kv.first);
    if ((uint64_t)(off->arr[1] - off->arr[0]) != (uint64_t)numel * elsz)
      throw std::runtime_error("safetensors: bytes != shape*dtype for " + kv.first);
    es.push_back(Entry{off->arr[0], off->arr[1], kv.first, sh->arr, d,
                       (size_t)(off->arr[1] - off->arr[0])});
  }
  std::sort(es.begin(), es.end(),
            [](const Entry& a, const Entry& c) { return a.begin < c.begin; });
  int64_t cur = 0;
  for (const Entry& e : es) {
    if (e.begin != cur)
      throw std::runtime_error("safetensors: data_offsets not contiguous at " + e.name);
    cur = e.end;
  }
  char* data_base = (char*)base + 8 + n;
  for (const Entry& e : es) {
    Tensor t;  // becomes a non-owning view into the mapping
    t.shape = e.shape;
    t.dtype = e.dt;
    t.data = data_base + e.begin;
    t.nbytes = e.bytes;
    t.owning = false;
    out.tensors.emplace(e.name, std::move(t));
  }
  return out;
}

void write_safetensors(const std::string& path,
                       const std::unordered_map<std::string, Tensor>& ts) {
  std::vector<std::pair<std::string, const Tensor*>> order;
  for (const auto& kv : ts) order.emplace_back(kv.first, &kv.second);

  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : order) {
    const char* dts = t->dtype == DType::F32 ? "F32" : t->dtype == DType::BF16 ? "BF16" : "I8";
    if (!first) hdr += ",";
    first = false;
    hdr += "\"" + escape_name(name) + "\":{\"dtype\":\"" + dts + "\",\"shape\":[";
    for (size_t i = 0; i < t->shape.size(); i++)
      hdr += (i ? "," : "") + std::to_string(t->shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + t->nbytes) + "]}";
    off += t->nbytes;
  }
  hdr += "}";

  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) throw std::runtime_error("safetensors: cannot write " + path);
  struct Guard { int f; ~Guard() { close(f); } } g{fd};
  uint64_t n = hdr.size();
  char len8[8];
  for (int i = 0; i < 8; i++) len8[i] = (char)(n >> (8 * i));
  write_all(fd, len8, 8);
  write_all(fd, hdr.data(), hdr.size());
  for (const auto& [name, t] : order) {
    (void)name;
    write_all(fd, t->data, t->nbytes);
  }
}

}  // namespace sd

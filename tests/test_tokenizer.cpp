// Golden test: C++ UMT5 sentencepiece tokenizer vs HF transformers output.
#include <cstdio>
#include <string>
#include <vector>

#include "test_support/npz.h"
#include "tokenizer/spm.h"

namespace {

// Minimal JSON array-of-strings decoder (json.dumps output: \", \\, \uXXXX).
std::vector<std::string> ParseJsonStrings(const std::string& b) {
  std::vector<std::string> out;
  size_t p = b.find('[');
  while (true) {
    p = b.find('"', p);
    if (p == std::string::npos) break;
    ++p;
    std::string s;
    while (b[p] != '"') {
      if (b[p] == '\\') {
        if (b[p + 1] == 'u') {
          unsigned v = (unsigned)strtoul(b.substr(p + 2, 4).c_str(), nullptr, 16);
          if (v < 0x80) s += (char)v;
          else if (v < 0x800) { s += (char)(0xc0 | v >> 6); s += (char)(0x80 | (v & 63)); }
          else { s += (char)(0xe0 | v >> 12); s += (char)(0x80 | ((v >> 6) & 63)); s += (char)(0x80 | (v & 63)); }
          p += 6;
        } else {
          s += b[p + 1];
          p += 2;
        }
      } else {
        s += b[p++];
      }
    }
    out.push_back(s);
    ++p;
    size_t comma = b.find_first_not_of(" \t\r\n", p);
    if (comma == std::string::npos || b[comma] == ']') break;
    p = comma;
  }
  return out;
}

}  // namespace

int main() {
  spm::SpmModel m;
  if (!m.load("ckpts/umt5_tokenizer.model")) {
    fprintf(stderr, "failed to load ckpts/umt5_tokenizer.model\n");
    return 1;
  }
  printf("vocab=%d unk=%d bos=%d eos=%d pad=%d byte_fallback=%d\n", m.vocab_size(),
         m.unk_id(), m.bos_id(), m.eos_id(), m.pad_id(), (int)m.byte_fallback());

  std::unordered_map<std::string, sd::Tensor> g;
  try { g = sd::load_npz("tools/golden/tokenizer_golden.npz"); }
  catch (const std::exception& e) {
    fprintf(stderr, "load_npz: %s (run tools/golden/gen_tokenizer.py)\n", e.what());
    return 1;
  }
  // i64 goldens load as F32; token ids < 2^24 so conversion is exact.
  const float* flat = g.at("ids_flat").ptr<float>();
  const float* off = g.at("offsets").ptr<float>();
  auto prompts = ParseJsonStrings(
      std::string(g.at("prompts_json").ptr<int8_t>(),
                  g.at("prompts_json").ptr<int8_t>() + g.at("prompts_json").numel()));
  if (prompts.size() + 1 != (size_t)g.at("offsets").numel()) {
    fprintf(stderr, "prompt/offset mismatch\n");
    return 1;
  }

  int pass = 0, fail = 0;
  for (size_t i = 0; i < prompts.size(); ++i) {
    std::vector<int> got = m.encode(prompts[i]);
    std::vector<int> want(flat + (size_t)off[i], flat + (size_t)off[i + 1]);
    if (got == want) {
      ++pass;
      printf("PASS [%2zu] %3zu ids  %.48s\n", i, want.size(), prompts[i].c_str());
    } else {
      ++fail;
      size_t d = 0;
      while (d < got.size() && d < want.size() && got[d] == want[d]) ++d;
      printf("FAIL [%2zu] first divergence at %zu: got %d want %d  %.48s\n", i, d,
             d < got.size() ? got[d] : -1, d < want.size() ? (int)want[d] : -1,
             prompts[i].c_str());
      printf("   got : ");
      for (int x : got) printf("%d ", x);
      printf("\n   want: ");
      for (int x : want) printf("%d ", x);
      printf("\n");
    }
  }
  printf("== %d/%zu passed\n", pass, prompts.size());
  return fail ? 1 : 0;
}

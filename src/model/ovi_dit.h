// Ovi dual-backbone (video+audio) fusion DiT — full forward, structure port of
// Ovi/ovi/modules/fusion.py (FusionModel) + model.py (WanModel). fp32 compute;
// BF16 checkpoint views are converted at load. Quantized .sdcpp packs: #15.
#pragma once
#include "model/blocks.h"
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ovi {

// Dims from Ovi/ovi/configs/model/dit/{video,audio}.json.
struct BackboneCfg {
  int dim = 3072, ffn_dim = 14336, num_heads = 24, num_layers = 30;
  int freq_dim = 256, text_dim = 4096, text_len = 512;
  int in_dim = 48, out_dim = 48;
  int pt = 1, ph = 2, pw = 2;  // video patch (1,2,2); audio is (1,1,1) no patch
  float rope_scaling = 1.0f;   // audio.json temporal_rope_scaling_factor
};

struct OviDitCfg {
  BackboneCfg video, audio;
  // Block indices whose cross-attn carries the fusion k/v injection. Real
  // model: all 30 (fusion.py:286-302 loops every block). Test cfg uses a
  // subset to exercise both paths.
  std::vector<int> fusion_layers;
  int slg_layer = -1;  // skip-layer guidance: drop block i==slg_layer>0 (fusion.py:290)
  bool quantized = false;  // TODO(#15): route Linears through sd::qgemm

  static OviDitCfg real();
  static OviDitCfg test();  // scaled-down golden config
};

struct SelfAttnW {
  blk::Linear q, k, v, o;
  Tensor nq_w, nk_w;  // RMSNorm weights [dim] (model.py:220-221)
};

// Fusion k/v injected into a block's cross_attn (fusion.py:40-52).
struct FusionKV {
  blk::Linear k_fusion, v_fusion;
  Tensor norm_k_fusion_w;            // RMSNorm [dim]
  Tensor pnf_w, pnf_b;               // pre_attn_norm_fusion affine LN [dim]
};

struct CrossAttnW {
  blk::Linear q, k, v, o;
  Tensor nq_w, nk_w;
  FusionKV fusion;  // consumed only on fusion_layers
};

struct BlockW {
  SelfAttnW sa;
  CrossAttnW ca;
  blk::FFN ffn;
  Tensor ln3_w, ln3_b;  // affine LN, cross_attn_norm=true (model.py:401-403)
  Tensor mod_bias;      // [6,dim] ModulationAdd bias (model.py:368-374)
};

struct BackboneW {
  // video: patch_embedding Conv3d flattened+transposed to [in*pt*ph*pw, dim]
  Tensor pe_mat;
  // audio: patch_embedding.0 Conv1d weight [dim,in,7] (view, torch layout)
  Tensor pe0;
  Tensor pe_b;  // conv bias [dim] (video Conv3d / audio Conv1d)
  // audio ConvMLP conv1d weights: w1/w3 [hidden,dim,7], w2 [dim,hidden,7]
  Tensor ac_w1, ac_w2, ac_w3;
  blk::Linear te0, te2;   // text_embedding.0/.2 (model.py:627-629)
  blk::Linear tm0, tm2;   // time_embedding.0/.2 (model.py:631-632)
  blk::Linear tp1;        // time_projection.1 (model.py:633)
  blk::Linear head;       // head.head (model.py:474-501)
  Tensor head_mod;        // [2,dim] head.modulation
  std::vector<BlockW> blocks;
};

struct OviDit {
  OviDitCfg cfg;
  BackboneW video, audio;
  // #17: true after load_quantized_pack — Linear weights may be I8 views
  // (torch [out,in] + per-row scale packed after the i8 blob) and forward
  // routes them through sd::qgemm; fp32 tensors keep the matmul path.
  bool use_int8 = false;

  void init(const OviDitCfg& c);
  // Weight VIEWS from a safetensors-style name->Tensor map (torch layout
  // [out,in] Linears; bf16 converted). Throws on missing/mismatched tensor.
  void load(const std::unordered_map<std::string, Tensor>& tensors);
  // Every checkpoint key load() consumes — for coverage checks.
  std::vector<std::string> expected_keys() const;
  // .sdcpp pack: I8+scales for big Linears, raw F32 for the rest (issue #15).
  void load_quantized_pack(const std::string& path);

  // noise_video [C,F,H,W], noise_audio [L,C]; t scalar timestep (timestep
  // SHIFT lives in the sampler — fm_solvers_unipc.py:114 — not here);
  // vid_ctx/aud_ctx [Lc,text_dim] raw text embeddings, zero-padded to
  // text_len (model.py:784-789 — padded rows ARE attended). CFG/ctx_neg is
  // two forward calls, the sampler's job.
  void forward(const Tensor& noise_video, const Tensor& noise_audio, float t,
               const Tensor& vid_ctx, const Tensor& aud_ctx, Tensor& out_video,
               Tensor& out_audio) const;

 private:
  std::deque<Tensor> pool_;  // f32 conversions made at load (stable refs)
  std::shared_ptr<void> pack_owner_;  // sdcpp mmap keepalive for I8 views
  void load_impl(const std::unordered_map<std::string, Tensor>& m);
};

// #17 bench hooks: rough wall-time split of forward() into sdpa (attention)
// vs gemm (linear) sections. section_timers(true) resets the accumulators.
void section_timers(bool on);
double section_ms_attn();
double section_ms_lin();

}  // namespace ovi
}  // namespace sd

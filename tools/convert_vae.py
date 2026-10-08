# Convert Wan2.2-TI2V-5B VAE .pth -> ckpts/vae_dec.safetensors (bf16).
#
# The TI2V-5B VAE is vae2_2.py (NOT vae.py — that's the Wan2.1 16ch one):
# Wan2_2_VAE -> WanVAE_(dim=160 enc / dec_dim=256, z_dim=48) with
# temperal_downsample=[False,True,True]. Decoder path only: conv2 + decoder.
#
# Keys keep torch state_dict names under "dec." / "conv2." prefixes. Conv
# weights stay in torch layout [Cout,Cin,kt,kh,kw]; the C++ side permutes to
# matmul-ready [Cin*kt*kh*kw, Cout] at load.
#
# RAM: torch.load peak ~3 GB (fp32 pth, meta-device assign).
import importlib.util
import os
import sys
import torch

SRC = "ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DST = "ckpts/vae_dec.safetensors"

spec = importlib.util.spec_from_file_location("vae2_2", "Ovi/ovi/modules/vae2_2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod._video_vae(
    pretrained_path=SRC, z_dim=48, dim=160,
    dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True]).eval()

dec = model.decoder
print(f"z_dim={model.z_dim} dec_dim={dec.dim} dim_mult={dec.dim_mult} "
      f"num_res_blocks={dec.num_res_blocks} "
      f"temperal_upsample={dec.temperal_upsample}")

# arch facts straight from the loaded module
conv3d = [m for m in dec.modules() if isinstance(m, mod.CausalConv3d)]
assert all(m.groups == 1 for m in conv3d), "groups>1 conv found"
for name in ["conv1", "head.2", "upsamples.0.upsamples.3.time_conv",
             "upsamples.2.upsamples.3.resample.1"]:
    m = dec.get_submodule(name)
    print(f"  {name}: {tuple(m.weight.shape)}")
print(f"decoder: {len(conv3d)} CausalConv3d, "
      f"{sum(isinstance(m, mod.ResidualBlock) for m in dec.modules())} resblocks, "
      f"{sum(isinstance(m, mod.AttentionBlock) for m in dec.modules())} attn blocks")
print(f"pth dtype: {dec.conv1.weight.dtype}")

out = {}
for k, v in model.conv2.state_dict().items():
    out[f"conv2.{k}"] = v.to(torch.bfloat16).contiguous()
for k, v in dec.state_dict().items():
    out[f"dec.{k}"] = v.to(torch.bfloat16).contiguous()

from safetensors.torch import save_file
save_file(out, DST)
n = sum(v.numel() for v in out.values())
print(f"wrote {DST}: {len(out)} tensors, {n/1e6:.1f}M params, "
      f"{os.path.getsize(DST)/1e6:.1f} MB")

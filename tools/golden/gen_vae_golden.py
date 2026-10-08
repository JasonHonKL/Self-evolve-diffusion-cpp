# Golden for the Wan2.2-TI2V-5B VAE decoder (issue #12).
# Runs the reference torch decoder (Ovi/ovi/modules/vae2_2.py) on a seeded
# random latent [48,2,8,8] and saves f32:
#   latent [48,2,8,8]  (bf16-rounded values as f32: exactly what C++ reads)
#   out    [3,5,128,128]
# Compute dtype: weights are rounded to bf16 (matching the C++ safetensors)
# but the math runs in fp32 — torch's CPU bf16 conv3d is pathologically
# slow, and bf16-weights/fp32-accum is exactly what the C++ side does, so
# this is the apples-to-apples golden (residual diff = accum order only).
# Compression: 16x spatial, temporal 4F-3 frames (anchor frame -> 1 frame,
# each later latent frame -> 4). np.savez = STORED entries (npz.h contract).
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("vae2_2", "Ovi/ovi/modules/vae2_2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

v = mod.Wan2_2_VAE(z_dim=48, c_dim=160,
                   vae_pth="ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                   dtype=torch.float32, device="cpu")
v.model.to(torch.bfloat16).to(torch.float32)  # bf16-rounded weights, fp32 math

torch.manual_seed(1234)
z = torch.randn(1, 48, 2, 8, 8).bfloat16().float()
with torch.no_grad():
    out = v.model.decode(z, v.scale).float()

lat = z.float().numpy()[0]
out = out.numpy()[0]
print("latent", lat.shape, "out", out.shape,
      f"out range [{out.min():.3f},{out.max():.3f}] mean {out.mean():.3f}")
np.savez("tools/golden/vae_golden.npz", latent=lat, out=out)
print("wrote tools/golden/vae_golden.npz")

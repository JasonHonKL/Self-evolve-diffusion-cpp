#!/usr/bin/env python3
# Golden parity data for the C++ vocoder port (issue #13).
# Torch fp32 CPU forward of the exact Ovi path:
#   audio latent [1,20,T@31.25Hz] -> MMAudio VAE decoder -> mel [1,80,2T@62.5Hz]
#     -> BigVGAN best_netG -> waveform [1,1,512T @16kHz]
# (Ovi itself runs this in bf16 autocast on CUDA; fp32 golden is the tighter
# parity target for the fp32 C++ port.)
# Usage: python3 tools/golden/gen_voc_golden.py   (after convert_vocoder.py)
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "convert_vocoder", Path(__file__).resolve().parent.parent / "convert_vocoder.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)
import_bigvgan, import_vae = cv.import_bigvgan, cv.import_vae
from argparse import Namespace

ROOT = Path(__file__).resolve().parent.parent.parent
EXT = ROOT / "ckpts/MMAudio/ext_weights"


def build():
    vae = import_vae().VAE_16k()
    vae.load_state_dict(torch.load(EXT / "v1-16.pth", map_location="cpu", weights_only=True), strict=True)
    vae.remove_weight_norm()
    vae.eval()

    BigVGANVocoder = import_bigvgan()
    h = Namespace(resblock="1", num_mels=80, upsample_rates=[4, 4, 2, 2, 2, 2],
                  upsample_kernel_sizes=[8, 8, 4, 4, 4, 4], upsample_initial_channel=1536,
                  resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3,
                  activation="snakebeta", snake_logscale=True)
    net = BigVGANVocoder(h)
    net.load_state_dict(torch.load(EXT / "best_netG.pt", map_location="cpu", weights_only=True)["generator"],
                        strict=True)
    net.remove_weight_norm()
    net.eval()
    return vae, net


def main():
    vae, net = build()
    torch.manual_seed(1234)
    z = torch.randn(1, 20, 64)  # T=64 latents = ~2.05s of audio

    snaps = {}

    def snap(name, clamp=False):
        def hook(mod, inp, out):
            v = out[0] if isinstance(out, tuple) else out
            v = v.detach().float()
            if clamp:
                v = v.clamp(-256, 256)  # C++ traces post-clamp
            snaps[name] = v.numpy()[0].astype(np.float32)
        return hook

    d = vae.decoder
    d.conv_in.register_forward_hook(snap("dec.conv_in"))
    d.mid.block_1.register_forward_hook(snap("dec.mid.b1"))
    d.mid.attn_1.register_forward_hook(snap("dec.mid.attn"))
    d.mid.block_2.register_forward_hook(snap("dec.mid.b2", clamp=True))
    # up containers are plain nn.Module (no forward): hook last block of each level
    for lvl in (2, 1, 0):
        d.up[lvl].block[2].register_forward_hook(snap(f"dec.up{lvl}", clamp=True))
    d.up[1].upsample.register_forward_hook(snap("dec.up1.up"))
    d.register_forward_hook(snap("dec.dec_out"))
    net.conv_pre.register_forward_hook(snap("g.conv_pre"))
    for i, up in enumerate(net.ups):
        up[0].register_forward_hook(snap(f"g.upT{i}"))  # raw convT out (pre-resblocks)
    for i in range(6):
        net.resblocks[i * 3].register_forward_hook(snap(f"g.rb{i}_0"))
    net.activation_post.register_forward_hook(snap("g.act_post"))
    net.conv_post.register_forward_hook(snap("g.conv_post"))

    with torch.inference_mode():
        mel = vae.decode(z)          # [1, 80, 128] unnormalized log10-mel
        wav = net(mel)               # [1, 1, 32768]
    print("z", tuple(z.shape), "mel", tuple(mel.shape), "wav", tuple(wav.shape),
          "range", wav.min().item(), wav.max().item())

    out = ROOT / "tests/golden/voc_golden.npz"
    np.savez(out, z=z.numpy()[0].astype(np.float32),
             mel=mel.numpy()[0].astype(np.float32),
             wav=wav.numpy()[0, 0].astype(np.float32), **snaps)
    print("wrote", out, f"({len(snaps)} stage snapshots)")


if __name__ == "__main__":
    main()

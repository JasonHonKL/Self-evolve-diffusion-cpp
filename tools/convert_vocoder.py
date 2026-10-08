#!/usr/bin/env python3
# Convert MMAudio audio-latent decoder (v1-16.pth VAE decoder) + BigVGAN vocoder
# (best_netG.pt generator) to a single safetensors for the C++ port (issue #13).
#
# Weight folding done here so C++ sees only plain conv weights:
#   - weight_norm (g,v) -> folded W for BigVGAN convs/convT (via remove_weight_norm)
#   - MPConv1D magnitude-preserving normalization -> folded W for VAE decoder
#   - SnakeBeta alpha/beta exported pre-exp()'d (alpha_logscale=True at runtime)
#   - kaiser sinc anti-alias filters exported as flat [12] vectors
#
# Usage: python3 tools/convert_vocoder.py
#   reads  ckpts/MMAudio/ext_weights/{v1-16.pth,best_netG.pt}
#   writes ckpts/mmaudio_vocoder.safetensors
import sys, types, importlib.util
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
OVI = ROOT / "Ovi"
BG = OVI / "ovi/modules/mmaudio/ext/bigvgan"


def load_pkg(name, path, pkg=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if pkg is not None:
        m.__path__ = [str(pkg)]
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def import_bigvgan():
    # Bypass ovi/modules/__init__.py (pulls diffusers): register a synthetic
    # 'bigvgan' package rooted at the vendored ext/bigvgan directory.
    pkg = types.ModuleType("bigvgan")
    pkg.__path__ = [str(BG)]
    sys.modules["bigvgan"] = pkg
    for fn in ("activations", "utils", "models"):
        m = load_pkg(f"bigvgan.{fn}", BG / f"{fn}.py")
        setattr(pkg, fn, m)
    af = types.ModuleType("bigvgan.alias_free_torch")
    af.__path__ = [str(BG / "alias_free_torch")]
    sys.modules["bigvgan.alias_free_torch"] = af
    for fn in ("filter", "resample", "act"):
        m = load_pkg(f"bigvgan.alias_free_torch.{fn}", BG / "alias_free_torch" / f"{fn}.py")
        setattr(af, fn, m)
    load_pkg("bigvgan.alias_free_torch", BG / "alias_free_torch" / "__init__.py", BG / "alias_free_torch")
    return sys.modules["bigvgan.models"].BigVGANVocoder


def import_vae():
    # Synthetic 'aem' package rooted at ext/autoencoder for the same reason.
    AE = OVI / "ovi/modules/mmaudio/ext/autoencoder"
    pkg = types.ModuleType("aem")
    pkg.__path__ = [str(AE)]
    sys.modules["aem"] = pkg
    for fn in ("edm2_utils", "vae_modules", "distributions", "vae"):
        m = load_pkg(f"aem.{fn}", AE / f"{fn}.py")
        setattr(pkg, fn, m)
    return sys.modules["aem.vae"]


def main():
    ext = ROOT / "ckpts/MMAudio/ext_weights"
    out = {}

    # ---- VAE decoder (v1-16): latent [20ch @31.25Hz] -> mel [80ch @62.5Hz] ----
    vae_mod = import_vae()
    vae = vae_mod.VAE_16k()
    sd = torch.load(ext / "v1-16.pth", map_location="cpu", weights_only=True)
    vae.load_state_dict(sd, strict=True)
    vae.remove_weight_norm()
    dec = vae.state_dict()
    for k, v in dec.items():
        if k.startswith("decoder."):
            out["dec." + k[len("decoder."):]] = v.float().contiguous()
    out["dec_gain"] = dec["decoder.learnable_gain"].float().clone()  # conv_out gain = lg + 1
    out["data_mean"] = vae.data_mean.reshape(80).float().contiguous()
    out["data_std"] = vae.data_std.reshape(80).float().contiguous()
    ndec = sum(v.numel() for v in out.values())
    print(f"VAE decoder: {ndec/1e6:.1f}M params")

    # ---- BigVGAN generator: mel [80 @62.5Hz] -> waveform [1ch @16kHz] ----
    BigVGANVocoder = import_bigvgan()
    # values from Ovi/ovi/modules/mmaudio/ext/bigvgan/bigvgan_vocoder.yml
    h = Namespace(resblock="1", num_mels=80, upsample_rates=[4, 4, 2, 2, 2, 2],
                  upsample_kernel_sizes=[8, 8, 4, 4, 4, 4], upsample_initial_channel=1536,
                  resblock_kernel_sizes=[3, 7, 11],
                  resblock_dilation_sizes=[[1, 3, 5]] * 3,
                  activation="snakebeta", snake_logscale=True)
    net = BigVGANVocoder(h)
    gen = torch.load(ext / "best_netG.pt", map_location="cpu", weights_only=True)["generator"]
    net.load_state_dict(gen, strict=True)
    net.remove_weight_norm()
    for k, v in net.state_dict().items():
        if k.endswith(".act.alpha") or k.endswith(".act.beta"):
            out["g." + k + "_exp"] = torch.exp(v.float()).contiguous()  # logscale -> exp at runtime
        elif k.endswith(".filter"):
            out["g." + k] = v.reshape(-1).float().contiguous()
        else:
            out["g." + k] = v.float().contiguous()
    print(f"BigVGAN gen: {sum(v.numel() for k, v in out.items() if k.startswith('g.'))/1e6:.1f}M params")

    dst = ROOT / "ckpts/mmaudio_vocoder.safetensors"
    save_file(out, str(dst))
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB, {len(out)} tensors)")


if __name__ == "__main__":
    main()

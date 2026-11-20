"""Export a tiny autoencoder's decoder (diffusers AutoencoderTiny) to a QNN context binary for sd-cli --qnn-tae.

Works for the decoder built into SDXS (its vae/ folder) and for TAESD (madebyollin/taesd). The graph has the
names the runtime expects:
  latent [1, 4, H/8, W/8] -> image [1, 3, H, W], in [0, 1] before sd.cpp clamps it

This is the same graph as sd.cpp's TinyDecoder (and script/coreml/convert_tae_decoder.py): the tanh clamp, then
the decoder layers, without the x * 2 - 1 that diffusers' DecoderTiny.forward ends with. It is only
convolutions, ReLU, nearest upsampling and one tanh, all of which run on the Hexagon NPU in fp16.

The binary is compiled ahead of time for one Snapdragon SoC (--soc, default sm8750: Snapdragon 8 Elite, as in
the Galaxy S25 Ultra) and checked against PyTorch fp32 on the x86 HTP simulator, which computes what the NPU
computes, but slowly (12 minutes at 512x512 on a 2-core machine; --no-check skips it).

Example (SDXS-512-DreamShaper):
  python convert_tae_decoder.py --vae sdxs-512-dreamshaper/vae --out sdxs_tae_decoder.bin

Needs QNN_SDK_ROOT (QAIRT SDK on x86_64 Linux, Python 3.10) and
pip install torch==2.5.1 diffusers==0.31.0 onnx==1.16.1 numpy==1.26.4
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderTiny

import qnn_tools


class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.layers = vae.decoder.layers

    def forward(self, latent):
        # DecoderTiny.forward without its final x * 2 - 1, so the image stays in [0, 1].
        return self.layers(torch.tanh(latent / 3) * 3)


def parse_size(text):
    width, height = (int(v) for v in text.lower().split("x"))
    if width % 8 or height % 8:
        raise argparse.ArgumentTypeError(f"{text}: width and height must be multiples of 8")
    return width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", required=True, help="diffusers AutoencoderTiny folder (SDXS's vae/ or madebyollin/taesd)")
    parser.add_argument("--out", required=True, help="output context binary (.bin)")
    parser.add_argument("--sizes", default="512x512",
                        help="comma-separated image sizes as WIDTHxHEIGHT; each becomes a graph of the binary")
    parser.add_argument("--soc", choices=sorted(qnn_tools.SOCS), default=qnn_tools.DEFAULT_SOC,
                        help="Snapdragon SoC to compile for: " +
                             "; ".join(f"{k} = {v[2]}" for k, v in qnn_tools.SOCS.items()))
    parser.add_argument("--share-weights", action="store_true",
                        help="with several --sizes, keep one copy of the weights (cannot be checked on the simulator)")
    parser.add_argument("--no-check", action="store_true", help="skip the check on the x86 HTP simulator")
    args = parser.parse_args()

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    vae = AutoencoderTiny.from_pretrained(args.vae, torch_dtype=torch.float32).eval()
    if vae.config.latent_channels != 4:
        raise SystemExit(f"error: expected 4 latent channels, got {vae.config.latent_channels}")
    wrapper = DecoderWrapper(vae).eval()

    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dlcs, checks = [], []
        for width, height in sizes:
            graph = f"tae_{width}x{height}"
            torch.manual_seed(0)
            latent = torch.randn(1, 4, height // 8, width // 8)
            with torch.no_grad():
                reference = wrapper(latent).numpy()
            onnx_path = tmp / f"{graph}.onnx"
            t0 = time.time()
            torch.onnx.export(wrapper, (latent,), str(onnx_path), input_names=["latent"], output_names=["image"],
                              opset_version=17)
            dlcs.append(qnn_tools.onnx_to_dlc(onnx_path, tmp / f"{graph}.dlc", "fp16"))
            print(f"{width}x{height}: converted in {time.time() - t0:.1f} s")
            checks.append((graph, latent.numpy(), reference))

        t0 = time.time()
        qnn_tools.dlcs_to_context_binary(dlcs, out, args.soc, args.share_weights)
        print(f"compiled {out} for {args.soc} ({qnn_tools.SOCS[args.soc][2]}) in {time.time() - t0:.1f} s, "
              f"{out.stat().st_size / 1e6:.1f} MB")

    if args.no_check or (args.share_weights and len(sizes) > 1):
        return
    for graph, latent, reference in checks:
        t0 = time.time()
        image = qnn_tools.run_on_host(out, {"latent": latent}, graph=graph)["image"]
        ref = np.clip(reference, 0, 1)
        print(f"{graph} on the HTP simulator ({time.time() - t0:.0f} s): max abs err "
              f"{np.abs(image - reference).max():.4f}, PSNR vs PyTorch fp32 {qnn_tools.psnr(np.clip(image, 0, 1), ref):.1f} dB")


if __name__ == "__main__":
    main()

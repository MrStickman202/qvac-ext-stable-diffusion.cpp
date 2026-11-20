"""Builds the tiny QNN context binaries that tests/qnn/test_qnn.cpp runs.

They have the I/O of the real sidecars but a few ops each, with outputs test_qnn.cpp computes exactly
(the same functions as tests/coreml/make_fixtures.py):
  unet:  noise_pred[b] = 0.5 * sample[b] + sum_{t,d} ehs[b, t, d] * W[t, d] + 0.001 * timestep[b]
         with W[t, d] = 0.1 * (t + 1) + 0.01 * (d + 1), so a wrong context layout changes the result
  tae:   image = nearest 8x upsampling of latent channels 0..2

They are compiled for the Snapdragon 8 Elite's HTP (v79) unless --soc says otherwise. test-qnn runs them on the
x86 HTP simulator that ships with the QAIRT SDK, so the tests need no phone; the same binaries load on a Galaxy S25.

  python tests/qnn/make_fixtures.py OUT_DIR [--soc sm8650]

Needs QNN_SDK_ROOT (QAIRT SDK, x86_64 Linux, Python 3.10) and pip install torch onnx numpy.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "script" / "qnn"))
import qnn_tools  # noqa: E402

C, H, W = 4, 8, 8  # latent
T, D = 5, 6  # text tokens, embedding width; T != D so a transposed context gives a different result


class FixtureUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = 0.1 * (torch.arange(T)[:, None] + 1) + 0.01 * (torch.arange(D)[None, :] + 1)
        self.register_buffer("w", w.float())

    def forward(self, sample, timestep, encoder_hidden_states):
        c = (encoder_hidden_states * self.w).sum(dim=(1, 2))
        bias = (c + 0.001 * timestep).reshape(-1, 1, 1, 1)
        return 0.5 * sample + bias




class FixtureTAE(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, latent):
        return torch.nn.functional.interpolate(latent[:, :3], scale_factor=self.scale, mode="nearest")


def export_unet(path, batch, h=H, w=W, names=("sample", "timestep", "encoder_hidden_states")):
    args = (torch.randn(batch, C, h, w), torch.full((batch,), 500.0), torch.randn(batch, T, D))
    torch.onnx.export(FixtureUNet().eval(), args, str(path), input_names=list(names), output_names=["noise_pred"],
                      opset_version=17)


def export_tae(path, scale):
    torch.onnx.export(FixtureTAE(scale).eval(), (torch.rand(1, C, H, W),), str(path), input_names=["latent"],
                      output_names=["image"], opset_version=17)


def unet_calibration(n=8):
    rng = np.random.default_rng(0)
    return [{
        "sample": rng.uniform(-1, 1, (1, C, H, W)).astype(np.float32),
        "timestep": np.array([rng.uniform(0, 1000)], np.float32),
        "encoder_hidden_states": rng.uniform(-0.5, 0.5, (1, T, D)).astype(np.float32),
    } for _ in range(n)]


def build(out, tmp, name, graphs, soc):
    """graphs: list of (graph name, ONNX exporter, precision, preserve_io)."""
    dlcs = []
    for graph, export, precision, preserve_io in graphs:
        onnx_path = tmp / f"{graph}.onnx"
        export(onnx_path)
        dlc = tmp / f"{graph}.dlc"
        calibration = unet_calibration() if precision == "w8a16" else None
        qnn_tools.onnx_to_dlc(onnx_path, dlc, precision, calibration, preserve_io)
        dlcs.append(dlc)
    path = qnn_tools.dlcs_to_context_binary(dlcs, out / f"{name}.bin", soc=soc)
    print(f"wrote {path}: {', '.join(f'{g}' for g in qnn_tools.describe(path))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output folder")
    parser.add_argument("--soc", choices=sorted(qnn_tools.SOCS), default=qnn_tools.DEFAULT_SOC)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    soc = args.soc
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build(out, tmp, "unet_b1", [("unet_64x64_b1", lambda p: export_unet(p, 1), "fp16", False)], soc)
        build(out, tmp, "unet_b1_f32io", [("unet_64x64_b1", lambda p: export_unet(p, 1), "fp16", True)], soc)
        build(out, tmp, "unet_b1_w8a16", [("unet_64x64_b1", lambda p: export_unet(p, 1), "w8a16", False)], soc)
        build(out, tmp, "unet_b2_vpred", [("unet_64x64_b2_vpred", lambda p: export_unet(p, 2), "fp16", False)], soc)
        # Two sizes in one binary: 64x64 and 128x96 images (8x8 and 12x16 latents). Without weight sharing,
        # which the x86 HTP simulator cannot load.
        build(out, tmp, "unet_multi", [
            ("unet_64x64_b1_eps", lambda p: export_unet(p, 1), "fp16", False),
            ("unet_128x96_b1_eps", lambda p: export_unet(p, 1, h=12, w=16), "fp16", False),
        ], soc)
        build(out, tmp, "unet_bad_names", [
            ("unet_64x64_b1", lambda p: export_unet(p, 1, names=("x", "t", "context")), "fp16", False)], soc)
        build(out, tmp, "tae", [("tae_64x64", lambda p: export_tae(p, 8), "fp16", False)], soc)
        # [1, 4, 8, 8] -> [1, 3, 32, 32]: not a tiny decoder's 8x.
        build(out, tmp, "tae_wrong_scale", [("tae_32x32", lambda p: export_tae(p, 4), "fp16", False)], soc)
    (out / "not_a_binary.bin").write_bytes(b"this is not a QNN context binary\n" * 64)
    print(f"wrote {out / 'not_a_binary.bin'}")


if __name__ == "__main__":
    main()

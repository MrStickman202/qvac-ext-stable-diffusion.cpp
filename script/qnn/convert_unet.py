"""Export a diffusers SD 1.x/2.x UNet to a QNN context binary for sd-cli --qnn-unet (Qualcomm Hexagon NPU).

The graph has the names the runtime expects:
  sample [N, 4, H/8, W/8], timestep [N], encoder_hidden_states [N, T, D] -> noise_pred [N, 4, H/8, W/8]

The NPU runs fixed-shape graphs compiled ahead of time for one SoC: --soc picks it (default sm8750, the Snapdragon
8 Elite in the Galaxy S25 / S25 Ultra). A binary compiled for one SoC fails to load on another. Each --sizes entry
becomes a graph named unet_<W>x<H>_b<N>_<eps|vpred>; the runtime picks the graph whose shapes match and reads the
prediction type from the name. Several sizes keep separate weights unless --share-weights is given (HTP weight
sharing: one copy of the weights, but the x86 simulator cannot load such a binary, so it is only checked on a phone).

Only list sizes the model was trained at. SDXS-512 is distilled at 512x512 only; this script refuses other sizes
for it unless --allow-untrained-sizes is given (see docs/coreml.md for what goes wrong).

Precision:
  fp16   weights and activations in fp16. Right for small UNets such as SDXS (about 0.6 GB).
  w8a16  8-bit weights (one scale per output channel), 16-bit activations: half the size, and the format
         Qualcomm uses for SD 1.5/2.1 on the NPU. Activations are calibrated on real UNet inputs, recorded by
         running the diffusers pipeline next to --unet on the CPU for --calibration-prompts prompts (the text
         encoder, scheduler and tokenizer are read from that pipeline). That takes a while: about 20 s per prompt
         and step for SD 2.1 on a laptop CPU.

--check runs the binary once on the x86 HTP simulator and compares it with PyTorch fp32. The simulator computes
exactly what the NPU computes, but a full UNet takes a long time on it (tens of minutes for SDXS).

Examples:
  python convert_unet.py --unet sdxs-512-dreamshaper/unet --out sdxs_unet_fp16.bin
  python convert_unet.py --unet sd21-base/unet --out sd21base_unet_w8a16.bin --precision w8a16 --batch 2
  python convert_unet.py --unet sdxs-512-dreamshaper/unet --out sdxs_unet_s24.bin --soc sm8650

Needs QNN_SDK_ROOT (QAIRT SDK on x86_64 Linux, Python 3.10) and
pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 onnx==1.16.1 numpy==1.26.4
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel

import qnn_tools

CALIBRATION_PROMPTS = [
    "portrait of a lovely cat",
    "a cozy mountain cabin at sunset, snow, warm window lights, photorealistic",
    "a large billboard on a city street showing an advertisement with the text \"COFFEE\", photo, daytime",
    "a bowl of ramen on a wooden table, studio lighting",
    "an astronaut riding a horse on the moon, digital art",
    "a watercolor painting of a lighthouse on a cliff",
    "close-up photo of a red rose with water droplets",
    "a busy market street in Marrakech, golden hour",
    "a futuristic city skyline at night, neon lights, rain",
    "a golden retriever puppy playing in autumn leaves",
    "an old library with tall bookshelves and a ladder",
    "a minimalist living room with large windows, interior design photo",
]


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states, return_dict=False)[0]


def parse_size(text):
    width, height = (int(v) for v in text.lower().split("x"))
    if width % 8 or height % 8:
        raise argparse.ArgumentTypeError(f"{text}: width and height must be multiples of 8")
    return width, height


def native_size(cfg):
    side = cfg.sample_size * 8
    return side, side


def is_sdxs(cfg):
    # Same test as script/coreml/convert_unet.py: no mid block, three levels, no attention at the top level.
    return (cfg.mid_block_type is None and len(cfg.block_out_channels) == 3 and
            list(cfg.down_block_types)[0] == "DownBlock2D")


def prediction_type(unet_dir, override):
    """epsilon or v_prediction, from --prediction or the pipeline's scheduler config next to unet/."""
    if override:
        return override
    sched = Path(unet_dir).resolve().parent / "scheduler" / "scheduler_config.json"
    if sched.exists():
        return json.loads(sched.read_text()).get("prediction_type", "epsilon")
    raise SystemExit(f"error: {sched} not found; pass --prediction epsilon or --prediction v_prediction")


def check_sizes(cfg, sizes, allow_untrained):
    native = native_size(cfg)
    off = [s for s in sizes if s != native]
    if not off:
        return
    listed = ", ".join(f"{w}x{h}" for w, h in off)
    if is_sdxs(cfg) and not allow_untrained:
        raise SystemExit(f"error: this looks like SDXS, which only gives good images at {native[0]}x{native[1]}; "
                         f"refusing {listed} (pass --allow-untrained-sizes to export anyway)")
    print(f"warning: {listed} differs from the UNet's native {native[0]}x{native[1]}; check image quality at those sizes")


def graph_name(width, height, batch, prediction):
    return f"unet_{width}x{height}_b{batch}_{'vpred' if prediction == 'v_prediction' else 'eps'}"


def record_calibration(unet_dir, unet, sizes, batch, context_len, n_prompts, steps, per_prompt):
    """Runs the diffusers pipeline around the UNet on the CPU and records UNet inputs at evenly spaced steps."""
    from diffusers import StableDiffusionPipeline

    pipe_dir = Path(unet_dir).resolve().parent
    pipe = StableDiffusionPipeline.from_pretrained(pipe_dir, unet=unet, torch_dtype=torch.float32,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    samples = {size: [] for size in sizes}
    keep_steps = set(np.linspace(0, steps - 1, num=min(per_prompt, steps)).round().astype(int).tolist())
    state = {"step": 0, "size": None}

    def hook(module, args, kwargs):
        sample, timestep = args[0], args[1]
        ehs = kwargs.get("encoder_hidden_states", args[2] if len(args) > 2 else None)
        if state["step"] in keep_steps:
            t = timestep.float().reshape(-1)
            if t.numel() == 1:
                t = t.expand(sample.shape[0])
            # The pipeline runs CFG as a batch of 2 (uncond, cond); a batch-1 graph takes each half on its own.
            for b0 in range(0, sample.shape[0], batch):
                samples[state["size"]].append({
                    "sample": sample[b0:b0 + batch].numpy(),
                    "timestep": t[b0:b0 + batch].numpy(),
                    "encoder_hidden_states": ehs[b0:b0 + batch, :context_len].numpy(),
                })
        state["step"] += 1

    handle = unet.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for i in range(n_prompts):
            for width, height in sizes:
                state.update(step=0, size=(width, height))
                t0 = time.time()
                with torch.no_grad():
                    pipe(CALIBRATION_PROMPTS[i % len(CALIBRATION_PROMPTS)], width=width, height=height,
                         num_inference_steps=steps, guidance_scale=7.0 if steps > 1 else 1.0,
                         generator=torch.Generator().manual_seed(i), output_type="latent")
                print(f"  calibration prompt {i + 1}/{n_prompts} at {width}x{height}: {time.time() - t0:.0f} s")
    finally:
        handle.remove()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet", required=True, help="diffusers UNet folder (the pipeline's unet/ directory)")
    parser.add_argument("--out", required=True, help="output context binary (.bin)")
    parser.add_argument("--precision", choices=["fp16", "w8a16"], default="fp16",
                        help="fp16 (default, small UNets such as SDXS) or w8a16 (SD 1.x/2.x)")
    parser.add_argument("--soc", choices=sorted(qnn_tools.SOCS), default=qnn_tools.DEFAULT_SOC,
                        help="Snapdragon SoC to compile for: " +
                             "; ".join(f"{k} = {v[2]}" for k, v in qnn_tools.SOCS.items()))
    parser.add_argument("--batch", type=int, choices=[1, 2], default=1, help="UNet batch: 1, or 2 for batched CFG")
    parser.add_argument("--sizes", default="512x512",
                        help="comma-separated image sizes as WIDTHxHEIGHT; each becomes a graph of the binary")
    parser.add_argument("--share-weights", action="store_true",
                        help="with several --sizes, keep one copy of the weights (HTP weight sharing)")
    parser.add_argument("--prediction", choices=["epsilon", "v_prediction"],
                        help="prediction type to record (default: from the pipeline's scheduler config)")
    parser.add_argument("--allow-untrained-sizes", action="store_true",
                        help="export sizes the model was not trained at (refused for SDXS by default)")
    parser.add_argument("--context-len", type=int, default=77, help="text tokens; sd.cpp uses 77 per prompt chunk")
    parser.add_argument("--calibration-prompts", type=int, default=8, help="w8a16: prompts to calibrate on")
    parser.add_argument("--calibration-steps", type=int, default=20, help="w8a16: sampling steps per prompt")
    parser.add_argument("--calibration-per-prompt", type=int, default=5,
                        help="w8a16: steps recorded per prompt, spread over the schedule")
    parser.add_argument("--check", action="store_true",
                        help="run the binary on the x86 HTP simulator and compare with PyTorch fp32 (slow)")
    parser.add_argument("--keep-onnx", help="also keep the exported ONNX models in this folder")
    args = parser.parse_args()

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    # Some pipelines ship only the fp16 variant (diffusion_pytorch_model.fp16.safetensors); export in fp32 either way.
    has_default = any((Path(args.unet) / f).exists() for f in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"))
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=torch.float32,
                                                variant=None if has_default else "fp16").eval()
    wrapper = UNetWrapper(unet).eval()
    cfg = unet.config
    if cfg.in_channels != 4 or getattr(cfg, "addition_embed_type", None) or getattr(cfg, "class_embed_type", None):
        raise SystemExit("error: only SD 1.x/2.x-style UNets (4 latent channels, text conditioning only) are supported")
    check_sizes(cfg, sizes, args.allow_untrained_sizes)
    prediction = prediction_type(args.unet, args.prediction)
    print(f"prediction type: {prediction}, SoC {args.soc} ({qnn_tools.SOCS[args.soc][2]})")

    calibration = None
    if args.precision == "w8a16":
        print(f"recording calibration inputs ({args.calibration_prompts} prompts x {args.calibration_steps} steps)")
        calibration = record_calibration(args.unet, unet, sizes, args.batch, args.context_len,
                                         args.calibration_prompts, args.calibration_steps, args.calibration_per_prompt)

    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dlcs, checks = [], []
        for width, height in sizes:
            graph = graph_name(width, height, args.batch, prediction)
            torch.manual_seed(0)
            example = (
                torch.randn(args.batch, cfg.in_channels, height // 8, width // 8),
                torch.full((args.batch,), 999.0),
                torch.randn(args.batch, args.context_len, cfg.cross_attention_dim),
            )
            with torch.no_grad():
                reference = wrapper(*example).numpy()
            onnx_dir = Path(args.keep_onnx) if args.keep_onnx else tmp
            onnx_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = onnx_dir / f"{graph}.onnx"
            t0 = time.time()
            with torch.no_grad():
                torch.onnx.export(wrapper, example, str(onnx_path),
                                  input_names=["sample", "timestep", "encoder_hidden_states"],
                                  output_names=["noise_pred"], opset_version=17)
            print(f"{width}x{height}: exported ONNX in {time.time() - t0:.0f} s")
            t0 = time.time()
            dlc = qnn_tools.onnx_to_dlc(onnx_path, tmp / f"{graph}.dlc", args.precision,
                                        calibration[(width, height)] if calibration else None)
            print(f"{width}x{height}: converted to {args.precision} in {time.time() - t0:.0f} s")
            dlcs.append(dlc)
            checks.append((graph, example, reference))

        t0 = time.time()
        qnn_tools.dlcs_to_context_binary(dlcs, out, args.soc, args.share_weights)
        print(f"compiled {out} in {time.time() - t0:.0f} s: {out.stat().st_size / 1e6:.0f} MB, "
              f"graphs {', '.join(qnn_tools.describe(out))}")

    if not args.check:
        return
    if args.share_weights and len(sizes) > 1:
        print("check skipped: the HTP simulator cannot load weight-shared binaries")
        return
    for graph, example, reference in checks:
        t0 = time.time()
        inputs = {"sample": example[0].numpy(), "timestep": example[1].numpy(),
                  "encoder_hidden_states": example[2].numpy()}
        noise = qnn_tools.run_on_host(out, inputs, graph=graph)["noise_pred"]
        rel = float(np.abs(noise - reference).max() / max(np.abs(reference).max(), 1e-12))
        cos = float((noise * reference).sum() / (np.linalg.norm(noise) * np.linalg.norm(reference) + 1e-12))
        print(f"{graph} on the HTP simulator ({time.time() - t0:.0f} s): max relative error {rel:.2e}, "
              f"cosine vs PyTorch fp32 {cos:.6f}")


if __name__ == "__main__":
    main()

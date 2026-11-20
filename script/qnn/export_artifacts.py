"""Regenerate the Android (Qualcomm Hexagon NPU) diffusion artifacts from pinned upstream sources.

Every file below is rebuilt from the original weights at the same Hugging Face revisions as
script/coreml/export_artifacts.py, with the converters in this directory and sd-cli. The context binaries are
compiled for one Snapdragon SoC (--soc, default sm8750: Snapdragon 8 Elite, Galaxy S25 / S25 Ultra).

  sdxs        SDXS-512-DreamShaper, 512x512, 1 step, no CFG
    sdxs-f16.gguf                  text encoder + UNet + tiny decoder for sd.cpp (sd-cli -M convert)
    sdxs_unet_fp16.bin             UNet on the NPU, fp16                            --qnn-unet
    sdxs_tae_decoder.bin           built-in tiny decoder on the NPU                 --qnn-tae
  sd21base    SD 2.1-base, 512x512, 20 steps, CFG > 1 (batched)
    sd21base-Q8_0.gguf             text encoder + UNet + VAE for sd.cpp (sd-cli -M convert); -m
    sd21base_unet_w8a16.bin        UNet on the NPU, 8-bit weights, 16-bit activations, batch 2  --qnn-unet
    taesd.safetensors              TAESD weights (sd.cpp still loads them)          --taesd
    taesd_decoder.bin              TAESD decoder on the NPU                         --qnn-tae

With --soc other than sm8750 the .bin files get the SoC as a suffix (sdxs_unet_fp16_sm8650.bin).

Run on x86_64 Linux (Ubuntu 22.04, or WSL) with the QAIRT SDK:
  pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 onnx==1.16.1 numpy==1.26.4 huggingface_hub
  export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
  python script/qnn/export_artifacts.py --out artifacts --sd-cli build/bin/sd-cli
  python script/qnn/export_artifacts.py --out artifacts --models sdxs --soc sm8650 --dry-run

The SD 2.1-base w8a16 export records calibration inputs by running the diffusers pipeline on the CPU
(8 prompts x 20 steps with CFG), which takes about an hour on a laptop, and needs about 16 GB of RAM.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qnn_tools  # noqa: E402

# Pinned sources, shared with script/coreml/export_artifacts.py. Licenses: SDXS and SD 2.1-base are OpenRAIL++,
# TAESD is MIT.
SOURCES = {
    "sdxs": {"repo": "IDKiro/sdxs-512-dreamshaper", "revision": "76f720262bb051da75666b22c902a78c8e16c763",
             "include": ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "unet/*", "vae/*",
                         "feature_extractor/*"]},
    "sd21base": {"repo": "sd2-community/stable-diffusion-2-1-base", "revision": "4e63672c03103b6c636b8fb4119ba982469b2955",
                 "include": ["model_index.json", "scheduler/*", "tokenizer/*", "unet/config.json",
                             "unet/diffusion_pytorch_model.fp16.safetensors", "text_encoder/config.json",
                             "text_encoder/model.fp16.safetensors", "vae/config.json",
                             "vae/diffusion_pytorch_model.fp16.safetensors"]},
    "taesd": {"repo": "madebyollin/taesd", "revision": "614f76814bbe30edbe2e627ace1c2234c81a2c0e",
              "include": ["config.json", "diffusion_pytorch_model.safetensors"]},
}

SKIP_EXISTING = False


def exists(path):
    if SKIP_EXISTING and Path(path).exists():
        print(f"  exists, skipped: {path}")
        return True
    return False


def run(cmd, dry_run):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def download(name, sources_dir, dry_run):
    src = SOURCES[name]
    dest = sources_dir / name
    run(["hf", "download", src["repo"], "--revision", src["revision"], "--local-dir", dest, "--include", *src["include"]],
        dry_run)
    for file, expected in src.get("sha256", {}).items():
        if dry_run:
            continue
        digest = hashlib.sha256()
        with open(dest / file, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise SystemExit(f"error: {dest / file} has sha256 {digest.hexdigest()}, expected {expected}")
    return dest


def convert(script, args, dry_run):
    run([sys.executable, HERE / script, *args], dry_run)


def fp16_view(src, dest, dry_run):
    """sd-cli reads unet/diffusion_pytorch_model.safetensors etc.; link the pipeline's fp16 variants under those names."""
    links = {"unet/diffusion_pytorch_model.safetensors": "unet/diffusion_pytorch_model.fp16.safetensors",
             "vae/diffusion_pytorch_model.safetensors": "vae/diffusion_pytorch_model.fp16.safetensors",
             "text_encoder/model.safetensors": "text_encoder/model.fp16.safetensors"}
    for name, target in links.items():
        print(f"+ ln -sf {src / target} {dest / name}")
        if not dry_run:
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).unlink(missing_ok=True)
            (dest / name).symlink_to(src / target)
    for name in ("model_index.json", "scheduler", "tokenizer", "unet/config.json", "vae/config.json", "text_encoder/config.json"):
        if not dry_run and not (dest / name).exists():
            (dest / name).symlink_to(src / name)
    return dest


def binary(out, stem, soc):
    return out / (f"{stem}.bin" if soc == qnn_tools.DEFAULT_SOC else f"{stem}_{soc}.bin")


def export_sdxs(out, sources, sd_cli, soc, dry_run):
    src = download("sdxs", sources, dry_run)
    if not exists(out / "sdxs-f16.gguf"):
        run([sd_cli, "-M", "convert", "-m", src, "-o", out / "sdxs-f16.gguf", "--type", "f16"], dry_run)
    unet = binary(out, "sdxs_unet_fp16", soc)
    if not exists(unet):
        convert("convert_unet.py", ["--unet", src / "unet", "--out", unet, "--soc", soc], dry_run)
    tae = binary(out, "sdxs_tae_decoder", soc)
    if not exists(tae):
        convert("convert_tae_decoder.py", ["--vae", src / "vae", "--out", tae, "--soc", soc, "--no-check"], dry_run)


def export_sd21base(out, sources, sd_cli, soc, dry_run):
    src = download("sd21base", sources, dry_run)
    if not exists(out / "sd21base-Q8_0.gguf"):
        # Q8_0 like the CoreML artifacts; sd-cli keeps convolutions, norms and the text encoder in F16.
        view = fp16_view(src, sources / "sd21base-diffusers", dry_run)
        run([sd_cli, "-M", "convert", "-m", view, "-o", out / "sd21base-Q8_0.gguf", "--type", "q8_0"], dry_run)
    taesd = download("taesd", sources, dry_run)
    if not exists(out / "taesd.safetensors"):
        run(["cp", taesd / "diffusion_pytorch_model.safetensors", out / "taesd.safetensors"], dry_run)
    tae = binary(out, "taesd_decoder", soc)
    if not exists(tae):
        convert("convert_tae_decoder.py", ["--vae", taesd, "--out", tae, "--soc", soc, "--no-check"], dry_run)
    unet = binary(out, "sd21base_unet_w8a16", soc)
    if not exists(unet):
        # The calibration pipeline reads the text encoder etc. next to unet/, through the fp32-named view.
        view = fp16_view(src, sources / "sd21base-diffusers", dry_run)
        convert("convert_unet.py", ["--unet", view / "unet", "--out", unet, "--precision", "w8a16", "--batch", "2",
                                    "--soc", soc], dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for the artifacts")
    parser.add_argument("--sources", help="directory for downloaded sources (default: <out>/sources)")
    parser.add_argument("--models", default="sdxs,sd21base", help="comma-separated: sdxs, sd21base")
    parser.add_argument("--soc", choices=sorted(qnn_tools.SOCS), default=qnn_tools.DEFAULT_SOC,
                        help="Snapdragon SoC to compile for (default sm8750: Galaxy S25 / S25 Ultra)")
    parser.add_argument("--sd-cli", default="build/bin/sd-cli", help="sd-cli (host build), used to convert to GGUF")
    parser.add_argument("--skip-existing", action="store_true", help="skip artifacts that already exist in --out")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    global SKIP_EXISTING
    SKIP_EXISTING = args.skip_existing
    out = Path(args.out).resolve()
    sources = Path(args.sources).resolve() if args.sources else out / "sources"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - {"sdxs", "sd21base"}
    if unknown:
        raise SystemExit(f"error: unknown models {sorted(unknown)}")
    if not args.dry_run:
        qnn_tools.sdk_root()
        out.mkdir(parents=True, exist_ok=True)
        if not Path(args.sd_cli).exists():
            raise SystemExit(f"error: {args.sd_cli} not found; build sd-cli for this machine first or pass --sd-cli")

    if "sdxs" in models:
        export_sdxs(out, sources, Path(args.sd_cli).resolve(), args.soc, args.dry_run)
    if "sd21base" in models:
        export_sd21base(out, sources, Path(args.sd_cli).resolve(), args.soc, args.dry_run)
    print(f"done: artifacts in {out}")


if __name__ == "__main__":
    main()

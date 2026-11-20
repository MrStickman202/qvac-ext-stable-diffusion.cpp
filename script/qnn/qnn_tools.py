"""Helpers shared by the QNN export scripts: ONNX -> DLC -> HTP context binary, and running a context binary
on the x86 HTP simulator to check it against PyTorch.

Needs the Qualcomm AI Runtime (QAIRT) SDK with QNN_SDK_ROOT set, on x86_64 Linux (Ubuntu 22.04 and Python 3.10
are what Qualcomm supports; WSL works on Windows). The tools are run as subprocesses; this module sets PATH,
LD_LIBRARY_PATH and PYTHONPATH for them, so sourcing $QNN_SDK_ROOT/bin/envsetup.sh is not required.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Snapdragon SoCs with an HTP that runs fp16 graphs (v73 and newer): SoC model id and Hexagon arch,
# from the "Supported Snapdragon devices" table of the QAIRT docs.
SOCS = {
    "sm8750": (69, "v79", "Snapdragon 8 Elite: Galaxy S25 / S25 Ultra, Z Fold7"),
    "sm8850": (87, "v81", "Snapdragon 8 Elite Gen 5"),
    "sm8650": (57, "v75", "Snapdragon 8 Gen 3: Galaxy S24 / S24 Ultra, Z Fold6"),
    "sm8550": (43, "v73", "Snapdragon 8 Gen 2: Galaxy S23 / S23 Ultra, Z Fold5"),
}
DEFAULT_SOC = "sm8750"


def sdk_root():
    root = os.environ.get("QNN_SDK_ROOT")
    if not root or not Path(root, "include", "QNN", "QnnInterface.h").exists():
        raise SystemExit("error: set QNN_SDK_ROOT to the Qualcomm AI Runtime (QAIRT) SDK root "
                         "(the folder with include/QNN and bin/x86_64-linux-clang)")
    return Path(root)


def tool_env():
    root = sdk_root()
    env = dict(os.environ)
    host = "x86_64-linux-clang"
    env["PATH"] = f"{root / 'bin' / host}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{root / 'lib' / host}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{root / 'lib' / 'python'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def run(cmd, quiet=True):
    proc = subprocess.run([str(c) for c in cmd], env=tool_env(), capture_output=quiet, text=True)
    if proc.returncode != 0:
        if quiet:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
        raise SystemExit(f"error: {Path(str(cmd[0])).name} failed (exit {proc.returncode})")
    return proc


def onnx_to_dlc(onnx_path, dlc_path, precision="fp16", calibration=None, preserve_io=False):
    """Converts an ONNX model to a DLC.

    precision: fp16 (weights and activations in fp16) or w8a16 (8-bit weights, 16-bit activations; needs
    calibration, a list of {input name: float32 array} dicts that exercise the model like real inputs).
    preserve_io: keep float32 graph inputs and outputs instead of the graph's own type (fp16 or uint16).
    """
    dlc_path = Path(dlc_path)
    if precision == "fp16":
        cmd = ["qairt-converter", "--input_network", onnx_path, "--output_path", dlc_path, "--float_bitwidth", "16"]
        if preserve_io:
            cmd += ["--preserve_io_datatype"]
        run(cmd)
        return dlc_path
    if precision != "w8a16":
        raise ValueError(precision)
    if not calibration:
        raise SystemExit("error: w8a16 needs calibration inputs")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        float_dlc = tmp / dlc_path.name  # the graph is named after the DLC file
        run(["qairt-converter", "--input_network", onnx_path, "--output_path", float_dlc])
        input_list = write_input_list(tmp / "calibration", calibration)
        cmd = ["qairt-quantizer", "--input_dlc", float_dlc, "--output_dlc", dlc_path, "--input_list", input_list,
               "--weights_bitwidth", "8", "--act_bitwidth", "16", "--bias_bitwidth", "32",
               "--use_per_channel_quantization"]
        if preserve_io:
            cmd += ["--preserve_io_datatype"]
        run(cmd)
    return dlc_path


def write_input_list(folder, samples):
    """Writes float32 .raw files and an input list in the "name:=path" format the QNN tools read."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, sample in enumerate(samples):
        parts = []
        for name, value in sample.items():
            path = folder / f"{i}_{name}.raw"
            np.ascontiguousarray(value, dtype=np.float32).tofile(path)
            parts.append(f"{name}:={path}")
        lines.append(" ".join(parts))
    input_list = folder / "input_list.txt"
    input_list.write_text("\n".join(lines) + "\n")
    return input_list


def htp_config(folder, graph_names, soc, weight_sharing):
    soc_model, arch, _ = SOCS[soc]
    folder = Path(folder)
    config = {
        "graphs": [{"graph_names": list(graph_names), "vtcm_mb": 0, "O": 3}],
        "devices": [{"soc_model": soc_model, "dsp_arch": arch}],
    }
    if weight_sharing:
        config["context"] = {"weight_sharing_enabled": True}
    (folder / "htp_config.json").write_text(json.dumps(config, indent=2))
    (folder / "backend_extensions.json").write_text(json.dumps({
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": str(folder / "htp_config.json"),
        }
    }, indent=2))
    return folder / "backend_extensions.json"


def dlcs_to_context_binary(dlcs, out_path, soc=DEFAULT_SOC, share_weights=False):
    """Compiles one or more DLCs into one HTP context binary for `soc`. Each DLC becomes a graph named after
    the DLC file (without .dlc). With share_weights, several graphs keep one copy of their common weights
    (HTP weight sharing: smaller files, but the x86 simulator cannot load such binaries, so they are only
    checked on a phone); without it, each graph carries its own copy."""
    out_path = Path(out_path)
    dlcs = [Path(d) for d in dlcs]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = htp_config(tmp, [d.stem for d in dlcs], soc, weight_sharing=share_weights and len(dlcs) > 1)
        run(["qnn-context-binary-generator", "--backend", "libQnnHtp.so", "--model", "libQnnModelDlc.so",
             "--dlc_path", ",".join(str(d) for d in dlcs), "--config_file", config,
             "--binary_file", out_path.stem, "--output_dir", tmp / "out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / "out" / (out_path.stem + ".bin")), out_path)
    return out_path


def describe(binary):
    """Graph names and the name, shape and type of every input and output, from qnn-context-binary-utility."""
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "info.json"
        run(["qnn-context-binary-utility", "--context_binary", binary, "--json_file", js])
        info = json.loads(js.read_text())["info"]
    graphs = {}
    for g in info["graphs"]:
        gi = g["info"]
        graphs[gi["graphName"]] = {
            "inputs": [(t["info"]["name"], t["info"]["dimensions"], t["info"]["dataType"]) for t in gi["graphInputs"]],
            "outputs": [(t["info"]["name"], t["info"]["dimensions"], t["info"]["dataType"]) for t in gi["graphOutputs"]],
        }
    return graphs


def run_on_host(binary, inputs, graph=None):
    """Runs a context binary on the x86 HTP simulator (bit-faithful to the NPU's arithmetic, but slow) and
    returns {output name: float32 array}. inputs maps input names to float32 arrays."""
    graphs = describe(binary)
    if graph is None:
        if len(graphs) != 1:
            raise ValueError(f"{binary} has {len(graphs)} graphs; pass graph=")
        graph = next(iter(graphs))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_list = write_input_list(tmp / "in", [inputs])
        # One input list per graph, in the binary's graph order; "__" skips a graph.
        lists = ",".join(str(input_list) if name == graph else "__" for name in graphs)
        run(["qnn-net-run", "--backend", "libQnnHtp.so", "--retrieve_context", binary, "--input_list", lists,
             "--output_dir", tmp / "out"])
        outputs = {}
        for name, dims, _ in graphs[graph]["outputs"]:
            path = next((tmp / "out").rglob(f"{name}.raw"))
            outputs[name] = np.fromfile(path, dtype=np.float32).reshape(dims)
    return outputs


def psnr(a, b, peak=1.0):
    mse = float(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)

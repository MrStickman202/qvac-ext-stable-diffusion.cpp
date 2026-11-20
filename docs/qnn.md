## Running the UNet and the tiny decoder on the Qualcomm Hexagon NPU (QNN sidecars)

On Android phones with a Snapdragon 8 Gen 2 or newer (Galaxy S23, S24 and S25 series, Z Fold5/6/7), two parts of the pipeline can run on the Hexagon NPU through Qualcomm's QNN runtime: the UNet of SD 1.x/2.x models (`--qnn-unet`), and the tiny autoencoder's decoder used by SDXS and `--taesd` (`--qnn-tae`). This is the Android counterpart of the [CoreML sidecars](coreml.md); both implement the same interfaces (`src/sidecar.h`), so the pipeline treats them alike. The text encoder, tokenizer and scheduler stay on ggml (CPU, or Vulkan/OpenCL). The ggml UNet weights are not loaded when the UNet runs on the NPU.

Each sidecar is a QNN *context binary* (`.bin`): the model compiled ahead of time for one Snapdragon SoC. The default target is the SM8750 (Snapdragon 8 Elite, Hexagon v79) in the Galaxy S25 / S25 Ultra; `--soc` in the export scripts picks another.

### Status

| | Checked |
| --- | --- |
| Runtime (`src/qnn_*.cpp`): loading, graph selection, fp32 / fp16 / 16-bit fixed-point I/O, batched CFG, error paths | `test-qnn` on the x86 HTP simulator, which computes what the NPU computes |
| SDXS tiny decoder, exported with `convert_tae_decoder.py` for SM8750 | 62.2 dB PSNR against PyTorch fp32 on the HTP simulator |
| Whole pipeline (`new_sd_ctx` / `generate_image`) with the QNN decoder: SDXS, 1 step | `test-qnn` with `SD_TEST_SDXS_DIR`, on the simulator: 62.9 dB against the ggml decoder |
| SDXS with the UNet **and** the decoder on QNN (`sd-cli --qnn-unet --qnn-tae`, fp16 UNet for SM8750) | 43.0 dB PSNR against all-ggml (CPU) on the simulator, same seed and prompt; the images look the same |
| Builds for Android arm64 (NDK r27) | CI |
| Speed on a phone, and the UNet exports on a phone | **not measured yet**: run [examples/android-bench](../examples/android-bench/README.md) on a device |

### Build

The QNN libraries are opened at run time, so building only needs the headers of the [Qualcomm AI Runtime (QAIRT) SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community) (free download; 2.40 was used here):

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DSD_QNN=ON -DGGML_OPENMP=OFF
cmake --build build-android --config Release
```

`SD_QNN` builds on Android and Linux (Linux runs the x86 HTP simulator, for tests). It adds `--qnn-unet` and `--qnn-tae` to `sd-cli` (`qnn_unet_path` and `qnn_tae_decoder_path` in `sd_ctx_params_t`).

At run time sd.cpp loads `libQnnHtp.so` and `libQnnSystem.so`, and the NPU loads `libQnnHtpV79Skel.so` (for v79). Ship these from the QAIRT SDK version that compiled the `.bin` files:

| File | From the SDK |
| --- | --- |
| `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV79Stub.so` | `lib/aarch64-android/` |
| `libQnnHtpV79Skel.so` | `lib/hexagon-v79/unsigned/` |

In an app, put them in `jniLibs/arm64-v8a` and set `SD_QNN_LIB_DIR` to the app's `nativeLibraryDir`; sd.cpp then also sets `ADSP_LIBRARY_PATH` to it, which is where the NPU looks for the skel library. From `adb shell`, set `SD_QNN_LIB_DIR` to the folder holding them (`examples/android-bench/run_on_device.sh` does this).

| Environment variable | |
| --- | --- |
| `SD_QNN_LIB_DIR` | folder with the QNN libraries (default: the normal library search path) |
| `SD_QNN_PERF` | NPU clock vote: `burst` (default), `high`, `balanced`, or `default` (leave it to the driver) |
| `SD_QNN_LOG` | QNN's own log level: `error`, `warn` (default), `info`, `verbose`, `debug` |
| `SD_QNN_BACKEND` | backend library (default `libQnnHtp.so`) |

### Export the artifacts

The export runs on x86_64 Linux (Ubuntu 22.04 with Python 3.10 is what Qualcomm supports; WSL on Windows works) with the QAIRT SDK. `script/qnn/export_artifacts.py` rebuilds every model file from the same pinned Hugging Face revisions as the CoreML artifacts:

```bash
pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 onnx==1.16.1 numpy==1.26.4 huggingface_hub
# plus what the QAIRT tools import: $QNN_SDK_ROOT/bin/check-python-dependency
python script/qnn/export_artifacts.py --out artifacts --sd-cli build/bin/sd-cli            # S25 / S25 Ultra
python script/qnn/export_artifacts.py --out artifacts --models sdxs --soc sm8650           # S24 (Snapdragon)
python script/qnn/export_artifacts.py --out artifacts --dry-run                            # print the commands
```

| Model | Artifact | Used as |
| --- | --- | --- |
| SDXS-512-DreamShaper | `sdxs-f16.gguf` (sd-cli -M convert) | `-m` |
| | `sdxs_unet_fp16.bin` (636 MB) | `--qnn-unet` |
| | `sdxs_tae_decoder.bin` (3.4 MB) | `--qnn-tae` |
| SD 2.1-base | `sd21base-Q8_0.gguf` | `-m` |
| | `sd21base_unet_w8a16.bin` (8-bit weights, 16-bit activations, batch 2) | `--qnn-unet` |
| | `taesd.safetensors`, `taesd_decoder.bin` | `--taesd`, `--qnn-tae` |

The pipeline runs ONNX export, `qairt-converter` (and `qairt-quantizer` for w8a16), then `qnn-context-binary-generator` with the HTP backend, all inside `script/qnn/qnn_tools.py`.

#### UNet

```bash
python script/qnn/convert_unet.py --unet sdxs-512-dreamshaper/unet --out sdxs_unet_fp16.bin
python script/qnn/convert_unet.py --unet sd21-base/unet --out sd21base_unet_w8a16.bin --precision w8a16 --batch 2
```

| `--precision` | Weights / activations | Use for |
| --- | --- | --- |
| `fp16` (default) | fp16 / fp16 | Small UNets such as SDXS. |
| `w8a16` | int8 per output channel / 16-bit fixed point | SD 1.x/2.x. This is the format Qualcomm uses for its own SD 1.5/2.1 NPU releases. |

`w8a16` calibrates the activation ranges on real UNet inputs: the script runs the diffusers pipeline next to `--unet` on the CPU for `--calibration-prompts` prompts (default 8, 20 steps, CFG 7) and records the UNet's inputs at 5 steps per prompt. The graph's inputs and outputs are then 16-bit fixed point; the runtime converts to and from float32.

Each `--sizes` entry becomes one fixed-shape graph named `unet_<W>x<H>_b<N>_<eps|vpred>`; the runtime picks the graph whose shapes match, and reads the prediction type from the name (so SD 2.x needs no `--prediction` with an exported UNet). Without `--share-weights` every graph carries its own copy of the weights. With it, HTP weight sharing keeps one copy, but the x86 simulator cannot load such binaries, so they have not been checked. The same SDXS size rules as for CoreML apply: SDXS is refused at sizes other than 512x512 unless `--allow-untrained-sizes` is given.

`--check` runs the exported UNet once on the x86 HTP simulator against PyTorch fp32. The simulator is exact but slow: on a 2-core machine one SDXS UNet call takes about 40 minutes and the tiny decoder 12 minutes at 512x512. The SDXS fp16 export itself takes about 80 s (ONNX export, conversion and HTP compile).

#### Tiny decoder

```bash
python script/qnn/convert_tae_decoder.py --vae sdxs-512-dreamshaper/vae --out sdxs_tae_decoder.bin
```

Same graph as the CoreML export and sd.cpp's `TinyDecoder` (no final `x * 2 - 1`). It runs in fp16 on the NPU and is checked on the simulator unless `--no-check` is given: 62.2 dB PSNR against PyTorch fp32 for SDXS's decoder (max abs error 0.013 on a [0, 1] image).

### Run

```bash
sd-cli -m sdxs-f16.gguf --qnn-unet sdxs_unet_fp16.bin --qnn-tae sdxs_tae_decoder.bin \
       -p "portrait of a lovely cat" --cfg-scale 1 --steps 1
```

The rules are the same as for the CoreML sidecars:

- `-m` still points at the full model: the text encoder (and the VAE, unless a tiny decoder is used) come from it, and its UNet tensors are skipped.
- `--qnn-tae` needs the image tiny autoencoder (SDXS, or `--taesd`). With `--vae-tiling` the decoder falls back to ggml.
- Circular padding fails at load, and a LoRA that changes UNet weights fails the generation (text-encoder LoRAs work).
- Prompts longer than 77 tokens, ControlNet, and models that take `c_concat` or `y` fail with an error.
- A batch-2 UNet runs both CFG passes in one call. It needs a CFG scale above 1 and no ControlNet, image CFG, skip layers or step caching.
- A CoreML and a QNN path for the same part cannot both be set.

The context binary is compiled for one SoC. On another SoC, loading fails with a message naming the SoC it was built for. It also fails when the QNN libraries come from a different QAIRT version than the converter that built the `.bin`. Unlike CoreML there is no on-device compile step: loading takes about as long as reading the file.

### Tests

`test-qnn` (built with `-DSD_QNN=ON` and static libraries) runs tiny context binaries through the runtime: fp16, float32 and w8a16 I/O, a two-size binary, prediction type from the graph name, batched CFG against separate passes, shape rejection, files that are not usable sidecars, and decoder output checks. On x86_64 Linux it runs on the HTP simulator, so no phone is needed. With `SD_TEST_SDXS_DIR` set to the SDXS artifacts it also runs the pipeline checks: circular padding and conflicting CoreML/QNN paths are rejected, `--vae-tiling` falls back to the ggml decoder with the same image, and (with `sdxs_unet_fp16.bin` present) UNet LoRAs are rejected.

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
python tests/qnn/make_fixtures.py build/qnn-fixtures
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang build/bin/test-qnn build/qnn-fixtures
SD_TEST_SDXS_DIR=artifacts LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang build/bin/test-qnn build/qnn-fixtures
```

The same binary and fixtures also run on a phone: `examples/android-bench/run_on_device.sh --push build/qnn-fixtures -- test qnn-fixtures`.

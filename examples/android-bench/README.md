## sd-bench on Android: ggml CPU vs the Hexagon NPU (QNN sidecars)

The Android counterpart of [ios-bench](../ios-bench/README.md): the same `sd-bench` (`examples/ios-bench/bench.cpp`) built for arm64-v8a, run over adb. It loads the model once, generates 1 warm-up and 3 timed images (512x512, seed 42), and prints the text encoder, UNet, decoder and total time for each. A UNet or decoder path ending in `.bin` runs on the NPU through the [QNN sidecars](../../docs/qnn.md); `-` runs the UNet on ggml (CPU).

You need a phone with a Snapdragon 8 Gen 2 or newer (Galaxy S23/S24/S25 series with Snapdragon) with USB debugging on, `adb`, the Android NDK, and the [QAIRT SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community) whose version compiled the `.bin` files. Building and running work on macOS and Linux; exporting the `.bin` files needs x86_64 Linux or WSL (see [docs/qnn.md](../../docs/qnn.md)).

### Build

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
cmake -S examples/android-bench -B build-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31
cmake --build build-android
```

This builds `sd-bench`, `sd-cli` and `test-qnn` in `build-android/bin`.

### Run

`run_on_device.sh` pushes the binaries, the QNN libraries for the phone's Hexagon version (detected from `ro.soc.model`) and any `--push` files to `/data/local/tmp/sdbench`, then runs the command after `--` there:

```bash
B=examples/android-bench/run_on_device.sh
$B --push sdxs-f16.gguf --push sdxs_unet_fp16.bin --push sdxs_tae_decoder.bin -- \
   bench sdxs-f16.gguf - s25_cpu                                              # all ggml (CPU)
$B -- bench sdxs-f16.gguf sdxs_unet_fp16.bin s25_npu                          # UNet on the NPU
$B -- bench sdxs-f16.gguf sdxs_unet_fp16.bin s25_npu_tae sdxs_tae_decoder.bin # UNet and decoder on the NPU
```

Files only need pushing once (`adb push --sync` skips unchanged ones). Copy images back with `adb pull /data/local/tmp/sdbench/s25_npu_cat.png`.

SD 2.1-base with the batch-2 w8a16 UNet and TAESD on the NPU:

```bash
$B --push sd21base-Q8_0.gguf --push sd21base_unet_w8a16.bin --push taesd.safetensors --push taesd_decoder.bin -- \
   bench sd21base-Q8_0.gguf sd21base_unet_w8a16.bin sd21_npu taesd_decoder.bin \
   --taesd taesd.safetensors --steps 20 --cfg 7 --sampler euler
```

`cli ARGS` runs `sd-cli` instead, and `test DIR` runs `test-qnn` on fixtures pushed with `--push build/qnn-fixtures`. They are compiled for the Snapdragon 8 Elite, so on other phones run `make_fixtures.py` with that SoC. `SD_QNN_PERF=balanced` (or `high`, `default`) changes the NPU clock vote, `SD_QNN_LOG=info` shows QNN's own log lines, and `SDBENCH_VERBOSE=1` shows every sd.cpp log line.

On a Mac: `brew install android-platform-tools` for adb, and the NDK from Android Studio's SDK Manager (`$HOME/Library/Android/sdk/ndk/<version>`).

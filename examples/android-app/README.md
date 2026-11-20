## SD NPU: an Android app for the QNN sidecars

Type a prompt, tap Generate, get a 512x512 image from SDXS-512-DreamShaper, with the UNet and the tiny decoder running on the Snapdragon NPU through the [QNN sidecars](../../docs/qnn.md). A switch runs the same model on the CPU (ggml) for comparison, and the app shows how long the text encoder, UNet and decoder took. Images can be saved to the gallery (Pictures/SD NPU).

It is the app version of [android-bench](../android-bench/README.md): no adb, no command line.

### Phones

The bundled NPU files are compiled for the **SM8750** (Snapdragon 8 Elite: Galaxy S25, S25+, S25 Ultra). On other phones the app runs on the CPU only, and says so. For another Snapdragon, export the `.bin` files with `--soc` (`script/qnn/export_artifacts.py`) and change `Models.kt`. The APK carries the QNN libraries for Hexagon v73, v75, v79 and v81.

### Model files

On first launch the app downloads about 2.4 GB in the background with Android's download manager:

| File | From |
| --- | --- |
| `sdxs/text_encoder/model.safetensors`, `sdxs/unet/diffusion_pytorch_model.safetensors`, `sdxs/vae/diffusion_pytorch_model.safetensors` | [IDKiro/sdxs-512-dreamshaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper) on Hugging Face, pinned revision. sd.cpp reads this diffusers layout directly; the UNet weights are only read on the CPU path. |
| `sdxs_unet_fp16.bin` (7 parts, joined on the phone), `sdxs_tae_decoder.bin` | the `sdnpu-models` branch of this fork, pinned to a commit (GitHub serves files up to 100 MB, hence the parts) |

Every file is checked against its SHA-256 after downloading. They are stored in `Android/data/io.tether.sdnpu/files/models/`, so you can also copy them there over USB (same names and folders) and skip the download.

### Build

Needs the Android SDK (platform 35), NDK 27.2.12479018, and the [QAIRT SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community) 2.40.0.251030, the version the `.bin` files were compiled with. Open this folder in Android Studio, or:

```bash
cd examples/android-app
printf 'sdk.dir=%s\nqnn.sdk.dir=%s\n' "$ANDROID_HOME" /path/to/qairt/2.40.0.251030 > local.properties
./gradlew assembleRelease            # add -PnativeJobs=2 on machines with less than 16 GB of RAM
adb install -r app/build/outputs/apk/release/app-release.apk
```

The APK is signed with `sdnpu-dev.keystore`, a development key committed on purpose (password `android`) so that builds from any machine can update an installed copy. Use your own key for anything you publish.

### How the NPU is reached

Gradle copies `libQnnHtp.so`, `libQnnSystem.so` and the HTP stub and skel libraries from the QAIRT SDK into the APK. The native libraries are extracted on install (`useLegacyPackaging`), because the NPU loads the skel library from a file path; `sd_jni.cpp` points `SD_QNN_LIB_DIR` and `ADSP_LIBRARY_PATH` at the app's native library folder. The manifest declares `libcdsprpc.so`, the vendor library the stub uses to talk to the NPU, which apps targeting Android 12 or newer may only load when they declare it. If the NPU cannot load the model, the app shows QNN's error and switches to the CPU.

`adb logcat -s sdnpu` shows sd.cpp's log, including the QNN lines.

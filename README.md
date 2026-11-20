# Model files for the SD NPU Android app

This branch only holds files for `examples/android-app` on the `qnn-sidecar` branch: the SDXS-512-DreamShaper UNet
and tiny decoder compiled as QNN context binaries for the **SM8750** NPU (Snapdragon 8 Elite, Galaxy S25 series)
with QAIRT 2.40.0.251030, by `script/qnn/export_artifacts.py`. The app downloads them on first launch; the rest of
SDXS comes from Hugging Face. GitHub refuses files over 100 MB, so the UNet is split into parts:

    cat sdxs_unet_fp16.bin.0* > sdxs_unet_fp16.bin
    sha256sum sdxs_unet_fp16.bin   # e340619a262790fff010816f7f13b96bab8a1d9973d1c580029f3e6b6a5be81c

| File | SHA-256 |
| --- | --- |
| sdxs_unet_fp16.bin (635,904,000 bytes, 7 parts) | e340619a262790fff010816f7f13b96bab8a1d9973d1c580029f3e6b6a5be81c |
| sdxs_tae_decoder.bin (3,424,256 bytes) | 92ab9270e8cff6ea51067460301edef652a3649fd954b0b16315d6d5900a5268 |

Source model: [IDKiro/sdxs-512-dreamshaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper) (revision 76f72026),
CreativeML OpenRAIL++-M license, which applies to these derived files too. Delete this branch to remove them; the
app then cannot download them.

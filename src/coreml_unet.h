#ifndef __COREML_UNET_H__
#define __COREML_UNET_H__

#include <cstdint>
#include <memory>
#include <string>

#include "sidecar.h"

// Runs a UNet exported to CoreML (see script/coreml/convert_unet.py) in place of the ggml UNet.
// A multifunction model (iOS 18+) holds one fixed-shape function per resolution; each call uses
// the function whose input shapes match, loading it on first use.
// Expected model I/O (fixed shapes; float32 or float16 inputs):
//   sample                [N, C, H, W]
//   timestep              [N]
//   encoder_hidden_states [N, T, D], or [N, D, 1, T] as Apple's ml-stable-diffusion exports it
//   noise_pred            [N, C, H, W]
// Compute units come from SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
class CoreMLUNet : public UNetSidecar {
public:
    static std::unique_ptr<CoreMLUNet> load(const std::string& path);
};

#endif  // __COREML_UNET_H__

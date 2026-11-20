#ifndef __COREML_TAE_H__
#define __COREML_TAE_H__

#include <cstdint>
#include <memory>
#include <string>

#include "sidecar.h"

// Runs the tiny autoencoder's decoder (TAESD, or the one built into SDXS) exported to CoreML
// (see script/coreml/convert_tae_decoder.py) in place of the ggml TinyDecoder.
// Expected model I/O (float32, fixed shapes):
//   latent [N, 4, h, w] -> image [N, 3, 8h, 8w], in [0, 1] before sd.cpp clamps it
// Compute units come from SD_COREML_COMPUTE_UNITS: all | cpu_ne (default) | cpu_gpu | cpu.
class CoreMLTAEDecoder : public TAEDecoderSidecar {
public:
    static std::unique_ptr<CoreMLTAEDecoder> load(const std::string& path);
};

#endif  // __COREML_TAE_H__

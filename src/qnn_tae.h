#ifndef __QNN_TAE_H__
#define __QNN_TAE_H__

#include <memory>
#include <string>

#include "sidecar.h"

// Runs the tiny autoencoder's decoder (TAESD, or the one built into SDXS) compiled to a QNN context binary
// (see script/qnn/convert_tae_decoder.py) on the Hexagon NPU in place of the ggml TinyDecoder.
// Expected graph I/O (fixed shapes):
//   latent [N, 4, h, w] -> image [N, 3, 8h, 8w], in [0, 1] before sd.cpp clamps it
// A binary with several graphs (one per size) uses the graph whose latent shape matches.
class QnnTAEDecoder : public TAEDecoderSidecar {
public:
    static std::unique_ptr<QnnTAEDecoder> load(const std::string& path);
};

#endif  // __QNN_TAE_H__

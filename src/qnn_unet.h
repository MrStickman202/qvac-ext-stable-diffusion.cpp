#ifndef __QNN_UNET_H__
#define __QNN_UNET_H__

#include <memory>
#include <string>

#include "sidecar.h"

// Runs a UNet compiled to a QNN context binary (see script/qnn/convert_unet.py) on the Qualcomm Hexagon NPU
// in place of the ggml UNet. A context binary can hold several graphs that share one copy of the weights,
// one fixed-shape graph per resolution; each call uses the graph whose input shapes match.
// Expected graph I/O (fixed shapes; float32, float16 or 8/16-bit fixed point):
//   sample                [N, C, H, W]
//   timestep              [N]
//   encoder_hidden_states [N, T, D]
//   noise_pred            [N, C, H, W]
// convert_unet.py names graphs unet_<W>x<H>_b<N>_<eps|vpred>; the last part records the prediction type.
// The runtime is set up by qnn_runtime.h (SD_QNN_LIB_DIR, SD_QNN_PERF, SD_QNN_LOG).
class QnnUNet : public UNetSidecar {
public:
    static std::unique_ptr<QnnUNet> load(const std::string& path);
};

#endif  // __QNN_UNET_H__

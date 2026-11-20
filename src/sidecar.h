#ifndef __SIDECAR_H__
#define __SIDECAR_H__

// Interfaces for "sidecars": parts of the pipeline that run as fixed-shape models on a vendor runtime
// instead of ggml. CoreML (Apple Neural Engine, coreml_unet.h / coreml_tae.h) and QNN (Qualcomm Hexagon
// NPU, qnn_unet.h / qnn_tae.h) implement them; the pipeline only sees these interfaces.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "util.h"

// A UNet run in place of the ggml UNet. Model I/O (fixed shapes):
//   sample [N, C, H, W], timestep [N], encoder_hidden_states [N, T, D] -> noise_pred [N, C, H, W]
class UNetSidecar {
public:
    virtual ~UNetSidecar() = default;

    // x and out are [N, C, H, W], context is [N, T, D], all contiguous float32.
    virtual bool predict(const float* x,
                         const int64_t x_shape[4],
                         const float* timesteps,
                         const float* context,
                         const int64_t context_shape[3],
                         float* out) = 0;

    virtual std::string get_desc() = 0;

    // "epsilon" or "v_prediction" as recorded by the export script, or "" if the model does not record it.
    virtual std::string prediction_type() = 0;

    // The batch the model was exported for: 1, or 2 for models that run CFG's two passes in one call.
    virtual int64_t batch_size() = 0;

    // Runs CFG's conditional and unconditional passes as one batch-2 prediction on a batch-2 model.
    // x [1, C, H, W] and the timestep are shared; cond, uncond, out_cond and out_uncond are [1, ...].
    bool predict_cfg_batch(const float* x,
                           const int64_t x_shape[4],
                           float timestep,
                           const float* cond,
                           const float* uncond,
                           const int64_t context_shape[3],
                           float* out_cond,
                           float* out_uncond) {
        if (x_shape[0] != 1 || context_shape[0] != 1) {
            LOG_ERROR("unet sidecar: batched CFG needs a batch-1 latent and batch-1 contexts");
            return false;
        }
        const size_t nx = x_shape[1] * x_shape[2] * x_shape[3];
        const size_t nc = context_shape[1] * context_shape[2];
        std::vector<float> xs(2 * nx), ts(2, timestep), ctx(2 * nc), out(2 * nx);
        std::copy(x, x + nx, xs.begin());
        std::copy(x, x + nx, xs.begin() + nx);
        std::copy(cond, cond + nc, ctx.begin());
        std::copy(uncond, uncond + nc, ctx.begin() + nc);
        const int64_t xs_shape[4]  = {2, x_shape[1], x_shape[2], x_shape[3]};
        const int64_t ctx_shape[3] = {2, context_shape[1], context_shape[2]};
        if (!predict(xs.data(), xs_shape, ts.data(), ctx.data(), ctx_shape, out.data())) {
            return false;
        }
        std::copy(out.begin(), out.begin() + nx, out_cond);
        std::copy(out.begin() + nx, out.end(), out_uncond);
        return true;
    }
};

// The tiny autoencoder's decoder (TAESD, or the one built into SDXS) run in place of the ggml TinyDecoder.
// Model I/O (fixed shapes): latent [N, 4, h, w] -> image [N, 3, 8h, 8w], in [0, 1] before sd.cpp clamps it.
class TAEDecoderSidecar {
public:
    virtual ~TAEDecoderSidecar() = default;

    // latent is [N, C, h, w] and image is [N, 3, 8h, 8w], both contiguous float32. Both shapes must match
    // the model's, so the decoder never writes more than image_shape holds.
    virtual bool decode(const float* latent, const int64_t latent_shape[4], float* image, const int64_t image_shape[4]) = 0;
};

// Shared by the sidecars' log messages.
inline std::string sidecar_shape_str(const int64_t* shape, size_t n) {
    std::string s = "[";
    for (size_t i = 0; i < n; i++) {
        s += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

#endif  // __SIDECAR_H__

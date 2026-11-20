#include "qnn_tae.h"

#include <algorithm>
#include <chrono>

#include "qnn_runtime.h"

using namespace sd_qnn;

namespace {

struct DecoderGraph {
    Graph* graph = nullptr;
    int latent = -1, image = -1;

    const std::vector<int64_t>& latent_shape() const { return graph->inputs[latent].shape; }
    const std::vector<int64_t>& image_shape() const { return graph->outputs[image].shape; }
};

class QnnTAEDecoderImpl : public QnnTAEDecoder {
public:
    std::unique_ptr<Context> ctx;
    std::vector<DecoderGraph> graphs;

    bool decode(const float* latent, const int64_t shape[4], float* image, const int64_t out_shape[4]) override {
        const DecoderGraph* g = nullptr;
        for (const auto& candidate : graphs) {
            if (std::equal(candidate.latent_shape().begin(), candidate.latent_shape().end(), shape)) {
                g = &candidate;
                break;
            }
        }
        if (g == nullptr) {
            std::string available;
            for (const auto& candidate : graphs) {
                available += " " + shape_str(candidate.latent_shape());
            }
            LOG_ERROR("qnn tae: no graph for latent %s; the binary has:%s", sidecar_shape_str(shape, 4).c_str(),
                      available.c_str());
            return false;
        }
        if (!std::equal(g->image_shape().begin(), g->image_shape().end(), out_shape)) {
            LOG_ERROR("qnn tae: image buffer %s does not match the graph's output %s",
                      sidecar_shape_str(out_shape, 4).c_str(), shape_str(g->image_shape()).c_str());
            return false;
        }
        std::vector<const float*> in(g->graph->inputs.size());
        in[g->latent] = latent;
        std::vector<float*> outs(g->graph->outputs.size(), nullptr);
        outs[g->image] = image;
        return g->graph->execute(in, outs, "qnn tae");
    }
};

}  // namespace

std::unique_ptr<QnnTAEDecoder> QnnTAEDecoder::load(const std::string& path) {
    auto t0  = std::chrono::steady_clock::now();
    auto ctx = Context::load(path, "qnn tae");
    if (!ctx) {
        return nullptr;
    }
    auto impl = std::make_unique<QnnTAEDecoderImpl>();
    std::string sizes;
    for (auto& graph : ctx->graphs) {
        DecoderGraph g;
        g.graph  = graph.get();
        g.latent = graph->input_index("latent");
        g.image  = graph->output_index("image");
        if (g.latent < 0 || g.image < 0 || graph->inputs.size() != 1) {
            LOG_ERROR("qnn tae: graph %s of %s does not have a single 'latent' input and an 'image' output "
                      "(export it with script/qnn/convert_tae_decoder.py)",
                      graph->name.c_str(), path.c_str());
            return nullptr;
        }
        const auto& z = g.latent_shape();
        if (z.size() != 4) {
            LOG_ERROR("qnn tae: graph %s of %s has latent %s; expected [N, 4, h, w]", graph->name.c_str(), path.c_str(),
                      shape_str(z).c_str());
            return nullptr;
        }
        const std::vector<int64_t> expected = {z[0], 3, 8 * z[2], 8 * z[3]};
        if (g.image_shape() != expected) {
            LOG_ERROR("qnn tae: graph %s of %s maps latent %s to image %s, expected %s", graph->name.c_str(),
                      path.c_str(), shape_str(z).c_str(), shape_str(g.image_shape()).c_str(),
                      shape_str(expected).c_str());
            return nullptr;
        }
        sizes += " " + shape_str(z);
        impl->graphs.push_back(g);
    }
    LOG_INFO("qnn tae: %s ready in %.2f s (%s), latent:%s", path.c_str(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ctx->summary.c_str(),
             sizes.c_str());
    impl->ctx = std::move(ctx);
    return impl;
}

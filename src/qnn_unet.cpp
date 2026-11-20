#include "qnn_unet.h"

#include <algorithm>
#include <chrono>
#include <sstream>

#include "qnn_runtime.h"

using namespace sd_qnn;

namespace {

// The prediction type a graph name records (unet_512x512_b1_eps, unet_512x512_b2_vpred), or "" if none.
std::string prediction_from_graph_name(const std::string& name) {
    std::stringstream ss(name);
    std::string token;
    std::string found;
    while (std::getline(ss, token, '_')) {
        if (token == "eps" || token == "epsilon") {
            found = "epsilon";
        } else if (token == "vpred" || token == "v") {
            found = "v_prediction";
        }
    }
    return found;
}

struct UNetGraph {
    Graph* graph = nullptr;
    int sample = -1, timestep = -1, context = -1, noise_pred = -1;

    const std::vector<int64_t>& sample_shape() const { return graph->inputs[sample].shape; }
    const std::vector<int64_t>& context_shape() const { return graph->inputs[context].shape; }

    bool matches(const int64_t x_shape[4], const int64_t ctx_shape[3]) const {
        return std::equal(sample_shape().begin(), sample_shape().end(), x_shape) &&
               std::equal(context_shape().begin(), context_shape().end(), ctx_shape);
    }
};

class QnnUNetImpl : public QnnUNet {
public:
    std::unique_ptr<Context> ctx;
    std::vector<UNetGraph> graphs;
    std::string desc;
    std::string prediction;

    bool predict(const float* x,
                 const int64_t x_shape[4],
                 const float* timesteps,
                 const float* context,
                 const int64_t ctx_shape[3],
                 float* out) override {
        const UNetGraph* g = nullptr;
        for (const auto& candidate : graphs) {
            if (candidate.matches(x_shape, ctx_shape)) {
                g = &candidate;
                break;
            }
        }
        if (g == nullptr) {
            std::string available;
            for (const auto& candidate : graphs) {
                available += " " + shape_str(candidate.sample_shape()) + "/" + shape_str(candidate.context_shape());
            }
            LOG_ERROR("qnn unet: no graph for sample %s, context %s; the binary has:%s",
                      sidecar_shape_str(x_shape, 4).c_str(), sidecar_shape_str(ctx_shape, 3).c_str(), available.c_str());
            return false;
        }

        std::vector<const float*> in(g->graph->inputs.size());
        in[g->sample]   = x;
        in[g->timestep] = timesteps;
        in[g->context]  = context;
        std::vector<float*> outs(g->graph->outputs.size(), nullptr);
        outs[g->noise_pred] = out;
        return g->graph->execute(in, outs, "qnn unet");
    }

    std::string get_desc() override {
        return desc;
    }

    std::string prediction_type() override {
        return prediction;
    }

    int64_t batch_size() override {
        return graphs[0].sample_shape()[0];
    }
};

}  // namespace

std::unique_ptr<QnnUNet> QnnUNet::load(const std::string& path) {
    auto t0  = std::chrono::steady_clock::now();
    auto ctx = Context::load(path, "qnn unet");
    if (!ctx) {
        return nullptr;
    }

    auto impl = std::make_unique<QnnUNetImpl>();
    for (auto& graph : ctx->graphs) {
        UNetGraph g;
        g.graph      = graph.get();
        g.sample     = graph->input_index("sample");
        g.timestep   = graph->input_index("timestep");
        g.context    = graph->input_index("encoder_hidden_states");
        g.noise_pred = graph->output_index("noise_pred");
        if (g.sample < 0 || g.timestep < 0 || g.context < 0 || g.noise_pred < 0 || graph->inputs.size() != 3) {
            std::string names;
            for (const auto& t : graph->inputs) {
                names += " " + t.name;
            }
            names += " ->";
            for (const auto& t : graph->outputs) {
                names += " " + t.name;
            }
            LOG_ERROR("qnn unet: graph %s of %s has I/O%s; expected sample, timestep, encoder_hidden_states -> "
                      "noise_pred (export it with script/qnn/convert_unet.py)",
                      graph->name.c_str(), path.c_str(), names.c_str());
            return nullptr;
        }
        const auto& s = g.sample_shape();
        const auto& c = g.context_shape();
        if (s.size() != 4 || c.size() != 3 || c[0] != s[0] || graph->inputs[g.timestep].elements() != s[0] ||
            graph->outputs[g.noise_pred].shape != s) {
            LOG_ERROR("qnn unet: graph %s of %s has sample %s, timestep %s, encoder_hidden_states %s, noise_pred %s; "
                      "expected [N, C, H, W], [N], [N, T, D] and [N, C, H, W]",
                      graph->name.c_str(), path.c_str(), shape_str(s).c_str(),
                      shape_str(graph->inputs[g.timestep].shape).c_str(), shape_str(c).c_str(),
                      shape_str(graph->outputs[g.noise_pred].shape).c_str());
            return nullptr;
        }
        if (!impl->graphs.empty() && s[0] != impl->graphs[0].sample_shape()[0]) {
            LOG_ERROR("qnn unet: the graphs of %s have different batch sizes; export one batch size per binary",
                      path.c_str());
            return nullptr;
        }
        const std::string pred = prediction_from_graph_name(graph->name);
        if (!impl->graphs.empty() && pred != impl->prediction) {
            LOG_ERROR("qnn unet: the graphs of %s record different prediction types", path.c_str());
            return nullptr;
        }
        impl->prediction = pred;
        impl->graphs.push_back(g);
    }

    std::string sizes, types;
    for (const auto& g : impl->graphs) {
        sizes += " " + std::to_string(g.sample_shape()[3] * 8) + "x" + std::to_string(g.sample_shape()[2] * 8);
    }
    const UNetGraph& first = impl->graphs[0];
    types = first.graph->inputs[first.sample].type_name() + " in, " + first.graph->outputs[first.noise_pred].type_name() + " out";
    impl->desc = "QNN UNet (" + ctx->summary + ")";
    impl->ctx  = std::move(ctx);
    LOG_INFO("qnn unet: %s ready in %.2f s, %zu graph(s), WxH:%s, batch %lld, %s, prediction %s",
             path.c_str(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
             impl->graphs.size(), sizes.c_str(), (long long)impl->batch_size(), types.c_str(),
             impl->prediction.empty() ? "not recorded" : impl->prediction.c_str());
    return impl;
}

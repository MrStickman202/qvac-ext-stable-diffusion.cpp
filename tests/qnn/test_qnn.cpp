// Tests for the QNN sidecars (SD_QNN builds).
//
// Unit tests run the tiny context binaries from tests/qnn/make_fixtures.py through QnnUNet and QnnTAEDecoder.
// On x86_64 Linux they run on the HTP simulator that ships with the QAIRT SDK (libQnnHtp.so for x86_64), so no
// phone is needed; on a phone (adb shell, with SD_QNN_LIB_DIR set) they run on the NPU.
// Pipeline tests run only when SD_TEST_SDXS_DIR points at the SDXS artifacts from script/qnn/export_artifacts.py
// (sdxs-f16.gguf, sdxs_tae_decoder.bin, and optionally sdxs_unet_fp16.bin).
//
//   python tests/qnn/make_fixtures.py build/qnn-fixtures
//   LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang build/bin/test-qnn build/qnn-fixtures

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "src/qnn_tae.h"
#include "src/qnn_unet.h"
#include "stable-diffusion.h"

namespace {

int g_failures = 0;
std::string g_log;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                          \
        }                                                                          \
    } while (0)

void log_cb(sd_log_level_t level, const char* text, void*) {
    g_log += text;
    if (level >= SD_LOG_WARN) {
        fputs(text, stderr);
    }
}

bool log_has(const char* text) {
    return g_log.find(text) != std::string::npos;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

constexpr int64_t C = 4, H = 8, W = 8, T = 5, D = 6;

std::vector<float> ramp(size_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = offset + scale * (float)((i * 37) % 101) / 101.0f;
    }
    return v;
}

// What the fixture UNet computes (see make_fixtures.py), for row-major [N, C, H, W] and [N, T, D] inputs.
std::vector<float> reference_unet(const std::vector<float>& x, const std::vector<float>& t, const std::vector<float>& ctx, int64_t n) {
    const int64_t per = (int64_t)x.size() / n;
    std::vector<float> out(x.size());
    for (int64_t b = 0; b < n; b++) {
        double c = 0;
        for (int64_t i = 0; i < T; i++) {
            for (int64_t j = 0; j < D; j++) {
                c += ctx[(b * T + i) * D + j] * (0.1 * (i + 1) + 0.01 * (j + 1));
            }
        }
        for (int64_t k = 0; k < per; k++) {
            out[b * per + k] = 0.5f * x[b * per + k] + (float)(c + 0.001 * t[b]);
        }
    }
    return out;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = a.size() == b.size() ? 0.0f : INFINITY;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        m = std::fmax(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// The NPU computes these graphs in fp16 (or w8a16), so results match the float64 reference to fp16 precision.
void test_unet_variants(const std::string& dir) {
    const int64_t x_shape[4]   = {1, C, H, W};
    const int64_t ctx_shape[3] = {1, T, D};
    auto x                     = ramp(C * H * W, 2.0f, -1.0f);
    auto ctx                   = ramp(T * D, 1.0f, -0.5f);
    std::vector<float> t       = {999.0f};
    auto expected              = reference_unet(x, t, ctx, 1);

    // fp16 graph I/O (converted by the runtime), float32 I/O, and 16-bit fixed-point I/O with 8-bit weights.
    for (auto [name, tol] : {std::pair<const char*, float>{"unet_b1", 1e-2f},
                             {"unet_b1_f32io", 1e-2f},
                             {"unet_b1_w8a16", 6e-2f}}) {
        auto unet = QnnUNet::load(dir + "/" + name + ".bin");
        CHECK(unet != nullptr);
        if (!unet) {
            continue;
        }
        std::vector<float> out(x.size(), NAN);
        CHECK(unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, out.data()));
        float err = max_abs_diff(out, expected);
        printf("  %s: max abs error %.2e\n", name, err);
        CHECK(err < tol);
        CHECK(unet->batch_size() == 1);
        CHECK(unet->prediction_type().empty());
        CHECK(unet->get_desc().find("QNN UNet") == 0);
    }
}

void test_unet_shape_rejection(const std::string& dir) {
    auto unet = QnnUNet::load(dir + "/unet_b1.bin");
    CHECK(unet != nullptr);
    if (!unet) {
        return;
    }
    std::vector<float> x(C * 16 * 16), ctx(T * D), out(C * 16 * 16), t = {1.0f};
    const int64_t big_x[4]     = {1, C, 16, 16};
    const int64_t ctx_shape[3] = {1, T, D};
    g_log.clear();
    CHECK(!unet->predict(x.data(), big_x, t.data(), ctx.data(), ctx_shape, out.data()));
    CHECK(log_has("no graph for sample [1, 4, 16, 16]"));
    const int64_t x_shape[4]  = {1, C, H, W};
    const int64_t long_ctx[3] = {1, T + 1, D};
    std::vector<float> long_data((T + 1) * D);
    CHECK(!unet->predict(x.data(), x_shape, t.data(), long_data.data(), long_ctx, out.data()));
}

void test_prediction_from_graph_name(const std::string& dir) {
    auto vpred = QnnUNet::load(dir + "/unet_b2_vpred.bin");
    CHECK(vpred != nullptr);
    if (vpred) {
        CHECK(vpred->prediction_type() == "v_prediction");
        CHECK(vpred->batch_size() == 2);
    }
    auto eps = QnnUNet::load(dir + "/unet_multi.bin");
    CHECK(eps != nullptr);
    if (eps) {
        CHECK(eps->prediction_type() == "epsilon");
    }
}

// A binary with one graph per size uses the graph that matches the latent.
void test_multi_graph(const std::string& dir) {
    auto unet = QnnUNet::load(dir + "/unet_multi.bin");
    CHECK(unet != nullptr);
    if (!unet) {
        return;
    }
    const int64_t ctx_shape[3] = {1, T, D};
    auto ctx                   = ramp(T * D, 1.0f, -0.5f);
    std::vector<float> t       = {250.0f};
    for (auto [h, w] : {std::pair<int64_t, int64_t>{H, W}, {12, 16}}) {
        const int64_t x_shape[4] = {1, C, h, w};
        auto x                   = ramp(C * h * w, 2.0f, -1.0f);
        std::vector<float> out(x.size(), NAN);
        CHECK(unet->predict(x.data(), x_shape, t.data(), ctx.data(), ctx_shape, out.data()));
        float err = max_abs_diff(out, reference_unet(x, t, ctx, 1));
        printf("  %lldx%lld latent: max abs error %.2e\n", (long long)h, (long long)w, err);
        CHECK(err < 1e-2f);
    }
}

// Batched CFG (one batch-2 call) must give the same outputs as separate conditional and unconditional calls.
void test_cfg_batch(const std::string& dir) {
    auto batched = QnnUNet::load(dir + "/unet_b2_vpred.bin");
    auto single  = QnnUNet::load(dir + "/unet_b1.bin");
    CHECK(batched && single);
    if (!batched || !single) {
        return;
    }
    const int64_t x_shape[4]   = {1, C, H, W};
    const int64_t ctx_shape[3] = {1, T, D};
    auto x                     = ramp(C * H * W, 2.0f, -1.0f);
    auto cond                  = ramp(T * D, 1.0f, -0.5f);
    auto uncond                = ramp(T * D, -0.3f, 0.2f);
    float t                    = 500.0f;
    std::vector<float> out_cond(x.size(), NAN), out_uncond(x.size(), NAN);
    CHECK(batched->predict_cfg_batch(x.data(), x_shape, t, cond.data(), uncond.data(), ctx_shape,
                                     out_cond.data(), out_uncond.data()));

    for (auto [context, batched_out] : {std::pair{&cond, &out_cond}, std::pair{&uncond, &out_uncond}}) {
        std::vector<float> separate(x.size(), NAN);
        CHECK(single->predict(x.data(), x_shape, &t, context->data(), ctx_shape, separate.data()));
        float err = max_abs_diff(*batched_out, separate);
        printf("  batched vs separate CFG: max abs difference %.2e\n", err);
        CHECK(err < 1e-2f);
        CHECK(max_abs_diff(*batched_out, reference_unet(x, {t}, *context, 1)) < 1e-2f);
    }

    // Only a batch-1 latent can be batched, and a batch-1 model cannot take both passes.
    const int64_t x2_shape[4] = {2, C, H, W};
    std::vector<float> x2(2 * x.size());
    CHECK(!batched->predict_cfg_batch(x2.data(), x2_shape, t, cond.data(), uncond.data(), ctx_shape,
                                      out_cond.data(), out_uncond.data()));
    CHECK(!single->predict_cfg_batch(x.data(), x_shape, t, cond.data(), uncond.data(), ctx_shape,
                                     out_cond.data(), out_uncond.data()));
}

// Files that are not usable sidecars fail at load with a message, not at the first generation.
void test_bad_binaries(const std::string& dir) {
    g_log.clear();
    CHECK(QnnUNet::load(dir + "/missing.bin") == nullptr);
    CHECK(log_has("cannot read"));
    g_log.clear();
    CHECK(QnnUNet::load(dir + "/not_a_binary.bin") == nullptr);
    CHECK(log_has("is not a QNN context binary"));
    g_log.clear();
    CHECK(QnnUNet::load(dir + "/unet_bad_names.bin") == nullptr);
    CHECK(log_has("expected sample, timestep, encoder_hidden_states -> noise_pred"));
    CHECK(QnnUNet::load(dir + "/tae.bin") == nullptr);
    CHECK(QnnTAEDecoder::load(dir + "/unet_b1.bin") == nullptr);
}

void test_tae(const std::string& dir) {
    auto tae = QnnTAEDecoder::load(dir + "/tae.bin");
    CHECK(tae != nullptr);
    if (tae) {
        const int64_t latent_shape[4] = {1, C, H, W};
        const int64_t image_shape[4]  = {1, 3, 8 * H, 8 * W};
        auto latent                   = ramp(C * H * W, 1.0f, 0.0f);
        std::vector<float> image(3 * 64 * H * W, NAN);
        CHECK(tae->decode(latent.data(), latent_shape, image.data(), image_shape));
        // Upsampling copies values, so the only error is the fp16 rounding of the graph's I/O.
        float err = 0;
        for (int64_t c = 0; c < 3; c++) {
            for (int64_t y = 0; y < 8 * H; y++) {
                for (int64_t x = 0; x < 8 * W; x++) {
                    err = std::fmax(err, std::fabs(image[(c * 8 * H + y) * 8 * W + x] - latent[(c * H + y / 8) * W + x / 8]));
                }
            }
        }
        printf("  tae: max abs error %.2e\n", err);
        CHECK(err < 1e-3f);

        // A destination of a different size is rejected before anything is written.
        const int64_t small_image[4] = {1, 3, 4 * H, 4 * W};
        std::vector<float> small(3 * 16 * H * W, -7.0f);
        CHECK(!tae->decode(latent.data(), latent_shape, small.data(), small_image));
        CHECK(small[0] == -7.0f);
        const int64_t other_latent[4] = {1, C, 2 * H, 2 * W};
        std::vector<float> big_latent(C * 4 * H * W);
        CHECK(!tae->decode(big_latent.data(), other_latent, image.data(), image_shape));
    }

    // A model whose output is not [N, 3, 8h, 8w] is rejected at load.
    CHECK(QnnTAEDecoder::load(dir + "/tae_wrong_scale.bin") == nullptr);
}

// ---- pipeline tests with the real SDXS artifacts ----

// A one-tensor LoRA for an SD 1.x UNet layer (SDXS keeps the SD 1.x names), in safetensors format.
std::string write_unet_lora(const std::string& path) {
    const std::string name = "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight";
    const size_t n         = 320 * 4;
    std::string header = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[320,4,1,1],\"data_offsets\":[0," +
                         std::to_string(n * 4) + "]}}";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream f(path, std::ios::binary);
    uint64_t len = header.size();
    f.write((const char*)&len, 8);
    f.write(header.data(), header.size());
    std::vector<float> data(n, 0.01f);
    f.write((const char*)data.data(), n * 4);
    return path;
}

std::string g_unet_path;  // empty when the SDXS UNet binary is not there: the UNet then runs on ggml

sd_ctx_params_t sdxs_params(const std::string& dir) {
    static std::string model, tae;
    model = dir + "/sdxs-f16.gguf";
    tae   = dir + "/sdxs_tae_decoder.bin";
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.model_path              = model.c_str();
    p.qnn_unet_path           = g_unet_path.c_str();
    p.qnn_tae_decoder_path    = tae.c_str();
    p.n_threads               = 4;
    p.free_params_immediately = false;  // each context generates more than once
    return p;
}

sd_image_t* generate(sd_ctx_t* sd, const sd_lora_t* loras, uint32_t n_loras, bool tiled) {
    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt                         = "a red apple on a table";
    g.width                          = 512;
    g.height                         = 512;
    g.seed                           = 42;
    g.sample_params.sample_steps     = 1;
    g.sample_params.guidance.txt_cfg = 1.0f;
    g.loras                          = loras;
    g.lora_count                     = n_loras;
    g.vae_tiling_params.enabled      = tiled;
    return generate_image(sd, &g);
}

void free_image(sd_image_t* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

void test_pipeline(const std::string& dir) {
    if (file_exists(dir + "/sdxs_unet_fp16.bin")) {
        g_unet_path = dir + "/sdxs_unet_fp16.bin";
    }
    printf("  UNet on %s\n", g_unet_path.empty() ? "ggml (no sdxs_unet_fp16.bin)" : "QNN");

    // Circular padding cannot be applied to fixed QNN graphs.
    for (bool tae_only : {false, true}) {
        if (!tae_only && g_unet_path.empty()) {
            continue;
        }
        sd_ctx_params_t p = sdxs_params(dir);
        if (tae_only) {
            p.qnn_unet_path = "";
        } else {
            p.qnn_tae_decoder_path = "";
        }
        p.circular_x = true;
        g_log.clear();
        sd_ctx_t* sd = new_sd_ctx(&p);
        CHECK(sd == nullptr);
        CHECK(log_has("does not support circular padding"));
        if (sd) {
            free_sd_ctx(sd);
        }
    }

    // A CoreML path and a QNN path for the same part are rejected.
    {
        sd_ctx_params_t p         = sdxs_params(dir);
        p.coreml_tae_decoder_path = "sdxs_tae_decoder.mlmodelc";
        g_log.clear();
        sd_ctx_t* sd = new_sd_ctx(&p);
        CHECK(sd == nullptr);
        CHECK(log_has("are both set"));
        if (sd) {
            free_sd_ctx(sd);
        }
    }

    // With VAE tiling the ggml tiny decoder runs instead of the QNN one, and gives the same image.
    sd_ctx_params_t p = sdxs_params(dir);
    sd_ctx_t* sd      = new_sd_ctx(&p);
    CHECK(sd != nullptr);
    if (sd) {
        sd_image_t* qnn   = generate(sd, nullptr, 0, false);
        sd_image_t* tiled = generate(sd, nullptr, 0, true);
        CHECK(qnn && tiled);
        if (qnn && tiled) {
            size_t n   = (size_t)qnn->width * qnn->height * qnn->channel;
            double mse = 0;
            for (size_t i = 0; i < n; i++) {
                double d = (double)qnn->data[i] - tiled->data[i];
                mse += d * d;
            }
            double psnr = 10 * std::log10(255.0 * 255.0 / std::fmax(mse / n, 1e-12));
            printf("  QNN vs tiled ggml tiny decoder: PSNR %.1f dB\n", psnr);
            CHECK(psnr > 25.0);
        }
        free_image(qnn);
        free_image(tiled);
        free_sd_ctx(sd);
    }

    // A LoRA that changes the UNet is rejected in both apply modes.
    if (!g_unet_path.empty()) {
        std::string lora_path = write_unet_lora(std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/sd_qnn_test_lora.safetensors");
        for (lora_apply_mode_t mode : {LORA_APPLY_IMMEDIATELY, LORA_APPLY_AT_RUNTIME}) {
            sd_ctx_params_t lp = sdxs_params(dir);
            lp.lora_apply_mode = mode;
            sd_ctx_t* lsd      = new_sd_ctx(&lp);
            CHECK(lsd != nullptr);
            if (!lsd) {
                continue;
            }
            sd_lora_t lora = {false, 1.0f, lora_path.c_str()};
            g_log.clear();
            sd_image_t* img = generate(lsd, &lora, 1, false);
            CHECK(img == nullptr);
            CHECK(log_has("cannot take LoRAs"));
            free_image(img);
            free_sd_ctx(lsd);
        }
        remove(lora_path.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE_DIR (from tests/qnn/make_fixtures.py)\n", argv[0]);
        return 2;
    }
    sd_set_log_callback(log_cb, nullptr);
    const std::string dir = argv[1];
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"unet variants", [&] { test_unet_variants(dir); }},
        {"unet shape rejection", [&] { test_unet_shape_rejection(dir); }},
        {"prediction from graph name", [&] { test_prediction_from_graph_name(dir); }},
        {"multi-graph binary", [&] { test_multi_graph(dir); }},
        {"batched cfg", [&] { test_cfg_batch(dir); }},
        {"bad binaries", [&] { test_bad_binaries(dir); }},
        {"tae", [&] { test_tae(dir); }},
    };
    for (auto& [name, fn] : tests) {
        printf("%s\n", name);
        fn();
    }
    if (const char* sdxs = getenv("SD_TEST_SDXS_DIR")) {
        printf("pipeline (SD_TEST_SDXS_DIR=%s)\n", sdxs);
        test_pipeline(sdxs);
    } else {
        printf("pipeline: skipped (set SD_TEST_SDXS_DIR to the SDXS artifacts to run it)\n");
    }
    printf(g_failures ? "FAILED: %d checks\n" : "all tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}

// JNI bridge between the app (SdNative.kt) and sd.cpp. One generation context at a time: load() builds it,
// generate() runs text-to-image with it, release() frees it. The QNN sidecars find their libraries in the app's
// native library directory, where Gradle packs libQnnHtp.so, libQnnSystem.so and the HTP stub/skel libraries.
#include <android/log.h>
#include <jni.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "stable-diffusion.h"

namespace {

const char* TAG = "sdnpu";

std::mutex g_mutex;
std::string g_errors;  // ERROR lines since the last load() or generate()
double g_text_ms = 0, g_unet_ms = 0, g_decode_ms = 0;

double number_after(const char* text, const char* key) {
    const char* p = strstr(text, key);
    return p ? atof(p + strlen(key)) : -1;
}

void log_cb(enum sd_log_level_t level, const char* text, void*) {
    int prio = level == SD_LOG_ERROR ? ANDROID_LOG_ERROR
               : level == SD_LOG_WARN ? ANDROID_LOG_WARN
               : level == SD_LOG_INFO ? ANDROID_LOG_INFO
                                      : ANDROID_LOG_DEBUG;
    __android_log_write(prio, TAG, text);
    double v;
    if ((v = number_after(text, "get_learned_condition completed, taking ")) >= 0) {
        g_text_ms = v;  // ms
    } else if ((v = number_after(text, "sampling completed, taking ")) >= 0) {
        g_unet_ms = v * 1000;  // s
    } else if ((v = number_after(text, "decode_first_stage completed, taking ")) >= 0) {
        g_decode_ms = v * 1000;  // s
    }
    // The diffusers loader always probes text_encoder_2/ (SDXL's second encoder), which SD 1.x/SDXS do not have.
    if (level == SD_LOG_ERROR && strstr(text, "text_encoder_2/") == nullptr) {
        // Strip sd.cpp's "[ERROR] file.cpp:123  - " prefix for the message shown in the app.
        const char* msg = strstr(text, " - ");
        g_errors += msg ? msg + 3 : text;
    }
}

std::string str(JNIEnv* env, jstring s) {
    if (s == nullptr) {
        return "";
    }
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_io_tether_sdnpu_SdNative_load(JNIEnv* env, jclass, jstring lib_dir, jstring model,
                                                           jstring qnn_unet, jstring qnn_tae, jint threads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_errors.clear();
    sd_set_log_callback(log_cb, nullptr);

    // The QNN libraries live in the app's native library directory. ADSP_LIBRARY_PATH is where the NPU side looks
    // for libQnnHtpV*Skel.so; the system folders after it are the defaults it would use otherwise.
    const std::string dir = str(env, lib_dir);
    setenv("SD_QNN_LIB_DIR", dir.c_str(), 1);
    const std::string adsp = dir + ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);

    const std::string model_path = str(env, model);
    const std::string unet_path  = str(env, qnn_unet);
    const std::string tae_path   = str(env, qnn_tae);

    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.model_path              = model_path.c_str();
    p.qnn_unet_path           = unet_path.c_str();
    p.qnn_tae_decoder_path    = tae_path.c_str();
    p.n_threads               = threads > 0 ? threads : sd_get_num_physical_cores();
    p.enable_mmap             = true;   // mapped weights do not count against the app's heap
    p.free_params_immediately = false;  // the context generates many images
    sd_ctx_t* sd = new_sd_ctx(&p);
    if (sd == nullptr && g_errors.empty()) {
        g_errors = "failed to load the model";
    }
    return (jlong)(intptr_t)sd;
}

JNIEXPORT jbyteArray JNICALL Java_io_tether_sdnpu_SdNative_generate(JNIEnv* env, jclass, jlong handle, jstring prompt,
                                                                    jstring negative, jlong seed, jint steps,
                                                                    jfloat cfg, jint width, jint height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* sd = (sd_ctx_t*)(intptr_t)handle;
    g_errors.clear();
    g_text_ms = g_unet_ms = g_decode_ms = 0;
    if (sd == nullptr) {
        g_errors = "the model is not loaded";
        return nullptr;
    }
    const std::string prompt_s   = str(env, prompt);
    const std::string negative_s = str(env, negative);

    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt                         = prompt_s.c_str();
    g.negative_prompt                = negative_s.c_str();
    g.width                          = width;
    g.height                         = height;
    g.seed                           = seed;
    g.batch_count                    = 1;
    g.sample_params.sample_steps     = steps;
    g.sample_params.guidance.txt_cfg = cfg;
    g.sample_params.sample_method    = sd_get_default_sample_method(sd);
    g.sample_params.scheduler        = sd_get_default_scheduler(sd, g.sample_params.sample_method);

    sd_image_t* img = generate_image(sd, &g);
    if (img == nullptr || img->data == nullptr) {
        if (g_errors.empty()) {
            g_errors = "generation failed";
        }
        free(img);
        return nullptr;
    }
    const jsize n   = (jsize)img->width * img->height * img->channel;
    jbyteArray out  = env->NewByteArray(n);
    env->SetByteArrayRegion(out, 0, n, (const jbyte*)img->data);
    free(img->data);
    free(img);
    return out;
}

// {text encoder ms, UNet ms, decoder ms} of the last generate().
JNIEXPORT jdoubleArray JNICALL Java_io_tether_sdnpu_SdNative_timings(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    jdoubleArray out = env->NewDoubleArray(3);
    const jdouble v[3] = {g_text_ms, g_unet_ms, g_decode_ms};
    env->SetDoubleArrayRegion(out, 0, 3, v);
    return out;
}

JNIEXPORT jstring JNICALL Java_io_tether_sdnpu_SdNative_lastError(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_errors.c_str());
}

JNIEXPORT void JNICALL Java_io_tether_sdnpu_SdNative_release(JNIEnv*, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (handle != 0) {
        free_sd_ctx((sd_ctx_t*)(intptr_t)handle);
    }
}

}  // extern "C"

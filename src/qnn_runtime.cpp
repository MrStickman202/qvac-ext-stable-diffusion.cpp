#include "qnn_runtime.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnProperty.h"
#include "ggml.h"
#include "util.h"

namespace sd_qnn {

namespace {

typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t*** providers, uint32_t* n);
typedef Qnn_ErrorHandle_t (*GetSystemProvidersFn)(const QnnSystemInterface_t*** providers, uint32_t* n);

const char* env_or(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return (v != nullptr && *v != '\0') ? v : fallback;
}

void qnn_log_callback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    (void)timestamp;
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);
    size_t n = strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
        buf[--n] = '\0';
    }
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            LOG_ERROR("qnn: %s", buf);
            break;
        case QNN_LOG_LEVEL_WARN:
            LOG_WARN("qnn: %s", buf);
            break;
        case QNN_LOG_LEVEL_INFO:
            LOG_INFO("qnn: %s", buf);
            break;
        default:
            LOG_DEBUG("qnn: %s", buf);
            break;
    }
}

QnnLog_Level_t log_level_from_env() {
    const char* v = env_or("SD_QNN_LOG", "warn");
    if (strcmp(v, "error") == 0) return QNN_LOG_LEVEL_ERROR;
    if (strcmp(v, "warn") == 0) return QNN_LOG_LEVEL_WARN;
    if (strcmp(v, "info") == 0) return QNN_LOG_LEVEL_INFO;
    if (strcmp(v, "verbose") == 0) return QNN_LOG_LEVEL_VERBOSE;
    if (strcmp(v, "debug") == 0) return QNN_LOG_LEVEL_DEBUG;
    LOG_WARN("SD_QNN_LOG=%s is not one of error, warn, info, verbose, debug; using warn", v);
    return QNN_LOG_LEVEL_WARN;
}

size_t element_size(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_FLOAT_32:
            return 4;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
            return 2;
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
            return 1;
        default:
            return 0;  // not supported as a graph input or output
    }
}

// The fields shared by every Qnn_Tensor_t version (V2 starts with V1's fields).
const Qnn_TensorV1_t& v1(const Qnn_Tensor_t& t) {
    return t.v1;
}
Qnn_TensorV1_t& v1(Qnn_Tensor_t& t) {
    return t.v1;
}

template <typename Q>
void quantize(const float* src, Q* dst, int64_t n, float scale, int32_t offset) {
    const double lo = (double)std::numeric_limits<Q>::min();
    const double hi = (double)std::numeric_limits<Q>::max();
    for (int64_t i = 0; i < n; i++) {
        double q = std::nearbyint((double)src[i] / scale) - offset;
        dst[i]   = (Q)std::min(hi, std::max(lo, q));
    }
}

template <typename Q>
void dequantize(const Q* src, float* dst, int64_t n, float scale, int32_t offset) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = scale * (float)((int64_t)src[i] + offset);
    }
}

void to_device(const TensorInfo& t, const float* src, uint8_t* dst) {
    const int64_t n = t.elements();
    switch (t.type) {
        case QNN_DATATYPE_FLOAT_32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case QNN_DATATYPE_FLOAT_16:
            ggml_fp32_to_fp16_row(src, (ggml_fp16_t*)dst, n);
            break;
        case QNN_DATATYPE_UFIXED_POINT_8:
            quantize(src, (uint8_t*)dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_SFIXED_POINT_8:
            quantize(src, (int8_t*)dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_UFIXED_POINT_16:
            quantize(src, (uint16_t*)dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_SFIXED_POINT_16:
            quantize(src, (int16_t*)dst, n, t.scale, t.offset);
            break;
        default:
            break;  // rejected at load
    }
}

void from_device(const TensorInfo& t, const uint8_t* src, float* dst) {
    const int64_t n = t.elements();
    switch (t.type) {
        case QNN_DATATYPE_FLOAT_32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case QNN_DATATYPE_FLOAT_16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)src, dst, n);
            break;
        case QNN_DATATYPE_UFIXED_POINT_8:
            dequantize((const uint8_t*)src, dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_SFIXED_POINT_8:
            dequantize((const int8_t*)src, dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_UFIXED_POINT_16:
            dequantize((const uint16_t*)src, dst, n, t.scale, t.offset);
            break;
        case QNN_DATATYPE_SFIXED_POINT_16:
            dequantize((const int16_t*)src, dst, n, t.scale, t.offset);
            break;
        default:
            break;
    }
}

// A context binary mapped into memory (read-only), unmapped when done.
struct MappedFile {
    void* data  = nullptr;
    size_t size = 0;

    bool open(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size == 0) {
            ::close(fd);
            return false;
        }
        size = (size_t)st.st_size;
        // MAP_PRIVATE: QnnSystemContext_getBinaryInfo takes a non-const pointer, but does not need to write the file.
        data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (data == MAP_FAILED) {
            data = nullptr;
            return false;
        }
        return true;
    }

    ~MappedFile() {
        if (data != nullptr) {
            munmap(data, size);
        }
    }
};

}  // namespace

// One QNN backend for the process: the libraries, the backend and device handles, and the HTP power vote.
// The UNet and decoder sidecars share it; it is released when the last context using it is freed.
class Backend {
public:
    QNN_INTERFACE_VER_TYPE qnn{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
    uint32_t backend_id         = 0;
    std::string backend_name;
    Qnn_LogHandle_t log         = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device   = nullptr;
    void* backend_lib           = nullptr;
    void* system_lib            = nullptr;
    QnnHtpDevice_PerfInfrastructure_t perf{};
    uint32_t power_config_id = 0;
    bool has_power_config    = false;

    static std::shared_ptr<Backend> get() {
        static std::mutex mutex;
        static std::weak_ptr<Backend> current;
        std::lock_guard<std::mutex> lock(mutex);
        std::shared_ptr<Backend> b = current.lock();
        if (b) {
            return b;
        }
        b = std::make_shared<Backend>();
        if (!b->init()) {
            return nullptr;
        }
        current = b;
        return b;
    }

    ~Backend() {
        if (has_power_config && perf.destroyPowerConfigId != nullptr) {
            perf.destroyPowerConfigId(power_config_id);
        }
        if (device != nullptr && qnn.deviceFree != nullptr) {
            qnn.deviceFree(device);
        }
        if (backend != nullptr && qnn.backendFree != nullptr) {
            qnn.backendFree(backend);
        }
        if (log != nullptr && qnn.logFree != nullptr) {
            qnn.logFree(log);
        }
        // The libraries stay loaded: the HTP library keeps threads and FastRPC sessions that do not
        // always survive being unloaded and reloaded in one process.
    }

private:
    static std::string lib_path(const char* name) {
        const char* dir = getenv("SD_QNN_LIB_DIR");
        if (dir == nullptr || *dir == '\0' || strchr(name, '/') != nullptr) {
            return name;
        }
        return std::string(dir) + "/" + name;
    }

    bool init() {
        const char* dir = getenv("SD_QNN_LIB_DIR");
        if (dir != nullptr && *dir != '\0' && getenv("ADSP_LIBRARY_PATH") == nullptr) {
            // Where the Hexagon side looks for libQnnHtpV*Skel.so; must be set before the HTP library starts.
            setenv("ADSP_LIBRARY_PATH", dir, 0);
        }

        const std::string backend_path = lib_path(env_or("SD_QNN_BACKEND", "libQnnHtp.so"));
        backend_lib                    = dlopen(backend_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (backend_lib == nullptr) {
            LOG_ERROR("qnn: cannot load %s: %s (put the QNN libraries next to the app's native libraries, "
                      "or set SD_QNN_LIB_DIR)",
                      backend_path.c_str(), dlerror());
            return false;
        }
        auto get_providers = (GetProvidersFn)dlsym(backend_lib, "QnnInterface_getProviders");
        const QnnInterface_t** providers = nullptr;
        uint32_t n_providers             = 0;
        if (get_providers == nullptr || get_providers(&providers, &n_providers) != QNN_SUCCESS || n_providers == 0) {
            LOG_ERROR("qnn: %s has no QNN interface providers", backend_path.c_str());
            return false;
        }
        const QnnInterface_t* chosen = nullptr;
        for (uint32_t i = 0; i < n_providers; i++) {
            const Qnn_Version_t& v = providers[i]->apiVersion.coreApiVersion;
            if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
                chosen = providers[i];
                break;
            }
        }
        if (chosen == nullptr) {
            const Qnn_Version_t& v = providers[0]->apiVersion.coreApiVersion;
            LOG_ERROR("qnn: %s implements QNN API %u.%u, this build needs %d.%d or newer: use the libraries from the "
                      "QAIRT SDK version sd.cpp was built with, or a newer one",
                      backend_path.c_str(), v.major, v.minor, QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
            return false;
        }
        qnn          = chosen->QNN_INTERFACE_VER_NAME;
        backend_id   = chosen->backendId;
        backend_name = chosen->providerName ? chosen->providerName : backend_path;

        const std::string system_path = lib_path("libQnnSystem.so");
        system_lib                    = dlopen(system_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (system_lib == nullptr) {
            LOG_ERROR("qnn: cannot load %s: %s", system_path.c_str(), dlerror());
            return false;
        }
        auto get_system = (GetSystemProvidersFn)dlsym(system_lib, "QnnSystemInterface_getProviders");
        const QnnSystemInterface_t** sys_providers = nullptr;
        uint32_t n_sys                             = 0;
        if (get_system == nullptr || get_system(&sys_providers, &n_sys) != QNN_SUCCESS || n_sys == 0) {
            LOG_ERROR("qnn: %s has no QNN system interface providers", system_path.c_str());
            return false;
        }
        const QnnSystemInterface_t* sys_chosen = nullptr;
        for (uint32_t i = 0; i < n_sys; i++) {
            const Qnn_Version_t& v = sys_providers[i]->systemApiVersion;
            if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
                sys_chosen = sys_providers[i];
                break;
            }
        }
        if (sys_chosen == nullptr) {
            LOG_ERROR("qnn: %s is older than the QNN system API this build needs (%d.%d)", system_path.c_str(),
                      QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
            return false;
        }
        sys = sys_chosen->QNN_SYSTEM_INTERFACE_VER_NAME;

        if (qnn.logCreate != nullptr && qnn.logCreate(qnn_log_callback, log_level_from_env(), &log) != QNN_SUCCESS) {
            log = nullptr;  // run without QNN's own log lines
        }
        if (qnn.backendCreate(log, nullptr, &backend) != QNN_SUCCESS) {
            LOG_ERROR("qnn: failed to create the %s backend", backend_name.c_str());
            return false;
        }

        // The HTP device on Android; the x86 HTP simulator and some other backends have no device API.
        if (qnn.propertyHasCapability != nullptr &&
            qnn.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE) == QNN_PROPERTY_SUPPORTED) {
            if (qnn.deviceCreate(log, nullptr, &device) != QNN_SUCCESS) {
                LOG_WARN("qnn: failed to create a device; continuing without one");
                device = nullptr;
            }
        }
        if (backend_id == QNN_BACKEND_ID_HTP && device != nullptr) {
            set_htp_performance();
        }
        LOG_INFO("qnn: %s backend ready (QNN API %u.%u)", backend_name.c_str(),
                 chosen->apiVersion.coreApiVersion.major, chosen->apiVersion.coreApiVersion.minor);
        return true;
    }

    // Votes for NPU clocks. Without a vote the HTP runs at whatever DCVS picks, which is often a low corner
    // for the short bursts a single UNet call makes.
    void set_htp_performance() {
        const char* mode = env_or("SD_QNN_PERF", "burst");
        if (strcmp(mode, "default") == 0) {
            return;
        }
        const bool burst = strcmp(mode, "burst") == 0;
        const bool high  = strcmp(mode, "high") == 0;
        if (!burst && !high && strcmp(mode, "balanced") != 0) {
            LOG_WARN("SD_QNN_PERF=%s is not one of burst, high, balanced, default; using burst", mode);
            return set_htp_performance_mode(true, false);
        }
        set_htp_performance_mode(burst, high);
    }

    void set_htp_performance_mode(bool burst, bool high) {
        QnnDevice_Infrastructure_t infra = nullptr;
        if (qnn.deviceGetInfrastructure == nullptr || qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || infra == nullptr) {
            LOG_WARN("qnn: no HTP performance infrastructure; NPU clocks are left to the driver");
            return;
        }
        auto* htp_infra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
        if (htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            return;
        }
        perf = htp_infra->perfInfra;
        if (perf.createPowerConfigId == nullptr || perf.setPowerConfig == nullptr ||
            perf.createPowerConfigId(0, 0, &power_config_id) != QNN_SUCCESS) {
            LOG_WARN("qnn: cannot create an HTP power config; NPU clocks are left to the driver");
            return;
        }
        has_power_config = true;

        const auto corner = burst  ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER
                            : high ? DCVS_VOLTAGE_VCORNER_TURBO
                                   : DCVS_VOLTAGE_VCORNER_NOM;

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        dcvs.option                                 = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        auto& d                                     = dcvs.dcvsV3Config;
        d.contextId                                 = power_config_id;
        d.setDcvsEnable                             = 1;
        d.dcvsEnable                                = (burst || high) ? 0 : 1;
        d.powerMode                                 = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        d.setSleepLatency                           = 1;
        d.sleepLatency                              = burst ? 40 : 100;  // us
        d.setSleepDisable                           = 1;
        d.sleepDisable                              = burst ? 1 : 0;
        d.setBusParams                              = 1;
        d.busVoltageCornerMin                       = corner;
        d.busVoltageCornerTarget                    = corner;
        d.busVoltageCornerMax                       = corner;
        d.setCoreParams                             = 1;
        d.coreVoltageCornerMin                      = corner;
        d.coreVoltageCornerTarget                   = corner;
        d.coreVoltageCornerMax                      = corner;

        QnnHtpPerfInfrastructure_PowerConfig_t latency = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        latency.option                  = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        latency.rpcControlLatencyConfig = 100;  // us

        QnnHtpPerfInfrastructure_PowerConfig_t polling = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        polling.option               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        polling.rpcPollingTimeConfig = burst ? 9999 : 0;  // us; polling keeps the CPU awake between calls

        const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &latency, &polling, nullptr};
        if (perf.setPowerConfig(power_config_id, configs) != QNN_SUCCESS) {
            LOG_WARN("qnn: the HTP rejected the %s power config", burst ? "burst" : (high ? "high" : "balanced"));
            return;
        }
        LOG_INFO("qnn: HTP performance mode %s", burst ? "burst" : (high ? "high" : "balanced"));
    }
};

int64_t TensorInfo::elements() const {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

std::string TensorInfo::type_name() const {
    switch (type) {
        case QNN_DATATYPE_FLOAT_32:
            return "float32";
        case QNN_DATATYPE_FLOAT_16:
            return "float16";
        case QNN_DATATYPE_UFIXED_POINT_8:
            return "ufixed8";
        case QNN_DATATYPE_SFIXED_POINT_8:
            return "sfixed8";
        case QNN_DATATYPE_UFIXED_POINT_16:
            return "ufixed16";
        case QNN_DATATYPE_SFIXED_POINT_16:
            return "sfixed16";
        default:
            return "type 0x" + std::to_string((int)type);
    }
}

std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        s += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

int Graph::input_index(const std::string& n) const {
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].name == n) {
            return (int)i;
        }
    }
    return -1;
}

int Graph::output_index(const std::string& n) const {
    for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i].name == n) {
            return (int)i;
        }
    }
    return -1;
}

bool Graph::execute(const std::vector<const float*>& in, const std::vector<float*>& out, const char* tag) {
    if (in.size() != inputs.size() || out.size() != outputs.size()) {
        LOG_ERROR("%s: graph %s takes %zu inputs and %zu outputs", tag, name.c_str(), inputs.size(), outputs.size());
        return false;
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        to_device(inputs[i], in[i], in_bufs[i].data());
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        // The backend may shrink dataSize for dynamic outputs; restore it on every call.
        v1(out_tensors[i]).clientBuf.dataSize = (uint32_t)out_bufs[i].size();
    }
    auto t0 = std::chrono::steady_clock::now();
    Qnn_ErrorHandle_t err = backend->qnn.graphExecute(handle, in_tensors.data(), (uint32_t)in_tensors.size(),
                                                      out_tensors.data(), (uint32_t)out_tensors.size(), nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (err != QNN_SUCCESS) {
        LOG_ERROR("%s: graph %s failed to execute (QNN error %lu)", tag, name.c_str(), (unsigned long)err);
        return false;
    }
    LOG_DEBUG("%s: graph %s took %.2f ms", tag, name.c_str(), std::chrono::duration<double, std::milli>(t1 - t0).count());
    for (size_t i = 0; i < outputs.size(); i++) {
        if (out[i] != nullptr) {
            from_device(outputs[i], out_bufs[i].data(), out[i]);
        }
    }
    return true;
}

std::unique_ptr<Context> Context::load(const std::string& path, const char* tag) {
    auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<Backend> b = Backend::get();
    if (!b) {
        return nullptr;
    }

    MappedFile file;
    if (!file.open(path)) {
        LOG_ERROR("%s: cannot read %s", tag, path.c_str());
        return nullptr;
    }

    QnnSystemContext_Handle_t sys_ctx = nullptr;
    if (b->sys.systemContextCreate(&sys_ctx) != QNN_SUCCESS) {
        LOG_ERROR("%s: failed to create a QNN system context", tag);
        return nullptr;
    }
    struct SysCtxGuard {
        Backend* b;
        QnnSystemContext_Handle_t h;
        ~SysCtxGuard() { b->sys.systemContextFree(h); }
    } sys_guard{b.get(), sys_ctx};

    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t info_size         = 0;
    if (b->sys.systemContextGetBinaryInfo(sys_ctx, file.data, file.size, &info, &info_size) != QNN_SUCCESS || info == nullptr) {
        LOG_ERROR("%s: %s is not a QNN context binary (export it with script/qnn/convert_unet.py or "
                  "qnn-context-binary-generator)",
                  tag, path.c_str());
        return nullptr;
    }

    uint32_t num_graphs                          = 0;
    const QnnSystemContext_GraphInfo_t* g_infos  = nullptr;
    const char* soc_version                      = nullptr;
    switch (info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            num_graphs  = info->contextBinaryInfoV1.numGraphs;
            g_infos     = info->contextBinaryInfoV1.graphs;
            soc_version = info->contextBinaryInfoV1.socVersion;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            num_graphs  = info->contextBinaryInfoV2.numGraphs;
            g_infos     = info->contextBinaryInfoV2.graphs;
            soc_version = info->contextBinaryInfoV2.socVersion;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            num_graphs  = info->contextBinaryInfoV3.numGraphs;
            g_infos     = info->contextBinaryInfoV3.graphs;
            soc_version = info->contextBinaryInfoV3.socVersion;
            break;
        default:
            LOG_ERROR("%s: %s uses context binary info version %d, which this build does not know; rebuild sd.cpp "
                      "with the QAIRT SDK that produced it",
                      tag, path.c_str(), (int)info->version);
            return nullptr;
    }
    if (num_graphs == 0) {
        LOG_ERROR("%s: %s holds no graphs", tag, path.c_str());
        return nullptr;
    }

    auto ctx     = std::unique_ptr<Context>(new Context());
    ctx->backend = b;
    ctx->soc     = soc_version ? soc_version : "";

    // Describe every graph before creating the context, so a binary with unsupported I/O fails fast.
    for (uint32_t gi = 0; gi < num_graphs; gi++) {
        const QnnSystemContext_GraphInfo_t& gsrc = g_infos[gi];
        // All graph info versions start with name, inputs and outputs.
        const QnnSystemContext_GraphInfoV1_t& g = gsrc.graphInfoV1;
        auto graph                             = std::make_unique<Graph>();
        graph->backend                         = b.get();
        graph->name                            = g.graphName ? g.graphName : "";

        const uint32_t n_tensors = g.numGraphInputs + g.numGraphOutputs;
        graph->names.reserve(n_tensors);
        graph->dims.reserve(n_tensors);
        graph->dynamic_dims.reserve(n_tensors);

        auto add = [&](const Qnn_Tensor_t& src, std::vector<TensorInfo>& infos, std::vector<Qnn_Tensor_t>& tensors,
                       std::vector<std::vector<uint8_t>>& bufs) -> bool {
            const Qnn_TensorV1_t& s = v1(src);
            TensorInfo t;
            t.name = s.name ? s.name : "";
            t.type = s.dataType;
            for (uint32_t d = 0; d < s.rank; d++) {
                t.shape.push_back(s.dimensions[d]);
            }
            if (src.version == QNN_TENSOR_VERSION_2 && src.v2.isDynamicDimensions != nullptr) {
                for (uint32_t d = 0; d < s.rank; d++) {
                    if (src.v2.isDynamicDimensions[d]) {
                        LOG_ERROR("%s: %s: tensor %s of graph %s has dynamic dimensions; export fixed shapes", tag,
                                  path.c_str(), t.name.c_str(), graph->name.c_str());
                        return false;
                    }
                }
            }
            if (element_size(t.type) == 0) {
                LOG_ERROR("%s: %s: tensor %s of graph %s is %s; graph inputs and outputs must be float32, float16 "
                          "or 8/16-bit fixed point",
                          tag, path.c_str(), t.name.c_str(), graph->name.c_str(), t.type_name().c_str());
                return false;
            }
            const bool fixed = t.type == QNN_DATATYPE_UFIXED_POINT_8 || t.type == QNN_DATATYPE_SFIXED_POINT_8 ||
                               t.type == QNN_DATATYPE_UFIXED_POINT_16 || t.type == QNN_DATATYPE_SFIXED_POINT_16;
            if (fixed) {
                const Qnn_QuantizeParams_t& q = s.quantizeParams;
                if (q.encodingDefinition != QNN_DEFINITION_DEFINED ||
                    q.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    LOG_ERROR("%s: %s: tensor %s of graph %s is quantized per axis or per block; graph inputs and "
                              "outputs need one scale and offset",
                              tag, path.c_str(), t.name.c_str(), graph->name.c_str());
                    return false;
                }
                t.scale  = q.scaleOffsetEncoding.scale;
                t.offset = q.scaleOffsetEncoding.offset;
            }

            // Our own copy of the tensor description: the system context's memory goes away after loading.
            graph->names.push_back(t.name);
            graph->dims.emplace_back(s.dimensions, s.dimensions + s.rank);
            Qnn_Tensor_t dst = src;
            Qnn_TensorV1_t& d = v1(dst);
            d.name            = graph->names.back().c_str();
            d.dimensions      = graph->dims.back().data();
            if (dst.version == QNN_TENSOR_VERSION_2) {
                if (src.v2.isDynamicDimensions != nullptr) {
                    graph->dynamic_dims.emplace_back(s.rank, 0);
                    dst.v2.isDynamicDimensions = graph->dynamic_dims.back().data();
                }
                dst.v2.sparseParams = QNN_SPARSE_PARAMS_INIT;
            }
            bufs.emplace_back(t.elements() * element_size(t.type));
            d.memType           = QNN_TENSORMEMTYPE_RAW;
            d.clientBuf.data    = bufs.back().data();
            d.clientBuf.dataSize = (uint32_t)bufs.back().size();
            tensors.push_back(dst);
            infos.push_back(t);
            return true;
        };

        graph->in_bufs.reserve(g.numGraphInputs);
        graph->out_bufs.reserve(g.numGraphOutputs);
        for (uint32_t i = 0; i < g.numGraphInputs; i++) {
            if (!add(g.graphInputs[i], graph->inputs, graph->in_tensors, graph->in_bufs)) {
                return nullptr;
            }
        }
        for (uint32_t i = 0; i < g.numGraphOutputs; i++) {
            if (!add(g.graphOutputs[i], graph->outputs, graph->out_tensors, graph->out_bufs)) {
                return nullptr;
            }
        }
        ctx->graphs.push_back(std::move(graph));
    }

    if (b->qnn.contextCreateFromBinary(b->backend, b->device, nullptr, file.data, file.size, &ctx->handle, nullptr) !=
        QNN_SUCCESS) {
        LOG_ERROR("%s: failed to load %s on %s%s%s: check that it was compiled for this device's NPU (convert_unet.py "
                  "--soc) and that the QNN libraries come from the QAIRT SDK version that compiled it",
                  tag, path.c_str(), b->backend_name.c_str(), ctx->soc.empty() ? "" : "; it was compiled for SoC ",
                  ctx->soc.c_str());
        ctx->handle = nullptr;
        return nullptr;
    }
    for (auto& graph : ctx->graphs) {
        if (b->qnn.graphRetrieve(ctx->handle, graph->name.c_str(), &graph->handle) != QNN_SUCCESS) {
            LOG_ERROR("%s: cannot retrieve graph %s from %s", tag, graph->name.c_str(), path.c_str());
            return nullptr;
        }
    }

    ctx->summary = b->backend_name + ", " + std::to_string(ctx->graphs.size()) + " graph(s)" +
                   (ctx->soc.empty() ? "" : ", built for SoC " + ctx->soc);
    LOG_INFO("%s: loaded %s in %.2f s (%s)", tag, path.c_str(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ctx->summary.c_str());
    return ctx;
}

Context::~Context() {
    if (handle != nullptr && backend) {
        backend->qnn.contextFree(handle, nullptr);
    }
}

}  // namespace sd_qnn

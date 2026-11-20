#ifndef __QNN_RUNTIME_H__
#define __QNN_RUNTIME_H__

// Shared runtime for the QNN sidecars (qnn_unet.cpp, qnn_tae.cpp): loads the QNN libraries at run time,
// creates one backend and device for the process, and runs the graphs of QNN context binaries with float32
// host buffers, converting to and from the types the graphs take (float32, float16, 8/16-bit fixed point).
//
// Nothing from the QNN SDK is linked: libQnnHtp.so and libQnnSystem.so are opened with dlopen. On Android
// they are found through the app's native library directory or LD_LIBRARY_PATH; SD_QNN_LIB_DIR names a
// directory to load them from instead, and also becomes ADSP_LIBRARY_PATH (where the Hexagon side looks for
// libQnnHtpV79Skel.so) unless that is already set.
//
// Environment:
//   SD_QNN_LIB_DIR  directory holding libQnnHtp.so, libQnnSystem.so, the HTP stub and skel libraries
//   SD_QNN_BACKEND  backend library (default libQnnHtp.so)
//   SD_QNN_PERF     HTP performance mode: burst (default) | high | balanced | default (leave the NPU's voting alone)
//   SD_QNN_LOG      QNN log level: error | warn (default) | info | verbose | debug

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace sd_qnn {

class Backend;

// An input or output of a graph, as the context binary declares it.
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    Qnn_DataType_t type = QNN_DATATYPE_UNDEFINED;
    float scale         = 1.0f;  // fixed-point types: real = scale * (q + offset)
    int32_t offset      = 0;

    int64_t elements() const;
    std::string type_name() const;
};

// One graph of a loaded context binary, retrieved and ready to execute.
class Graph {
public:
    std::string name;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;

    // Index of the input or output with this name, or -1.
    int input_index(const std::string& name) const;
    int output_index(const std::string& name) const;

    // Runs the graph. in[i] holds inputs[i].elements() float32 values and out[i] receives outputs[i].elements();
    // a null out[i] skips that output.
    bool execute(const std::vector<const float*>& in, const std::vector<float*>& out, const char* tag);

private:
    friend class Context;
    Backend* backend         = nullptr;
    Qnn_GraphHandle_t handle = nullptr;
    // Tensor descriptions handed to graphExecute; they point into the storage below.
    std::vector<Qnn_Tensor_t> in_tensors, out_tensors;
    std::vector<std::string> names;  // owned copies of tensor names
    std::vector<std::vector<uint32_t>> dims;
    std::vector<std::vector<uint8_t>> dynamic_dims;
    std::vector<std::vector<uint8_t>> in_bufs, out_bufs;  // device-typed staging buffers
};

// A loaded QNN context binary (.bin from qnn-context-binary-generator or Qualcomm AI Hub) and its graphs.
class Context {
public:
    static std::unique_ptr<Context> load(const std::string& path, const char* tag);
    ~Context();

    std::vector<std::unique_ptr<Graph>> graphs;
    std::string soc;      // SoC the binary was generated for, as the binary records it
    std::string summary;  // backend, SoC and graph count, for log lines

private:
    std::shared_ptr<Backend> backend;
    Qnn_ContextHandle_t handle = nullptr;
};

std::string shape_str(const std::vector<int64_t>& shape);

}  // namespace sd_qnn

#endif  // __QNN_RUNTIME_H__

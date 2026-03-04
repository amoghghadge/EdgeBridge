#include "GemmaEngine.hpp"
#include "MetalSampler.hpp" // Our custom Zero-Copy GPU shader bridge

#include "tflite/model.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include <iostream>

// --- THE SHIELD ---
// We manually declare the C-API signatures to bypass Objective-C header conflicts
extern "C" {
    struct TfLiteDelegate;
    
    // 1. Metal GPU Options
    struct TFLGpuDelegateOptions {
        bool allow_precision_loss;
        int32_t wait_type;
        bool enable_quantization;
    };
    TFLGpuDelegateOptions TFLGpuDelegateOptionsDefault();
    TfLiteDelegate* TFLGpuDelegateCreate(const TFLGpuDelegateOptions* options);
    void TFLGpuDelegateDelete(TfLiteDelegate* delegate);

    // 2. Core ML NPU Options
    struct TfLiteCoreMlDelegateOptions {
        int enabled_devices;
        int coreml_version;
        int max_delegated_partitions;
        int min_nodes_per_partition;
    };
    TfLiteDelegate* TfLiteCoreMlDelegateCreate(const TfLiteCoreMlDelegateOptions* options);
    void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);
}
// ------------------

GemmaEngine::GemmaEngine() {}

GemmaEngine::~GemmaEngine() {
    if (hardware_delegate) {
        if (current_backend == 1) {
            TFLGpuDelegateDelete(static_cast<TfLiteDelegate*>(hardware_delegate));
        } else if (current_backend == 2) {
            TfLiteCoreMlDelegateDelete(static_cast<TfLiteDelegate*>(hardware_delegate));
        }
    }
    if (interpreter) { delete static_cast<tflite::Interpreter*>(interpreter); }
    if (model) { delete static_cast<tflite::FlatBufferModel*>(model); }
}

bool GemmaEngine::loadModel(const std::string& modelPath, int backend) {
    current_backend = backend;
    
    // Use mmap to map the file into virtual memory (bypasses Jetsam limits)
    auto mmap_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!mmap_model) {
        std::cerr << "Failed to mmap model at: " << modelPath << std::endl;
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    
    // 1. Initialize the Builder first
    tflite::InterpreterBuilder builder(*mmap_model, resolver);
    
    // 2. Attach the Hardware Delegates to the BUILDER
    if (backend == 1) {
        TFLGpuDelegateOptions gpu_opts = TFLGpuDelegateOptionsDefault();
        gpu_opts.allow_precision_loss = true;
        gpu_opts.enable_quantization = true;
        
        TfLiteDelegate* delegate = TFLGpuDelegateCreate(&gpu_opts);
        builder.AddDelegate(delegate);
        hardware_delegate = delegate;
        std::cout << "Metal (GPU) Delegate Attached!" << std::endl;
        
    } else if (backend == 2) {
        TfLiteCoreMlDelegateOptions coreml_opts = {0, 0, 0, 0};
        TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&coreml_opts);
        builder.AddDelegate(delegate);
        hardware_delegate = delegate;
        std::cout << "Core ML (NPU) Delegate Attached!" << std::endl;
    } else {
        std::cout << "Running on standard CPU." << std::endl;
    }

    // 3. Build the interpreter with the delegates baked in
    builder(&local_interpreter);
    
    if (!local_interpreter) { return false; }

    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    model = mmap_model.release();
    interpreter = local_interpreter.release();
    return true;
}

std::string GemmaEngine::testBridge() const {
    return "Hello from the C++ Engine!";
}

// ------------------------------------------------------------------
// Custom GPU Sampling Pipeline (Bypassing the CPU Bottleneck)
// ------------------------------------------------------------------
int GemmaEngine::generateNextToken() {
    if (!interpreter) return -1;
    
    tflite::Interpreter* local_interpreter = static_cast<tflite::Interpreter*>(interpreter);
    
    // 1. Execute the heavy math (this runs on CPU XNNPACK due to the STABLEHLO error)
    if (local_interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter." << std::endl;
        return -1;
    }
    
    // 2. Get the output tensor (the massive 250k logits array)
    int output_tensor_index = local_interpreter->outputs()[0];
    TfLiteTensor* output_tensor = local_interpreter->tensor(output_tensor_index);
    float* logits = output_tensor->data.f;
    
    // Gemma 3 1B vocabulary size
    int vocab_size = 256000;
    
    // 3. Bypass the CPU and force the GPU to pick the winner
    MetalSampler sampler;
    int winning_token_id = sampler.sampleToken(logits, vocab_size);
    
    std::cout << "GPU Selected Token ID: " << winning_token_id << std::endl;
    
    return winning_token_id;
}

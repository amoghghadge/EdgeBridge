#include "GemmaEngine.hpp"

// All the heavy Google headers are safely hidden in here
#include "tflite/model.h"
#include "tflite/interpreter.h"
#include <iostream>

GemmaEngine::GemmaEngine() {}

// We manually cast the void pointers back to their true forms to delete them cleanly
GemmaEngine::~GemmaEngine() {
    if (interpreter) {
        delete static_cast<tflite::Interpreter*>(interpreter);
    }
    if (model) {
        delete static_cast<tflite::FlatBufferModel*>(model);
    }
}

bool GemmaEngine::loadModel(const std::string& modelPath) {
    // BuildFromFile natively executes the mmap() syscall on iOS.
    auto mmap_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    
    if (!mmap_model) {
        std::cerr << "Failed to mmap model at: " << modelPath << std::endl;
        return false;
    }
    
    // Release the unique_ptr and save it into our void* shield
    model = mmap_model.release();
    
    std::cout << "Successfully memory-mapped Gemma 3!" << std::endl;
    return true;
}

std::string GemmaEngine::testBridge() const {
    return "Hello from the C++ Engine!";
}

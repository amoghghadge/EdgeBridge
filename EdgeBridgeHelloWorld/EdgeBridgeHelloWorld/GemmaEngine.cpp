//
//  GemmaEngine.cpp
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

#include "GemmaEngine.hpp"
#include <iostream>

GemmaEngine::GemmaEngine() {
    // We will initialize the interpreter here in Step 2
}

bool GemmaEngine::loadModel(const std::string& modelPath) {
    // BuildFromFile implicitly executes the mmap() syscall under the hood.
    // It maps the FlatBuffer weights directly from the iOS file system without copying them into the heap.
    model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    
    if (!model) {
        std::cerr << "Failed to mmap model at: " << modelPath << std::endl;
        return false;
    }
    
    std::cout << "Successfully memory-mapped Gemma 3!" << std::endl;
    return true;
}

std::string GemmaEngine::testBridge() const {
    return "Hello from the C++ Engine!";
}

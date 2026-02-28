//
//  GemmaEngine.hpp
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

#ifndef GemmaEngine_hpp
#define GemmaEngine_hpp

#include <stdio.h>
#include <string>
#include <memory>
// Import the LiteRT model definition
#include "tensorflow/lite/model.h"

class GemmaEngine {
private:
    // We use a unique_ptr to automatically handle memory cleanup if the engine is destroyed
    std::unique_ptr<tflite::FlatBufferModel> model;
    
public:
    // Constructor
    GemmaEngine();
    
    // Our new function to safely memory-map the model
    bool loadModel(const std::string& modelPath);
    
    // A simple test function we will call from Swift
    std::string testBridge() const;
};

#endif /* GemmaEngine_hpp */

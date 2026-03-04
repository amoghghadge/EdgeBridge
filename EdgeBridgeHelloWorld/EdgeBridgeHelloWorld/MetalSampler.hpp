//
//  MetalSampler.hpp
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 3/3/26.
//

#ifndef MetalSampler_hpp
#define MetalSampler_hpp

#include <vector>
#include <cstdint>

class MetalSampler {
public:
    MetalSampler();
    ~MetalSampler();
    
    // Takes the raw float pointer from the LiteRT tensor and the size of the vocabulary
    int sampleToken(const float* logits, int vocab_size);

private:
    // Opaque pointer to hide Objective-C instances from C++
    void* mtl_context;
};

#endif /* MetalSampler_hpp */

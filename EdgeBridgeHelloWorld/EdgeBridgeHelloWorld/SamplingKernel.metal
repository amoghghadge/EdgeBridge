//
//  SamplingKernel.metal
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 3/3/26.
//

#include <metal_stdlib>
using namespace metal;

// The GPU Compute Kernel
// It takes a massive array of floats (the logits) and outputs a single integer (the winning token)
kernel void argmax_sampling(device const float* logits [[buffer(0)]],
                            device int* output_token [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    
    // NOTE: For Gemma 3 1B, the vocabulary size is typically 256,000.
    // We assign one GPU thread to handle the entire reduction for simplicity in V1.
    // (In a production environment, this would use threadgroup memory for parallel reduction)
    if (id == 0) {
        int max_index = 0;
        float max_value = -INFINITY;
        
        // Scan the entire vocabulary array directly in GPU memory
        for (int i = 0; i < 256000; i++) {
            if (logits[i] > max_value) {
                max_value = logits[i];
                max_index = i;
            }
        }
        
        // Write the single winning token ID to the output buffer
        output_token[0] = max_index;
    }
}

//
//  MetalSampler.mm
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 3/3/26.
//

#import <Metal/Metal.h>
#include "MetalSampler.hpp"
#include <iostream>

// Internal Objective-C struct to hold our Apple-specific objects
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    id<MTLBuffer> outputBuffer;
};

MetalSampler::MetalSampler() {
    mtl_context = new MetalContext();
    MetalContext* ctx = static_cast<MetalContext*>(mtl_context);
    
    // 1. Grab the iPhone's physical GPU
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        return;
    }
    
    ctx->commandQueue = [ctx->device newCommandQueue];
    
    // 2. Load the custom shader we wrote in Step 1
    id<MTLLibrary> defaultLibrary = [ctx->device newDefaultLibrary];
    id<MTLFunction> samplingFunction = [defaultLibrary newFunctionWithName:@"argmax_sampling"];
    
    NSError* error = nil;
    ctx->pipelineState = [ctx->device newComputePipelineStateWithFunction:samplingFunction error:&error];
    
    if (error) {
        std::cerr << "Failed to create compute pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    // 3. Pre-allocate a tiny 4-byte buffer for the winning Token ID
    ctx->outputBuffer = [ctx->device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
}

MetalSampler::~MetalSampler() {
    MetalContext* ctx = static_cast<MetalContext*>(mtl_context);
    delete ctx;
}

int MetalSampler::sampleToken(const float* logits, int vocab_size) {
    MetalContext* ctx = static_cast<MetalContext*>(mtl_context);
    if (!ctx->pipelineState) return -1;
    
    // Create a GPU command buffer
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:ctx->pipelineState];
    
    // Wrap the raw C++ logits array in a Metal Buffer WITHOUT copying the memory (Zero-Copy)
    id<MTLBuffer> logitsBuffer = [ctx->device newBufferWithBytesNoCopy:(void*)logits
                                                                length:vocab_size * sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
    
    // Bind the buffers to the shader
    [computeEncoder setBuffer:logitsBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:ctx->outputBuffer offset:0 atIndex:1];
    
    // Dispatch the GPU threads
    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [computeEncoder endEncoding];
    
    // Execute and wait for the GPU to finish
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Read the single winning token ID back to the CPU
    int* result = (int*)[ctx->outputBuffer contents];
    return result[0];
}

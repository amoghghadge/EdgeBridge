#ifndef GemmaEngine_hpp
#define GemmaEngine_hpp

#include <string>

class GemmaEngine {
private:
    // We use "Opaque Pointers" (void*) to hide Google's typedefs from Swift
    void* model = nullptr;
    void* interpreter = nullptr;
    void* hardware_delegate = nullptr;
    
    // 0 = CPU, 1 = Metal (GPU), 2 = Core ML (NPU)
    int current_backend = 0;

public:
    GemmaEngine();
    ~GemmaEngine();
    
    bool loadModel(const std::string& modelPath, int backend);
    std::string testBridge() const;
    
    // Declare the generation method so the .cpp file can implement it
    int generateNextToken();
};

#endif /* GemmaEngine_hpp */

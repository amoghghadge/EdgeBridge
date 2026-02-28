#ifndef GemmaEngine_hpp
#define GemmaEngine_hpp

#include <string>

class GemmaEngine {
private:
    // We use "Opaque Pointers" (void*).
    // This completely hides the underlying Google types from Swift!
    void* model = nullptr;
    void* interpreter = nullptr;

public:
    GemmaEngine();
    ~GemmaEngine();
    
    bool loadModel(const std::string& modelPath);
    std::string testBridge() const;
};

#endif /* GemmaEngine_hpp */

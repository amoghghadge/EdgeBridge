//
//  EdgeBridgeEngine.hpp
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

#ifndef EdgeBridgeEngine_hpp
#define EdgeBridgeEngine_hpp

// ============================================================================
// EdgeBridgeEngine.hpp
//
// This is the C++ header that Swift will import directly via C++ interop.
// For now, it's a minimal skeleton that proves the interop works.
// We'll replace the stub implementations with real LiteRT-LM calls later.
// ============================================================================

#include <string>

// ---------------------------------------------------------------------------
// Backend enum — mirrors litert::lm::Backend from LiteRT-LM.
// We define our own for now so we can build without LiteRT-LM linked.
// Later, we'll switch this to use the real enum from the LiteRT-LM headers.
// ---------------------------------------------------------------------------
enum class InferenceBackend {
    CPU,
    GPU
};

// ---------------------------------------------------------------------------
// BenchmarkResult — captures performance metrics from a single inference run.
// Swift will be able to read these fields directly through C++ interop.
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    double prefill_time_ms;     // Time to process the input prompt
    double decode_time_ms;      // Time to generate the full response
    double tokens_per_second;   // Decode throughput
    int    tokens_generated;    // Number of output tokens
};

// ---------------------------------------------------------------------------
// EdgeBridgeEngine — the main interface between Swift and the C++ runtime.
//
// Swift calls methods on this class directly. No Objective-C wrappers needed.
// The class manages the lifecycle of the LiteRT-LM Engine and Conversation
// objects, handles backend switching, and will eventually contain the
// agentic loop and constrained decoder.
// ---------------------------------------------------------------------------
class EdgeBridgeEngine {
public:
    // Constructor: takes the path to a .litertlm model bundle and a backend.
    EdgeBridgeEngine(const std::string& model_path, InferenceBackend backend);
    
    // Destructor: cleans up Engine and Conversation objects.
    ~EdgeBridgeEngine();
    
    // Send a user message and get the model's response.
    // This is a blocking call — it returns when generation is complete.
    // Later, we'll add an async streaming version.
    std::string sendMessage(const std::string& user_message);
    
    // Switch the backend (CPU ↔ GPU). This tears down and recreates the
    // engine, which is an expensive operation — don't call it mid-conversation.
    void switchBackend(InferenceBackend new_backend);
    
    // Get the benchmark result from the most recent inference.
    BenchmarkResult getLastBenchmark() const;
    
    // Check if the engine is ready to accept messages.
    bool isReady() const;
    
    // Get the currently active backend.
    InferenceBackend getCurrentBackend() const;

private:
    std::string   m_model_path;
    InferenceBackend m_backend;
    bool          m_is_ready;
    BenchmarkResult m_last_benchmark;
    
    // These will become real LiteRT-LM pointers later:
    // std::unique_ptr<litert::lm::Engine> m_engine;
    // std::unique_ptr<litert::lm::Conversation> m_conversation;
    
    // Internal: (re)initializes the engine with the current backend setting.
    void initializeEngine();
};

#endif /* EdgeBridgeEngine_hpp */

// EdgeBridge C API — thin wrapper around LiteRT-LM Conversation API
// for integration into iOS/macOS apps via Xcode.
//
// This header is the ONLY LiteRT-LM header that Xcode needs to see.
// All complex C++ dependencies (absl, protobuf, nlohmann/json) stay
// hidden behind this C interface.

#ifndef LITERT_LM_RUNTIME_BRIDGE_LITERT_BRIDGE_API_H_
#define LITERT_LM_RUNTIME_BRIDGE_LITERT_BRIDGE_API_H_

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types — callers never see the C++ internals.
typedef void* LiteRTEngineHandle;
typedef void* LiteRTConversationHandle;

// Backend selection for inference hardware.
typedef enum {
  LITERT_BACKEND_CPU = 0,
  LITERT_BACKEND_GPU = 1,
} LiteRTBackend;

// Status codes returned by all API functions.
typedef enum {
  LITERT_OK = 0,
  LITERT_ERROR_INVALID_ARGUMENT = 1,
  LITERT_ERROR_MODEL_LOAD_FAILED = 2,
  LITERT_ERROR_INFERENCE_FAILED = 3,
  LITERT_ERROR_NOT_INITIALIZED = 4,
  LITERT_ERROR_UNKNOWN = 99,
} LiteRTStatus;

// Benchmark metrics from the most recent inference call.
typedef struct {
  float time_to_first_token_ms;
  float prefill_tokens_per_sec;
  float decode_tokens_per_sec;
  int prefill_token_count;
  int decode_token_count;
  float total_inference_ms;
} LiteRTBenchmarkInfo;

// Callback for streaming token-by-token output.
// - token: the next chunk of generated text (UTF-8, null-terminated).
//          NULL signals end-of-stream.
// - context: user-provided pointer passed through from the async call.
typedef void (*LiteRTStreamCallback)(const char* token, void* context);

// --- Engine lifecycle ---

// Create an Engine by loading a .litertlm model file.
// The engine is a heavyweight object holding model weights.
// Returns LITERT_OK on success, populating *engine_out.
LiteRTStatus litert_engine_create(
    const char* model_path,
    LiteRTBackend backend,
    LiteRTEngineHandle* engine_out);

// Destroy an Engine and free all associated resources.
void litert_engine_destroy(LiteRTEngineHandle engine);

// --- Conversation lifecycle ---

// Create a Conversation from an existing Engine.
// system_prompt may be NULL for no system instruction.
// tools_json may be NULL for no tool declarations;
// otherwise it should be a JSON array string of tool schemas.
LiteRTStatus litert_conversation_create(
    LiteRTEngineHandle engine,
    const char* system_prompt,
    const char* tools_json,
    LiteRTConversationHandle* conversation_out);

// Destroy a Conversation.
void litert_conversation_destroy(LiteRTConversationHandle conversation);

// --- Messaging ---

// Send a user message (blocking). Returns the complete model response.
// The returned string is owned by the library; caller must copy it
// before the next call or before destroying the conversation.
// If the model produces a tool call, the response will be a JSON
// string containing a "tool_calls" array.
LiteRTStatus litert_conversation_send(
    LiteRTConversationHandle conversation,
    const char* user_message,
    const char** response_out);

// Send a user message with streaming callback (non-blocking).
// The callback is invoked on a background thread for each token chunk.
// A NULL token signals completion. The callback must be thread-safe.
LiteRTStatus litert_conversation_send_async(
    LiteRTConversationHandle conversation,
    const char* user_message,
    LiteRTStreamCallback callback,
    void* context);

// Send a tool response back to the model after executing a tool call.
// tool_result_json should be a JSON string with the tool's output.
LiteRTStatus litert_conversation_send_tool_response(
    LiteRTConversationHandle conversation,
    const char* tool_result_json,
    const char** response_out);

// --- Metrics ---

// Retrieve benchmark info from the most recent inference.
LiteRTStatus litert_get_benchmark_info(
    LiteRTConversationHandle conversation,
    LiteRTBenchmarkInfo* info_out);

// --- Utility ---

// Wait for any outstanding async operation to complete.
// timeout_ms of 0 means wait indefinitely.
LiteRTStatus litert_engine_wait(LiteRTEngineHandle engine, int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif  // LITERT_LM_RUNTIME_BRIDGE_LITERT_BRIDGE_API_H_

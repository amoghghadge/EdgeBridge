//
//  EngineViewModel.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

import Foundation

// ============================================================================
// EngineViewModel.swift  (v4 — Real LiteRT-LM Integration)
//
// This ViewModel drives the SwiftUI chat interface by calling the C bridge
// functions defined in litert_bridge_api.h. Those C functions are compiled
// inside libLiteRTLM.a by Bazel and delegate to the full LiteRT-LM
// Conversation API (Engine, Conversation, Session) internally.
//
// The bridging header (EdgeBridge-Bridging-Header.h) imports the C header,
// making all litert_* functions directly callable from Swift with no
// additional imports needed.
//
// ARCHITECTURE:
//   SwiftUI ChatView
//     → EngineViewModel (this file)
//       → litert_engine_create()      [C function in libLiteRTLM.a]
//       → litert_conversation_create() [C function in libLiteRTLM.a]
//       → litert_conversation_send()   [C function in libLiteRTLM.a]
//         → Engine::CreateSession()    [C++ inside the archive]
//         → Conversation::SendMessage()[C++ inside the archive]
//         → XNNPACK / Metal inference  [deep inside the archive]
//
// STREAMING:
//   For the streaming path, we use litert_conversation_send_async() with
//   a C callback. Since C function pointers can't capture Swift closures,
//   we use the standard Unmanaged<> pattern to pass a Swift object as the
//   void* context parameter.
// ============================================================================

import SwiftUI
import Foundation

// MARK: - EngineViewModel

@Observable
class EngineViewModel {
    
    // -- Published State for SwiftUI (same interface as v3) --
    var messages: [ChatMessage] = []
    var isGenerating: Bool = false
    var isSwitchingBackend: Bool = false
    var isEngineReady: Bool = false
    var currentBackend: String = "CPU"
    var benchmarkText: String = ""
    var statusMessage: String = "No model loaded"
    
    // -- Internal State --
    // These are the opaque handles returned by the C API.
    // They wrap C++ unique_ptr<Engine> and unique_ptr<Conversation>
    // inside the static archive — we never see the C++ types.
    private var engineHandle: LiteRTEngineHandle?
    private var conversationHandle: LiteRTConversationHandle?
    private var currentModelPath: String?
    private var currentUseGPU: Bool = false
    
    // -----------------------------------------------------------------------
    // initialize — loads a .litertlm model and creates the engine.
    //
    // This is called from ChatView.onAppear. It runs the heavyweight
    // model loading on a background thread so the UI stays responsive.
    // Model loading involves reading the .litertlm file (potentially
    // hundreds of MB), setting up the XNNPACK/Metal executor, and
    // allocating KV-cache memory.
    // -----------------------------------------------------------------------
    func initialize(modelPath: String, useGPU: Bool) {
        // Don't re-initialize if already loaded with the same config.
        guard engineHandle == nil else { return }
        
        currentModelPath = modelPath
        currentUseGPU = useGPU
        statusMessage = "Loading model..."
        
        Task.detached { [self] in
            let backend: LiteRTBackend = useGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            // Step 1: Create the Engine (heavyweight — loads model weights).
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(modelPath, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    self.statusMessage = "Failed to load model (error \(engineStatus.rawValue))"
                    self.isEngineReady = false
                }
                return
            }
            
            // Step 2: Create a Conversation (lightweight — sets up prompt
            // template, tokenizer state, and optional tool declarations).
            var conversation: LiteRTConversationHandle?
            let convStatus = litert_conversation_create(
                engine,
                nil,    // No system prompt for now; add later for agentic mode.
                nil,    // No tool declarations for now; add later for tool calling.
                &conversation
            )
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    self.statusMessage = "Failed to create conversation (error \(convStatus.rawValue))"
                    self.isEngineReady = false
                }
                return
            }
            
            // Step 3: Update UI state on the main thread.
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                self.statusMessage = "Model loaded — ready to chat"
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // sendMessage — sends user input to the model and streams the response.
    //
    // Uses the async streaming path (litert_conversation_send_async) so
    // tokens appear in the UI as they're generated, giving the user
    // immediate feedback. Falls back to blocking send if streaming fails.
    // -----------------------------------------------------------------------
    func sendMessage(_ text: String) {
        guard let engine = engineHandle,
              let conversation = conversationHandle,
              isEngineReady else { return }
        
        // Add the user's message to the chat immediately.
        let userMessage = ChatMessage(role: .user, content: text)
        messages.append(userMessage)
        isGenerating = true
        
        // Create a placeholder for the assistant's streaming response.
        // We'll accumulate tokens into this and update the UI progressively.
        let assistantId = UUID()
        let placeholder = ChatMessage(role: .assistant, content: "", id: assistantId)
        messages.append(placeholder)
        
        Task.detached { [self] in
            // We use the streaming API with a C callback. The callback
            // receives each token chunk on a background thread, and we
            // dispatch to MainActor to update the UI.
            
            // StreamContext holds the accumulated text and a reference
            // back to the ViewModel so the C callback can update state.
            let context = StreamContext(
                viewModel: self,
                messageId: assistantId
            )
            
            // Retain the context so it survives across callback invocations.
            // We'll release it when streaming completes (nil token).
            let contextPtr = Unmanaged.passRetained(context).toOpaque()
            
            let status = litert_conversation_send_async(
                conversation,
                text,
                streamCallbackFunction,  // Global C-compatible function defined below.
                contextPtr
            )
            
            if status != LITERT_OK {
                // If async fails, fall back to blocking send.
                Unmanaged<StreamContext>.fromOpaque(contextPtr).release()
                self.fallbackBlockingSend(conversation: conversation, engine: engine, text: text, assistantId: assistantId)
                return
            }
            
            // Wait for the model to finish generating.
            _ = litert_engine_wait(engine, 600_000)  // 10 minute timeout.
            
            // Update benchmark info after generation completes.
            await MainActor.run {
                self.isGenerating = false
                self.updateBenchmark()
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // fallbackBlockingSend — used if async streaming isn't available.
    //
    // Calls litert_conversation_send() which blocks until the full
    // response is generated, then updates the UI all at once.
    // -----------------------------------------------------------------------
    private func fallbackBlockingSend(
        conversation: LiteRTConversationHandle,
        engine: LiteRTEngineHandle,
        text: String,
        assistantId: UUID
    ) {
        var responsePtr: UnsafePointer<CChar>?
        let status = litert_conversation_send(conversation, text, &responsePtr)
        
        Task { @MainActor in
            if status == LITERT_OK, let ptr = responsePtr {
                let response = String(cString: ptr)
                // Replace the placeholder message with the full response.
                if let idx = self.messages.firstIndex(where: { $0.id == assistantId }) {
                    self.messages[idx] = ChatMessage(
                        role: .assistant,
                        content: response,
                        id: assistantId
                    )
                }
            } else {
                if let idx = self.messages.firstIndex(where: { $0.id == assistantId }) {
                    self.messages[idx] = ChatMessage(
                        role: .assistant,
                        content: "[Error: inference failed with code \(status.rawValue)]",
                        id: assistantId
                    )
                }
            }
            self.isGenerating = false
            self.updateBenchmark()
        }
    }
    
    // -----------------------------------------------------------------------
    // toggleBackend — tears down the current engine and recreates it
    // with the opposite backend (CPU ↔ GPU).
    //
    // This is the real deal — it destroys the Conversation and Engine
    // C++ objects (freeing model weights from memory), then reloads
    // everything with the new backend configuration.
    // -----------------------------------------------------------------------
    func toggleBackend(useGPU: Bool) {
        guard let modelPath = currentModelPath else { return }
        
        isSwitchingBackend = true
        
        Task.detached { [self] in
            // Destroy existing conversation and engine.
            if let conv = self.conversationHandle {
                litert_conversation_destroy(conv)
            }
            if let eng = self.engineHandle {
                litert_engine_destroy(eng)
            }
            
            await MainActor.run {
                self.conversationHandle = nil
                self.engineHandle = nil
                self.isEngineReady = false
            }
            
            // Recreate with the new backend.
            let backend: LiteRTBackend = useGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(modelPath, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    self.statusMessage = "Failed to switch backend"
                    self.isSwitchingBackend = false
                }
                return
            }
            
            var conversation: LiteRTConversationHandle?
            let convStatus = litert_conversation_create(engine, nil, nil, &conversation)
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    self.statusMessage = "Failed to create conversation after backend switch"
                    self.isSwitchingBackend = false
                }
                return
            }
            
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.currentUseGPU = useGPU
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                self.isSwitchingBackend = false
                self.statusMessage = "Backend switched to \(self.currentBackend)"
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // updateBenchmark — retrieves timing metrics from the last inference.
    // -----------------------------------------------------------------------
    private func updateBenchmark() {
        guard let conversation = conversationHandle else { return }
        var info = LiteRTBenchmarkInfo()
        let status = litert_get_benchmark_info(conversation, &info)
        if status == LITERT_OK {
            // Only show non-zero metrics (the C struct may be zeroed
            // if benchmark data isn't available yet).
            if info.prefill_tokens_per_sec > 0 || info.decode_tokens_per_sec > 0 {
                benchmarkText = String(
                    format: "TTFT: %.0fms | Prefill: %.0f tok/s | Decode: %.1f tok/s",
                    info.time_to_first_token_ms,
                    info.prefill_tokens_per_sec,
                    info.decode_tokens_per_sec
                )
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // cleanup — called when the view disappears or the app terminates.
    // Ensures all C++ resources are properly freed.
    // -----------------------------------------------------------------------
    func cleanup() {
        if let conv = conversationHandle {
            litert_conversation_destroy(conv)
            conversationHandle = nil
        }
        if let eng = engineHandle {
            litert_engine_destroy(eng)
            engineHandle = nil
        }
        isEngineReady = false
    }
    
    deinit {
        cleanup()
    }
}

// MARK: - Streaming Callback Infrastructure

/// StreamContext holds state that the C streaming callback needs access to.
/// It's passed as an opaque void* pointer through the C API, then recovered
/// inside the callback using Unmanaged<>.
///
/// This is the standard pattern for bridging C callbacks to Swift — since
/// C function pointers can't capture context (no closures), you smuggle
/// a reference to a Swift object through the void* parameter.
class StreamContext {
    weak var viewModel: EngineViewModel?
    let messageId: UUID
    var accumulatedText: String = ""
    
    init(viewModel: EngineViewModel, messageId: UUID) {
        self.viewModel = viewModel
        self.messageId = messageId
    }
}

/// The global C-compatible callback function for streaming tokens.
///
/// This function has the signature `void (*)(const char*, void*)` matching
/// the LiteRTStreamCallback typedef. It's called on a background thread
/// by the LiteRT-LM engine for each generated token chunk.
///
/// - token: The next chunk of UTF-8 text, or NULL to signal end-of-stream.
/// - rawContext: An opaque pointer to our StreamContext object.
private let streamCallbackFunction: LiteRTStreamCallback = { token, rawContext in
    guard let rawContext = rawContext else { return }
    
    // Recover our Swift StreamContext from the opaque pointer.
    // We use takeUnretainedValue() during streaming (don't release yet)
    // and takeRetainedValue() on the final nil-token call (releases).
    if let token = token {
        let context = Unmanaged<StreamContext>.fromOpaque(rawContext).takeUnretainedValue()
        let chunk = String(cString: token)
        context.accumulatedText += chunk
        
        // Dispatch UI update to the main thread.
        let text = context.accumulatedText
        let msgId = context.messageId
        
        DispatchQueue.main.async {
            guard let vm = context.viewModel else { return }
            if let idx = vm.messages.firstIndex(where: { $0.id == msgId }) {
                vm.messages[idx] = ChatMessage(
                    role: .assistant,
                    content: text,
                    id: msgId
                )
            }
        }
    } else {
        // nil token = end of stream. Release the retained context.
        let context = Unmanaged<StreamContext>.fromOpaque(rawContext).takeRetainedValue()
        _ = context  // Allow ARC to clean up.
    }
}

// MARK: - ChatMessage

struct ChatMessage: Identifiable {
    let id: UUID
    let role: Role
    let content: String
    let timestamp = Date()
    
    /// Convenience initializer that auto-generates an ID.
    init(role: Role, content: String, id: UUID = UUID()) {
        self.id = id
        self.role = role
        self.content = content
    }
    
    enum Role {
        case user
        case assistant
        case toolResult
        case system
    }
}

// MARK: - Model Discovery Helper

/// Utility to find .litertlm model files in the app's Documents directory.
///
/// For development, you copy the model to the device via Xcode's
/// "Devices and Simulators" window or via the Files app (if document
/// sharing is enabled in Info.plist). For production, you'd download
/// from HuggingFace or Firebase on first launch.
struct ModelDiscovery {
    
    /// Returns the path to the first .litertlm file found in Documents,
    /// or nil if no model is available.
    static func findModel() -> String? {
        let docs = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: docs,
            includingPropertiesForKeys: nil
        )) ?? []
        
        return contents
            .first(where: { $0.pathExtension == "litertlm" })?
            .path
    }
    
    /// Returns all .litertlm files in Documents, for a model picker UI.
    static func findAllModels() -> [(name: String, path: String)] {
        let docs = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: docs,
            includingPropertiesForKeys: nil
        )) ?? []
        
        return contents
            .filter { $0.pathExtension == "litertlm" }
            .map { (name: $0.deletingPathExtension().lastPathComponent, path: $0.path) }
    }
}

//
//  EngineViewModel.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

import Foundation

// ============================================================================
// EngineViewModel.swift
//
// This is the Swift-side wrapper that imports the C++ EdgeBridgeEngine
// and exposes it as an @Observable class for SwiftUI to bind to.
//
// THE KEY LINE is `import EdgeBridgeCxx` — this imports the C++ module
// we defined in module.modulemap. Swift can then directly construct
// C++ objects, call C++ methods, and read C++ struct fields.
//
// No Objective-C bridging header. No @objc wrappers. No C shims.
// This is true zero-cost Swift/C++ interoperability.
//
// IMPORTANT NOTE ON C++ VALUE SEMANTICS:
// Swift treats imported C++ classes as value types (like structs).
// This means if you do `var copy = engine`, you get an independent copy.
// Mutating the copy does NOT affect the original. This is why we always
// operate on `self.engine` directly instead of binding it to a local
// variable. When we integrate real LiteRT-LM (which holds gigabytes of
// model weights), we'll need to refactor to a pointer/reference pattern
// to avoid accidentally copying the engine.
// ============================================================================

// ============================================================================
// EngineViewModel.swift
//
// Swift-side wrapper that imports the C++ EdgeBridgeEngine and exposes
// it as an @Observable class for SwiftUI to bind to.
//
// CHANGES in this version:
// - Split isLoading into isGenerating + isSwitchingBackend so the UI
//   can show appropriate feedback for each operation independently.
// ============================================================================

import SwiftUI
import EdgeBridgeCxx

@Observable
class EngineViewModel {
    
    // -- Published State for SwiftUI --
    var messages: [ChatMessage] = []
    var isGenerating: Bool = false          // True during inference only
    var isSwitchingBackend: Bool = false     // True during backend switch only
    var isEngineReady: Bool = false
    var currentBackend: String = "CPU"
    var benchmarkText: String = ""
    
    // -- The C++ Engine --
    private var engine: EdgeBridgeEngine?
    
    // -----------------------------------------------------------------------
    // initialize — creates the C++ engine with a model path and backend.
    // -----------------------------------------------------------------------
    func initialize(modelPath: String, useGPU: Bool) {
        let backend: InferenceBackend = useGPU ? .GPU : .CPU
        engine = EdgeBridgeEngine(std.string(modelPath), backend)
        isEngineReady = engine?.isReady() ?? false
        currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
    }
    
    // -----------------------------------------------------------------------
    // sendMessage — sends user input to the C++ engine and updates the UI.
    // -----------------------------------------------------------------------
    func sendMessage(_ text: String) {
        guard engine != nil, engine!.isReady() else { return }
        
        let userMessage = ChatMessage(role: .user, content: text)
        messages.append(userMessage)
        isGenerating = true   // <-- Only this flag, not a generic "loading"
        
        Task.detached { [self] in
            let response = self.engine!.sendMessage(std.string(text))
            let benchmark = self.engine!.getLastBenchmark()
            
            await MainActor.run {
                let assistantMessage = ChatMessage(
                    role: .assistant,
                    content: String(response)
                )
                self.messages.append(assistantMessage)
                self.isGenerating = false   // <-- Clear only this flag
                
                self.benchmarkText = String(
                    format: "Prefill: %.1fms | Decode: %.1fms | %.1f tok/s | %d tokens",
                    benchmark.prefill_time_ms,
                    benchmark.decode_time_ms,
                    benchmark.tokens_per_second,
                    benchmark.tokens_generated
                )
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // toggleBackend — switches between CPU and GPU.
    //
    // Uses isSwitchingBackend instead of isGenerating, so the chat area
    // won't show a "Generating..." spinner during a backend switch.
    // The header area shows its own small spinner instead.
    // -----------------------------------------------------------------------
    func toggleBackend(useGPU: Bool) {
        guard engine != nil else { return }
        
        isSwitchingBackend = true   // <-- Separate flag for backend switching
        let newBackend: InferenceBackend = useGPU ? .GPU : .CPU
        
        Task.detached { [self] in
            self.engine!.switchBackend(newBackend)
            
            await MainActor.run {
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                self.isSwitchingBackend = false   // <-- Clear only this flag
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    let timestamp = Date()
    
    enum Role {
        case user
        case assistant
        case toolResult
    }
}

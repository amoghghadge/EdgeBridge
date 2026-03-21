//
//  EngineViewModel.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// EngineViewModel.swift  (v6 — Dual-Model Cascade + Constrained Decoding)
//
// WHAT'S NEW IN v6:
//   - Smart Mode: automatic routing between two models
//     • Qwen2.5-3B (fine-tuned) → calendar tool-calling queries
//     • Gemma-3n-E2B → general conversation + constrained decoding demo
//   - Keyword-based router decides which model handles each message
//   - Load-on-demand: only one model in memory at a time
//   - Model attribution: each response tagged with which model produced it
//   - Constrained decoding A/B demo runs on Gemma (general-purpose model)
//   - Manual model picker still works for individual model selection
//
// ARCHITECTURE:
//   User message → Router (keyword match)
//     ├─ Calendar keywords → load Qwen if needed → agentic tool loop
//     └─ General query    → load Gemma if needed → direct response
//
// MEMORY: Only one model loaded at a time (~3GB). Switching takes ~10-20s
// for XNNPACK cache rebuild on first load, faster on subsequent loads.
// ============================================================================

import SwiftUI
import Foundation

// MARK: - Model Role

/// Which model is currently loaded or needed.
enum ModelRole: String {
    case calendar = "Calendar Agent"   // Qwen2.5-3B fine-tuned
    case general  = "General"          // Gemma-3n-E2B
    case standalone = "Standalone"     // Any manually-selected model
}

// MARK: - EngineViewModel

@Observable
class EngineViewModel {
    
    // -- Published State for SwiftUI --
    var messages: [ChatMessage] = []
    var isGenerating: Bool = false
    var isSwitchingModel: Bool = false
    var isEngineReady: Bool = false
    var currentBackend: String = "CPU"
    var benchmarkText: String = ""
    var statusMessage: String = "No model loaded"
    var currentModelName: String = ""
    
    /// Which model role is currently active.
    var activeModelRole: ModelRole = .standalone
    
    /// Smart Mode: when true, automatically routes between Qwen (calendar)
    /// and Gemma (general). When false, uses whatever model is manually selected.
    var smartModeEnabled: Bool = false
    
    /// Whether Smart Mode is available (both model files found on device).
    var smartModeAvailable: Bool = false
    
    // -- Internal State --
    private var engineHandle: LiteRTEngineHandle?
    private var conversationHandle: LiteRTConversationHandle?
    private var currentModelPath: String?
    
    /// Public accessor for the model picker checkmark.
    var loadedModelPath: String? { currentModelPath }
    private var currentUseGPU: Bool = false
    
    /// Incremented on every load/cleanup — stale background Tasks check this
    /// before setting handles, preventing race conditions on rapid model switches.
    private var loadGeneration: Int = 0
    
    // Smart Mode model paths (discovered at startup).
    private var calendarModelPath: String?  // Qwen2.5-3B fine-tuned
    private var generalModelPath: String?   // Gemma-3n-E2B
    
    // The calendar tool executor — interfaces with real EventKit.
    private let calendarExecutor = CalendarToolExecutor()
    
    // Whether the current model has tool calling enabled.
    var toolCallingEnabled: Bool = false
    
    // Maximum number of tool-call rounds per user message.
    private let maxToolRounds = 100
    
    // MARK: - Calendar Keyword Router
    
    /// Keywords that indicate a calendar-related query.
    /// If any of these appear in the user's message, route to Qwen.
    private let calendarKeywords: Set<String> = [
        "schedule", "calendar", "event", "events", "meeting", "meetings",
        "appointment", "appointments", "free", "busy", "available",
        "availability", "book", "create", "cancel", "reschedule",
        "move", "delete", "remove", "when am i", "what's on",
        "what do i have", "today", "tomorrow", "this week", "next week",
        "morning", "afternoon", "evening", "slot", "slots", "conflict",
        "overlap", "block", "schedule for"
    ]
    
    /// Determines if a message should go to the calendar model.
    private func isCalendarQuery(_ text: String) -> Bool {
        let lower = text.lowercased()
        return calendarKeywords.contains(where: { lower.contains($0) })
    }
    
    // MARK: - Initialization
    
    /// Discovers available models and sets up Smart Mode if both are found.
    func discoverModelsAndInitialize(useGPU: Bool) {
        currentUseGPU = useGPU
        let allModels = ModelDiscovery.findAllModels()
        
        // Look for Smart Mode pair using exact filename prefixes:
        //   calendar-qwen25-3b  → fine-tuned calendar agent
        //   gemma-3n-E2B        → general conversation
        //   Qwen2.5-3B-Instruct → base model (individual selection only)
        for model in allModels {
            let name = model.name.lowercased()
            if name.contains("calendar-qwen") {
                calendarModelPath = model.path
            } else if name.contains("gemma") {
                generalModelPath = model.path
            }
            // Note: Qwen2.5-3B-Instruct is NOT part of Smart Mode —
            // it's the base pretrained model for individual comparison.
        }
        
        // Enable Smart Mode if both models are available.
        smartModeAvailable = (calendarModelPath != nil && generalModelPath != nil)
        
        if smartModeAvailable {
            // Default to Smart Mode with calendar model loaded first.
            smartModeEnabled = true
            loadModel(path: calendarModelPath!, role: .calendar, useGPU: useGPU)
        } else if let modelPath = ModelDiscovery.findModel() {
            // Fallback: load whatever model is available.
            smartModeEnabled = false
            loadModel(path: modelPath, role: .standalone, useGPU: useGPU)
        } else {
            statusMessage = "No model found — copy a .litertlm file to the app's Documents folder"
        }
    }
    
    /// Determines if a model should have calendar tools enabled.
    /// All models EXCEPT Gemma get calendar tools — this lets you compare
    /// the fine-tuned Qwen (works great) vs base Qwen (struggles).
    private func shouldEnableToolCalling(for path: String) -> Bool {
        let name = (path as NSString).lastPathComponent.lowercased()
        return !name.contains("gemma")
    }
    
    /// Loads a specific model with the given role.
    func loadModel(path: String, role: ModelRole, useGPU: Bool) {
        // If already loaded, switch instead.
        if engineHandle != nil {
            switchToModel(path: path, role: role, useGPU: useGPU)
            return
        }
        
        loadGeneration += 1
        let myGeneration = loadGeneration
        
        currentModelPath = path
        currentUseGPU = useGPU
        activeModelRole = role
        toolCallingEnabled = shouldEnableToolCalling(for: path)
        statusMessage = "Loading..."
        
        currentModelName = (path as NSString).lastPathComponent
            .replacingOccurrences(of: ".litertlm", with: "")
        
        Task.detached { [self] in
            if self.toolCallingEnabled {
                _ = await self.calendarExecutor.requestAccess()
            }
            
            let backend: LiteRTBackend = useGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(path, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    if self.loadGeneration == myGeneration {
                        self.statusMessage = "Failed to load model (error \(engineStatus.rawValue))"
                        self.isEngineReady = false
                    }
                }
                return
            }
            
            // Check if a newer load was requested while we were loading.
            let stale = await MainActor.run { self.loadGeneration != myGeneration }
            if stale {
                litert_engine_destroy(engine)
                return
            }
            
            var conversation: LiteRTConversationHandle?
            let convStatus: LiteRTStatus
            
            if self.toolCallingEnabled {
                convStatus = litert_conversation_create(
                    engine,
                    ToolDeclarations.getDynamicSystemPrompt(),
                    ToolDeclarations.calendarToolsJSON,
                    &conversation
                )
            } else {
                convStatus = litert_conversation_create_ex(
                    engine, nil, nil, 1, &conversation
                )
            }
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    if self.loadGeneration == myGeneration {
                        self.statusMessage = "Failed to create conversation"
                        self.isEngineReady = false
                    }
                }
                return
            }
            
            // Final stale check before committing.
            let stale2 = await MainActor.run { self.loadGeneration != myGeneration }
            if stale2 {
                litert_conversation_destroy(conversation)
                litert_engine_destroy(engine)
                return
            }
            
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                
                switch role {
                case .calendar:
                    self.statusMessage = "Calendar agent ready"
                case .general:
                    self.statusMessage = "General assistant ready"
                case .standalone:
                    if self.toolCallingEnabled {
                        self.statusMessage = "Calendar tools active (base)"
                    } else {
                        self.statusMessage = "Ready"
                    }
                }
            }
        }
    }
    
    /// Tears down the current model and loads a different one.
    private func switchToModel(path: String, role: ModelRole, useGPU: Bool) {
        isSwitchingModel = true
        
        let roleLabel = role == .calendar ? "📅 Calendar Agent" : "💬 General Assistant"
        statusMessage = "Switching..."
        
        // Show a system message so the user knows what's happening.
        messages.append(ChatMessage(
            role: .system,
            content: "🔄 Switching to \(roleLabel) model..."
        ))
        
        Task.detached { [self] in
            // Tear down current model.
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
            
            // Load the new model.
            self.currentModelPath = path
            self.activeModelRole = role
            self.toolCallingEnabled = self.shouldEnableToolCalling(for: path)
            
            self.currentModelName = (path as NSString).lastPathComponent
                .replacingOccurrences(of: ".litertlm", with: "")
            
            if self.toolCallingEnabled {
                _ = await self.calendarExecutor.requestAccess()
            }
            
            let backend: LiteRTBackend = useGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(path, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    self.statusMessage = "Failed to load model"
                    self.isSwitchingModel = false
                }
                return
            }
            
            var conversation: LiteRTConversationHandle?
            let convStatus: LiteRTStatus
            
            if self.toolCallingEnabled {
                convStatus = litert_conversation_create(
                    engine,
                    ToolDeclarations.getDynamicSystemPrompt(),
                    ToolDeclarations.calendarToolsJSON,
                    &conversation
                )
            } else {
                convStatus = litert_conversation_create_ex(
                    engine, nil, nil, 1, &conversation
                )
            }
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    self.statusMessage = "Failed to create conversation after switch"
                    self.isSwitchingModel = false
                }
                return
            }
            
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.isSwitchingModel = false
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                
                switch role {
                case .calendar:
                    self.statusMessage = "Calendar agent ready"
                case .general:
                    self.statusMessage = "General assistant ready"
                case .standalone:
                    self.statusMessage = "Model loaded — ready to chat"
                }
                
                self.messages.append(ChatMessage(
                    role: .system,
                    content: "✅ \(roleLabel) model loaded"
                ))
            }
        }
    }
    
    // MARK: - Send Message (with Smart Routing)
    
    func sendMessage(_ text: String) {
        guard isEngineReady || smartModeEnabled else { return }
        
        // In Smart Mode, check if we need to switch models.
        if smartModeEnabled {
            let needsCalendar = isCalendarQuery(text)
            let currentIsCalendar = (activeModelRole == .calendar)
            
            if needsCalendar && !currentIsCalendar {
                // Need calendar model but general is loaded — switch then send.
                messages.append(ChatMessage(role: .user, content: text))
                switchAndSend(text: text, targetRole: .calendar, path: calendarModelPath!)
                return
            } else if !needsCalendar && currentIsCalendar {
                // Need general model but calendar is loaded — switch then send.
                messages.append(ChatMessage(role: .user, content: text))
                switchAndSend(text: text, targetRole: .general, path: generalModelPath!)
                return
            }
        }
        
        // Model is already correct — send directly.
        sendMessageDirect(text)
    }
    
    /// Switches model then sends the message after loading completes.
    private func switchAndSend(text: String, targetRole: ModelRole, path: String) {
        isGenerating = true
        isSwitchingModel = true
        
        let roleLabel = targetRole == .calendar ? "📅 Calendar Agent" : "💬 General Assistant"
        statusMessage = "Switching..."
        messages.append(ChatMessage(
            role: .system,
            content: "🔄 Routing to \(roleLabel)..."
        ))
        
        Task.detached { [self] in
            // Tear down.
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
            
            self.currentModelPath = path
            self.activeModelRole = targetRole
            self.toolCallingEnabled = self.shouldEnableToolCalling(for: path)
            self.currentModelName = (path as NSString).lastPathComponent
                .replacingOccurrences(of: ".litertlm", with: "")
            
            if self.toolCallingEnabled {
                _ = await self.calendarExecutor.requestAccess()
            }
            
            let backend: LiteRTBackend = self.currentUseGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(path, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    self.statusMessage = "Failed to load model"
                    self.isSwitchingModel = false
                    self.isGenerating = false
                }
                return
            }
            
            var conversation: LiteRTConversationHandle?
            let convStatus: LiteRTStatus
            
            if self.toolCallingEnabled {
                convStatus = litert_conversation_create(
                    engine,
                    ToolDeclarations.getDynamicSystemPrompt(),
                    ToolDeclarations.calendarToolsJSON,
                    &conversation
                )
            } else {
                convStatus = litert_conversation_create_ex(
                    engine, nil, nil, 1, &conversation
                )
            }
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    self.statusMessage = "Failed to create conversation"
                    self.isSwitchingModel = false
                    self.isGenerating = false
                }
                return
            }
            
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.isSwitchingModel = false
                self.currentBackend = self.currentUseGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                
                switch targetRole {
                case .calendar:
                    self.statusMessage = "Calendar agent ready"
                case .general:
                    self.statusMessage = "General assistant ready"
                case .standalone:
                    self.statusMessage = "Model loaded"
                }
            }
            
            // Now send the original message on the newly loaded model.
            self.sendMessageDirectFromBackground(text)
        }
    }
    
    /// Sends a message directly (model already loaded). Called from main context.
    private func sendMessageDirect(_ text: String) {
        guard let conversation = conversationHandle, isEngineReady else { return }
        
        var actualText = text
        if messages.isEmpty && toolCallingEnabled {
            let context = "[Context: \(ToolDeclarations.getDynamicSystemPrompt())]\n\n"
            actualText = context + text
        } else if toolCallingEnabled {
            // For subsequent messages, still prepend context if this is
            // the first message after a model switch.
            let hasUserMessages = messages.contains(where: {
                $0.role == .user && $0.modelName == nil
            })
            if !hasUserMessages {
                let context = "[Context: \(ToolDeclarations.getDynamicSystemPrompt())]\n\n"
                actualText = context + text
            }
        }
        
        messages.append(ChatMessage(role: .user, content: text))
        isGenerating = true
        
        Task.detached { [self] in
            self.performInference(text: actualText, conversation: conversation)
        }
    }
    
    /// Sends a message from within an already-running background Task.
    private func sendMessageDirectFromBackground(_ text: String) {
        guard let conversation = conversationHandle else {
            Task { @MainActor in self.isGenerating = false }
            return
        }
        
        var actualText = text
        if toolCallingEnabled {
            let context = "[Context: \(ToolDeclarations.getDynamicSystemPrompt())]\n\n"
            actualText = context + text
        }
        
        performInference(text: actualText, conversation: conversation)
    }
    
    /// Core inference logic — shared by direct and switch-then-send paths.
    private func performInference(text: String, conversation: LiteRTConversationHandle) {
        let modelTag = self.currentModelName
        
        var responsePtr: UnsafePointer<CChar>?
        let status = litert_conversation_send(conversation, text, &responsePtr)
        
        guard status == LITERT_OK, let ptr = responsePtr else {
            Task { @MainActor in
                self.messages.append(ChatMessage(
                    role: .assistant,
                    content: "[Error: inference failed with code \(status.rawValue)]",
                    modelName: modelTag
                ))
                self.isGenerating = false
            }
            return
        }
        
        var responseText = String(cString: ptr).trimmingCharacters(in: .whitespacesAndNewlines)
        var toolRound = 0
        
        // === AGENTIC LOOP ===
        while self.isToolCallResponse(responseText) && toolRound < self.maxToolRounds {
            toolRound += 1
            let toolCalls = self.parseToolCalls(responseText)
            
            for toolCall in toolCalls {
                let toolCallDisplay = self.formatToolCallDisplay(toolCall)
                Task { @MainActor in
                    self.messages.append(ChatMessage(
                        role: .toolCall,
                        content: toolCallDisplay,
                        toolName: toolCall.name,
                        modelName: modelTag
                    ))
                }
                
                // Use a semaphore to bridge async tool execution.
                let semaphore = DispatchSemaphore(value: 0)
                var toolResult = ""
                Task {
                    toolResult = await self.calendarExecutor.execute(
                        functionName: toolCall.name,
                        arguments: toolCall.arguments
                    )
                    semaphore.signal()
                }
                semaphore.wait()
                
                let resultDisplay = self.formatToolResultDisplay(toolCall.name, result: toolResult)
                Task { @MainActor in
                    self.messages.append(ChatMessage(
                        role: .toolResult,
                        content: resultDisplay,
                        toolName: toolCall.name,
                        modelName: modelTag
                    ))
                }
                
                var nextResponsePtr: UnsafePointer<CChar>?
                let nextStatus = litert_conversation_send_tool_response(
                    conversation, toolResult, &nextResponsePtr
                )
                
                if nextStatus == LITERT_OK, let nextPtr = nextResponsePtr {
                    responseText = String(cString: nextPtr)
                } else {
                    responseText = "[Error: failed to send tool response]"
                    break
                }
            }
        }
        
        // === FINAL RESPONSE ===
        let trimmedResponse = responseText.trimmingCharacters(in: .whitespacesAndNewlines)
        Task { @MainActor in
            self.messages.append(ChatMessage(
                role: .assistant,
                content: trimmedResponse,
                modelName: modelTag
            ))
            self.isGenerating = false
            self.updateBenchmark()
        }
    }
    
    // MARK: - Constrained Decoding Demo
    
    /// JSON schema for the constrained decoding A/B test.
    private let demoJsonSchema = """
    {
      "type": "object",
      "properties": {
        "answer": {
          "type": "string"
        },
        "confidence": {
          "type": "string",
          "enum": ["high", "medium", "low"]
        },
        "keywords": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 2,
          "maxItems": 4
        }
      },
      "required": ["answer", "confidence", "keywords"],
      "additionalProperties": false
    }
    """
    
    /// Runs the constrained decoding A/B demo on the general model (Gemma).
    /// If Gemma isn't loaded, switches to it first.
    func runConstrainedDecodingDemo() {
        guard smartModeEnabled, let gemmaPath = generalModelPath else {
            messages.append(ChatMessage(
                role: .system,
                content: "⚠️ Constrained decoding demo requires Smart Mode with Gemma model."
            ))
            return
        }
        
        isGenerating = true
        
        messages.append(ChatMessage(
            role: .system,
            content: "🧪 Constrained Decoding Demo\n\nSending the same prompt twice:\n1. Free generation (unconstrained)\n2. JSON Schema enforced (constrained)\n\nConstrained decoding masks invalid tokens at the logit level during inference — the model literally cannot produce output that violates the schema."
        ))
        
        // Ensure Gemma is loaded.
        if activeModelRole != .general {
            // Need to switch to Gemma first, then run the demo.
            Task.detached { [self] in
                // Tear down current model.
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
                    self.statusMessage = "Loading..."
                    self.messages.append(ChatMessage(
                        role: .system,
                        content: "🔄 Loading Gemma-3n-E2B for demo..."
                    ))
                }
                
                self.activeModelRole = .general
                self.toolCallingEnabled = false
                self.currentModelPath = gemmaPath
                self.currentModelName = (gemmaPath as NSString).lastPathComponent
                    .replacingOccurrences(of: ".litertlm", with: "")
                
                let backend: LiteRTBackend = self.currentUseGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
                
                var engine: LiteRTEngineHandle?
                let engineStatus = litert_engine_create(gemmaPath, backend, &engine)
                guard engineStatus == LITERT_OK, let engine = engine else {
                    await MainActor.run {
                        self.statusMessage = "Failed to load Gemma"
                        self.isGenerating = false
                    }
                    return
                }
                
                var conversation: LiteRTConversationHandle?
                let convStatus = litert_conversation_create_ex(engine, nil, nil, 1, &conversation)
                guard convStatus == LITERT_OK, let conversation = conversation else {
                    litert_engine_destroy(engine)
                    await MainActor.run {
                        self.statusMessage = "Failed to create conversation"
                        self.isGenerating = false
                    }
                    return
                }
                
                await MainActor.run {
                    self.engineHandle = engine
                    self.conversationHandle = conversation
                    self.isEngineReady = true
                    self.statusMessage = "General assistant ready"
                }
                
                self.executeConstrainedDemo(conversation: conversation)
            }
        } else {
            // Gemma already loaded.
            guard let conversation = conversationHandle else { return }
            Task.detached { [self] in
                self.executeConstrainedDemo(conversation: conversation)
            }
        }
    }
    
    /// Executes the actual A/B constrained decoding test.
    private func executeConstrainedDemo(conversation: LiteRTConversationHandle) {
        let prompt = "What is the tallest building in the world?"
        let modelTag = self.currentModelName
        
        // Part 1: Unconstrained.
        Task { @MainActor in
            self.messages.append(ChatMessage(
                role: .system,
                content: "── Part 1: Free Generation (no constraint) ──"
            ))
            self.messages.append(ChatMessage(role: .user, content: prompt))
        }
        
        var responsePtr: UnsafePointer<CChar>?
        let status1 = litert_conversation_send(conversation, prompt, &responsePtr)
        let freeResponse = (status1 == LITERT_OK && responsePtr != nil)
            ? String(cString: responsePtr!).trimmingCharacters(in: .whitespacesAndNewlines)
            : "[Error: inference failed]"
        
        Task { @MainActor in
            self.messages.append(ChatMessage(
                role: .assistant, content: freeResponse, modelName: modelTag
            ))
            self.messages.append(ChatMessage(
                role: .system,
                content: "── Part 2: JSON Schema Enforced ──\nSchema: {answer: string, confidence: high|medium|low, keywords: string[]}"
            ))
            self.messages.append(ChatMessage(role: .user, content: prompt))
        }
        
        // Part 2: Constrained.
        var constrainedPtr: UnsafePointer<CChar>?
        let status2 = litert_conversation_send_constrained(
            conversation, prompt,
            LITERT_CONSTRAINT_JSON_SCHEMA,
            self.demoJsonSchema,
            &constrainedPtr
        )
        let constrainedResponse = (status2 == LITERT_OK && constrainedPtr != nil)
            ? String(cString: constrainedPtr!).trimmingCharacters(in: .whitespacesAndNewlines)
            : "[Error: constrained inference failed (code \(status2.rawValue))]"
        
        let isValid = isValidJson(constrainedResponse)
        
        Task { @MainActor in
            self.messages.append(ChatMessage(
                role: .assistant, content: constrainedResponse, modelName: modelTag
            ))
            
            let verdict = isValid
                ? "✅ Valid JSON! The LLGuidance engine masked invalid tokens at each step, guaranteeing the output conforms to the schema. ~50μs overhead per token."
                : "⚠️ Output may not be valid JSON. The constraint provider may not have loaded — check that libGemmaModelConstraintProvider.dylib is embedded."
            self.messages.append(ChatMessage(role: .system, content: verdict))
            self.isGenerating = false
        }
    }
    
    private func isValidJson(_ string: String) -> Bool {
        guard let data = string.data(using: .utf8) else { return false }
        return (try? JSONSerialization.jsonObject(with: data)) != nil
    }
    
    // MARK: - Manual Model Loading (for model picker)
    
    /// Loads a manually-selected model, disabling Smart Mode.
    func loadManualModel(path: String, useGPU: Bool) {
        smartModeEnabled = false
        cleanup()
        messages = []  // Clear conversation history on model switch.
        ModelDiscovery.saveLastUsedModel(path: path)
        
        // Determine role based on filename.
        let name = (path as NSString).lastPathComponent.lowercased()
        let role: ModelRole
        if name.contains("calendar-qwen") {
            role = .calendar
        } else if name.contains("gemma") {
            role = .general
        } else {
            role = .standalone
        }
        
        loadModel(path: path, role: role, useGPU: useGPU)
    }
    
    /// Re-enables Smart Mode.
    func enableSmartMode(useGPU: Bool) {
        guard smartModeAvailable else { return }
        cleanup()
        messages = []  // Clear conversation history on mode switch.
        smartModeEnabled = true
        loadModel(path: calendarModelPath!, role: .calendar, useGPU: useGPU)
    }
    
    // MARK: - Tool Call Detection and Parsing
    
    private func isToolCallResponse(_ response: String) -> Bool {
        return response.contains("\"tool_calls\"")
    }
    
    struct ToolCall {
        let name: String
        let arguments: [String: Any]
    }
    
    private func parseToolCalls(_ response: String) -> [ToolCall] {
        guard let data = response.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let toolCalls = json["tool_calls"] as? [[String: Any]]
        else { return [] }
        
        return toolCalls.compactMap { call in
            guard let function = call["function"] as? [String: Any],
                  let name = function["name"] as? String
            else { return nil }
            let arguments = function["arguments"] as? [String: Any] ?? [:]
            return ToolCall(name: name, arguments: arguments)
        }
    }
    
    // MARK: - Tool Call Display Formatting
    
    private func formatToolCallDisplay(_ toolCall: ToolCall) -> String {
        switch toolCall.name {
        case "get_events", "get_todays_events", "get_events_for_date":
            let date = toolCall.arguments["date"] as? String ?? "today"
            return "📅 Checking schedule for \(date)..."
        case "get_week_events":
            let start = toolCall.arguments["start_date"] as? String ?? "this week"
            return "📅 Getting weekly schedule from \(start)..."
        case "find_free_slots":
            let date = toolCall.arguments["date"] as? String ?? "today"
            let duration = toolCall.arguments["duration_minutes"] as? Int ?? 60
            return "🔍 Finding \(duration)-minute free slots on \(date)..."
        case "create_event":
            let title = toolCall.arguments["title"] as? String ?? "New Event"
            return "➕ Creating event: \(title)..."
        case "modify_event":
            let newTitle = toolCall.arguments["new_title"] as? String
            return "✏️ Modifying event\(newTitle.map { ": \($0)" } ?? "")..."
        case "delete_event":
            return "🗑️ Deleting event..."
        case "get_upcoming_events":
            let count = toolCall.arguments["count"] as? Int ?? 5
            return "📋 Getting next \(count) upcoming events..."
        case "search_events":
            let query = toolCall.arguments["query"] as? String ?? ""
            return "🔎 Searching for '\(query)'..."
        case "check_conflicts":
            return "⚠️ Checking for scheduling conflicts..."
        default:
            return "🔧 Calling \(toolCall.name)..."
        }
    }
    
    private func formatToolResultDisplay(_ toolName: String, result: String) -> String {
        guard let data = result.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "✅ Tool executed" }
        
        if let error = json["error"] as? String {
            return "❌ \(error)"
        }
        
        switch toolName {
        case "get_events", "get_todays_events", "get_events_for_date", "get_upcoming_events":
            let count = json["event_count"] as? Int ?? json["count_returned"] as? Int ?? 0
            let date = json["date"] as? String ?? ""
            return "✅ Found \(count) event\(count == 1 ? "" : "s")\(date.isEmpty ? "" : " on \(date)")"
        case "get_week_events":
            let total = json["total_events"] as? Int ?? 0
            let days = json["days_with_events"] as? Int ?? 0
            return "✅ Found \(total) events across \(days) days"
        case "find_free_slots":
            let count = json["free_slots_count"] as? Int ?? 0
            return "✅ Found \(count) available slot\(count == 1 ? "" : "s")"
        case "create_event":
            let title = json["title"] as? String ?? "Event"
            return "✅ Created: \(title)"
        case "modify_event":
            return "✅ Event updated"
        case "delete_event":
            let title = json["deleted_title"] as? String ?? "Event"
            return "✅ Deleted: \(title)"
        case "search_events":
            let count = json["match_count"] as? Int ?? 0
            return "✅ Found \(count) matching event\(count == 1 ? "" : "s")"
        case "check_conflicts":
            let hasConflicts = json["has_conflicts"] as? Bool ?? false
            let count = json["conflict_count"] as? Int ?? 0
            return hasConflicts ? "⚠️ \(count) conflict\(count == 1 ? "" : "s") found" : "✅ No conflicts"
        default:
            return "✅ Done"
        }
    }
    
    // MARK: - Backend Toggle
    
    func toggleBackend(useGPU: Bool) {
        guard let modelPath = currentModelPath else { return }
        currentUseGPU = useGPU
        let role = activeModelRole
        cleanup()
        loadModel(path: modelPath, role: role, useGPU: useGPU)
    }
    
    // MARK: - Benchmark
    
    private func updateBenchmark() {
        guard let conversation = conversationHandle else { return }
        var info = LiteRTBenchmarkInfo()
        let status = litert_get_benchmark_info(conversation, &info)
        if status == LITERT_OK {
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
    
    // MARK: - Cleanup
    
    func cleanup() {
        loadGeneration += 1  // Invalidate any in-flight background loads.
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

// MARK: - ChatMessage

struct ChatMessage: Identifiable {
    let id: UUID
    let role: Role
    let content: String
    let timestamp = Date()
    let toolName: String?
    let modelName: String?  // Which model produced this response.
    
    init(role: Role, content: String, id: UUID = UUID(),
         toolName: String? = nil, modelName: String? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.toolName = toolName
        self.modelName = modelName
    }
    
    enum Role {
        case user
        case assistant
        case toolCall
        case toolResult
        case system
    }
}

// MARK: - Model Discovery

struct ModelDiscovery {
    private static let lastUsedModelKey = "LastUsedModelName"
    
    static func findModel() -> String? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let contents = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
        let allModels = contents.filter { $0.pathExtension == "litertlm" }
        guard !allModels.isEmpty else { return nil }
        
        if let lastModelName = UserDefaults.standard.string(forKey: lastUsedModelKey),
           let matchedModel = allModels.first(where: { $0.lastPathComponent == lastModelName }) {
            return matchedModel.path
        }
        return allModels.first?.path
    }
    
    static func findAllModels() -> [(name: String, path: String)] {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let contents = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
        return contents
            .filter { $0.pathExtension == "litertlm" }
            .map { (name: $0.deletingPathExtension().lastPathComponent, path: $0.path) }
            .sorted { $0.name < $1.name }
    }
    
    static func saveLastUsedModel(path: String) {
        let url = URL(fileURLWithPath: path)
        UserDefaults.standard.set(url.lastPathComponent, forKey: lastUsedModelKey)
    }
}

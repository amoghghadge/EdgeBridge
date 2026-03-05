//
//  EngineViewModel.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// EngineViewModel.swift  (v5 — Agentic Calendar Assistant)
//
// Extends v4 with the agentic tool-calling loop. When the model responds
// with a tool_calls JSON structure, the ViewModel dispatches to
// CalendarToolExecutor, sends the result back to the model via
// litert_conversation_send_tool_response(), and repeats until the model
// produces a final text response.
//
// AGENTIC LOOP ARCHITECTURE:
//   1. User sends message
//   2. Model responds (might be text OR tool_calls)
//   3. If tool_calls: parse → execute via CalendarToolExecutor → send result back → goto 2
//   4. If text: display to user, loop ends
//
// The tool-calling loop uses blocking sends (litert_conversation_send)
// rather than async streaming, because we need to inspect each response
// to decide whether to continue the loop or display the result. The
// final text response to the user uses the blocking path as well for
// simplicity — we can upgrade to streaming for the final response later.
// ============================================================================

import SwiftUI
import Foundation

// MARK: - EngineViewModel

@Observable
class EngineViewModel {
    
    // -- Published State for SwiftUI --
    var messages: [ChatMessage] = []
    var isGenerating: Bool = false
    var isSwitchingBackend: Bool = false
    var isEngineReady: Bool = false
    var currentBackend: String = "CPU"
    var benchmarkText: String = ""
    var statusMessage: String = "No model loaded"
    var currentModelName: String = ""
    
    // -- Internal State --
    private var engineHandle: LiteRTEngineHandle?
    private var conversationHandle: LiteRTConversationHandle?
    private var currentModelPath: String?
    private var currentUseGPU: Bool = false
    
    // The calendar tool executor — interfaces with real EventKit.
    private let calendarExecutor = CalendarToolExecutor()
    
    // Whether to enable tool calling (calendar agent mode).
    // When false, the model runs as a plain chatbot.
    var toolCallingEnabled: Bool = true
    
    // Maximum number of tool-call rounds per user message to prevent
    // infinite loops if the model keeps producing tool calls.
    private let maxToolRounds = 5
    
    // MARK: - Initialization
    
    /// Loads a .litertlm model and creates the engine + conversation.
    /// When toolCallingEnabled is true, the conversation is created with
    /// calendar tool declarations and a system prompt.
    func initialize(modelPath: String, useGPU: Bool) {
        guard engineHandle == nil else { return }
        
        currentModelPath = modelPath
        currentUseGPU = useGPU
        statusMessage = "Loading model..."
        
        // Extract model name from the file path for display.
        currentModelName = (modelPath as NSString).lastPathComponent
            .replacingOccurrences(of: ".litertlm", with: "")
        
        Task.detached { [self] in
            // Request calendar permission early so it's ready when needed.
            if self.toolCallingEnabled {
                _ = await self.calendarExecutor.requestAccess()
            }
            
            let backend: LiteRTBackend = useGPU ? LITERT_BACKEND_GPU : LITERT_BACKEND_CPU
            
            // Step 1: Create the Engine.
            var engine: LiteRTEngineHandle?
            let engineStatus = litert_engine_create(modelPath, backend, &engine)
            
            guard engineStatus == LITERT_OK, let engine = engine else {
                await MainActor.run {
                    self.statusMessage = "Failed to load model (error \(engineStatus.rawValue))"
                    self.isEngineReady = false
                }
                return
            }
            
            // Step 2: Create the Conversation with optional tools.
            var conversation: LiteRTConversationHandle?
            let convStatus: LiteRTStatus
            
            if self.toolCallingEnabled {
                // Pass the calendar tools JSON and system prompt.
                convStatus = litert_conversation_create(
                    engine,
                    ToolDeclarations.calendarSystemPrompt,
                    ToolDeclarations.calendarToolsJSON,
                    &conversation
                )
            } else {
                // Plain chatbot mode — no tools, no system prompt.
                convStatus = litert_conversation_create(
                    engine, nil, nil, &conversation
                )
            }
            
            guard convStatus == LITERT_OK, let conversation = conversation else {
                litert_engine_destroy(engine)
                await MainActor.run {
                    self.statusMessage = "Failed to create conversation (error \(convStatus.rawValue))"
                    self.isEngineReady = false
                }
                return
            }
            
            await MainActor.run {
                self.engineHandle = engine
                self.conversationHandle = conversation
                self.isEngineReady = true
                self.currentBackend = useGPU ? "GPU (Metal)" : "CPU (XNNPACK)"
                self.statusMessage = self.toolCallingEnabled
                    ? "Calendar assistant ready"
                    : "Model loaded — ready to chat"
            }
        }
    }
    
    // MARK: - Send Message (with Agentic Loop)
    
    /// Sends a user message and runs the agentic tool-calling loop.
    /// The loop continues until the model produces a text response
    /// (no tool calls) or we hit the maximum number of tool rounds.
    func sendMessage(_ text: String) {
        guard let engine = engineHandle,
              let conversation = conversationHandle,
              isEngineReady else { return }
        
        // Add the user's message to the chat.
        let userMessage = ChatMessage(role: .user, content: text)
        messages.append(userMessage)
        isGenerating = true
        
        Task.detached { [self] in
            // Send the user's message to the model.
            var responsePtr: UnsafePointer<CChar>?
            let status = litert_conversation_send(conversation, text, &responsePtr)
            
            guard status == LITERT_OK, let ptr = responsePtr else {
                await MainActor.run {
                    self.messages.append(ChatMessage(
                        role: .assistant,
                        content: "[Error: inference failed with code \(status.rawValue)]"
                    ))
                    self.isGenerating = false
                }
                return
            }
            
            var responseText = String(cString: ptr)
            var toolRound = 0
            
            // === AGENTIC LOOP ===
            // Keep looping while the model produces tool calls.
            while self.isToolCallResponse(responseText) && toolRound < self.maxToolRounds {
                toolRound += 1
                
                // Parse the tool call(s) from the model's response.
                let toolCalls = self.parseToolCalls(responseText)
                
                for toolCall in toolCalls {
                    // Show the tool call in the chat UI.
                    let toolCallDisplay = self.formatToolCallDisplay(toolCall)
                    await MainActor.run {
                        self.messages.append(ChatMessage(
                            role: .toolCall,
                            content: toolCallDisplay,
                            toolName: toolCall.name
                        ))
                    }
                    
                    // Execute the tool via CalendarToolExecutor.
                    let toolResult = await self.calendarExecutor.execute(
                        functionName: toolCall.name,
                        arguments: toolCall.arguments
                    )
                    
                    // Show the tool result in the chat UI.
                    let resultDisplay = self.formatToolResultDisplay(toolCall.name, result: toolResult)
                    await MainActor.run {
                        self.messages.append(ChatMessage(
                            role: .toolResult,
                            content: resultDisplay,
                            toolName: toolCall.name
                        ))
                    }
                    
                    // Send the tool result back to the model.
                    var nextResponsePtr: UnsafePointer<CChar>?
                    let nextStatus = litert_conversation_send_tool_response(
                        conversation, toolResult, &nextResponsePtr
                    )
                    
                    if nextStatus == LITERT_OK, let nextPtr = nextResponsePtr {
                        responseText = String(cString: nextPtr)
                    } else {
                        // Tool response failed — break out of the loop.
                        responseText = "[Error: failed to send tool response]"
                        break
                    }
                }
            }
            
            // === FINAL RESPONSE ===
            // The model has produced a text response (not a tool call).
            await MainActor.run {
                self.messages.append(ChatMessage(
                    role: .assistant,
                    content: responseText
                ))
                self.isGenerating = false
                self.updateBenchmark()
            }
        }
    }
    
    // MARK: - Tool Call Detection and Parsing
    
    /// Checks if a response string contains tool calls.
    /// The C bridge returns tool_calls as a JSON string when the model
    /// wants to invoke a function.
    private func isToolCallResponse(_ response: String) -> Bool {
        return response.contains("\"tool_calls\"")
    }
    
    /// Represents a single parsed tool call from the model.
    struct ToolCall {
        let name: String
        let arguments: [String: Any]
    }
    
    /// Parses tool calls from the model's JSON response.
    /// Expected format (from litert_bridge_api.cc's ExtractTextFromMessage):
    /// {"tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}
    private func parseToolCalls(_ response: String) -> [ToolCall] {
        guard let data = response.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let toolCalls = json["tool_calls"] as? [[String: Any]]
        else {
            return []
        }
        
        return toolCalls.compactMap { call in
            guard let function = call["function"] as? [String: Any],
                  let name = function["name"] as? String
            else {
                return nil
            }
            let arguments = function["arguments"] as? [String: Any] ?? [:]
            return ToolCall(name: name, arguments: arguments)
        }
    }
    
    // MARK: - Tool Call Display Formatting
    
    /// Formats a tool call for display in the chat UI.
    /// Shows a user-friendly description of what the model is doing.
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
    
    /// Formats a tool result for display in the chat UI.
    /// Provides a brief summary rather than showing raw JSON.
    private func formatToolResultDisplay(_ toolName: String, result: String) -> String {
        guard let data = result.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return "✅ Tool executed"
        }
        
        if let error = json["error"] as? String {
            return "❌ \(error)"
        }
        
        switch toolName {
        case "get_events", "get_todays_events", "get_events_for_date", "get_upcoming_events":
            let count = json["event_count"] as? Int ?? json["count_returned"] as? Int ?? 0
            let date = json["date"] as? String ?? json["day_of_week"] as? String ?? ""
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
        
        isSwitchingBackend = true
        
        Task.detached { [self] in
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
            let convStatus: LiteRTStatus
            
            if self.toolCallingEnabled {
                convStatus = litert_conversation_create(
                    engine,
                    ToolDeclarations.calendarSystemPrompt,
                    ToolDeclarations.calendarToolsJSON,
                    &conversation
                )
            } else {
                convStatus = litert_conversation_create(engine, nil, nil, &conversation)
            }
            
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
                self.statusMessage = self.toolCallingEnabled
                    ? "Calendar assistant ready"
                    : "Backend switched to \(self.currentBackend)"
            }
        }
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
    let toolName: String?  // For tool call/result messages.
    
    init(role: Role, content: String, id: UUID = UUID(), toolName: String? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.toolName = toolName
    }
    
    enum Role {
        case user
        case assistant
        case toolCall     // "📅 Checking schedule..." — what the model is doing.
        case toolResult   // "✅ Found 3 events" — what happened.
        case system
    }
}

// MARK: - Model Discovery

struct ModelDiscovery {
    
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

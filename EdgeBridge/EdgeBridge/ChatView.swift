//
//  ChatView.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// ChatView.swift  (v4 — Dual-Model Cascade + Constrained Decoding Demo)
//
// WHAT'S NEW IN v4:
//   - Model attribution: assistant bubbles show which model responded
//   - Smart Mode indicator and toggle in the header
//   - Constrained Decoding demo button (runs on Gemma)
//   - Model picker now has "Smart Mode" option at top
//   - Routing indicator shows when model is switching
// ============================================================================

import Foundation
import SwiftUI

struct ChatView: View {
    @State private var viewModel = EngineViewModel()
    @State private var inputText = ""
    @State private var useGPU = false
    @State private var availableModels: [(name: String, path: String)] = []
    @State private var showModelPicker = false
    
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            statusBar
            benchmarkBar
            messageList
            Divider()
            inputBar
        }
        .onAppear {
            viewModel.discoverModelsAndInitialize(useGPU: useGPU)
        }
        .onDisappear {
            viewModel.cleanup()
        }
        .sheet(isPresented: $showModelPicker) {
            modelPickerSheet
        }
    }
    
    // MARK: - Header
    
    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("EdgeBridge")
                        .font(.headline)
                    if viewModel.smartModeEnabled {
                        Text("SMART")
                            .font(.system(size: 8, weight: .bold))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.purple.opacity(0.2))
                            .foregroundStyle(.purple)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
                if !viewModel.currentModelName.isEmpty {
                    Text(viewModel.currentModelName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            
            Spacer()
            
            if viewModel.isSwitchingModel {
                ProgressView()
                    .scaleEffect(0.7)
            }
            
            Text(viewModel.currentBackend)
                .font(.caption)
                .foregroundStyle(.secondary)
            
            Toggle("GPU", isOn: $useGPU)
                .toggleStyle(.switch)
                .labelsHidden()
                .disabled(viewModel.isSwitchingModel || !viewModel.isEngineReady)
                .onChange(of: useGPU) { _, newValue in
                    viewModel.toggleBackend(useGPU: newValue)
                }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
    
    // MARK: - Status Bar
    
    private var statusBar: some View {
        HStack(spacing: 8) {
            // Status icon + text.
            HStack(spacing: 5) {
                if viewModel.isEngineReady {
                    // Model role icon.
                    Image(systemName: viewModel.activeModelRole == .general
                          ? "bubble.left.and.bubble.right.fill"
                          : "calendar.badge.checkmark")
                        .font(.caption)
                        .foregroundStyle(viewModel.activeModelRole == .general ? .blue : .green)
                        .frame(width: 16, alignment: .center)
                } else if viewModel.statusMessage.contains("Loading") || viewModel.statusMessage.contains("Switching") {
                    // Spinner replaces the icon while loading.
                    ProgressView()
                        .scaleEffect(0.6)
                        .frame(width: 16, height: 14)
                }
                Text(viewModel.statusMessage)
                    .font(.caption)
                    .foregroundStyle(
                        viewModel.statusMessage.contains("Failed") ? .red : .secondary
                    )
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
            
            // Constrained Decoding demo button.
            Button(action: {
                viewModel.runConstrainedDecodingDemo()
            }) {
                HStack(spacing: 3) {
                    Image(systemName: "lock.shield")
                    Text("CD Test")
                }
                .font(.caption)
            }
            .disabled(!viewModel.smartModeEnabled || !viewModel.isEngineReady || viewModel.isGenerating)
            
            // Model selector button.
            Button(action: {
                availableModels = ModelDiscovery.findAllModels()
                showModelPicker = true
            }) {
                HStack(spacing: 3) {
                    Image(systemName: "doc.badge.gearshape")
                    Text("Models")
                }
                .font(.caption)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .frame(minHeight: 30)
        .background(Color(.systemGray6))
    }
    
    // MARK: - Benchmark Bar
    
    @ViewBuilder
    private var benchmarkBar: some View {
        if !viewModel.benchmarkText.isEmpty {
            Text(viewModel.benchmarkText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
                .padding(.vertical, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.systemGray6).opacity(0.5))
        }
    }
    
    // MARK: - Message List
    
    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(viewModel.messages) { message in
                        MessageRow(message: message)
                            .id(message.id)
                    }
                    
                    if viewModel.isGenerating {
                        HStack(spacing: 8) {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(viewModel.isSwitchingModel ? "Switching model..." : "Thinking...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal)
                    }
                    
                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: viewModel.messages.count) { _, _ in
                withAnimation {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: viewModel.isGenerating) { _, _ in
                withAnimation {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        }
    }
    
    // MARK: - Input Bar
    
    private var inputBar: some View {
        HStack(spacing: 12) {
            TextField(
                viewModel.smartModeEnabled
                    ? "Ask anything..."
                    : viewModel.toolCallingEnabled
                        ? "Ask about your schedule..."
                        : "Ask something...",
                text: $inputText
            )
            .textFieldStyle(.plain)
            .focused($isInputFocused)
            .onSubmit { send() }
            .disabled(!viewModel.isEngineReady && !viewModel.smartModeEnabled)
            
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(
                        inputText.isEmpty || (!viewModel.isEngineReady && !viewModel.isSwitchingModel)
                            ? .gray : .blue
                    )
            }
            .disabled(
                inputText.isEmpty ||
                viewModel.isGenerating ||
                (!viewModel.isEngineReady && !viewModel.smartModeEnabled)
            )
        }
        .padding()
    }
    
    // MARK: - Model Picker
    
    private var modelPickerSheet: some View {
        NavigationStack {
            List {
                // Smart Mode option (if both models available).
                if viewModel.smartModeAvailable {
                    Section("Smart Mode") {
                        Button(action: {
                            showModelPicker = false
                            viewModel.enableSmartMode(useGPU: useGPU)
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 4) {
                                        Text("Auto-Route (Qwen + Gemma)")
                                            .font(.body)
                                        if viewModel.smartModeEnabled {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundStyle(.green)
                                                .font(.caption)
                                        }
                                    }
                                    Text("Calendar queries → Qwen 3B | General → Gemma E2B")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                        }
                    }
                }
                
                // Individual models.
                Section("Individual Models") {
                    ForEach(availableModels, id: \.path) { model in
                        Button(action: {
                            showModelPicker = false
                            viewModel.loadManualModel(path: model.path, useGPU: useGPU)
                        }) {
                            VStack(alignment: .leading) {
                                HStack(spacing: 4) {
                                    Text(model.name)
                                        .font(.body)
                                    if !viewModel.smartModeEnabled &&
                                        viewModel.loadedModelPath == model.path {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(.green)
                                            .font(.caption)
                                    }
                                }
                                Text(model.path)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showModelPicker = false }
                }
            }
        }
        .presentationDetents([.medium])
    }
    
    // MARK: - Actions
    
    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        isInputFocused = false
        viewModel.sendMessage(text)
    }
}

// MARK: - MessageRow

struct MessageRow: View {
    let message: ChatMessage
    
    var body: some View {
        switch message.role {
        case .user:
            UserBubble(content: message.content)
        case .assistant:
            AssistantBubble(content: message.content, modelName: message.modelName)
        case .toolCall:
            ToolActionCard(content: message.content, isResult: false)
        case .toolResult:
            ToolActionCard(content: message.content, isResult: true)
        case .system:
            SystemMessage(content: message.content)
        }
    }
}

// MARK: - User Bubble

struct UserBubble: View {
    let content: String
    
    var body: some View {
        HStack {
            Spacer(minLength: 60)
            VStack(alignment: .trailing, spacing: 4) {
                Text("You")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        }
    }
}

// MARK: - Assistant Bubble (with model attribution)

struct AssistantBubble: View {
    let content: String
    let modelName: String?
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Text("EdgeBridge")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let name = modelName {
                        Text("· \(shortenModelName(name))")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(modelColor(name))
                    }
                }
                Text(content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray5))
                    .foregroundStyle(.primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            Spacer(minLength: 60)
        }
    }
    
    /// Shortens the model filename for display.
    private func shortenModelName(_ name: String) -> String {
        let lower = name.lowercased()
        if lower.contains("calendar-qwen") { return "Qwen 3B (fine-tuned)" }
        if lower.contains("qwen2.5-3b-instruct") { return "Qwen 3B (base)" }
        if lower.contains("qwen") { return "Qwen" }
        if lower.contains("gemma") && lower.contains("e2b") { return "Gemma E2B" }
        if lower.contains("gemma") && lower.contains("e4b") { return "Gemma E4B" }
        if lower.contains("gemma") && lower.contains("1b") { return "Gemma 1B" }
        if lower.contains("gemma") { return "Gemma" }
        if lower.contains("phi") { return "Phi-4" }
        return String(name.prefix(15))
    }
    
    /// Color-codes model attribution.
    private func modelColor(_ name: String) -> Color {
        let lower = name.lowercased()
        if lower.contains("qwen") { return .green }
        if lower.contains("gemma") { return .blue }
        return .secondary
    }
}

// MARK: - Tool Action Card

struct ToolActionCard: View {
    let content: String
    let isResult: Bool
    
    var body: some View {
        HStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 2)
                .fill(isResult ? Color.green : Color.orange)
                .frame(width: 3)
            
            Text(content)
                .font(.caption)
                .foregroundStyle(.secondary)
            
            Spacer()
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6).opacity(0.7))
        )
        .padding(.horizontal, 4)
    }
}

// MARK: - System Message

struct SystemMessage: View {
    let content: String
    
    var body: some View {
        Text(content)
            .font(.caption)
            .foregroundStyle(.secondary)
            .italic()
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 4)
    }
}

// MARK: - Preview

#Preview {
    ChatView()
}

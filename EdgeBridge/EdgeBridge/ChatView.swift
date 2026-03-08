//
//  ChatView.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

import Foundation

// ============================================================================
// ChatView.swift  (v3 — Agentic Calendar Assistant)
//
// Updated to support the agentic tool-calling loop. Key additions:
//   - Tool call messages shown as action cards (distinct from chat bubbles)
//   - Tool result messages shown as compact status cards
//   - Calendar agent mode toggle in the header
//   - Model name display
//   - Info.plist must include NSCalendarsFullAccessUsageDescription for EventKit
// ============================================================================

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
            discoverAndLoadModel()
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
                Text("EdgeBridge")
                    .font(.headline)
                if !viewModel.currentModelName.isEmpty {
                    Text(viewModel.currentModelName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            
            Spacer()
            
            if viewModel.isSwitchingBackend {
                ProgressView()
                    .scaleEffect(0.7)
            }
            
            Text(viewModel.currentBackend)
                .font(.caption)
                .foregroundStyle(.secondary)
            
            Toggle("GPU", isOn: $useGPU)
                .toggleStyle(.switch)
                .labelsHidden()
                .disabled(viewModel.isSwitchingBackend || !viewModel.isEngineReady)
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
            // Loading spinner.
            if !viewModel.isEngineReady && viewModel.statusMessage.contains("Loading") {
                ProgressView()
                    .scaleEffect(0.6)
            }
            
            // Calendar agent indicator.
            if viewModel.toolCallingEnabled && viewModel.isEngineReady {
                Image(systemName: "calendar.badge.checkmark")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
            
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundStyle(
                    viewModel.statusMessage.contains("Failed") ? .red : .secondary
                )
            
            Spacer()
            
            // Model selector button.
            Button(action: {
                availableModels = ModelDiscovery.findAllModels()
                if availableModels.count > 1 {
                    showModelPicker = true
                }
            }) {
                Image(systemName: "doc.badge.gearshape")
                    .font(.caption)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
        .background(Color(.systemGray6))
    }
    
    // MARK: - Benchmark Bar
    
    @ViewBuilder
    private var benchmarkBar: some View {
        if !viewModel.benchmarkText.isEmpty {
            Text(viewModel.benchmarkText)
                .font(.caption2)
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
                            Text("Thinking...")
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
            .onTapGesture {
                isInputFocused = false
            }
            .onChange(of: viewModel.messages.count) { _, _ in
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
                viewModel.toolCallingEnabled
                    ? "Ask about your schedule..."
                    : "Ask something...",
                text: $inputText
            )
            .textFieldStyle(.plain)
            .focused($isInputFocused)
            .onSubmit { send() }
            .disabled(!viewModel.isEngineReady)
            
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(
                        inputText.isEmpty || !viewModel.isEngineReady
                            ? .gray : .blue
                    )
            }
            .disabled(
                inputText.isEmpty ||
                viewModel.isGenerating ||
                !viewModel.isEngineReady
            )
        }
        .padding()
    }
    
    // MARK: - Model Picker
    
    private var modelPickerSheet: some View {
        NavigationStack {
            List(availableModels, id: \.path) { model in
                Button(action: {
                    showModelPicker = false
                    viewModel.cleanup()
                    
                    // Save the user's choice so it loads automatically next time!
                    ModelDiscovery.saveLastUsedModel(path: model.path)
                    
                    viewModel.initialize(modelPath: model.path, useGPU: useGPU)
                }) {
                    VStack(alignment: .leading) {
                        Text(model.name)
                            .font(.body)
                        Text(model.path)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
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
    
    private func discoverAndLoadModel() {
        if let modelPath = ModelDiscovery.findModel() {
            viewModel.initialize(modelPath: modelPath, useGPU: useGPU)
        } else {
            viewModel.statusMessage =
                "No model found — copy a .litertlm file to the app's Documents folder"
        }
    }
}

// MARK: - MessageRow

/// Routes each message to the appropriate visual component based on role.
struct MessageRow: View {
    let message: ChatMessage
    
    var body: some View {
        switch message.role {
        case .user:
            UserBubble(content: message.content)
        case .assistant:
            AssistantBubble(content: message.content)
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

// MARK: - Assistant Bubble

struct AssistantBubble: View {
    let content: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("EdgeBridge")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
}

// MARK: - Tool Action Card

/// Displays tool calls and results as compact, visually distinct cards.
/// Tool calls show what the model is doing (e.g., "📅 Checking schedule...").
/// Tool results show the outcome (e.g., "✅ Found 3 events").
struct ToolActionCard: View {
    let content: String
    let isResult: Bool
    
    var body: some View {
        HStack(spacing: 8) {
            // Accent bar on the left edge.
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

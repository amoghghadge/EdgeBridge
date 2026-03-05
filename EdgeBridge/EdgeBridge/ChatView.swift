//
//  ChatView.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

import Foundation

// ============================================================================
// ChatView.swift  (v2 — Real LiteRT-LM Integration)
//
// Updated to work with EngineViewModel v4 (C bridge API) instead of the
// old C++ stub. Key changes from v1:
//
// 1. Model discovery: scans the app's Documents directory for .litertlm
//    files and presents a picker if multiple are found.
// 2. Status bar: shows engine loading state, model name, and errors.
// 3. Streaming support: assistant messages update live as tokens arrive.
// 4. Cleanup on disappear: properly frees C++ engine resources.
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
            // -- Header with backend toggle --
            headerBar
            
            Divider()
            
            // -- Status bar (model loading, errors) --
            statusBar
            
            // -- Benchmark bar --
            if !viewModel.benchmarkText.isEmpty {
                Text(viewModel.benchmarkText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
                    .padding(.vertical, 4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.systemGray6))
            }
            
            // -- Message list --
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        
                        if viewModel.isGenerating {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Generating...")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal)
                        }
                        
                        // Invisible anchor for auto-scrolling.
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
                // Also scroll when the last message's content changes
                // (streaming updates).
                .onChange(of: viewModel.messages.last?.content ?? "") { _, _ in
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            
            Divider()
            
            // -- Input bar --
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
    
    // MARK: - Subviews
    
    private var headerBar: some View {
        HStack {
            Text("EdgeBridge")
                .font(.headline)
            
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
        .padding()
    }
    
    private var statusBar: some View {
        HStack(spacing: 8) {
            if !viewModel.isEngineReady && viewModel.statusMessage.contains("Loading") {
                ProgressView()
                    .scaleEffect(0.6)
            }
            
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundStyle(
                    viewModel.statusMessage.contains("Failed")
                        ? .red
                        : .secondary
                )
            
            Spacer()
            
            // Model selector button — tap to pick a different model.
            Button(action: {
                availableModels = ModelDiscovery.findAllModels()
                if availableModels.count > 1 {
                    showModelPicker = true
                }
            }) {
                Image(systemName: "doc.badge.gearshape")
                    .font(.caption)
            }
            .disabled(!viewModel.isEngineReady && !viewModel.statusMessage.contains("Failed"))
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
        .background(Color(.systemGray6))
    }
    
    private var inputBar: some View {
        HStack(spacing: 12) {
            TextField("Ask something...", text: $inputText)
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
    
    private var modelPickerSheet: some View {
        NavigationStack {
            List(availableModels, id: \.path) { model in
                Button(action: {
                    showModelPicker = false
                    viewModel.cleanup()
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
    
    /// Scans the Documents directory for .litertlm files and loads the
    /// first one found. If no models are found, shows instructions.
    private func discoverAndLoadModel() {
        if let modelPath = ModelDiscovery.findModel() {
            viewModel.initialize(modelPath: modelPath, useGPU: useGPU)
        } else {
            viewModel.statusMessage =
                "No model found — copy a .litertlm file to the app's Documents folder"
        }
    }
}

// MARK: - MessageBubble

struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }
            
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(roleLabel)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                
                Text(message.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(bubbleColor)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            
            if message.role != .user { Spacer(minLength: 60) }
        }
    }
    
    private var roleLabel: String {
        switch message.role {
        case .user: return "You"
        case .assistant: return "EdgeBridge"
        case .toolResult: return "Tool Result"
        case .system: return "System"
        }
    }
    
    private var bubbleColor: Color {
        switch message.role {
        case .user: return .blue
        case .assistant: return Color(.systemGray5)
        case .toolResult: return Color(.systemGreen).opacity(0.2)
        case .system: return Color(.systemOrange).opacity(0.2)
        }
    }
}

// MARK: - Preview

#Preview {
    ChatView()
}

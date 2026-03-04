//
//  ChatView.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

import Foundation

// ============================================================================
// ChatView.swift
//
// A minimal chat interface to verify the full pipeline works:
// SwiftUI → Swift EngineViewModel → C++ EdgeBridgeEngine → response back
//
// FIXES in this version:
// 1. Keyboard dismissal: tap anywhere on the chat area to dismiss keyboard
// 2. Separated loading states: backend switching no longer shows "Generating..."
// ============================================================================

import SwiftUI

struct ChatView: View {
    @State private var viewModel = EngineViewModel()
    @State private var inputText = ""
    @State private var useGPU = false
    
    // Access the keyboard dismiss action so we can hide it on tap.
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            // -- Header with backend toggle --
            HStack {
                Text("EdgeBridge")
                    .font(.headline)
                
                Spacer()
                
                // Backend indicator + switching state
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
                    .disabled(viewModel.isSwitchingBackend)
                    .onChange(of: useGPU) { _, newValue in
                        viewModel.toggleBackend(useGPU: newValue)
                    }
            }
            .padding()
            
            Divider()
            
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
            // The tap gesture on the ScrollView dismisses the keyboard
            // when the user taps on the chat area.
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        
                        // Only show "Generating..." during actual inference,
                        // not during backend switching.
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
                        
                        Color.clear
                            .frame(height: 1)
                            .id("bottomAnchor")
                    }
                    .padding()
                }
                .onTapGesture {
                    // Dismiss the keyboard when tapping on the chat area.
                    isInputFocused = false
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    withAnimation {
                        proxy.scrollTo("bottomAnchor", anchor: .bottom)
                    }
                }
            }
            
            Divider()
            
            // -- Input bar --
            HStack(spacing: 12) {
                TextField("Ask something or give a command...", text: $inputText)
                    .textFieldStyle(.plain)
                    .focused($isInputFocused)
                    .onSubmit { send() }
                
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(inputText.isEmpty ? .gray : .blue)
                }
                .disabled(inputText.isEmpty || viewModel.isGenerating)
            }
            .padding()
        }
        .onAppear {
            // Initialize the C++ engine when the view appears.
            // The model path is a placeholder — we'll use the real
            // .litertlm path once we integrate the models.
            viewModel.initialize(modelPath: "placeholder.litertlm", useGPU: false)
        }
    }
    
    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        
        // Dismiss keyboard after sending.
        isInputFocused = false
        
        viewModel.sendMessage(text)
    }
}

// ---------------------------------------------------------------------------
// MessageBubble — renders a single chat message.
// User messages are right-aligned, assistant messages are left-aligned.
// ---------------------------------------------------------------------------
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }
            
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : "EdgeBridge")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                
                Text(message.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        message.role == .user
                        ? Color.blue
                        : Color(.systemGray5)
                    )
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            
            if message.role != .user { Spacer(minLength: 60) }
        }
    }
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------
#Preview {
    ChatView()
}

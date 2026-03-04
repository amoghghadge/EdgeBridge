//
//  ContentView.swift
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

import SwiftUI
import Combine

class EngineWrapper: ObservableObject {
    let objectWillChange = ObservableObjectPublisher()
    var engine = GemmaEngine()
}

struct ContentView: View {
    @StateObject private var wrapper = EngineWrapper()
    @State private var responseText = "Waiting for engine..."
    
    // 0 = CPU, 1 = GPU (Metal), 2 = NPU (Core ML)
    @State private var selectedBackend = 0
    
    var body: some View {
        VStack (spacing: 20) {
            Image(systemName: "memorychip")
              .imageScale(.large)
              .font(.largeTitle)
              .foregroundStyle(.tint)
            
            Text(responseText)
              .font(.headline)
              .multilineTextAlignment(.center)
            
            // The Delegate Dilemma Toggle
            Picker("Hardware Backend", selection: $selectedBackend) {
                Text("CPU").tag(0)
                Text("GPU (Metal)").tag(1)
                Text("NPU (Core ML)").tag(2)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            
            Button("Load Model") {
                // Instantly update UI so the user knows it is working
                responseText = "Loading model in background...\nThis involves massive memory allocation."
                
                if let modelPath = Bundle.main.path(forResource: "gemma3_1b_int4", ofType: "tflite") {
                    let cppPath = std.string(modelPath)
                    let backendToUse = Int32(selectedBackend)
                    
                    // 1. Push the heavy C++ memory allocation to a background thread
                    DispatchQueue.global(qos: .userInitiated).async {
                        
                        let success = wrapper.engine.loadModel(cppPath, backendToUse)
                        
                        // 2. Push the UI update back to the Main Thread once C++ is done
                        DispatchQueue.main.async {
                            if success {
                                let backendName = selectedBackend == 1 ? "Metal GPU" : (selectedBackend == 2 ? "Neural Engine" : "CPU")
                                responseText = "Model loaded safely via mmap!\nRunning on: \(backendName)\n(Note: Unsupported GenAI ops will fallback to CPU)"
                            } else {
                                responseText = "Failed to load model or apply delegates."
                            }
                        }
                    }
                } else {
                    responseText = "Model file not found in Xcode bundle."
                }
            }
          .buttonStyle(.borderedProminent)
        }
      .padding()
    }
}

#Preview {
    ContentView()
}

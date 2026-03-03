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
    
    // NEW: State to track the selected hardware (0=CPU, 1=GPU, 2=NPU)
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
            
            // NEW: A toggle switch for the Delegate Dilemma
            Picker("Hardware Backend", selection: $selectedBackend) {
                Text("CPU").tag(0)
                Text("GPU (Metal)").tag(1)
                Text("NPU (Core ML)").tag(2)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            
            Button("Load Model") {
                // Instantly update UI so the user knows it is working
                responseText = "Loading model in background... This may take a moment."
                
                if let modelPath = Bundle.main.path(forResource: "gemma3_1b_int4", ofType: "tflite") {
                    let cppPath = std.string(modelPath)
                    let backendToUse = Int32(selectedBackend)
                    
                    // Push the heavy C++ initialization to a background thread
                    DispatchQueue.global(qos: .userInitiated).async {
                        let success = wrapper.engine.loadModel(cppPath, backendToUse)
                        
                        // Push the UI update back to the Main Thread once C++ is done
                        DispatchQueue.main.async {
                            if success {
                                let backendName = selectedBackend == 1 ? "Metal GPU" : (selectedBackend == 2 ? "Neural Engine" : "CPU")
                                responseText = "Model loaded safely via mmap!\nRunning on: \(backendName)"
                            } else {
                                responseText = "Failed to apply hardware delegate. Running on fallback CPU."
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

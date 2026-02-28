//
//  ContentView.swift
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

import SwiftUI
import Combine

// We create a Swift Class (Reference Type) to safely hold the C++ engine
class EngineWrapper: ObservableObject {
    // Manually provide the publisher to bypass the compiler bug
    let objectWillChange = ObservableObjectPublisher()
    
    var engine = GemmaEngine()
}

struct ContentView: View {
    // We use @StateObject so the engine is initialized exactly once and never copied
    @StateObject private var wrapper = EngineWrapper()
    
    @State private var responseText = "Waiting for engine..."
    
    var body: some View {
        VStack (spacing: 20) {
            Image(systemName: "memorychip")
               .imageScale(.large)
               .font(.largeTitle)
               .foregroundStyle(.tint)
            
            Text(responseText)
               .font(.headline)
               .multilineTextAlignment(.center)
            
            // call the C++ function inside an Action Closure
            Button("Load Model (mmap)") {
                // Find where iOS stored the 1GB file on the device's SSD
                if let modelPath = Bundle.main.path(forResource: "gemma3_1b_int4", ofType: "tflite") {
                    // Convert Swift String to C++ std::string using Swift 5.9 Interop
                    let cppPath = std.string(modelPath)
                    
                    // Call our C++ memory-mapping function
                    let success = wrapper.engine.loadModel(cppPath)
                    
                    if success {
                        responseText = "Model loaded safely via mmap!\nJetsam evaded."
                    } else {
                        responseText = "Failed to memory-map the model."
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

//
//  ContentView.swift
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

import SwiftUI

struct ContentView: View {
    // 1. We wrap the C++ engine in @State so SwiftUI knows it might change
    @State private var engine = GemmaEngine()
        
    // 2. A state variable to hold the C++ output
    @State private var responseText = "Waiting for engine..."
    
    var body: some View {
        VStack (spacing: 20) {
            Image(systemName: "cpu")
                .imageScale(.large)
                .font(.largeTitle)
                .foregroundStyle(.tint)
            
            Text(responseText)
                .font(.headline)
            
            // 3. We call the C++ function inside an Action Closure
            Button("Test C++ Bridge") {
                responseText = String(engine.testBridge())
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

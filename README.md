# EdgeBridge: Custom iOS AI Runtime for Gemma 3

Mobile applications are shifting from cloud-dependent API wrappers to autonomous, on-device AI agents. Running Large Language Models (LLMs) natively on mobile hardware provides three critical advantages:

1.  **Zero Latency:** No network roundtrips, enabling real-time streaming and seamless UX.
2.  **Absolute Privacy:** User data never leaves the device.
3.  **Zero Cloud Cost:** Compute is offloaded to the user's hardware (Apple Neural Engine / GPU).

However, achieving this on iOS in 2026 presents a massive systems engineering challenge. While Google recently released the **Gemma 3 1B** model—which is small enough to fit on a phone—the high-level, easy-to-use SDKs (like MediaPipe) currently lack full native support for Gemma 3 on iOS.

My goal with this project is to bridge this gap of deploying to edge devices, and build a native iOS application that acts as an intelligent, offline AI agent. It will use a "cascade" of two different Google AI models (FunctionGemma 270M and Gemma 3 1B) to handle tasks and chat, powered by a custom-built C++ inference engine that talks directly to Swift.

## Component 1: The Engine (Custom LiteRT Build & Quantization)

By relying purely on pre-compiled CocoaPods or standard Apple Core ML, I would be blocked. Apple's Core ML ecosystem can introduce behavioral inconsistencies when porting Google-trained models, and pre-compiled TensorFlow Lite pods are bloated with hundreds of unused legacy operations.

**The Solution:** EdgeBridge bypasses the high-level wrappers entirely. This project involves downloading Google's raw **LiteRT** (formerly TensorFlow Lite) C++ source code, utilizing Google's **Bazel** build system, and executing a **Selective Build**. By analyzing the specific mathematical operations required by the Gemma 3 model, I compiled a custom, stripped-down C++ inference engine tailored explicitly for the physical iPhone (arm64) architecture.

### **Architectural Trade-offs & Design Choices**

Before writing code, several deliberate infrastructure choices were made:

**1. Selective Compilation vs. Pre-compiled CocoaPods**

* **Trade-off:** Ease of use vs. Binary Size.
* **Decision:** I chose to compile the C++ engine from source using Bazel. Pre-compiled libraries contain massive amounts of overhead for legacy models (audio, image segmentation, etc.). By feeding the Gemma 3 model directly into the Bazel compiler, the build system selectively included *only* the specific GenAI operations needed (such as INT4 memory decompression).
* **Result:** The final compiled AI engine (`TensorFlowLiteC.framework`) is a remarkably lightweight **11.6 MB**. This avoids severe app bloat from a standard, pre-compiled "Fat" TensorFlow Lite framework with all architectures and operations that could easily be 50MB to 100MB+.

**2. INT4 Quantization Precision**

* **Trade-off:** Model Quality (Perplexity) vs. RAM footprint.
* **Decision:** I utilized the **INT4 Quantization-Aware Trained (QAT)** version of Gemma 3 1B.
* **Result:** At roughly ~540MB to ~892MB, this 4-bit integer compression allows the model to sit comfortably in the background of an iPhone 15/16 Pro (which has 8GB of total system RAM) without triggering the OS Jetsam process to terminate the app during heavy multitasking.

**3. Physical Device (`arm64`) vs. iOS Simulator (`x86_64`)**

* **Trade-off:** Developer convenience vs. Accurate Profiling & Build Stability.
* **Decision:** I explicitly stripped the `x86_64` (Intel Mac) architecture from the build target. Apple has deprecated "Fat Frameworks," and attempting to build them often causes SDK collisions in modern build systems. More importantly, testing an AI model on a Mac Simulator with 16GB+ of RAM defeats the purpose of on-device engineering.
* **Result:** The framework targets `arm64` exclusively, ensuring that when the model runs, it is accurately subjected to the iPhone's physical thermal throttling and strict iOS Jetsam memory eviction limits.

### **The Engineering Journey (Challenges & Solutions)**

Building Google's internal infrastructure on a local macOS environment is notoriously difficult. Below is the documentation of the system-level challenges encountered during the Bazel compilation phase and the architectural patches applied to solve them.

**Challenge 1: The `.task` Bundle Architecture**

**The Problem:** Google distributes Gemma 3 for edge devices as a `.task` file. However, the low-level C++ Bazel script (`build_frameworks.sh`) explicitly requires a `.tflite` FlatBuffer file to analyze the model's computational graph.
**The Solution:** A `.task` file is not a proprietary format; it is fundamentally a standard `.tflite` file with a ZIP archive (containing the tokenizer dictionary) appended to the end of the binary. By modifying the file extension directly to `.tflite`, the Bazel compiler was able to read the mathematical graph headers at the start of the file and successfully extract the required C++ operators, safely ignoring the trailing metadata.

**Challenge 2: The "LiteRT" Rebranding Desync**

**The Problem:** In late 2024/2025, Google rebranded "TensorFlow Lite" to "LiteRT". The new Python `./configure` script generates a hidden config file named `.litert_configure.bazelrc`. However, the older bash build scripts were hardcoded to `grep` for the legacy `.tf_configure.bazelrc` file and the specific `TF_CONFIGURE_IOS=1` flag. This resulted in an immediate build crash.
**The Solution:** Rather than waiting for an upstream fix, I manually bridged the gap between the scripts. I duplicated the configuration file to match the legacy naming convention and injected the exact string `TF_CONFIGURE_IOS=1` into the environment variables, bypassing the hardcoded validation check.

**Challenge 3: Toolchain Rot (`armv7`)**

**The Problem:** The default Google build script attempted to compile the C++ framework for `x86_64`, `arm64`, and `armv7`. Apple dropped 32-bit (`armv7`) support years ago. Modern Bazel Apple toolchains lack the definitions for it, causing a fatal `key "ios_armv7" not found in dictionary` crash.
**The Solution:** I overrode the script's internal variables via the command line (`--target_archs=arm64`), surgically removing the dead 32-bit architecture and forcing a pure, modern Apple Silicon build.

**Challenge 4: The Host vs. Target Sandbox Collision**

**The Problem:** Bazel compiles code inside an isolated, hermetic "sandbox." While compiling the 2,000+ C++ files for the iOS Target, it successfully found the iPhone SDK. However, at the very end of the build, it attempted to compile a small Apple Instruments profiling tool (`signpost_profiler.mm`) for the Mac Host. The sandbox stripped the `DEVELOPER_DIR` and `SDKROOT` environment variables, blinding the `wrapped_clang_pp` compiler to the location of Xcode. Attempting to forcefully inject the paths globally caused a fatal Java exception (`Multiple entries with same key: SDKROOT`) because Bazel tried to map the iOS SDK and Mac SDK to the same dictionary key.
**The Solution:**
1.  Used `sudo xcode-select -s` to register the Xcode path globally at the macOS level.
2.  Passed the `--noincompatible_strict_action_env` flag to Bazel, poking a hole in the sandbox so the host compiler could inherit the system environment naturally without colliding with the iOS target's internal paths.

**Challenge 5: Surgically Bypassing Broken Telemetry**

**The Problem:** Even with the sandbox resolved, the `signpost_profiler` continued to fail due to missing internal dependency declarations within the Google repo.
**The Solution:** Because `signpost_profiler` is solely used for Apple-specific debugging telemetry and is completely unnecessary for Gemma 3 inference, I chose to sever the knot. I used `sed` to dynamically remove the profiler target from the `BUILD` graph and stripped the `__APPLE__` macros from the C++ source code (`platform_profiler.cc`), forcing the compiler to skip it entirely.

### **Phase 1 Result**

The build completed successfully, generating a highly optimized, 11.6 MB `TensorFlowLiteC_framework.zip` containing the exact, custom-compiled C++ neural network operations required to run Gemma 3 1B locally on an iPhone.

## Component 2: The Core Bridge (Swift 5.9 C++ Interoperability)

Historically, Google's cross-platform mobile architecture (seen in YouTube, Chrome, and Search) relied heavily on shared C++ backends for core business logic. To connect this C++ logic to the iOS UI layer, engineers had to write extensive Objective-C++ ( `.mm` ) bridging headers. This legacy approach introduced significant overhead, increased binary size, and created a massive mental tax for developers constantly context-switching between Swift, Objective-C, and C++.

With the release of **Swift 5.9**, Apple introduced bidirectional C++ Interoperability. By leveraging this cutting-edge compiler feature, **EdgeBridge** directly instantiates and commands the custom C++ LiteRT inference engine from native Swift code. This eliminates the Objective-C middleman, resulting in lower memory overhead, faster execution times, and a highly modernized "Google-scale" codebase.

**The Goal:** Establish a direct, zero-cost communication layer between the native iOS UI (SwiftUI) and the custom LiteRT inference engine (C++), bypassing legacy wrapper architectures.

### **Architectural Trade-offs & Design Choices**

**1. Direct C++ Interop vs. Objective-C++ Wrappers**

* **Trade-off:** Developer velocity & modern tooling vs. Legacy compatibility.
* **Decision:** I chose to explicitly enable Swift's `C++ / Objective-C++` interoperability mode in the Xcode Build Settings.
* **Result:** This achieves "Zero-Cost Abstraction." Passing data between the UI and the AI engine no longer requires copying data into intermediate `NSString` or `NSObject` wrappers. This tight coupling is critical for minimizing "Time to First Token" (TTFT) latency during inference.

**2. Value Semantics (Swift) vs. Reference Semantics (C++)**

* **Trade-off:** Memory Safety vs. Mutability.
* **Decision:** By default, Swift 5.9 imports C++ classes as Swift `structs` (Value Types). I architected the SwiftUI layer to respect this strict value-semantic boundary by wrapping the C++ `GemmaEngine` in a SwiftUI `@State` property wrapper.
* **Result:** This guarantees memory safety and prevents the iOS UI from querying the engine while the C++ thread is actively mutating the KV cache during token generation.

### **The Engineering Journey (Challenges & Solutions)**

**Challenge 1: The `mutating` Member View Crash**

**The Problem:** During the initial bridge testing, attempting to call a C++ function (`testBridge()`) directly from the SwiftUI `body` resulted in a compiler error: `Cannot use mutating member on immutable value: 'self' is immutable`. Because Swift imports C++ classes as value types, it assumes any C++ member function without a `const` qualifier might mutate the object's underlying memory. This violates SwiftUI's strict declarative, immutable rendering loop.

**The Solution:**
1.  **C++ Const Correctness:** I refactored the C++ definitions, applying `const` qualifiers to deterministic methods (e.g., `std::string testBridge() const;`) so Swift could safely parse them as non-mutating.
2.  **Event-Driven Architecture:** For the future stateful generative loop (which *must* mutate the KV cache and tensor buffers), I decoupled the engine from the view hierarchy. The C++ instance is held in `@State`, and inference commands are dispatched strictly through asynchronous Action Closures (e.g., button presses), adhering to iOS concurrency rules.

**Challenge 2: Bare-Metal Execution & Provisioning Profiles**

**The Problem:** Because the custom LiteRT engine built in Phase 1 was compiled exclusively for the `arm64` architecture to accurately test memory limits, it could not be run on the Xcode iOS Simulator (which requires `x86_64` or simulator-specific binaries). Pushing the raw code to a physical iPhone 15 Pro resulted in an Apple security block: `Signing requires a development team`.
**The Solution:** I configured a local Personal Team provisioning profile in Xcode to generate a free signing certificate. To bypass Apple's strict sandboxing for sideloaded apps, I manually enabled "Developer Mode" deep within the iOS Privacy & Security settings and established certificate trust, allowing the highly-optimized `arm64` binary to execute natively on the iPhone's silicon.

### **Phase 2 Result**

The native iOS application now successfully compiles and runs native C++ code without Objective-C wrappers, seamlessly instantiating the engine and converting `std::string` outputs directly into Swift `String` types for rendering.
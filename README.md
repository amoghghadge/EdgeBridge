## EdgeBridge: Custom iOS AI Runtime for Gemma 3

📖 **Project Motivation (The "Why")**

Mobile applications are shifting from cloud-dependent API wrappers to autonomous, on-device AI agents. Running Large Language Models (LLMs) natively on mobile hardware provides three critical advantages:

1.  **Zero Latency:** No network roundtrips, enabling real-time streaming and seamless UX.
2.  **Absolute Privacy:** User data never leaves the device.
3.  **Zero Cloud Cost:** Compute is offloaded to the user's hardware (Apple Neural Engine / GPU).

However, achieving this on iOS in 2026 presents a massive systems engineering challenge. While Google recently released the **Gemma 3 1B** model—which is small enough to fit on a phone—the high-level, easy-to-use SDKs (like MediaPipe) currently lack full native support for Gemma 3 on iOS.

My goal with this project is to bridge this gap of deploying to edge devices, and build a native iOS application that acts as an intelligent, offline AI agent. It will use a "cascade" of two different Google AI models (FunctionGemma 270M and Gemma 3 1B) to handle tasks and chat, powered by a custom-built C++ inference engine that talks directly to Swift.

### Component 1: The Engine (Custom LiteRT Build & Quantization)

By relying purely on pre-compiled CocoaPods or standard Apple Core ML, I would be blocked. Apple's Core ML ecosystem can introduce behavioral inconsistencies when porting Google-trained models, and pre-compiled TensorFlow Lite pods are bloated with hundreds of unused legacy operations.

**The Solution:** EdgeBridge bypasses the high-level wrappers entirely. This project involves downloading Google's raw **LiteRT** (formerly TensorFlow Lite) C++ source code, utilizing Google's **Bazel** build system, and executing a **Selective Build**. By analyzing the specific mathematical operations required by the Gemma 3 model, I compiled a custom, stripped-down C++ inference engine tailored explicitly for the physical iPhone (arm64) architecture.

#### **Architectural Trade-offs & Design Choices**

Before writing code, several deliberate infrastructure choices were made:

**1. Selective Compilation vs. Pre-compiled CocoaPods**

* **Trade-off:** Ease of use vs. Binary Size.
* **Decision:** I chose to compile the C++ engine from source using Bazel. Pre-compiled libraries contain massive amounts of overhead for legacy models (audio, image segmentation, etc.). By feeding the Gemma 3 model directly into the Bazel compiler, the build system selectively included *only* the specific GenAI operations needed (such as INT4 memory decompression).
* **Result:** Result: The final compiled AI engine (`TensorFlowLiteC.framework`) is a remarkably lightweight **11.6 MB**, avoiding severe app bloat.

**2. Physical Device (`arm64`) vs. iOS Simulator (`x86_64`)**

* **Trade-off:** Developer convenience vs. Accurate Profiling & Build Stability.
* **Decision:** I explicitly stripped the `x86_64` (Intel Mac) architecture from the build target. Apple has deprecated "Fat Frameworks," and attempting to build them often causes SDK collisions in modern build systems. More importantly, testing an AI model on a Mac Simulator with 16GB+ of RAM defeats the purpose of on-device engineering.
* **Result:** The framework targets `arm64` exclusively, ensuring that when the model runs, it is accurately subjected to the iPhone's physical thermal throttling and strict iOS Jetsam memory eviction limits.

**3. INT4 Quantization Precision**

* **Trade-off:** Model Quality (Perplexity) vs. RAM footprint.
* **Decision:** I utilized the **INT4 Quantization-Aware Trained (QAT)** version of Gemma 3 1B.
* **Result:** At roughly ~540MB to ~892MB, this 4-bit integer compression allows the model to sit comfortably in the background of an iPhone 15/16 Pro (which has 8GB of total system RAM) without triggering the OS Jetsam process to terminate the app during heavy multitasking.

#### **The Engineering Journey (Challenges & Solutions)**

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

🎉 **Phase 1 Result**

The build completed successfully, generating a highly optimized, 11.6 MB `TensorFlowLiteC_framework.zip` containing the exact, custom-compiled C++ neural network operations required to run Gemma 3 1B locally on an iPhone.

**Next Steps (Phase 2):** Integrating this raw C++ engine directly into a native SwiftUI application using **Swift 5.9 C++ Interoperability**, bypassing Objective-C wrappers entirely to manage model memory-mapping (`mmap`) and Hardware Delegates (Metal/Neural Engine).
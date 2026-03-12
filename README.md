# EdgeBridge: Custom iOS AI Runtime for Gemma 3

Mobile applications are shifting from cloud-dependent API wrappers to autonomous, on-device AI agents. Running Large Language Models (LLMs) natively on mobile hardware provides three critical advantages:

1.  **Zero Latency:** No network roundtrips, enabling real-time streaming and seamless UX.
2.  **Absolute Privacy:** User data never leaves the device.
3.  **Zero Cloud Cost:** Compute is offloaded to the user's hardware (Apple Neural Engine / GPU).

However, achieving this on iOS in 2026 presents a massive systems engineering challenge. While Google recently released the **Gemma 3 1B** model—which is small enough to fit on a phone—the high-level, easy-to-use SDKs (like MediaPipe) currently lack full native support for Gemma 3 on iOS.

My goal with this project is to bridge this gap of deploying to edge devices, and build a native iOS application that acts as an intelligent, offline AI agent. It will use a "cascade" of two different Google AI models (FunctionGemma 270M and Gemma 3 1B) to handle tasks and chat, powered by a custom-built C++ inference engine that talks directly to Swift.

// ============================================================================
// EdgeBridge-Bridging-Header.h
//
// Objective-C Bridging Header for the EdgeBridge iOS app.
// This is the ONLY LiteRT-LM header that Xcode needs to see.
// All complex C++ dependencies (absl, protobuf, nlohmann/json,
// XNNPACK, etc.) are hidden inside libLiteRTLM.a behind this
// clean C interface.
//
// Configure in Xcode:
//   Build Settings → Swift Compiler - General →
//   Objective-C Bridging Header = EdgeBridge/EdgeBridge-Bridging-Header.h
// ============================================================================

#include "litert_bridge_api.h"

//
//  GemmaEngine.hpp
//  EdgeBridgeHelloWorld
//
//  Created by Amogh Ghadge on 2/27/26.
//

#ifndef GemmaEngine_hpp
#define GemmaEngine_hpp

#include <stdio.h>
#include <string>

class GemmaEngine {
public:
    // Constructor
    GemmaEngine();
    
    // A simple test function we will call from Swift
    std::string testBridge() const;
};

#endif /* GemmaEngine_hpp */

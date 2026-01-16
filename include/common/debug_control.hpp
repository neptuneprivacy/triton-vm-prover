#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace triton_vm {
namespace debug {

/**
 * Debug and Profile Control
 * 
 * Environment variables:
 * - TRITON_VM_PROFILE: Enable/disable profiling output (timing measurements)
 *   Set to "1" or "true" to enable, "0" or "false" (or unset) to disable
 * 
 * - TRITON_VM_DEBUG: Enable/disable debug output (detailed state dumps)
 *   Set to "1" or "true" to enable, "0" or "false" (or unset) to disable
 */

// Check if profiling is enabled
inline bool is_profile_enabled() {
    static int cached = -1;
    if (cached == -1) {
        const char* env = std::getenv("TRITON_VM_PROFILE");
        cached = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return cached == 1;
}

// Check if debug printing is enabled
inline bool is_debug_enabled() {
    static int cached = -1;
    if (cached == -1) {
        const char* env = std::getenv("TRITON_VM_DEBUG");
        cached = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return cached == 1;
}

} // namespace debug
} // namespace triton_vm

// Convenience macros for profile and debug printing
// These can be used in both C++ and CUDA code

// Profile printing (timing measurements)
#define TRITON_PROFILE_ENABLED() (triton_vm::debug::is_profile_enabled())

#define TRITON_PROFILE_PRINT(...) \
    do { \
        if (triton_vm::debug::is_profile_enabled()) { \
            printf(__VA_ARGS__); \
        } \
    } while(0)

#define TRITON_PROFILE_COUT(expr) \
    do { \
        if (triton_vm::debug::is_profile_enabled()) { \
            std::cout << expr; \
        } \
    } while(0)

// Debug printing (detailed state dumps)
#define TRITON_DEBUG_ENABLED() (triton_vm::debug::is_debug_enabled())

#define TRITON_DEBUG_PRINT(...) \
    do { \
        if (triton_vm::debug::is_debug_enabled()) { \
            printf(__VA_ARGS__); \
        } \
    } while(0)

#define TRITON_DEBUG_COUT(expr) \
    do { \
        if (triton_vm::debug::is_debug_enabled()) { \
            std::cout << expr; \
        } \
    } while(0)

#define TRITON_DEBUG_FPRINTF(stream, ...) \
    do { \
        if (triton_vm::debug::is_debug_enabled()) { \
            fprintf(stream, __VA_ARGS__); \
        } \
    } while(0)

// Combined profile/debug conditional block
// Use this to wrap entire blocks of code that should only run when profiling/debugging
#define TRITON_IF_PROFILE if (triton_vm::debug::is_profile_enabled())
#define TRITON_IF_DEBUG if (triton_vm::debug::is_debug_enabled())

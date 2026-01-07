#pragma once

#include "types/b_field_element.hpp"

namespace triton_vm {

/**
 * ProcessorMainColumn - Enum for the 39 processor table columns
 * 
 * Matches Rust's ProcessorMainColumn enum.
 */
enum class ProcessorMainColumn {
    CLK = 0,
    IsPadding = 1,
    IP = 2,
    CI = 3,
    NIA = 4,
    IB0 = 5,
    IB1 = 6,
    IB2 = 7,
    IB3 = 8,
    IB4 = 9,
    IB5 = 10,
    IB6 = 11,
    JSP = 12,
    JSO = 13,
    JSD = 14,
    ST0 = 15,
    ST1 = 16,
    ST2 = 17,
    ST3 = 18,
    ST4 = 19,
    ST5 = 20,
    ST6 = 21,
    ST7 = 22,
    ST8 = 23,
    ST9 = 24,
    ST10 = 25,
    ST11 = 26,
    ST12 = 27,
    ST13 = 28,
    ST14 = 29,
    ST15 = 30,
    OpStackPointer = 31,
    HV0 = 32,
    HV1 = 33,
    HV2 = 34,
    HV3 = 35,
    HV4 = 36,
    HV5 = 37,
    ClockJumpDifferenceLookupMultiplicity = 38
};

constexpr size_t PROCESSOR_COLUMN_COUNT = 39;

/**
 * Get the index of a processor column
 */
inline size_t processor_column_index(ProcessorMainColumn col) {
    return static_cast<size_t>(col);
}

} // namespace triton_vm


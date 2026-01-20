// Bytecode Interpreter for Constraint Evaluation
// Compact stack-based virtual machine for evaluating algebraic constraints

#pragma once

#include "field_arithmetic.cuh"

// Bytecode instruction opcodes
enum BytecodeOp : uint8_t {
    // Stack operations
    OP_LOAD_MAIN_ROW = 0,      // Push main_row[u16] (as XField, lifted from BField)
    OP_LOAD_AUX_ROW = 1,       // Push aux_row[u16]
    OP_LOAD_CHALLENGE = 2,     // Push challenges[u16]
    OP_LOAD_NEXT_MAIN_ROW = 3, // Push next_main_row[u16] (as XField, lifted from BField)
    OP_LOAD_NEXT_AUX_ROW = 4,  // Push next_aux_row[u16]

    // Constant loading (followed by 3 × u64 for XField coefficients)
    OP_LOAD_CONST_XFIELD = 5,  // Push XFieldElement constant
    OP_LOAD_CONST_BFIELD = 6,  // Push BFieldElement constant (as XField with c1=c2=0)

    // Arithmetic operations (pop 2, push 1)
    OP_XFIELD_ADD = 10,        // xfield_add(pop(), pop())
    OP_XFIELD_MUL = 11,        // xfield_mul(pop(), pop())

    // Stack operations
    OP_DUP = 15,               // Duplicate top of stack: push(peek())

    // Output operations
    OP_STORE_OUTPUT = 20,      // output[u16] = pop()
};

// Bytecode program structure
struct BytecodeProgram {
    const uint8_t* instructions;   // Instruction stream
    uint32_t instruction_count;    // Number of bytes in instruction stream
    uint32_t output_count;         // Number of constraints to evaluate
};

// Stack for interpreter execution
// 1024 should be sufficient now that cache bug is fixed
#define MAX_STACK_DEPTH 12

// Interpreter device function (called from other GPU code)
// __noinline__ prevents compiler from inlining this function, which would cause
// invalid PTX due to excessive code expansion from loop unrolling

__device__ __forceinline__ XFieldElement lift(BFieldElement loaded){
    return XFieldElement(loaded, bfield_zero(), bfield_zero());
}

__device__ __noinline__ void interpret_constraints(
    const uint8_t* bytecode,
    uint32_t bytecode_length,
    const BFieldElement* main_row,
    const XFieldElement* aux_row,
    const BFieldElement* next_main_row,  // nullptr for init/cons/term
    const XFieldElement* next_aux_row,   // nullptr for init/cons/term
    const XFieldElement* challenges,
    XFieldElement* output,
    uint32_t output_count)
{
    // Stack-based virtual machine
   // __shared__ XFieldElement stacks[128][MAX_STACK_DEPTH];

    //XFieldElement * stack = stacks[threadIdx.x];
    XFieldElement stack[MAX_STACK_DEPTH];

    int sp = 0;  // Stack pointer

    uint32_t pc = 0;  // Program counter

    while (pc < bytecode_length) {
        uint8_t op = bytecode[pc++];

        switch (op) {
            case OP_LOAD_MAIN_ROW: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                // Lift BFieldElement to XFieldElement
                BFieldElement loaded = main_row[index];
                if (threadIdx.x == 0 && blockIdx.x == 0 && index == 0) {
                    //printf("[GPU Debug] LOAD_MAIN_ROW[%d]: raw value = %llu\n", index, (unsigned long long)loaded.value);
                }
                stack[sp++] = XFieldElement(loaded, bfield_zero(), bfield_zero());
                break;
            }

            case OP_LOAD_AUX_ROW: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = aux_row[index];
                break;
            }

            case OP_LOAD_CHALLENGE: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = challenges[index];
                break;
            }

            case OP_LOAD_NEXT_MAIN_ROW: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                // Lift BFieldElement to XFieldElement
                stack[sp++] = XFieldElement(next_main_row[index], bfield_zero(), bfield_zero());
                break;
            }

            case OP_LOAD_NEXT_AUX_ROW: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = next_aux_row[index];
                break;
            }

            case OP_LOAD_CONST_XFIELD: {
                // Read 3 × 8 bytes for XFieldElement coefficients
                uint64_t c0, c1, c2;

                c0 = (uint64_t(bytecode[pc+0]) << 56) | (uint64_t(bytecode[pc+1]) << 48) |
                     (uint64_t(bytecode[pc+2]) << 40) | (uint64_t(bytecode[pc+3]) << 32) |
                     (uint64_t(bytecode[pc+4]) << 24) | (uint64_t(bytecode[pc+5]) << 16) |
                     (uint64_t(bytecode[pc+6]) << 8)  | uint64_t(bytecode[pc+7]);

                c1 = (uint64_t(bytecode[pc+8]) << 56)  | (uint64_t(bytecode[pc+9]) << 48) |
                     (uint64_t(bytecode[pc+10]) << 40) | (uint64_t(bytecode[pc+11]) << 32) |
                     (uint64_t(bytecode[pc+12]) << 24) | (uint64_t(bytecode[pc+13]) << 16) |
                     (uint64_t(bytecode[pc+14]) << 8)  | uint64_t(bytecode[pc+15]);

                c2 = (uint64_t(bytecode[pc+16]) << 56) | (uint64_t(bytecode[pc+17]) << 48) |
                     (uint64_t(bytecode[pc+18]) << 40) | (uint64_t(bytecode[pc+19]) << 32) |
                     (uint64_t(bytecode[pc+20]) << 24) | (uint64_t(bytecode[pc+21]) << 16) |
                     (uint64_t(bytecode[pc+22]) << 8)  | uint64_t(bytecode[pc+23]);

                pc += 24;

                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = XFieldElement(
                    bfield_from_raw(c0),
                    bfield_from_raw(c1),
                    bfield_from_raw(c2)
                );
                break;
            }

            case OP_LOAD_CONST_BFIELD: {
                // Read 8 bytes for BFieldElement, lift to XField
                uint64_t value;
                value = (uint64_t(bytecode[pc+0]) << 56) | (uint64_t(bytecode[pc+1]) << 48) |
                        (uint64_t(bytecode[pc+2]) << 40) | (uint64_t(bytecode[pc+3]) << 32) |
                        (uint64_t(bytecode[pc+4]) << 24) | (uint64_t(bytecode[pc+5]) << 16) |
                        (uint64_t(bytecode[pc+6]) << 8)  | uint64_t(bytecode[pc+7]);
                pc += 8;

                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = XFieldElement(bfield_from_raw(value), bfield_zero(), bfield_zero());
                break;
            }

            case OP_XFIELD_ADD: {
                if (sp < 2) return;  // Stack underflow protection
                XFieldElement b = stack[--sp];
                XFieldElement a = stack[--sp];
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = xfield_add(a, b);
                break;
            }

            case OP_XFIELD_MUL: {
                if (sp < 2) return;  // Stack underflow protection
                XFieldElement b = stack[--sp];
                XFieldElement a = stack[--sp];
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp++] = xfield_mul(a, b);
                break;
            }

            case OP_DUP: {
                // Duplicate top of stack
                if (sp < 1) return;  // Stack underflow protection (need at least 1 element)
                if (sp >= MAX_STACK_DEPTH) return;  // Stack overflow protection
                stack[sp] = stack[sp - 1];  // Duplicate top element
                sp++;
                break;
            }

            case OP_STORE_OUTPUT: {
                uint16_t index = (uint16_t(bytecode[pc]) << 8) | uint16_t(bytecode[pc+1]);
                pc += 2;
                if (sp < 1) return;  // Stack underflow protection
                XFieldElement val = stack[--sp];
                if (threadIdx.x == 0 && blockIdx.x == 0 && index == 0) {
                    //printf("[GPU Debug] STORE_OUTPUT[%d]: XField = (%llu, %llu, %llu)\n",
                    //       index,
                    //       (unsigned long long)val.c0.value,
                    //       (unsigned long long)val.c1.value,
                    //       (unsigned long long)val.c2.value);
                }
                output[index] = val;
                break;
            }

            default:
                // Invalid opcode - should never happen
                return;
        }
    }
}

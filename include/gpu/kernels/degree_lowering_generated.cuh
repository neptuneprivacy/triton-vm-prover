#pragma once

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Generated from degree_lowering_table_fixed.rs

struct Bfe {
    uint64_t v;
    __device__ __forceinline__ Bfe() : v(0) {}
    __device__ __forceinline__ explicit Bfe(uint64_t x) : v(x) {}
};

// Rust BFieldElement::from_raw_u64() is Montgomery representation.
// We must decode it to canonical value: raw * INV_R mod p.
__device__ __forceinline__ Bfe bfe_from_raw(uint64_t raw) {
    // Observed via Rust FFI: from_raw(1).value() == 18446744065119617025
    constexpr uint64_t INV_R = 18446744065119617025ULL;
    return Bfe(bfield_mul_impl(raw, INV_R));
}

struct Xfe {
    uint64_t c0, c1, c2;
    __device__ __forceinline__ Xfe() : c0(0), c1(0), c2(0) {}
    __device__ __forceinline__ explicit Xfe(Bfe b) : c0(b.v), c1(0), c2(0) {}
    __device__ __forceinline__ explicit Xfe(uint64_t b) : c0(b), c1(0), c2(0) {}
    __device__ __forceinline__ Xfe(uint64_t a0, uint64_t a1, uint64_t a2) : c0(a0), c1(a1), c2(a2) {}
};

__device__ __forceinline__ Bfe operator+(Bfe a, Bfe b) { return Bfe(bfield_add_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator-(Bfe a, Bfe b) { return Bfe(bfield_sub_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator*(Bfe a, Bfe b) { return Bfe(bfield_mul_impl(a.v, b.v)); }

__device__ __forceinline__ Xfe operator+(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    xfield_add_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator-(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    xfield_sub_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    xfield_mul_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Xfe a, Bfe s) {
    uint64_t r0, r1, r2;
    xfield_scalar_mul_impl(a.c0, a.c1, a.c2, s.v, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Bfe s, Xfe a) { return a * s; }
__device__ __forceinline__ Xfe operator+(Xfe a, Bfe b) { return a + Xfe(b); }
__device__ __forceinline__ Xfe operator+(Bfe a, Xfe b) { return Xfe(a) + b; }
__device__ __forceinline__ Xfe operator-(Xfe a, Bfe b) { return a - Xfe(b); }
__device__ __forceinline__ Xfe operator-(Bfe a, Xfe b) { return Xfe(a) - b; }

__device__ __forceinline__ void degree_lowering_fill_row(
    const Bfe* main_cur,
    const Bfe* main_next,
    Xfe* aux_cur,           // length >= 87, indices 0..86 accessible
    const Xfe* aux_next,    // length >= 49, indices 0..48 accessible
    const Xfe* ch           // length >= 63
) {
    Xfe s0 = ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * (main_next[38]))) + ((ch[19]) * (main_next[37]))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(4294967295ULL))))) + ((ch[19]) * (main_next[36])))));
    aux_cur[49] = s0;
    Xfe s1 = ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * (main_cur[38]))) + ((ch[19]) * (main_cur[37]))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(4294967295ULL))))) + ((ch[19]) * (main_cur[36])))));
    aux_cur[50] = s1;
    Xfe s2 = (aux_cur[49]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(8589934590ULL))))) + ((ch[19]) * (main_next[35])))));
    aux_cur[51] = s2;
    Xfe s3 = (aux_cur[50]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(8589934590ULL))))) + ((ch[19]) * (main_cur[35])))));
    aux_cur[52] = s3;
    Xfe s4 = (aux_cur[51]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(12884901885ULL))))) + ((ch[19]) * (main_next[34])))));
    aux_cur[53] = s4;
    Xfe s5 = (aux_cur[52]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(12884901885ULL))))) + ((ch[19]) * (main_cur[34])))));
    aux_cur[54] = s5;
    Xfe s6 = (aux_cur[53]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(17179869180ULL))))) + ((ch[19]) * (main_next[33])))));
    aux_cur[55] = s6;
    Xfe s7 = ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_next[23]) * (ch[22]))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_next[24]) * (ch[22])))));
    aux_cur[56] = s7;
    Xfe s8 = ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + ((main_cur[22]) * (ch[21]))) + ((main_cur[23]) * (ch[22]))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + (((main_cur[22]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_cur[24]) * (ch[22])))));
    aux_cur[57] = s8;
    Xfe s9 = (aux_cur[6]) * (aux_cur[55]);
    aux_cur[58] = s9;
    Xfe s10 = (aux_cur[54]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(17179869180ULL))))) + ((ch[19]) * (main_cur[33])))));
    aux_cur[59] = s10;
    Xfe s11 = (aux_cur[56]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(12884901885ULL))) * (ch[21]))) + ((main_next[25]) * (ch[22])))));
    aux_cur[60] = s11;
    Xfe s12 = (aux_cur[57]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + (((main_cur[22]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_cur[25]) * (ch[22])))));
    aux_cur[61] = s12;
    Xfe s13 = (aux_cur[60]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(17179869180ULL))) * (ch[21]))) + ((main_next[26]) * (ch[22])))));
    aux_cur[62] = s13;
    Xfe s14 = (aux_cur[61]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + (((main_cur[22]) + (bfe_from_raw(12884901885ULL))) * (ch[21]))) + ((main_cur[26]) * (ch[22])))));
    aux_cur[63] = s14;
    Xfe s15 = (((aux_cur[55]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(21474836475ULL))))) + ((ch[19]) * (main_next[32])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(25769803770ULL))))) + ((ch[19]) * (main_next[31])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(30064771065ULL))))) + ((ch[19]) * (main_next[30])))));
    aux_cur[64] = s15;
    Xfe s16 = (((aux_cur[59]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(21474836475ULL))))) + ((ch[19]) * (main_cur[32])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(25769803770ULL))))) + ((ch[19]) * (main_cur[31])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(30064771065ULL))))) + ((ch[19]) * (main_cur[30])))));
    aux_cur[65] = s16;
    Xfe s17 = (aux_cur[6]) * (((aux_cur[64]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(34359738360ULL))))) + ((ch[19]) * (main_next[29])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_next[38]) + (bfe_from_raw(38654705655ULL))))) + ((ch[19]) * (main_next[28]))))));
    aux_cur[66] = s17;
    Xfe s18 = (aux_cur[6]) * (((aux_cur[65]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(34359738360ULL))))) + ((ch[19]) * (main_cur[29])))))) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * ((main_cur[38]) + (bfe_from_raw(38654705655ULL))))) + ((ch[19]) * (main_cur[28]))))));
    aux_cur[67] = s18;
    Xfe s19 = (main_cur[202]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * ((aux_cur[62]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(21474836475ULL))) * (ch[21]))) + ((main_next[27]) * (ch[22])))))))));
    aux_cur[68] = s19;
    Xfe s20 = (main_cur[202]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * ((aux_cur[63]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + (((main_cur[22]) + (bfe_from_raw(17179869180ULL))) * (ch[21]))) + ((main_cur[27]) * (ch[22])))))))));
    aux_cur[69] = s20;
    Xfe s21 = ((((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + ((main_cur[29]) * (ch[21]))) + ((main_cur[39]) * (ch[22]))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[29]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_cur[40]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[29]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_cur[41]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[29]) + (bfe_from_raw(12884901885ULL))) * (ch[21]))) + ((main_cur[42]) * (ch[22])))));
    aux_cur[70] = s21;
    Xfe s22 = ((((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + ((main_cur[22]) * (ch[21]))) + ((main_cur[39]) * (ch[22]))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[22]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_cur[40]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[22]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_cur[41]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + ((main_cur[23]) * (ch[21]))) + ((main_cur[42]) * (ch[22])))));
    aux_cur[71] = s22;
    Xfe s23 = ((((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + ((main_cur[22]) * (ch[21]))) + ((main_cur[39]) * (ch[22]))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + ((main_cur[23]) * (ch[21]))) + ((main_cur[40]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[23]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_cur[41]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[23]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_cur[42]) * (ch[22])))));
    aux_cur[72] = s23;
    Xfe s24 = (main_cur[195]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * (main_next[38]))) + ((ch[19]) * (main_next[37]))))))));
    aux_cur[73] = s24;
    Xfe s25 = (main_cur[196]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[49]))));
    aux_cur[74] = s25;
    Xfe s26 = (main_cur[198]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[51]))));
    aux_cur[75] = s26;
    Xfe s27 = (main_cur[200]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[53]))));
    aux_cur[76] = s27;
    Xfe s28 = (main_cur[195]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * ((ch[7]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((ch[16]) * (main_cur[7])) + ((ch[17]) * (main_cur[13]))) + ((ch[18]) * (main_cur[38]))) + ((ch[19]) * (main_cur[37]))))))));
    aux_cur[77] = s28;
    Xfe s29 = (main_cur[196]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[50]))));
    aux_cur[78] = s29;
    Xfe s30 = (main_cur[198]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[52]))));
    aux_cur[79] = s30;
    Xfe s31 = (main_cur[200]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[54]))));
    aux_cur[80] = s31;
    Xfe s32 = (main_cur[202]) * ((aux_next[6]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[6]) * (aux_cur[59]))));
    aux_cur[81] = s32;
    Xfe s33 = (aux_cur[7]) * (((aux_cur[71]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[23]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_cur[43]) * (ch[22])))))) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[23]) + (bfe_from_raw(8589934590ULL))) * (ch[21]))) + ((main_cur[44]) * (ch[22]))))));
    aux_cur[82] = s33;
    Xfe s34 = ((((aux_next[21]) + ((bfe_from_raw(18446744065119617026ULL)) * (aux_cur[21]))) * ((ch[11]) + ((bfe_from_raw(18446744065119617026ULL)) * ((main_next[50]) + ((bfe_from_raw(18446744065119617026ULL)) * (main_cur[50])))))) + (bfe_from_raw(18446744065119617026ULL))) * ((bfe_from_raw(4294967295ULL)) + ((bfe_from_raw(18446744065119617026ULL)) * (((main_next[52]) + ((bfe_from_raw(18446744065119617026ULL)) * (main_cur[52]))) * (main_cur[54]))));
    aux_cur[83] = s34;
    Xfe s35 = (main_cur[301]) * (((aux_cur[7]) * ((aux_cur[70]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_cur[29]) + (bfe_from_raw(17179869180ULL))) * (ch[21]))) + ((main_cur[43]) * (ch[22]))))))) + ((bfe_from_raw(18446744065119617026ULL)) * (aux_next[7])));
    aux_cur[84] = s35;
    Xfe s36 = (main_cur[220]) * ((((((main_cur[195]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * (((((main_cur[7]) * (ch[20])) + (ch[23])) + (((main_next[22]) + (bfe_from_raw(4294967295ULL))) * (ch[21]))) + ((main_next[23]) * (ch[22]))))))))) + ((main_cur[196]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[56])))))) + ((main_cur[198]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[60])))))) + ((main_cur[200]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[62])))))) + (aux_cur[68]));
    aux_cur[85] = s36;
    Xfe s37 = (main_cur[228]) * ((((((main_cur[195]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * ((ch[8]) + ((bfe_from_raw(18446744065119617026ULL)) * ((((main_cur[7]) * (ch[20])) + ((main_cur[22]) * (ch[21]))) + ((main_cur[23]) * (ch[22]))))))))) + ((main_cur[196]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[57])))))) + ((main_cur[198]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[61])))))) + ((main_cur[200]) * ((aux_next[7]) + ((bfe_from_raw(18446744065119617026ULL)) * ((aux_cur[7]) * (aux_cur[63])))))) + (aux_cur[69]));
    aux_cur[86] = s37;
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

/**
 * GPU Kernel for Main Table Degree Lowering (columns 149-378)
 * 
 * Ported from degree_lowering_main_cpp.cpp - pure arithmetic operations
 * perfect for GPU parallelization.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/degree_lowering_main_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include <iostream>
#include <chrono>
#include <cstdlib>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// BFieldElement wrapper (same as quotient kernels)
// ============================================================================

// INV_R for converting from_raw_u64 constants to standard form
static constexpr uint64_t INV_R = 18446744065119617025ULL;

struct Bfe {
    uint64_t v;
    __device__ __forceinline__ Bfe() : v(0) {}
    __device__ __forceinline__ explicit Bfe(uint64_t x) : v(x) {}
    __device__ __forceinline__ static Bfe zero() { return Bfe(0); }
    __device__ __forceinline__ static Bfe one() { return Bfe(1); }
    __device__ __forceinline__ static Bfe from_raw_u64(uint64_t raw) {
        return Bfe(bfield_mul_impl(raw, INV_R));
    }
};

__device__ __forceinline__ Bfe operator+(Bfe a, Bfe b) { return Bfe(bfield_add_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator-(Bfe a, Bfe b) { return Bfe(bfield_sub_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator*(Bfe a, Bfe b) { return Bfe(bfield_mul_impl(a.v, b.v)); }

// Pre-computed constants (from_raw_u64 values from Rust - will be converted in kernel)
__constant__ uint64_t d_C_raw[27];

// Host-side raw constants for upload
static const uint64_t h_C_raw[27] = {
    4294967295ULL,          // C_0
    4294967296ULL,          // C_1
    8589934590ULL,          // C_2
    12884901885ULL,         // C_3
    17179869180ULL,         // C_4
    21474836475ULL,         // C_5
    25769803770ULL,         // C_6
    30064771065ULL,         // C_7
    34359738360ULL,         // C_8
    38654705655ULL,         // C_9
    18446743828896415801ULL, // C_10
    18446743863256154161ULL, // C_11
    18446743897615892521ULL, // C_12
    18446743923385696291ULL, // C_13
    18446743931975630881ULL, // C_14
    18446743940565565471ULL, // C_15
    18446743949155500061ULL, // C_16
    18446743992105173011ULL, // C_17
    18446744000695107601ULL, // C_18
    18446744009285042191ULL, // C_19
    18446744017874976781ULL, // C_20
    18446744043644780551ULL, // C_21
    18446744047939747846ULL, // C_22
    18446744052234715141ULL, // C_23
    18446744056529682436ULL, // C_24
    18446744060824649731ULL, // C_25
    18446744065119617026ULL  // C_26 (NEG_ONE)
};

static bool constants_initialized = false;

void init_degree_lowering_constants() {
    if (!constants_initialized) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_C_raw, h_C_raw, sizeof(h_C_raw)));
        constants_initialized = true;
    }
}

// ============================================================================
// Kernel implementation with proper Bfe arithmetic
// ============================================================================

__global__ void degree_lowering_main_kernel(
    uint64_t* d_table,
    size_t num_rows,
    size_t num_cols
) {
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows) return;
    
    // Get row pointers
    uint64_t* cur_u64 = d_table + r * num_cols;
    const uint64_t* next_u64 = (r + 1 < num_rows) ? d_table + (r + 1) * num_cols : cur_u64;
    
    // Load constants and convert from raw to standard form
    Bfe C0 = Bfe::from_raw_u64(d_C_raw[0]);
    Bfe C1 = Bfe::from_raw_u64(d_C_raw[1]);
    Bfe C2 = Bfe::from_raw_u64(d_C_raw[2]);
    Bfe C10 = Bfe::from_raw_u64(d_C_raw[10]);
    Bfe C11 = Bfe::from_raw_u64(d_C_raw[11]);
    Bfe C12 = Bfe::from_raw_u64(d_C_raw[12]);
    Bfe C13 = Bfe::from_raw_u64(d_C_raw[13]);
    Bfe C14 = Bfe::from_raw_u64(d_C_raw[14]);
    Bfe C15 = Bfe::from_raw_u64(d_C_raw[15]);
    Bfe C16 = Bfe::from_raw_u64(d_C_raw[16]);
    Bfe C17 = Bfe::from_raw_u64(d_C_raw[17]);
    Bfe C18 = Bfe::from_raw_u64(d_C_raw[18]);
    Bfe C19 = Bfe::from_raw_u64(d_C_raw[19]);
    Bfe C20 = Bfe::from_raw_u64(d_C_raw[20]);
    Bfe C21 = Bfe::from_raw_u64(d_C_raw[21]);
    Bfe C22 = Bfe::from_raw_u64(d_C_raw[22]);
    Bfe C23 = Bfe::from_raw_u64(d_C_raw[23]);
    Bfe C24 = Bfe::from_raw_u64(d_C_raw[24]);
    Bfe C25 = Bfe::from_raw_u64(d_C_raw[25]);
    Bfe C26 = Bfe::from_raw_u64(d_C_raw[26]);  // NEG_ONE
    
    // Load current row values into Bfe
    #define LOAD_CUR(i) Bfe(cur_u64[i])
    #define LOAD_NEXT(i) Bfe(next_u64[i])
    #define STORE_CUR(i, val) cur_u64[i] = (val).v
    
    // =========================================================================
    // Phase 1: Columns 149-150 (depends only on cols 0-148)
    // =========================================================================
    
    // Column 149: ((row[12] + NEG_ONE) * row[13] * (row[14] + NEG_ONE) * (row[15] + NEG_ONE))
    Bfe col149 = (((LOAD_CUR(12) + C26) * LOAD_CUR(13)) * (LOAD_CUR(14) + C26)) * (LOAD_CUR(15) + C26);
    STORE_CUR(149, col149);
    
    // Column 150: (row[149] * row[16] * (row[17] + NEG_ONE) * (row[18] + NEG_ONE))
    Bfe col150 = ((col149 * LOAD_CUR(16)) * (LOAD_CUR(17) + C26)) * (LOAD_CUR(18) + C26);
    STORE_CUR(150, col150);
    
    // =========================================================================
    // Phase 2: Columns 151-168 (depends on cols 0-150)
    // =========================================================================
    
    Bfe r64 = LOAD_CUR(64);
    Bfe r62 = LOAD_CUR(62);
    Bfe r63 = LOAD_CUR(63);
    Bfe r139 = LOAD_CUR(139);
    Bfe r142 = LOAD_CUR(142);
    Bfe r143 = LOAD_CUR(143);
    Bfe r144 = LOAD_CUR(144);
    Bfe r145 = LOAD_CUR(145);
    Bfe r146 = LOAD_CUR(146);
    
    // Column 151
    Bfe col151 = r64 * (r64 + C26);
    STORE_CUR(151, col151);
    
    // Column 152
    Bfe col152 = (((r64 + C26) * (r64 + C25)) * (r64 + C24)) * (r64 + C23);
    STORE_CUR(152, col152);
    
    // Column 153
    Bfe col153 = ((r64 * (r64 + C25)) * (r64 + C24)) * (r64 + C23);
    STORE_CUR(153, col153);
    
    // Column 154
    Bfe col154 = col151 * (r64 + C25);
    STORE_CUR(154, col154);
    
    // Column 155
    Bfe col155 = ((col151 * (r64 + C24)) * (r64 + C23)) * (r64 + C22);
    STORE_CUR(155, col155);
    
    // Column 156
    Bfe col156 = (r142 + C23) * (r142 + C21);
    STORE_CUR(156, col156);
    
    // Column 157
    Bfe col157 = r143 * r144;
    STORE_CUR(157, col157);
    
    // Column 158
    Bfe col158 = (((r142 + C23) * (r142 + C19)) * (r142 + C20)) * (r142 + C15);
    STORE_CUR(158, col158);
    
    // Column 159
    Bfe col159 = col156 * (r142 + C19);
    STORE_CUR(159, col159);
    
    // Column 160
    Bfe col160 = r145 * r146;
    STORE_CUR(160, col160);
    
    // Column 161
    Bfe col161 = ((r62 + C26) * (r62 + C25)) * r62;
    STORE_CUR(161, col161);
    
    // Column 162
    Bfe col162 = col158 * (r142 + C16);
    STORE_CUR(162, col162);
    
    // Column 163
    Bfe col163 = ((col156 * (r142 + C20)) * (r142 + C15)) * (r142 + C16);
    STORE_CUR(163, col163);
    
    // Column 164
    Bfe col164 = (col159 * (r142 + C15)) * (r142 + C16);
    STORE_CUR(164, col164);
    
    // Column 165
    Bfe col165 = (((r62 + C26) * (r62 + C24)) * r62) * (r63 + C12);
    STORE_CUR(165, col165);
    
    // Column 166
    Bfe col166 = col159 * (r142 + C20);
    STORE_CUR(166, col166);
    
    // Column 167
    Bfe col167 = ((col162 * (r139 + C26)) * (C0 + (C26 * col157))) * (C0 + (C26 * col160));
    STORE_CUR(167, col167);
    
    // Column 168
    Bfe col168 = ((col162 * r139) * (C0 + (C26 * col157))) * (C0 + (C26 * col160));
    STORE_CUR(168, col168);
    
    // =========================================================================
    // Phase 3: Columns 169-378 (depends on cols 0-168 and next row)
    // Skip last row since it needs next row
    // =========================================================================
    
    if (r + 1 >= num_rows) {
        // Last row: copy from second-to-last (will be done by last thread if we had next row data)
        return;
    }
    
    // Load more current row values
    Bfe r12 = LOAD_CUR(12), r13 = LOAD_CUR(13), r14 = LOAD_CUR(14), r15 = LOAD_CUR(15);
    Bfe r16 = LOAD_CUR(16), r17 = LOAD_CUR(17), r18 = LOAD_CUR(18);
    Bfe r39 = LOAD_CUR(39), r40 = LOAD_CUR(40), r41 = LOAD_CUR(41), r42 = LOAD_CUR(42);
    Bfe r43 = LOAD_CUR(43), r44 = LOAD_CUR(44);
    Bfe r9 = LOAD_CUR(9), r10 = LOAD_CUR(10);
    Bfe r22 = LOAD_CUR(22), r23 = LOAD_CUR(23), r24 = LOAD_CUR(24), r25 = LOAD_CUR(25);
    Bfe r26 = LOAD_CUR(26), r27 = LOAD_CUR(27), r28 = LOAD_CUR(28);
    Bfe r57 = LOAD_CUR(57), r58 = LOAD_CUR(58), r59 = LOAD_CUR(59);
    Bfe r97 = LOAD_CUR(97), r98 = LOAD_CUR(98), r99 = LOAD_CUR(99);
    Bfe r100 = LOAD_CUR(100), r101 = LOAD_CUR(101), r102 = LOAD_CUR(102);
    Bfe r103 = LOAD_CUR(103), r104 = LOAD_CUR(104), r105 = LOAD_CUR(105);
    Bfe r106 = LOAD_CUR(106), r107 = LOAD_CUR(107), r108 = LOAD_CUR(108);
    Bfe r147 = LOAD_CUR(147);
    
    // Load next row values
    Bfe n9 = LOAD_NEXT(9);
    Bfe n12 = LOAD_NEXT(12), n13 = LOAD_NEXT(13), n14 = LOAD_NEXT(14), n15 = LOAD_NEXT(15);
    Bfe n16 = LOAD_NEXT(16), n17 = LOAD_NEXT(17), n18 = LOAD_NEXT(18);
    Bfe n22 = LOAD_NEXT(22), n23 = LOAD_NEXT(23), n24 = LOAD_NEXT(24), n25 = LOAD_NEXT(25), n26 = LOAD_NEXT(26);
    Bfe n39 = LOAD_NEXT(39), n40 = LOAD_NEXT(40), n41 = LOAD_NEXT(41), n42 = LOAD_NEXT(42), n43 = LOAD_NEXT(43);
    Bfe n44 = LOAD_NEXT(44);
    Bfe n57 = LOAD_NEXT(57), n59 = LOAD_NEXT(59);
    Bfe n62 = LOAD_NEXT(62), n63 = LOAD_NEXT(63), n64 = LOAD_NEXT(64);
    Bfe n139 = LOAD_NEXT(139), n142 = LOAD_NEXT(142), n143 = LOAD_NEXT(143), n144 = LOAD_NEXT(144);
    Bfe n145 = LOAD_NEXT(145), n147 = LOAD_NEXT(147);
    
    // Intermediate values
    Bfe col169 = (r12 + C26) * r13;
    Bfe col170 = (r12 * (r13 + C26)) * (r14 + C26);
    Bfe col171 = col169 * (r14 + C26);
    Bfe col172 = (r12 + C26) * (r13 + C26);
    Bfe col173 = (C0 + (C26 * r42)) * (C0 + (C26 * r41));
    Bfe col174 = col172 * (r14 + C26);
    Bfe col175 = (C0 + (C26 * r42)) * r41;
    Bfe col176 = col170 * (r15 + C26);
    Bfe col177 = col170 * r15;
    Bfe col178 = col171 * (r15 + C26);
    Bfe col179 = col173 * r40;
    Bfe col180 = col171 * r15;
    Bfe col181 = col175 * (C0 + (C26 * r40));
    Bfe col182 = col176 * (r16 + C26);
    Bfe col183 = col174 * (r15 + C26);
    Bfe col184 = col174 * r15;
    Bfe col185 = col173 * (C0 + (C26 * r40));
    Bfe col186 = (r12 * r13) * (r14 + C26);
    Bfe col187 = col177 * (r16 + C26);
    Bfe col188 = col169 * r14;
    Bfe col189 = col180 * (r16 + C26);
    Bfe col190 = col178 * (r16 + C26);
    Bfe col191 = col177 * r16;
    Bfe col192 = r42 * (C0 + (C26 * r41));
    Bfe col193 = r42 * r41;
    Bfe col194 = col172 * r14;
    Bfe col195 = col185 * r39;
    Bfe col196 = col179 * (C0 + (C26 * r39));
    Bfe col197 = col183 * (r16 + C26);
    Bfe col198 = col179 * r39;
    Bfe col199 = col178 * r16;
    Bfe col200 = col181 * (C0 + (C26 * r39));
    Bfe col201 = col182 * (r17 + C26);
    Bfe col202 = col181 * r39;
    Bfe col203 = col176 * r16;
    Bfe col204 = col190 * (r17 + C26);
    Bfe col205 = col184 * (r16 + C26);
    Bfe col206 = col180 * r16;
    Bfe col207 = col186 * (r15 + C26);
    Bfe col208 = col189 * (r17 + C26);
    Bfe col209 = col187 * (r17 + C26);
    Bfe col210 = col201 * (r18 + C26);
    Bfe col211 = (col182 * r17) * (r18 + C26);
    Bfe col212 = col184 * r16;
    Bfe col213 = col197 * (r17 + C26);
    Bfe col214 = col188 * (r15 + C26);
    Bfe col215 = ((col207 * (r16 + C26)) * (r17 + C26)) * (r18 + C26);
    Bfe col216 = col183 * r16;
    Bfe col217 = col194 * (r15 + C26);
    Bfe col218 = col204 * (r18 + C26);
    Bfe col219 = col188 * r15;
    Bfe col220 = (col191 * r17) * (r18 + C26);
    Bfe col221 = ((col186 * r15) * (r16 + C26)) * (r17 + C26);
    Bfe col222 = col199 * (r17 + C26);
    Bfe col223 = col209 * (r18 + C26);
    Bfe col224 = col205 * (r17 + C26);
    Bfe col225 = (col203 * (r17 + C26)) * (r18 + C26);
    Bfe col226 = col208 * (r18 + C26);
    Bfe col227 = (col191 * (r17 + C26)) * (r18 + C26);
    Bfe col228 = col221 * (r18 + C26);
    Bfe col229 = (col187 * r17) * (r18 + C26);
    Bfe col230 = col217 * (r16 + C26);
    Bfe col231 = col175 * r40;
    Bfe col232 = col192 * (C0 + (C26 * r40));
    Bfe col233 = col192 * r40;
    Bfe col234 = col193 * (C0 + (C26 * r40));
    Bfe col235 = col193 * r40;
    Bfe col236 = col194 * r15;
    Bfe col237 = (col189 * r17) * (r18 + C26);
    Bfe col238 = (col199 * r17) * (r18 + C26);
    Bfe col239 = col222 * (r18 + C26);
    Bfe col240 = col212 * (r17 + C26);
    Bfe col241 = (col206 * (r17 + C26)) * (r18 + C26);
    Bfe col242 = ((col214 * (r16 + C26)) * (r17 + C26)) * (r18 + C26);
    Bfe col243 = r41 * (r41 + C26);
    Bfe col244 = (col206 * r17) * (r18 + C26);
    Bfe col245 = (col230 * (r17 + C26)) * (r18 + C26);
    Bfe col246 = ((col219 * (r16 + C26)) * (r17 + C26)) * (r18 + C26);
    Bfe col247 = col213 * (r18 + C26);
    Bfe col248 = ((col214 * r16) * (r17 + C26)) * (r18 + C26);
    Bfe col249 = col216 * (r17 + C26);
    Bfe col250 = col224 * (r18 + C26);
    Bfe col251 = (col203 * r17) * (r18 + C26);
    Bfe col252 = (col197 * r17) * (r18 + C26);
    Bfe col253 = ((col219 * r16) * (r17 + C26)) * (r18 + C26);
    Bfe col254 = r42 * (r42 + C26);
    Bfe col255 = ((r97 * r97) * r97) * r97;
    Bfe col256 = col236 * (r16 + C26);
    Bfe col257 = ((r98 * r98) * r98) * r98;
    Bfe col258 = ((r99 * r99) * r99) * r99;
    Bfe col259 = col240 * (r18 + C26);
    Bfe col260 = ((r100 * r100) * r100) * r100;
    Bfe col261 = (col205 * r17) * (r18 + C26);
    Bfe col262 = (col190 * r17) * (r18 + C26);
    Bfe col263 = ((r101 * r101) * r101) * r101;
    Bfe col264 = (col216 * r17) * (r18 + C26);
    Bfe col265 = col201 * r18;
    Bfe col266 = (col212 * r17) * (r18 + C26);
    Bfe col267 = col213 * r18;
    Bfe col268 = ((r102 * r102) * r102) * r102;
    Bfe col269 = col249 * (r18 + C26);
    Bfe col270 = ((col217 * r16) * (r17 + C26)) * (r18 + C26);
    Bfe col271 = ((r103 * r103) * r103) * r103;
    Bfe col272 = col204 * r18;
    Bfe col273 = (col256 * (r17 + C26)) * (r18 + C26);
    Bfe col274 = r39 * (r28 + (C26 * r27));
    Bfe col275 = col208 * r18;
    Bfe col276 = ((r104 * r104) * r104) * r104;
    Bfe col277 = ((col236 * r16) * (r17 + C26)) * (r18 + C26);
    Bfe col278 = ((r105 * r105) * r105) * r105;
    Bfe col279 = ((col207 * r16) * (r17 + C26)) * (r18 + C26);
    Bfe col280 = ((r106 * r106) * r106) * r106;
    Bfe col281 = ((r107 * r107) * r107) * r107;
    Bfe col282 = r39 * r22;
    Bfe col283 = ((r108 * r108) * r108) * r108;
    Bfe col284 = col224 * r18;
    Bfe col285 = col185 * (C0 + (C26 * r39));
    Bfe col286 = col231 * (C0 + (C26 * r39));
    Bfe col287 = col231 * r39;
    Bfe col288 = col232 * (C0 + (C26 * r39));
    Bfe col289 = col232 * r39;
    Bfe col290 = col233 * (C0 + (C26 * r39));
    Bfe col291 = col233 * r39;
    Bfe col292 = col234 * (C0 + (C26 * r39));
    Bfe col293 = col234 * r39;
    Bfe col294 = col235 * (C0 + (C26 * r39));
    Bfe col295 = col235 * r39;
    Bfe col296 = col222 * r18;
    Bfe col297 = col209 * r18;
    Bfe col298 = ((n64 * (n64 + C26)) * (n64 + C25)) * (n64 + C24);
    Bfe col299 = r44 * (r44 + C26);
    Bfe col300 = (n62 * (n64 + C22)) * (n63 + C12);
    Bfe col301 = (col256 * r17) * (r18 + C26);
    Bfe col302 = (col230 * r17) * (r18 + C26);
    Bfe col303 = ((r43 * (r43 + C26)) * (r43 + C25)) * (r43 + C24);
    Bfe col304 = r39 * (r39 + C26);
    Bfe col305 = r40 * (r40 + C26);
    Bfe col306 = col249 * r18;
    Bfe col307 = col240 * r18;
    Bfe col308 = (((n142 + C23) * (n142 + C19)) * (n142 + C20)) * (n142 + C15);
    Bfe col309 = (n142 + C23) * (n142 + C21);
    Bfe col310 = (((n64 + C26) * (n64 + C25)) * (n64 + C24)) * (n64 + C23);
    Bfe col311 = ((col255 * r97) * r97) * r97;
    Bfe col312 = ((col257 * r98) * r98) * r98;
    Bfe col313 = ((col258 * r99) * r99) * r99;
    Bfe col314 = ((col260 * r100) * r100) * r100;
    Bfe col315 = ((col263 * r101) * r101) * r101;
    Bfe col316 = ((col268 * r102) * r102) * r102;
    Bfe col317 = ((col271 * r103) * r103) * r103;
    Bfe col318 = ((col276 * r104) * r104) * r104;
    Bfe col319 = ((col278 * r105) * r105) * r105;
    Bfe col320 = ((col280 * r106) * r106) * r106;
    Bfe col321 = ((col281 * r107) * r107) * r107;
    Bfe col322 = ((col283 * r108) * r108) * r108;
    Bfe col323 = (n139 + C26) * (col308 * (n142 + C16));
    Bfe col324 = col309 * (n142 + C19);
    Bfe col325 = col310 * (n64 + C22);
    Bfe col326 = ((n12 + C26) * (n13 + C26)) * n14;
    Bfe col327 = r39 * (r23 + (C26 * r22));
    Bfe col328 = (col323 * n147) * (n147 + C26);
    Bfe col329 = (((n12 + C26) * n13) * (n14 + C26)) * (n15 + C26);
    Bfe col330 = ((n62 + C26) * (n62 + C25)) * n62;
    Bfe col331 = r24 * r27;
    Bfe col332 = r24 * n24;
    Bfe col333 = col324 * (n142 + C20);
    Bfe col334 = (((r10 + C12) * (r10 + C13)) * (r10 + C11)) * (r10 + C10);
    Bfe col335 = (n139 + C26) * ((col324 * (n142 + C15)) * (n142 + C16));
    Bfe col336 = (C0 + (C26 * n44)) * n22;
    Bfe col337 = n44 * n39;
    Bfe col338 = (C0 + (C26 * n44)) * n23;
    Bfe col339 = n44 * n40;
    Bfe col340 = (C0 + (C26 * n44)) * n24;
    Bfe col341 = n44 * n41;
    Bfe col342 = (C0 + (C26 * n44)) * n25;
    Bfe col343 = n44 * n42;
    Bfe col344 = (C0 + (C26 * n44)) * n26;
    Bfe col345 = n44 * n43;
    Bfe col346 = (C0 + (C26 * n44)) * n39;
    Bfe col347 = n44 * n22;
    Bfe col348 = (C0 + (C26 * n44)) * n40;
    Bfe col349 = n44 * n23;
    Bfe col350 = (C0 + (C26 * n44)) * n41;
    Bfe col351 = n44 * n24;
    Bfe col352 = (C0 + (C26 * n44)) * n42;
    Bfe col353 = n44 * n25;
    Bfe col354 = (C0 + (C26 * n44)) * n43;
    Bfe col355 = n44 * n26;
    Bfe col356 = ((col329 * n16) * (n17 + C26)) * (n18 + C26);
    Bfe col357 = ((col326 * (n15 + C26)) * (n16 + C26)) * n17;
    Bfe col358 = r39 * r42;
    Bfe col359 = ((col326 * n15) * (n16 + C26)) * n17;
    Bfe col360 = col325 * (((n62 + C25) * (n62 + C24)) * n62);
    Bfe col361 = ((r62 + C25) * (r62 + C24)) * r62;
    Bfe col362 = col245 * (n22 * ((r39 * (n23 + C1)) + C26));
    Bfe col363_t1 = n9 + (C26 * r9);
    Bfe col363_t2 = (col363_t1 + C26) * r22;
    Bfe col363_t3 = ((col363_t1 + C25) * (col282 + C26)) * (r40 + C26);
    Bfe col363_t4 = ((col363_t1 + C24) * (col282 + C26)) * r40;
    Bfe col363 = col218 * (col363_t2 + col363_t3 + col363_t4);
    Bfe col364 = col218 * ((col243 * (r41 + C25)) * (r41 + C24));
    Bfe col365 = col218 * ((col254 * (r42 + C25)) * (r42 + C24));
    Bfe col366 = col218 * ((col299 * (r44 + C25)) * (r44 + C24));
    Bfe col367 = ((r64 * (r64 + C26)) * (r64 + C25)) * (r64 + C24);
    Bfe col368 = (((r62 + C26) * (r62 + C24)) * r62) * (n62 + C25);
    Bfe col369 = ((col309 * (n142 + C20)) * (n142 + C15)) * (n142 + C16);
    Bfe c370_1 = r143 + (C26 * (C2 * n143));
    Bfe c370_2 = r145 + (C26 * (C2 * n145));
    Bfe c370_3 = C0 + (C26 * c370_1);
    Bfe c370_4 = C26 * c370_2;  // Fixed: no C0 here
    Bfe c370_5 = (C2 * c370_1) * c370_2;
    Bfe col370 = col328 * (c370_3 + c370_4 + c370_5);
    Bfe col371 = (n139 + C26) * (col333 * (n142 + C16));
    Bfe t372_1 = n59 + (C26 * r59);
    Bfe col372 = (((t372_1 + C26) * (r58 + C18)) * (r58 + C14)) * ((n57 + (C26 * r57)) + C26);
    Bfe col373 = col361 * (((n62 + C26) * (n62 + C24)) * n62);
    Bfe col374 = (((r62 + C26) * (r62 + C25)) * r62) * (n62 + C24);
    Bfe col375 = ((col325 * (n62 + C24)) * n62) * (n63 + C12);
    Bfe col376 = col325 * (((n63 + C17) * (n63 + C12)) * (n63 + C13));
    Bfe col377 = (col335 * (C0 + (C26 * (n143 * n144)))) * r143;
    Bfe col378 = (n147 * n147) * r143;
    
    // Store all computed columns
    STORE_CUR(169, col169); STORE_CUR(170, col170); STORE_CUR(171, col171); STORE_CUR(172, col172);
    STORE_CUR(173, col173); STORE_CUR(174, col174); STORE_CUR(175, col175); STORE_CUR(176, col176);
    STORE_CUR(177, col177); STORE_CUR(178, col178); STORE_CUR(179, col179); STORE_CUR(180, col180);
    STORE_CUR(181, col181); STORE_CUR(182, col182); STORE_CUR(183, col183); STORE_CUR(184, col184);
    STORE_CUR(185, col185); STORE_CUR(186, col186); STORE_CUR(187, col187); STORE_CUR(188, col188);
    STORE_CUR(189, col189); STORE_CUR(190, col190); STORE_CUR(191, col191); STORE_CUR(192, col192);
    STORE_CUR(193, col193); STORE_CUR(194, col194); STORE_CUR(195, col195); STORE_CUR(196, col196);
    STORE_CUR(197, col197); STORE_CUR(198, col198); STORE_CUR(199, col199); STORE_CUR(200, col200);
    STORE_CUR(201, col201); STORE_CUR(202, col202); STORE_CUR(203, col203); STORE_CUR(204, col204);
    STORE_CUR(205, col205); STORE_CUR(206, col206); STORE_CUR(207, col207); STORE_CUR(208, col208);
    STORE_CUR(209, col209); STORE_CUR(210, col210); STORE_CUR(211, col211); STORE_CUR(212, col212);
    STORE_CUR(213, col213); STORE_CUR(214, col214); STORE_CUR(215, col215); STORE_CUR(216, col216);
    STORE_CUR(217, col217); STORE_CUR(218, col218); STORE_CUR(219, col219); STORE_CUR(220, col220);
    STORE_CUR(221, col221); STORE_CUR(222, col222); STORE_CUR(223, col223); STORE_CUR(224, col224);
    STORE_CUR(225, col225); STORE_CUR(226, col226); STORE_CUR(227, col227); STORE_CUR(228, col228);
    STORE_CUR(229, col229); STORE_CUR(230, col230); STORE_CUR(231, col231); STORE_CUR(232, col232);
    STORE_CUR(233, col233); STORE_CUR(234, col234); STORE_CUR(235, col235); STORE_CUR(236, col236);
    STORE_CUR(237, col237); STORE_CUR(238, col238); STORE_CUR(239, col239); STORE_CUR(240, col240);
    STORE_CUR(241, col241); STORE_CUR(242, col242); STORE_CUR(243, col243); STORE_CUR(244, col244);
    STORE_CUR(245, col245); STORE_CUR(246, col246); STORE_CUR(247, col247); STORE_CUR(248, col248);
    STORE_CUR(249, col249); STORE_CUR(250, col250); STORE_CUR(251, col251); STORE_CUR(252, col252);
    STORE_CUR(253, col253); STORE_CUR(254, col254); STORE_CUR(255, col255); STORE_CUR(256, col256);
    STORE_CUR(257, col257); STORE_CUR(258, col258); STORE_CUR(259, col259); STORE_CUR(260, col260);
    STORE_CUR(261, col261); STORE_CUR(262, col262); STORE_CUR(263, col263); STORE_CUR(264, col264);
    STORE_CUR(265, col265); STORE_CUR(266, col266); STORE_CUR(267, col267); STORE_CUR(268, col268);
    STORE_CUR(269, col269); STORE_CUR(270, col270); STORE_CUR(271, col271); STORE_CUR(272, col272);
    STORE_CUR(273, col273); STORE_CUR(274, col274); STORE_CUR(275, col275); STORE_CUR(276, col276);
    STORE_CUR(277, col277); STORE_CUR(278, col278); STORE_CUR(279, col279); STORE_CUR(280, col280);
    STORE_CUR(281, col281); STORE_CUR(282, col282); STORE_CUR(283, col283); STORE_CUR(284, col284);
    STORE_CUR(285, col285); STORE_CUR(286, col286); STORE_CUR(287, col287); STORE_CUR(288, col288);
    STORE_CUR(289, col289); STORE_CUR(290, col290); STORE_CUR(291, col291); STORE_CUR(292, col292);
    STORE_CUR(293, col293); STORE_CUR(294, col294); STORE_CUR(295, col295); STORE_CUR(296, col296);
    STORE_CUR(297, col297); STORE_CUR(298, col298); STORE_CUR(299, col299); STORE_CUR(300, col300);
    STORE_CUR(301, col301); STORE_CUR(302, col302); STORE_CUR(303, col303); STORE_CUR(304, col304);
    STORE_CUR(305, col305); STORE_CUR(306, col306); STORE_CUR(307, col307); STORE_CUR(308, col308);
    STORE_CUR(309, col309); STORE_CUR(310, col310); STORE_CUR(311, col311); STORE_CUR(312, col312);
    STORE_CUR(313, col313); STORE_CUR(314, col314); STORE_CUR(315, col315); STORE_CUR(316, col316);
    STORE_CUR(317, col317); STORE_CUR(318, col318); STORE_CUR(319, col319); STORE_CUR(320, col320);
    STORE_CUR(321, col321); STORE_CUR(322, col322); STORE_CUR(323, col323); STORE_CUR(324, col324);
    STORE_CUR(325, col325); STORE_CUR(326, col326); STORE_CUR(327, col327); STORE_CUR(328, col328);
    STORE_CUR(329, col329); STORE_CUR(330, col330); STORE_CUR(331, col331); STORE_CUR(332, col332);
    STORE_CUR(333, col333); STORE_CUR(334, col334); STORE_CUR(335, col335); STORE_CUR(336, col336);
    STORE_CUR(337, col337); STORE_CUR(338, col338); STORE_CUR(339, col339); STORE_CUR(340, col340);
    STORE_CUR(341, col341); STORE_CUR(342, col342); STORE_CUR(343, col343); STORE_CUR(344, col344);
    STORE_CUR(345, col345); STORE_CUR(346, col346); STORE_CUR(347, col347); STORE_CUR(348, col348);
    STORE_CUR(349, col349); STORE_CUR(350, col350); STORE_CUR(351, col351); STORE_CUR(352, col352);
    STORE_CUR(353, col353); STORE_CUR(354, col354); STORE_CUR(355, col355); STORE_CUR(356, col356);
    STORE_CUR(357, col357); STORE_CUR(358, col358); STORE_CUR(359, col359); STORE_CUR(360, col360);
    STORE_CUR(361, col361); STORE_CUR(362, col362); STORE_CUR(363, col363); STORE_CUR(364, col364);
    STORE_CUR(365, col365); STORE_CUR(366, col366); STORE_CUR(367, col367); STORE_CUR(368, col368);
    STORE_CUR(369, col369); STORE_CUR(370, col370); STORE_CUR(371, col371); STORE_CUR(372, col372);
    STORE_CUR(373, col373); STORE_CUR(374, col374); STORE_CUR(375, col375); STORE_CUR(376, col376);
    STORE_CUR(377, col377); STORE_CUR(378, col378);
    
    #undef LOAD_CUR
    #undef LOAD_NEXT
    #undef STORE_CUR
}

// Copy last row from second-to-last for phase 3 columns
__global__ void degree_lowering_copy_last_row_kernel(
    uint64_t* d_table,
    size_t num_rows,
    size_t num_cols
) {
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x + 169;
    if (col >= 379) return;
    if (num_rows < 2) return;
    
    d_table[(num_rows - 1) * num_cols + col] = d_table[(num_rows - 2) * num_cols + col];
}

// ============================================================================
// Host Function
// ============================================================================

void gpu_degree_lowering_main(
    uint64_t* d_table,
    size_t num_rows,
    size_t num_cols,
    cudaStream_t stream
) {
    if (num_rows == 0 || num_cols < 379) return;
    
    const bool profile = (std::getenv("TVM_PROFILE_GPU_DL") != nullptr);
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Initialize constants
    init_degree_lowering_constants();
    
    constexpr size_t BLOCK_SIZE = 256;
    size_t num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Single kernel computes all phases
    degree_lowering_main_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_table, num_rows, num_cols);
    
    // Copy last row
    size_t copy_blocks = (210 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    degree_lowering_copy_last_row_kernel<<<copy_blocks, BLOCK_SIZE, 0, stream>>>(
        d_table, num_rows, num_cols);
    
    if (profile) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[GPU DL Main] Degree lowering: " << ms << " ms" << std::endl;
    }
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

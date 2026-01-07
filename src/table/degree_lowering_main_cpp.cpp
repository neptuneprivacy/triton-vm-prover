// Auto-generated from degree_lowering_table_fixed.rs
// Pure C++ implementation of main table degree lowering columns (149-378)

#include "table/master_table.hpp"
#include "types/b_field_element.hpp"
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cstdlib>

namespace triton_vm {

// Pre-computed constants used in degree lowering
namespace degree_lowering_constants {
    // These are BFieldElement::from_raw_u64() values from the Rust code
    static const BFieldElement C_0 = BFieldElement::from_raw_u64(4294967295ULL);
    static const BFieldElement C_1 = BFieldElement::from_raw_u64(4294967296ULL);
    static const BFieldElement C_2 = BFieldElement::from_raw_u64(8589934590ULL);
    static const BFieldElement C_3 = BFieldElement::from_raw_u64(12884901885ULL);
    static const BFieldElement C_4 = BFieldElement::from_raw_u64(17179869180ULL);
    static const BFieldElement C_5 = BFieldElement::from_raw_u64(21474836475ULL);
    static const BFieldElement C_6 = BFieldElement::from_raw_u64(25769803770ULL);
    static const BFieldElement C_7 = BFieldElement::from_raw_u64(30064771065ULL);
    static const BFieldElement C_8 = BFieldElement::from_raw_u64(34359738360ULL);
    static const BFieldElement C_9 = BFieldElement::from_raw_u64(38654705655ULL);
    static const BFieldElement C_10 = BFieldElement::from_raw_u64(18446743828896415801ULL);
    static const BFieldElement C_11 = BFieldElement::from_raw_u64(18446743863256154161ULL);
    static const BFieldElement C_12 = BFieldElement::from_raw_u64(18446743897615892521ULL);
    static const BFieldElement C_13 = BFieldElement::from_raw_u64(18446743923385696291ULL);
    static const BFieldElement C_14 = BFieldElement::from_raw_u64(18446743931975630881ULL);
    static const BFieldElement C_15 = BFieldElement::from_raw_u64(18446743940565565471ULL);
    static const BFieldElement C_16 = BFieldElement::from_raw_u64(18446743949155500061ULL);
    static const BFieldElement C_17 = BFieldElement::from_raw_u64(18446743992105173011ULL);
    static const BFieldElement C_18 = BFieldElement::from_raw_u64(18446744000695107601ULL);
    static const BFieldElement C_19 = BFieldElement::from_raw_u64(18446744009285042191ULL);
    static const BFieldElement C_20 = BFieldElement::from_raw_u64(18446744017874976781ULL);
    static const BFieldElement C_21 = BFieldElement::from_raw_u64(18446744043644780551ULL);
    static const BFieldElement C_22 = BFieldElement::from_raw_u64(18446744047939747846ULL);
    static const BFieldElement C_23 = BFieldElement::from_raw_u64(18446744052234715141ULL);
    static const BFieldElement C_24 = BFieldElement::from_raw_u64(18446744056529682436ULL);
    static const BFieldElement C_25 = BFieldElement::from_raw_u64(18446744060824649731ULL);
    static const BFieldElement C_26 = BFieldElement::from_raw_u64(18446744065119617026ULL);
} // namespace degree_lowering_constants

using namespace degree_lowering_constants;

void fill_degree_lowering_main_columns_cpp(std::vector<std::vector<BFieldElement>>& data) {
    if (data.empty() || data[0].empty()) return;
    
    const bool profile = (std::getenv("TVM_PROFILE_PAD") != nullptr);
    auto t_start = std::chrono::high_resolution_clock::now();
    
    const size_t num_rows = data.size();
    const size_t num_cols = data[0].size();
    
    if (num_cols < 379) {
        std::cerr << "Error: Main table must have at least 379 columns for degree lowering" << std::endl;
        return;
    }
    
    // =========================================================================
    // Phase 1: Compute columns 149-150 (2 columns)
    // These depend only on columns 0-148
    // =========================================================================
    auto t_phase1_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t r = 0; r < num_rows; ++r) {
        auto& row = data[r];
        
        // Precompute common expressions to reduce redundant operations
        const BFieldElement r12_plus_c26 = row[12] + C_26;
        const BFieldElement r14_plus_c26 = row[14] + C_26;
        const BFieldElement r15_plus_c26 = row[15] + C_26;
        const BFieldElement r17_plus_c26 = row[17] + C_26;
        const BFieldElement r18_plus_c26 = row[18] + C_26;
        
        // Column 149: ((row[12] + NEG_ONE) * row[13] * (row[14] + NEG_ONE) * (row[15] + NEG_ONE))
        row[149] = ((r12_plus_c26 * row[13]) * r14_plus_c26) * r15_plus_c26;
        
        // Column 150: (row[149] * row[16] * (row[17] + NEG_ONE) * (row[18] + NEG_ONE))
        row[150] = ((row[149] * row[16]) * r17_plus_c26) * r18_plus_c26;
    }
    
    if (profile) {
        auto t_phase1_end = std::chrono::high_resolution_clock::now();
        std::cout << "      phase1 (cols 149-150): " 
                  << std::chrono::duration<double, std::milli>(t_phase1_end - t_phase1_start).count() 
                  << " ms" << std::endl;
    }
    
    // =========================================================================
    // Phase 2: Compute columns 151-168 (18 columns)
    // These depend on columns 0-150
    // =========================================================================
    auto t_phase2_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t r = 0; r < num_rows; ++r) {
        auto& row = data[r];
        
        // Column 151
        row[151] = (row[64]) * ((row[64]) + (C_26));

        // Column 152
        row[152] = ((((row[64]) + (C_26)) * ((row[64]) + (C_25))) * ((row[64]) + (C_24))) * ((row[64]) + (C_23));

        // Column 153
        row[153] = (((row[64]) * ((row[64]) + (C_25))) * ((row[64]) + (C_24))) * ((row[64]) + (C_23));

        // Column 154
        row[154] = (row[151]) * ((row[64]) + (C_25));

        // Column 155
        row[155] = (((row[151]) * ((row[64]) + (C_24))) * ((row[64]) + (C_23))) * ((row[64]) + (C_22));

        // Column 156
        row[156] = ((row[142]) + (C_23)) * ((row[142]) + (C_21));

        // Column 157
        row[157] = (row[143]) * (row[144]);

        // Column 158
        row[158] = ((((row[142]) + (C_23)) * ((row[142]) + (C_19))) * ((row[142]) + (C_20))) * ((row[142]) + (C_15));

        // Column 159
        row[159] = (row[156]) * ((row[142]) + (C_19));

        // Column 160
        row[160] = (row[145]) * (row[146]);

        // Column 161
        row[161] = (((row[62]) + (C_26)) * ((row[62]) + (C_25))) * (row[62]);

        // Column 162
        row[162] = (row[158]) * ((row[142]) + (C_16));

        // Column 163
        row[163] = (((row[156]) * ((row[142]) + (C_20))) * ((row[142]) + (C_15))) * ((row[142]) + (C_16));

        // Column 164
        row[164] = ((row[159]) * ((row[142]) + (C_15))) * ((row[142]) + (C_16));

        // Column 165
        row[165] = ((((row[62]) + (C_26)) * ((row[62]) + (C_24))) * (row[62])) * ((row[63]) + (C_12));

        // Column 166
        row[166] = (row[159]) * ((row[142]) + (C_20));

        // Column 167
        row[167] = (((row[162]) * ((row[139]) + (C_26))) * ((C_0) + ((C_26) * (row[157])))) * ((C_0) + ((C_26) * (row[160])));

        // Column 168
        row[168] = (((row[162]) * (row[139])) * ((C_0) + ((C_26) * (row[157])))) * ((C_0) + ((C_26) * (row[160])));

    }
    
    if (profile) {
        auto t_phase2_end = std::chrono::high_resolution_clock::now();
        std::cout << "      phase2 (cols 151-168): " 
                  << std::chrono::duration<double, std::milli>(t_phase2_end - t_phase2_start).count() 
                  << " ms" << std::endl;
    }
    
    // =========================================================================
    // Phase 3: Compute columns 169-378 (210 columns)
    // These depend on columns 0-168 and may use next row
    // =========================================================================
    auto t_phase3_start = std::chrono::high_resolution_clock::now();
    
    // Phase 3 uses current_main_row and next_main_row
    // Use dynamic scheduling with good chunk size for better load balancing
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t r = 0; r < num_rows - 1; ++r) {
        auto& cur = data[r];
        const auto& next = data[r + 1];
        
        // Column 169
        cur[169] = ((cur[12]) + (C_26)) * (cur[13]);

        // Column 170
        cur[170] = ((cur[12]) * ((cur[13]) + (C_26))) * ((cur[14]) + (C_26));

        // Column 171
        cur[171] = (cur[169]) * ((cur[14]) + (C_26));

        // Column 172
        cur[172] = ((cur[12]) + (C_26)) * ((cur[13]) + (C_26));

        // Column 173
        cur[173] = ((C_0) + ((C_26) * (cur[42]))) * ((C_0) + ((C_26) * (cur[41])));

        // Column 174
        cur[174] = (cur[172]) * ((cur[14]) + (C_26));

        // Column 175
        cur[175] = ((C_0) + ((C_26) * (cur[42]))) * (cur[41]);

        // Column 176
        cur[176] = (cur[170]) * ((cur[15]) + (C_26));

        // Column 177
        cur[177] = (cur[170]) * (cur[15]);

        // Column 178
        cur[178] = (cur[171]) * ((cur[15]) + (C_26));

        // Column 179
        cur[179] = (cur[173]) * (cur[40]);

        // Column 180
        cur[180] = (cur[171]) * (cur[15]);

        // Column 181
        cur[181] = (cur[175]) * ((C_0) + ((C_26) * (cur[40])));

        // Column 182
        cur[182] = (cur[176]) * ((cur[16]) + (C_26));

        // Column 183
        cur[183] = (cur[174]) * ((cur[15]) + (C_26));

        // Column 184
        cur[184] = (cur[174]) * (cur[15]);

        // Column 185
        cur[185] = (cur[173]) * ((C_0) + ((C_26) * (cur[40])));

        // Column 186
        cur[186] = ((cur[12]) * (cur[13])) * ((cur[14]) + (C_26));

        // Column 187
        cur[187] = (cur[177]) * ((cur[16]) + (C_26));

        // Column 188
        cur[188] = (cur[169]) * (cur[14]);

        // Column 189
        cur[189] = (cur[180]) * ((cur[16]) + (C_26));

        // Column 190
        cur[190] = (cur[178]) * ((cur[16]) + (C_26));

        // Column 191
        cur[191] = (cur[177]) * (cur[16]);

        // Column 192
        cur[192] = (cur[42]) * ((C_0) + ((C_26) * (cur[41])));

        // Column 193
        cur[193] = (cur[42]) * (cur[41]);

        // Column 194
        cur[194] = (cur[172]) * (cur[14]);

        // Column 195
        cur[195] = (cur[185]) * (cur[39]);

        // Column 196
        cur[196] = (cur[179]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 197
        cur[197] = (cur[183]) * ((cur[16]) + (C_26));

        // Column 198
        cur[198] = (cur[179]) * (cur[39]);

        // Column 199
        cur[199] = (cur[178]) * (cur[16]);

        // Column 200
        cur[200] = (cur[181]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 201
        cur[201] = (cur[182]) * ((cur[17]) + (C_26));

        // Column 202
        cur[202] = (cur[181]) * (cur[39]);

        // Column 203
        cur[203] = (cur[176]) * (cur[16]);

        // Column 204
        cur[204] = (cur[190]) * ((cur[17]) + (C_26));

        // Column 205
        cur[205] = (cur[184]) * ((cur[16]) + (C_26));

        // Column 206
        cur[206] = (cur[180]) * (cur[16]);

        // Column 207
        cur[207] = (cur[186]) * ((cur[15]) + (C_26));

        // Column 208
        cur[208] = (cur[189]) * ((cur[17]) + (C_26));

        // Column 209
        cur[209] = (cur[187]) * ((cur[17]) + (C_26));

        // Column 210
        cur[210] = (cur[201]) * ((cur[18]) + (C_26));

        // Column 211
        cur[211] = ((cur[182]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 212
        cur[212] = (cur[184]) * (cur[16]);

        // Column 213
        cur[213] = (cur[197]) * ((cur[17]) + (C_26));

        // Column 214
        cur[214] = (cur[188]) * ((cur[15]) + (C_26));

        // Column 215
        cur[215] = (((cur[207]) * ((cur[16]) + (C_26))) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 216
        cur[216] = (cur[183]) * (cur[16]);

        // Column 217
        cur[217] = (cur[194]) * ((cur[15]) + (C_26));

        // Column 218
        cur[218] = (cur[204]) * ((cur[18]) + (C_26));

        // Column 219
        cur[219] = (cur[188]) * (cur[15]);

        // Column 220
        cur[220] = ((cur[191]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 221
        cur[221] = (((cur[186]) * (cur[15])) * ((cur[16]) + (C_26))) * ((cur[17]) + (C_26));

        // Column 222
        cur[222] = (cur[199]) * ((cur[17]) + (C_26));

        // Column 223
        cur[223] = (cur[209]) * ((cur[18]) + (C_26));

        // Column 224
        cur[224] = (cur[205]) * ((cur[17]) + (C_26));

        // Column 225
        cur[225] = ((cur[203]) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 226
        cur[226] = (cur[208]) * ((cur[18]) + (C_26));

        // Column 227
        cur[227] = ((cur[191]) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 228
        cur[228] = (cur[221]) * ((cur[18]) + (C_26));

        // Column 229
        cur[229] = ((cur[187]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 230
        cur[230] = (cur[217]) * ((cur[16]) + (C_26));

        // Column 231
        cur[231] = (cur[175]) * (cur[40]);

        // Column 232
        cur[232] = (cur[192]) * ((C_0) + ((C_26) * (cur[40])));

        // Column 233
        cur[233] = (cur[192]) * (cur[40]);

        // Column 234
        cur[234] = (cur[193]) * ((C_0) + ((C_26) * (cur[40])));

        // Column 235
        cur[235] = (cur[193]) * (cur[40]);

        // Column 236
        cur[236] = (cur[194]) * (cur[15]);

        // Column 237
        cur[237] = ((cur[189]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 238
        cur[238] = ((cur[199]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 239
        cur[239] = (cur[222]) * ((cur[18]) + (C_26));

        // Column 240
        cur[240] = (cur[212]) * ((cur[17]) + (C_26));

        // Column 241
        cur[241] = ((cur[206]) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 242
        cur[242] = (((cur[214]) * ((cur[16]) + (C_26))) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 243
        cur[243] = (cur[41]) * ((cur[41]) + (C_26));

        // Column 244
        cur[244] = ((cur[206]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 245
        cur[245] = ((cur[230]) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 246
        cur[246] = (((cur[219]) * ((cur[16]) + (C_26))) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 247
        cur[247] = (cur[213]) * ((cur[18]) + (C_26));

        // Column 248
        cur[248] = (((cur[214]) * (cur[16])) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 249
        cur[249] = (cur[216]) * ((cur[17]) + (C_26));

        // Column 250
        cur[250] = (cur[224]) * ((cur[18]) + (C_26));

        // Column 251
        cur[251] = ((cur[203]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 252
        cur[252] = ((cur[197]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 253
        cur[253] = (((cur[219]) * (cur[16])) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 254
        cur[254] = (cur[42]) * ((cur[42]) + (C_26));

        // Column 255
        cur[255] = (((cur[97]) * (cur[97])) * (cur[97])) * (cur[97]);

        // Column 256
        cur[256] = (cur[236]) * ((cur[16]) + (C_26));

        // Column 257
        cur[257] = (((cur[98]) * (cur[98])) * (cur[98])) * (cur[98]);

        // Column 258
        cur[258] = (((cur[99]) * (cur[99])) * (cur[99])) * (cur[99]);

        // Column 259
        cur[259] = (cur[240]) * ((cur[18]) + (C_26));

        // Column 260
        cur[260] = (((cur[100]) * (cur[100])) * (cur[100])) * (cur[100]);

        // Column 261
        cur[261] = ((cur[205]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 262
        cur[262] = ((cur[190]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 263
        cur[263] = (((cur[101]) * (cur[101])) * (cur[101])) * (cur[101]);

        // Column 264
        cur[264] = ((cur[216]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 265
        cur[265] = (cur[201]) * (cur[18]);

        // Column 266
        cur[266] = ((cur[212]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 267
        cur[267] = (cur[213]) * (cur[18]);

        // Column 268
        cur[268] = (((cur[102]) * (cur[102])) * (cur[102])) * (cur[102]);

        // Column 269
        cur[269] = (cur[249]) * ((cur[18]) + (C_26));

        // Column 270
        cur[270] = (((cur[217]) * (cur[16])) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 271
        cur[271] = (((cur[103]) * (cur[103])) * (cur[103])) * (cur[103]);

        // Column 272
        cur[272] = (cur[204]) * (cur[18]);

        // Column 273
        cur[273] = ((cur[256]) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 274
        cur[274] = (cur[39]) * ((cur[28]) + ((C_26) * (cur[27])));

        // Column 275
        cur[275] = (cur[208]) * (cur[18]);

        // Column 276
        cur[276] = (((cur[104]) * (cur[104])) * (cur[104])) * (cur[104]);

        // Column 277
        cur[277] = (((cur[236]) * (cur[16])) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 278
        cur[278] = (((cur[105]) * (cur[105])) * (cur[105])) * (cur[105]);

        // Column 279
        cur[279] = (((cur[207]) * (cur[16])) * ((cur[17]) + (C_26))) * ((cur[18]) + (C_26));

        // Column 280
        cur[280] = (((cur[106]) * (cur[106])) * (cur[106])) * (cur[106]);

        // Column 281
        cur[281] = (((cur[107]) * (cur[107])) * (cur[107])) * (cur[107]);

        // Column 282
        cur[282] = (cur[39]) * (cur[22]);

        // Column 283
        cur[283] = (((cur[108]) * (cur[108])) * (cur[108])) * (cur[108]);

        // Column 284
        cur[284] = (cur[224]) * (cur[18]);

        // Column 285
        cur[285] = (cur[185]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 286
        cur[286] = (cur[231]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 287
        cur[287] = (cur[231]) * (cur[39]);

        // Column 288
        cur[288] = (cur[232]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 289
        cur[289] = (cur[232]) * (cur[39]);

        // Column 290
        cur[290] = (cur[233]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 291
        cur[291] = (cur[233]) * (cur[39]);

        // Column 292
        cur[292] = (cur[234]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 293
        cur[293] = (cur[234]) * (cur[39]);

        // Column 294
        cur[294] = (cur[235]) * ((C_0) + ((C_26) * (cur[39])));

        // Column 295
        cur[295] = (cur[235]) * (cur[39]);

        // Column 296
        cur[296] = (cur[222]) * (cur[18]);

        // Column 297
        cur[297] = (cur[209]) * (cur[18]);

        // Column 298
        cur[298] = (((next[64]) * ((next[64]) + (C_26))) * ((next[64]) + (C_25))) * ((next[64]) + (C_24));

        // Column 299
        cur[299] = (cur[44]) * ((cur[44]) + (C_26));

        // Column 300
        cur[300] = ((next[62]) * ((next[64]) + (C_22))) * ((next[63]) + (C_12));

        // Column 301
        cur[301] = ((cur[256]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 302
        cur[302] = ((cur[230]) * (cur[17])) * ((cur[18]) + (C_26));

        // Column 303
        cur[303] = (((cur[43]) * ((cur[43]) + (C_26))) * ((cur[43]) + (C_25))) * ((cur[43]) + (C_24));

        // Column 304
        cur[304] = (cur[39]) * ((cur[39]) + (C_26));

        // Column 305
        cur[305] = (cur[40]) * ((cur[40]) + (C_26));

        // Column 306
        cur[306] = (cur[249]) * (cur[18]);

        // Column 307
        cur[307] = (cur[240]) * (cur[18]);

        // Column 308
        cur[308] = ((((next[142]) + (C_23)) * ((next[142]) + (C_19))) * ((next[142]) + (C_20))) * ((next[142]) + (C_15));

        // Column 309
        cur[309] = ((next[142]) + (C_23)) * ((next[142]) + (C_21));

        // Column 310
        cur[310] = ((((next[64]) + (C_26)) * ((next[64]) + (C_25))) * ((next[64]) + (C_24))) * ((next[64]) + (C_23));

        // Column 311
        cur[311] = (((cur[255]) * (cur[97])) * (cur[97])) * (cur[97]);

        // Column 312
        cur[312] = (((cur[257]) * (cur[98])) * (cur[98])) * (cur[98]);

        // Column 313
        cur[313] = (((cur[258]) * (cur[99])) * (cur[99])) * (cur[99]);

        // Column 314
        cur[314] = (((cur[260]) * (cur[100])) * (cur[100])) * (cur[100]);

        // Column 315
        cur[315] = (((cur[263]) * (cur[101])) * (cur[101])) * (cur[101]);

        // Column 316
        cur[316] = (((cur[268]) * (cur[102])) * (cur[102])) * (cur[102]);

        // Column 317
        cur[317] = (((cur[271]) * (cur[103])) * (cur[103])) * (cur[103]);

        // Column 318
        cur[318] = (((cur[276]) * (cur[104])) * (cur[104])) * (cur[104]);

        // Column 319
        cur[319] = (((cur[278]) * (cur[105])) * (cur[105])) * (cur[105]);

        // Column 320
        cur[320] = (((cur[280]) * (cur[106])) * (cur[106])) * (cur[106]);

        // Column 321
        cur[321] = (((cur[281]) * (cur[107])) * (cur[107])) * (cur[107]);

        // Column 322
        cur[322] = (((cur[283]) * (cur[108])) * (cur[108])) * (cur[108]);

        // Column 323
        cur[323] = ((next[139]) + (C_26)) * ((cur[308]) * ((next[142]) + (C_16)));

        // Column 324
        cur[324] = (cur[309]) * ((next[142]) + (C_19));

        // Column 325
        cur[325] = (cur[310]) * ((next[64]) + (C_22));

        // Column 326
        cur[326] = (((next[12]) + (C_26)) * ((next[13]) + (C_26))) * (next[14]);

        // Column 327
        cur[327] = (cur[39]) * ((cur[23]) + ((C_26) * (cur[22])));

        // Column 328
        cur[328] = ((cur[323]) * (next[147])) * ((next[147]) + (C_26));

        // Column 329
        cur[329] = ((((next[12]) + (C_26)) * (next[13])) * ((next[14]) + (C_26))) * ((next[15]) + (C_26));

        // Column 330
        cur[330] = (((next[62]) + (C_26)) * ((next[62]) + (C_25))) * (next[62]);

        // Column 331
        cur[331] = (cur[24]) * (cur[27]);

        // Column 332
        cur[332] = (cur[24]) * (next[24]);

        // Column 333
        cur[333] = (cur[324]) * ((next[142]) + (C_20));

        // Column 334
        cur[334] = ((((cur[10]) + (C_12)) * ((cur[10]) + (C_13))) * ((cur[10]) + (C_11))) * ((cur[10]) + (C_10));

        // Column 335
        cur[335] = ((next[139]) + (C_26)) * (((cur[324]) * ((next[142]) + (C_15))) * ((next[142]) + (C_16)));

        // Column 336
        cur[336] = ((C_0) + ((C_26) * (next[44]))) * (next[22]);

        // Column 337
        cur[337] = (next[44]) * (next[39]);

        // Column 338
        cur[338] = ((C_0) + ((C_26) * (next[44]))) * (next[23]);

        // Column 339
        cur[339] = (next[44]) * (next[40]);

        // Column 340
        cur[340] = ((C_0) + ((C_26) * (next[44]))) * (next[24]);

        // Column 341
        cur[341] = (next[44]) * (next[41]);

        // Column 342
        cur[342] = ((C_0) + ((C_26) * (next[44]))) * (next[25]);

        // Column 343
        cur[343] = (next[44]) * (next[42]);

        // Column 344
        cur[344] = ((C_0) + ((C_26) * (next[44]))) * (next[26]);

        // Column 345
        cur[345] = (next[44]) * (next[43]);

        // Column 346
        cur[346] = ((C_0) + ((C_26) * (next[44]))) * (next[39]);

        // Column 347
        cur[347] = (next[44]) * (next[22]);

        // Column 348
        cur[348] = ((C_0) + ((C_26) * (next[44]))) * (next[40]);

        // Column 349
        cur[349] = (next[44]) * (next[23]);

        // Column 350
        cur[350] = ((C_0) + ((C_26) * (next[44]))) * (next[41]);

        // Column 351
        cur[351] = (next[44]) * (next[24]);

        // Column 352
        cur[352] = ((C_0) + ((C_26) * (next[44]))) * (next[42]);

        // Column 353
        cur[353] = (next[44]) * (next[25]);

        // Column 354
        cur[354] = ((C_0) + ((C_26) * (next[44]))) * (next[43]);

        // Column 355
        cur[355] = (next[44]) * (next[26]);

        // Column 356
        cur[356] = (((cur[329]) * (next[16])) * ((next[17]) + (C_26))) * ((next[18]) + (C_26));

        // Column 357
        cur[357] = (((cur[326]) * ((next[15]) + (C_26))) * ((next[16]) + (C_26))) * (next[17]);

        // Column 358
        cur[358] = (cur[39]) * (cur[42]);

        // Column 359
        cur[359] = (((cur[326]) * (next[15])) * ((next[16]) + (C_26))) * (next[17]);

        // Column 360
        cur[360] = (cur[325]) * ((((next[62]) + (C_25)) * ((next[62]) + (C_24))) * (next[62]));

        // Column 361
        cur[361] = (((cur[62]) + (C_25)) * ((cur[62]) + (C_24))) * (cur[62]);

        // Column 362
        cur[362] = (cur[245]) * ((next[22]) * (((cur[39]) * ((next[23]) + (C_1))) + (C_26)));

        // Column 363
        cur[363] = (cur[218]) * ((((((next[9]) + ((C_26) * (cur[9]))) + (C_26)) * (cur[22])) + (((((next[9]) + ((C_26) * (cur[9]))) + (C_25)) * ((cur[282]) + (C_26))) * ((cur[40]) + (C_26)))) + (((((next[9]) + ((C_26) * (cur[9]))) + (C_24)) * ((cur[282]) + (C_26))) * (cur[40])));

        // Column 364
        cur[364] = (cur[218]) * (((cur[243]) * ((cur[41]) + (C_25))) * ((cur[41]) + (C_24)));

        // Column 365
        cur[365] = (cur[218]) * (((cur[254]) * ((cur[42]) + (C_25))) * ((cur[42]) + (C_24)));

        // Column 366
        cur[366] = (cur[218]) * (((cur[299]) * ((cur[44]) + (C_25))) * ((cur[44]) + (C_24)));

        // Column 367
        cur[367] = (((cur[64]) * ((cur[64]) + (C_26))) * ((cur[64]) + (C_25))) * ((cur[64]) + (C_24));

        // Column 368
        cur[368] = ((((cur[62]) + (C_26)) * ((cur[62]) + (C_24))) * (cur[62])) * ((next[62]) + (C_25));

        // Column 369
        cur[369] = (((cur[309]) * ((next[142]) + (C_20))) * ((next[142]) + (C_15))) * ((next[142]) + (C_16));

        // Column 370
        cur[370] = (cur[328]) * ((((C_0) + ((C_26) * ((cur[143]) + ((C_26) * ((C_2) * (next[143])))))) + ((C_26) * ((cur[145]) + ((C_26) * ((C_2) * (next[145])))))) + (((C_2) * ((cur[143]) + ((C_26) * ((C_2) * (next[143]))))) * ((cur[145]) + ((C_26) * ((C_2) * (next[145]))))));

        // Column 371
        cur[371] = ((next[139]) + (C_26)) * ((cur[333]) * ((next[142]) + (C_16)));

        // Column 372
        cur[372] = (((((next[59]) + ((C_26) * (cur[59]))) + (C_26)) * ((cur[58]) + (C_18))) * ((cur[58]) + (C_14))) * (((next[57]) + ((C_26) * (cur[57]))) + (C_26));

        // Column 373
        cur[373] = (cur[361]) * ((((next[62]) + (C_26)) * ((next[62]) + (C_24))) * (next[62]));

        // Column 374
        cur[374] = ((((cur[62]) + (C_26)) * ((cur[62]) + (C_25))) * (cur[62])) * ((next[62]) + (C_24));

        // Column 375
        cur[375] = (((cur[325]) * ((next[62]) + (C_24))) * (next[62])) * ((next[63]) + (C_12));

        // Column 376
        cur[376] = (cur[325]) * ((((next[63]) + (C_17)) * ((next[63]) + (C_12))) * ((next[63]) + (C_13)));

        // Column 377
        cur[377] = ((cur[335]) * ((C_0) + ((C_26) * ((next[143]) * (next[144]))))) * (cur[143]);

        // Column 378
        cur[378] = ((next[147]) * (next[147])) * (cur[143]);


    }
    
    if (profile) {
        auto t_phase3_end = std::chrono::high_resolution_clock::now();
        std::cout << "      phase3 (cols 169-378): " 
                  << std::chrono::duration<double, std::milli>(t_phase3_end - t_phase3_start).count() 
                  << " ms" << std::endl;
    }
    
    // Last row: copy from second-to-last for phase 3 columns
    if (num_rows > 1) {
        for (size_t c = 169; c < 379; ++c) {
            data[num_rows - 1][c] = data[num_rows - 2][c];
        }
    }
    
    if (profile) {
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "      total cpp: " 
                  << std::chrono::duration<double, std::milli>(t_end - t_start).count() 
                  << " ms" << std::endl;
    }
}

} // namespace triton_vm

#pragma once
// Generated from src/quotient/constraint_evaluations.cpp
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py
//
// This file contains initial constraints.

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_initial_weighted(const Bfe* main_row, const Xfe* aux_row, const Xfe* challenges, const Xfe* weights) {
            const auto node_468 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[129]));
            const auto node_474 = ((challenges[52]) * (main_row[131]))
                + ((challenges[53]) * (main_row[133]));
            const auto node_477 = ((challenges[52]) * (main_row[130]))
                + ((challenges[53]) * (main_row[132]));
    Xfe acc = Xfe::zero();
    acc = acc + Xfe(main_row[0]) * weights[0];
    acc = acc + Xfe(main_row[3]) * weights[1];
    acc = acc + Xfe(main_row[5]) * weights[2];
    acc = acc + Xfe(main_row[7]) * weights[3];
    acc = acc + Xfe(main_row[9]) * weights[4];
    acc = acc + Xfe(main_row[19]) * weights[5];
    acc = acc + Xfe(main_row[20]) * weights[6];
    acc = acc + Xfe(main_row[21]) * weights[7];
    acc = acc + Xfe(main_row[22]) * weights[8];
    acc = acc + Xfe(main_row[23]) * weights[9];
    acc = acc + Xfe(main_row[24]) * weights[10];
    acc = acc + Xfe(main_row[25]) * weights[11];
    acc = acc + Xfe(main_row[26]) * weights[12];
    acc = acc + Xfe(main_row[27]) * weights[13];
    acc = acc + Xfe(main_row[28]) * weights[14];
    acc = acc + Xfe(main_row[29]) * weights[15];
    acc = acc + Xfe(main_row[30]) * weights[16];
    acc = acc + Xfe(main_row[31]) * weights[17];
    acc = acc + Xfe(main_row[32]) * weights[18];
    acc = acc + Xfe((main_row[38]) + (Bfe::from_raw_u64(18446744000695107601ULL))) * weights[19];
    acc = acc + Xfe((main_row[48]) + (Bfe::from_raw_u64(18446744000695107601ULL))) * weights[20];
    acc = acc + Xfe(main_row[55]) * weights[21];
    acc = acc + Xfe(main_row[57]) * weights[22];
    acc = acc + Xfe(main_row[59]) * weights[23];
    acc = acc + Xfe(main_row[60]) * weights[24];
    acc = acc + Xfe(main_row[61]) * weights[25];
    acc = acc + Xfe((main_row[62]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[26];
    acc = acc + Xfe(main_row[64]) * weights[27];
    acc = acc + Xfe(main_row[136]) * weights[28];
    acc = acc + Xfe((main_row[149])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((((main_row[12])
                            + (Bfe::from_raw_u64(18446744065119617026ULL)))
                            * (main_row[13]))
                            * ((main_row[14])
                                + (Bfe::from_raw_u64(18446744065119617026ULL))))
                            * ((main_row[15])
                                + (Bfe::from_raw_u64(18446744065119617026ULL)))))) * weights[29];
    acc = acc + Xfe((main_row[150])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[149]) * (main_row[16]))
                            * ((main_row[17])
                                + (Bfe::from_raw_u64(18446744065119617026ULL))))
                            * ((main_row[18])
                                + (Bfe::from_raw_u64(18446744065119617026ULL)))))) * weights[30];
    acc = acc + (aux_row[0]) * weights[31];
    acc = acc + (((aux_row[1])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[29])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (main_row[1]))) * weights[32];
    acc = acc + ((aux_row[2]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[33];
    acc = acc + (((((((((((challenges[0]) + (main_row[33])) * (challenges[0]))
                    + (main_row[34])) * (challenges[0])) + (main_row[35]))
                    * (challenges[0])) + (main_row[36])) * (challenges[0]))
                    + (main_row[37]))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[62]))) * weights[34];
    acc = acc + ((aux_row[3]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[35];
    acc = acc + (((aux_row[5])
                    * ((challenges[3])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[14]) * (main_row[10]))
                                + ((challenges[15]) * (main_row[11]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[36];
    acc = acc + ((aux_row[4]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[37];
    acc = acc + ((aux_row[6]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[38];
    acc = acc + ((aux_row[7]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[39];
    acc = acc + ((aux_row[8])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[9])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * ((challenges[25]) * (main_row[10])))))) * weights[40];
    acc = acc + (((aux_row[13]) * (challenges[11]))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (main_row[45]))) * weights[41];
    acc = acc + ((((main_row[10])
                    + (Bfe::from_raw_u64(18446743992105173011ULL)))
                    * ((aux_row[9])
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((main_row[150])
                        * ((aux_row[9])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (challenges[4]))))) * weights[42];
    acc = acc + ((aux_row[10]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[43];
    acc = acc + ((aux_row[11]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[44];
    acc = acc + (aux_row[12]) * weights[45];
    acc = acc + ((((aux_row[14])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[7])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((((challenges[16]) * (main_row[46]))
                                    + ((challenges[17]) * (main_row[47])))
                                    + ((challenges[18])
                                        * (Bfe::from_raw_u64(68719476720ULL))))
                                    + ((challenges[19]) * (main_row[49])))))))
                    * ((main_row[47])
                        + (Bfe::from_raw_u64(18446744060824649731ULL))))
                    + (((aux_row[14])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))
                        * ((main_row[47])
                            * ((main_row[47])
                                + (Bfe::from_raw_u64(18446744065119617026ULL)))))) * weights[46];
    acc = acc + (aux_row[15]) * weights[47];
    acc = acc + (aux_row[18]) * weights[48];
    acc = acc + ((aux_row[19])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (main_row[56]))) * weights[49];
    acc = acc + (((aux_row[16])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[12]))) + (main_row[52])) * weights[50];
    acc = acc + ((aux_row[17]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[51];
    acc = acc + (((((aux_row[20])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[8])))
                    + (((((main_row[50]) * (challenges[20]))
                        + ((main_row[51]) * (challenges[23])))
                        + ((main_row[52]) * (challenges[21])))
                        + ((main_row[53]) * (challenges[22]))))
                    * ((main_row[51])
                        + (Bfe::from_raw_u64(18446744060824649731ULL))))
                    + (((aux_row[20])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))
                        * (((main_row[51])
                            + (Bfe::from_raw_u64(18446744065119617026ULL)))
                            * (main_row[51])))) * weights[52];
    acc = acc + (aux_row[21]) * weights[53];
    acc = acc + ((aux_row[22])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[9])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * ((challenges[25]) * (main_row[58])))))) * weights[54];
    acc = acc + (aux_row[23]) * weights[55];
    acc = acc + ((aux_row[25]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[56];
    acc = acc + ((aux_row[26]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[57];
    acc = acc + ((aux_row[27]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[58];
    acc = acc + (((aux_row[24])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[30])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((((((((((((((((((challenges[29])
                            + ((((((main_row[65])
                                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                                + ((main_row[66])
                                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                                + ((main_row[67])
                                    * (Bfe::from_raw_u64(281474976645120ULL))))
                                + (main_row[68]))
                                * (Bfe::from_raw_u64(1ULL))))
                            * (challenges[29]))
                            + ((((((main_row[69])
                                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                                + ((main_row[70])
                                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                                + ((main_row[71])
                                    * (Bfe::from_raw_u64(281474976645120ULL))))
                                + (main_row[72]))
                                * (Bfe::from_raw_u64(1ULL))))
                            * (challenges[29]))
                            + ((((((main_row[73])
                                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                                + ((main_row[74])
                                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                                + ((main_row[75])
                                    * (Bfe::from_raw_u64(281474976645120ULL))))
                                + (main_row[76]))
                                * (Bfe::from_raw_u64(1ULL))))
                            * (challenges[29]))
                            + ((((((main_row[77])
                                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                                + ((main_row[78])
                                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                                + ((main_row[79])
                                    * (Bfe::from_raw_u64(281474976645120ULL))))
                                + (main_row[80]))
                                * (Bfe::from_raw_u64(1ULL))))
                            * (challenges[29])) + (main_row[97]))
                            * (challenges[29])) + (main_row[98]))
                            * (challenges[29])) + (main_row[99]))
                            * (challenges[29])) + (main_row[100]))
                            * (challenges[29])) + (main_row[101]))
                            * (challenges[29])) + (main_row[102])))) * weights[59];
    acc = acc + (((aux_row[28])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[65]))
                                + ((challenges[50]) * (main_row[81]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[60];
    acc = acc + (((aux_row[29])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[66]))
                                + ((challenges[50]) * (main_row[82]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[61];
    acc = acc + (((aux_row[30])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[67]))
                                + ((challenges[50]) * (main_row[83]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[62];
    acc = acc + (((aux_row[31])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[68]))
                                + ((challenges[50]) * (main_row[84]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[63];
    acc = acc + (((aux_row[32])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[69]))
                                + ((challenges[50]) * (main_row[85]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[64];
    acc = acc + (((aux_row[33])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[70]))
                                + ((challenges[50]) * (main_row[86]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[65];
    acc = acc + (((aux_row[34])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[71]))
                                + ((challenges[50]) * (main_row[87]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[66];
    acc = acc + (((aux_row[35])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[72]))
                                + ((challenges[50]) * (main_row[88]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[67];
    acc = acc + (((aux_row[36])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[73]))
                                + ((challenges[50]) * (main_row[89]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[68];
    acc = acc + (((aux_row[37])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[74]))
                                + ((challenges[50]) * (main_row[90]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[69];
    acc = acc + (((aux_row[38])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[75]))
                                + ((challenges[50]) * (main_row[91]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[70];
    acc = acc + (((aux_row[39])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[76]))
                                + ((challenges[50]) * (main_row[92]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[71];
    acc = acc + (((aux_row[40])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[77]))
                                + ((challenges[50]) * (main_row[93]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[72];
    acc = acc + (((aux_row[41])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[78]))
                                + ((challenges[50]) * (main_row[94]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[73];
    acc = acc + (((aux_row[42])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[79]))
                                + ((challenges[50]) * (main_row[95]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[74];
    acc = acc + (((aux_row[43])
                    * ((challenges[48])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((challenges[49]) * (main_row[80]))
                                + ((challenges[50]) * (main_row[96]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[75];
    acc = acc + (((node_468)
                    * (((aux_row[44])
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49])
                                    * (((Bfe::from_raw_u64(1099511627520ULL))
                                        * (main_row[130])) + (main_row[131])))
                                    + ((challenges[50])
                                        * (((Bfe::from_raw_u64(1099511627520ULL))
                                            * (main_row[132])) + (main_row[133])))))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (main_row[134]))))
                    + ((main_row[129]) * (aux_row[44]))) * weights[76];
    acc = acc + (((node_468)
                    * ((((((aux_row[45])
                        * ((challenges[51])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_474))))
                        * ((challenges[51])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_477))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((Bfe::from_raw_u64(8589934590ULL))
                                * (challenges[51])))) + (node_474)) + (node_477)))
                    + ((main_row[129]) * (aux_row[45]))) * weights[77];
    acc = acc + (((aux_row[46])
                    * ((challenges[51])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((main_row[137]) * (challenges[53])))))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (main_row[138]))) * weights[78];
    acc = acc + (((aux_row[47])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[54])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (main_row[137]))) * weights[79];
    acc = acc + ((((main_row[139])
                    + (Bfe::from_raw_u64(18446744065119617026ULL)))
                    * (aux_row[48]))
                    + ((main_row[139])
                        * (((aux_row[48])
                            * ((challenges[10])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((((challenges[55]) * (main_row[143]))
                                        + ((challenges[56]) * (main_row[145])))
                                        + ((challenges[57]) * (main_row[142])))
                                        + ((challenges[58]) * (main_row[147]))))))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (main_row[148]))))) * weights[80];
    return acc;
}
}}} // namespace triton_vm::gpu::quotient_gen

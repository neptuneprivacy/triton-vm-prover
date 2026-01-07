#pragma once
// Generated from src/quotient/constraint_evaluations.cpp
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py
//
// This file contains terminal constraints.

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_terminal_weighted(const Bfe* main_row, const Xfe* aux_row, const Xfe* challenges, const Xfe* weights) {

    Xfe acc = Xfe::zero();
    acc = acc + Xfe((main_row[5]) + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[573];
    acc = acc + Xfe(((main_row[3]) + (Bfe::from_raw_u64(18446744030759878666ULL)))
                    * ((main_row[6])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[574];
    acc = acc + Xfe(main_row[10]) * weights[575];
    acc = acc + Xfe(((main_row[62])
                    * ((main_row[63])
                        + (Bfe::from_raw_u64(18446743897615892521ULL))))
                    * ((main_row[64])
                        + (Bfe::from_raw_u64(18446744047939747846ULL)))) * weights[576];
    acc = acc + Xfe((main_row[143])
                    * ((main_row[142])
                        + (Bfe::from_raw_u64(18446743940565565471ULL)))) * weights[577];
    acc = acc + Xfe(main_row[145]) * weights[578];
    acc = acc + ((((aux_row[18]) * (aux_row[16]))
                    + ((aux_row[19]) * (aux_row[17])))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[579];
    acc = acc + (((((main_row[62])
                    + (Bfe::from_raw_u64(18446744060824649731ULL)))
                    * ((main_row[62])
                        + (Bfe::from_raw_u64(18446744056529682436ULL))))
                    * (main_row[62]))
                    * (((((((((((challenges[0])
                        + ((((((main_row[65])
                            * (Bfe::from_raw_u64(18446744069414518785ULL)))
                            + ((main_row[66])
                                * (Bfe::from_raw_u64(18446744069414584320ULL))))
                            + ((main_row[67])
                                * (Bfe::from_raw_u64(281474976645120ULL))))
                            + (main_row[68])) * (Bfe::from_raw_u64(1ULL))))
                        * (challenges[0]))
                        + ((((((main_row[69])
                            * (Bfe::from_raw_u64(18446744069414518785ULL)))
                            + ((main_row[70])
                                * (Bfe::from_raw_u64(18446744069414584320ULL))))
                            + ((main_row[71])
                                * (Bfe::from_raw_u64(281474976645120ULL))))
                            + (main_row[72])) * (Bfe::from_raw_u64(1ULL))))
                        * (challenges[0]))
                        + ((((((main_row[73])
                            * (Bfe::from_raw_u64(18446744069414518785ULL)))
                            + ((main_row[74])
                                * (Bfe::from_raw_u64(18446744069414584320ULL))))
                            + ((main_row[75])
                                * (Bfe::from_raw_u64(281474976645120ULL))))
                            + (main_row[76])) * (Bfe::from_raw_u64(1ULL))))
                        * (challenges[0]))
                        + ((((((main_row[77])
                            * (Bfe::from_raw_u64(18446744069414518785ULL)))
                            + ((main_row[78])
                                * (Bfe::from_raw_u64(18446744069414584320ULL))))
                            + ((main_row[79])
                                * (Bfe::from_raw_u64(281474976645120ULL))))
                            + (main_row[80])) * (Bfe::from_raw_u64(1ULL))))
                        * (challenges[0])) + (main_row[97]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (challenges[62])))) * weights[580];
    acc = acc + ((aux_row[47])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[61]))) * weights[581];
    acc = acc + ((aux_row[2])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[24]))) * weights[582];
    acc = acc + ((challenges[59])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[3]))) * weights[583];
    acc = acc + ((aux_row[4])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (challenges[60]))) * weights[584];
    acc = acc + ((aux_row[5])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[0]))) * weights[585];
    acc = acc + ((aux_row[6])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[14]))) * weights[586];
    acc = acc + ((aux_row[7])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[20]))) * weights[587];
    acc = acc + ((aux_row[8])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[22]))) * weights[588];
    acc = acc + ((aux_row[9])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[25]))) * weights[589];
    acc = acc + ((aux_row[26])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[10]))) * weights[590];
    acc = acc + ((aux_row[11])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[27]))) * weights[591];
    acc = acc + (((((((((((((((((aux_row[44])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[28])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[29])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[30])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[31])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[32])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[33])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[34])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[35])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[36])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[37])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[38])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[39])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[40])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[41])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[42])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[43]))) * weights[592];
    acc = acc + ((aux_row[45])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[46]))) * weights[593];
    acc = acc + ((aux_row[12])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[48]))) * weights[594];
    acc = acc + ((((aux_row[13])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[15])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[21])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (aux_row[23]))) * weights[595];
    return acc;
}
}}} // namespace triton_vm::gpu::quotient_gen

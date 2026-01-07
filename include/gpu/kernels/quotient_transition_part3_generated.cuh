#pragma once
// Generated from src/quotient/constraint_evaluations.cpp
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py
//
// This file contains transition constraints (part 4/4).

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_transition_part3_weighted(const Bfe* current_main_row, const Xfe* current_aux_row, const Bfe* next_main_row, const Xfe* next_aux_row, const Xfe* challenges, const Xfe* weights) {
            const auto node_120 = (next_main_row[19])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[19]));
            const auto node_813 = (challenges[47]) * (current_main_row[37]);
            const auto node_520 = (next_aux_row[3])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[3]));
            const auto node_524 = (next_aux_row[4])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[4]));
            const auto node_624 = (challenges[46]) * (current_main_row[36]);
            const auto node_124 = (next_main_row[20])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[20]));
            const auto node_128 = (next_main_row[21])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[21]));
            const auto node_812 = ((((((((((((((((challenges[32]) * (next_main_row[22]))
                + ((challenges[33]) * (next_main_row[23])))
                + ((challenges[34]) * (next_main_row[24])))
                + ((challenges[35]) * (next_main_row[25])))
                + ((challenges[36]) * (next_main_row[26])))
                + ((challenges[37]) * (next_main_row[27])))
                + ((challenges[38]) * (next_main_row[28])))
                + ((challenges[39]) * (next_main_row[29])))
                + ((challenges[40]) * (next_main_row[30])))
                + ((challenges[41]) * (next_main_row[31])))
                + ((challenges[42]) * (next_main_row[32])))
                + ((challenges[43]) * (next_main_row[33])))
                + ((challenges[44]) * (next_main_row[34])))
                + ((challenges[45]) * (next_main_row[35])))
                + ((challenges[46]) * (next_main_row[36])))
                + ((challenges[47]) * (next_main_row[37]));
            const auto node_622 = (challenges[45]) * (current_main_row[35]);
            const auto node_620 = (challenges[44]) * (current_main_row[34]);
            const auto node_516 = (next_aux_row[7])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[7]));
            const auto node_618 = (challenges[43]) * (current_main_row[33]);
            const auto node_616 = (challenges[42]) * (current_main_row[32]);
            const auto node_2752 = (current_main_row[18])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_614 = (challenges[41]) * (current_main_row[31]);
            const auto node_261 = (next_main_row[38])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[38]));
            const auto node_612 = (challenges[40]) * (current_main_row[30]);
            const auto node_4429 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[8]));
            const auto node_610 = (challenges[39]) * (current_main_row[29]);
            const auto node_1623 = ((next_main_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[9])))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_608 = (challenges[38]) * (current_main_row[28]);
            const auto node_606 = (challenges[37]) * (current_main_row[27]);
            const auto node_2750 = (current_main_row[17])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_604 = (challenges[36]) * (current_main_row[26]);
            const auto node_1129 = (next_aux_row[6])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[6]));
            const auto node_602 = (challenges[35]) * (current_main_row[25]);
            const auto node_272 = ((challenges[16]) * (current_main_row[7]))
                + ((challenges[17]) * (current_main_row[13]));
            const auto node_600 = (challenges[34]) * (current_main_row[24]);
            const auto node_6095 = (current_main_row[298])
                * ((next_main_row[64])
                    + (Bfe::from_raw_u64(18446744052234715141ULL)));
            const auto node_598 = (challenges[33]) * (current_main_row[23]);
            const auto node_6172 = (((next_main_row[63])
                + (Bfe::from_raw_u64(18446743992105173011ULL)))
                * ((next_main_row[63])
                    + (Bfe::from_raw_u64(18446743923385696291ULL))))
                * ((next_main_row[63])
                    + (Bfe::from_raw_u64(18446743828896415801ULL)));
            const auto node_5149 = (((((current_main_row[81])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[82])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[83])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[84])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5160 = (((((current_main_row[85])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[86])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[87])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[88])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5171 = (((((current_main_row[89])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[90])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[91])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[92])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5182 = (((((current_main_row[93])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[94])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[95])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[96])) * (Bfe::from_raw_u64(1ULL));
            const auto node_1715 = ((current_main_row[7]) * (challenges[20]))
                + (challenges[23]);
            const auto node_818 = (challenges[33]) * (current_main_row[22]);
            const auto node_1305 = (next_main_row[22])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[22]));
            const auto node_839 = (challenges[34]) * (current_main_row[23]);
            const auto node_860 = (challenges[35]) * (current_main_row[24]);
            const auto node_538 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[27]);
            const auto node_116 = ((next_main_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[9])))
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_881 = (challenges[36]) * (current_main_row[25]);
            const auto node_536 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[26]);
            const auto node_540 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[28]);
            const auto node_542 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[29]);
            const auto node_2748 = (current_main_row[16])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_902 = (challenges[37]) * (current_main_row[26]);
            const auto node_534 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[25]);
            const auto node_544 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[30]);
            const auto node_546 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[31]);
            const auto node_548 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[32]);
            const auto node_1673 = (next_main_row[25]) + (node_536);
            const auto node_1674 = (next_main_row[26]) + (node_538);
            const auto node_1675 = (next_main_row[27]) + (node_540);
            const auto node_1676 = (next_main_row[28]) + (node_542);
            const auto node_1677 = (next_main_row[29]) + (node_544);
            const auto node_1678 = (next_main_row[30]) + (node_546);
            const auto node_262 = (node_261) + (Bfe::from_raw_u64(4294967295ULL));
            const auto node_1679 = (next_main_row[31]) + (node_548);
            const auto node_1680 = (next_main_row[32])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[33]));
            const auto node_1681 = (next_main_row[33])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[34]));
            const auto node_1682 = (next_main_row[34])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[35]));
            const auto node_1683 = (next_main_row[35])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[36]));
            const auto node_1684 = (next_main_row[36])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[37]));
            const auto node_286 = (next_aux_row[6])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((current_aux_row[6])
                        * ((challenges[7])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((node_272)
                                    + ((challenges[18]) * (next_main_row[38])))
                                    + ((challenges[19]) * (next_main_row[37])))))));
            const auto node_6561 = (next_main_row[139])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_924 = (challenges[38]) * (current_main_row[27]);
            const auto node_550 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[33]);
            const auto node_1671 = (next_main_row[23])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[24]));
            const auto node_154 = ((((current_main_row[11])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((Bfe::from_raw_u64(34359738360ULL))
                        * (current_main_row[42]))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((Bfe::from_raw_u64(17179869180ULL))
                        * (current_main_row[41]))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((Bfe::from_raw_u64(8589934590ULL))
                        * (current_main_row[40]))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[39]));
            const auto node_1672 = (next_main_row[24]) + (node_534);
            const auto node_4916 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((next_main_row[52])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[52]))) * (current_main_row[54])));
            const auto node_4578 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[44]));
            const auto node_816 = (node_812)
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((((((((((((((((challenges[32]) * (current_main_row[22]))
                        + (node_598)) + (node_600)) + (node_602)) + (node_604)) + (node_606))
                        + (node_608)) + (node_610)) + (node_612)) + (node_614)) + (node_616))
                        + (node_618)) + (node_620)) + (node_622)) + (node_624))
                        + (node_813)));
            const auto node_946 = (challenges[39]) * (current_main_row[28]);
            const auto node_530 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[23]);
            const auto node_532 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[24]);
            const auto node_552 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[34]);
            const auto node_968 = (challenges[40]) * (current_main_row[29]);
            const auto node_554 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[35]);
            const auto node_341 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[39]));
            const auto node_990 = (challenges[41]) * (current_main_row[30]);
            const auto node_556 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[36]);
            const auto node_209 = (challenges[40]) * (next_main_row[30]);
            const auto node_212 = (challenges[41]) * (next_main_row[31]);
            const auto node_215 = (challenges[42]) * (next_main_row[32]);
            const auto node_218 = (challenges[43]) * (next_main_row[33]);
            const auto node_221 = (challenges[44]) * (next_main_row[34]);
            const auto node_224 = (challenges[45]) * (next_main_row[35]);
            const auto node_227 = (challenges[46]) * (next_main_row[36]);
            const auto node_811 = (challenges[47]) * (next_main_row[37]);
            const auto node_4785 = (next_aux_row[12])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[12]));
            const auto node_4913 = (next_main_row[52])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[52]));
            const auto node_5938 = (next_main_row[62])
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_6557 = (current_main_row[145])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((Bfe::from_raw_u64(8589934590ULL))
                        * (next_main_row[145])));
            const auto node_1012 = (challenges[42]) * (current_main_row[31]);
            const auto node_1349 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[37]);
            const auto node_203 = (challenges[38]) * (next_main_row[28]);
            const auto node_206 = (challenges[39]) * (next_main_row[29]);
            const auto node_512 = ((((((((((current_main_row[285])
                + (current_main_row[286])) + (current_main_row[287]))
                + (current_main_row[288])) + (current_main_row[289]))
                + (current_main_row[290])) + (current_main_row[291]))
                + (current_main_row[292])) + (current_main_row[293]))
                + (current_main_row[294])) + (current_main_row[295]);
            const auto node_6554 = (current_main_row[143])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((Bfe::from_raw_u64(8589934590ULL))
                        * (next_main_row[143])));
            const auto node_2746 = (current_main_row[15])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_1713 = (current_main_row[7]) * (challenges[20]);
            const auto node_525 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[11]);
            const auto node_1034 = (challenges[43]) * (current_main_row[32]);
            const auto node_200 = (challenges[37]) * (next_main_row[27]);
            const auto node_2490 = (challenges[1]) * (current_aux_row[3]);
            const auto node_5942 = (next_main_row[63])
                + (Bfe::from_raw_u64(18446743897615892521ULL));
            const auto node_1056 = (challenges[44]) * (current_main_row[33]);
            const auto node_1625 = (current_main_row[282])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_1696 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[274]));
            const auto node_528 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[22]);
            const auto node_113 = (next_main_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[9]));
            const auto node_531 = (next_main_row[24]) + (node_530);
            const auto node_533 = (next_main_row[25]) + (node_532);
            const auto node_535 = (next_main_row[26]) + (node_534);
            const auto node_537 = (next_main_row[27]) + (node_536);
            const auto node_194 = (challenges[35]) * (next_main_row[25]);
            const auto node_197 = (challenges[36]) * (next_main_row[26]);
            const auto node_539 = (next_main_row[28]) + (node_538);
            const auto node_2350 = ((((((((((((((((challenges[33]) * (next_main_row[23]))
                + ((challenges[34]) * (next_main_row[24]))) + (node_194))
                + (node_197)) + (node_200)) + (node_203)) + (node_206)) + (node_209))
                + (node_212)) + (node_215)) + (node_218)) + (node_221)) + (node_224))
                + (node_227)) + (node_811))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((((((((((((((node_598) + (node_600)) + (node_602)) + (node_604))
                        + (node_606)) + (node_608)) + (node_610)) + (node_612)) + (node_614))
                        + (node_616)) + (node_618)) + (node_620)) + (node_622)) + (node_624))
                        + (node_813)));
            const auto node_541 = (next_main_row[29]) + (node_540);
            const auto node_543 = (next_main_row[30]) + (node_542);
            const auto node_545 = (next_main_row[31]) + (node_544);
            const auto node_558 = (node_261)
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_547 = (next_main_row[32]) + (node_546);
            const auto node_549 = (next_main_row[33]) + (node_548);
            const auto node_551 = (next_main_row[34]) + (node_550);
            const auto node_553 = (next_main_row[35]) + (node_552);
            const auto node_555 = (next_main_row[36]) + (node_554);
            const auto node_557 = (next_main_row[37]) + (node_556);
            const auto node_567 = (next_aux_row[6])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((current_aux_row[6])
                        * ((challenges[7])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((node_272)
                                    + ((challenges[18]) * (current_main_row[38])))
                                    + ((challenges[19])
                                        * (current_main_row[37])))))));
            const auto node_5809 = (((((next_main_row[65])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((next_main_row[66])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((next_main_row[67])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (next_main_row[68])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5820 = (((((next_main_row[69])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((next_main_row[70])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((next_main_row[71])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (next_main_row[72])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5831 = (((((next_main_row[73])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((next_main_row[74])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((next_main_row[75])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (next_main_row[76])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5842 = (((((next_main_row[77])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((next_main_row[78])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((next_main_row[79])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (next_main_row[80])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5933 = (next_main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6501 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[135]));
            const auto node_6588 = (next_main_row[142])
                + (Bfe::from_raw_u64(18446743940565565471ULL));
            const auto node_2744 = (current_main_row[14])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_293 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[40]));
            const auto node_6592 = (next_main_row[142])
                + (Bfe::from_raw_u64(18446743949155500061ULL));
            const auto node_34 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((current_main_row[4])
                        * ((Bfe::from_raw_u64(38654705655ULL))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_main_row[3])))));
            const auto node_837 = ((current_main_row[285]) * (node_816))
                + ((current_main_row[195])
                    * ((node_812)
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((((((((((((((((challenges[32])
                                * (current_main_row[23])) + (node_818)) + (node_600))
                                + (node_602)) + (node_604)) + (node_606)) + (node_608))
                                + (node_610)) + (node_612)) + (node_614)) + (node_616))
                                + (node_618)) + (node_620)) + (node_622)) + (node_624))
                                + (node_813)))));
            const auto node_299 = (challenges[32]) * (current_main_row[24]);
            const auto node_346 = (challenges[32]) * (current_main_row[25]);
            const auto node_390 = (challenges[32]) * (current_main_row[26]);
            const auto node_433 = (challenges[32]) * (current_main_row[27]);
            const auto node_1002 = (challenges[32]) * (current_main_row[32]);
            const auto node_1078 = (challenges[45]) * (current_main_row[34]);
            const auto node_1307 = (next_main_row[22]) + (node_530);
            const auto node_2359 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[327]));
            const auto node_2411 = (next_main_row[24]) + (node_532);
            const auto node_2627 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (next_aux_row[7]);
            const auto node_2026 = (next_main_row[25]) + (node_540);
            const auto node_2027 = (next_main_row[26]) + (node_542);
            const auto node_191 = (challenges[34]) * (next_main_row[24]);
            const auto node_2028 = (next_main_row[27]) + (node_544);
            const auto node_2029 = (next_main_row[28]) + (node_546);
            const auto node_2261 = (((((((((((node_200) + (node_203)) + (node_206)) + (node_209))
                + (node_212)) + (node_215)) + (node_218)) + (node_221)) + (node_224))
                + (node_227)) + (node_811))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((((((((((node_606) + (node_608)) + (node_610)) + (node_612))
                        + (node_614)) + (node_616)) + (node_618)) + (node_620)) + (node_622))
                        + (node_624)) + (node_813)));
            const auto node_2030 = (next_main_row[29]) + (node_548);
            const auto node_2031 = (next_main_row[30]) + (node_550);
            const auto node_476 = (((((current_main_row[195]) * (node_262))
                + ((current_main_row[196])
                    * ((node_261) + (Bfe::from_raw_u64(8589934590ULL)))))
                + ((current_main_row[198])
                    * ((node_261) + (Bfe::from_raw_u64(12884901885ULL)))))
                + ((current_main_row[200])
                    * ((node_261) + (Bfe::from_raw_u64(17179869180ULL)))))
                + ((current_main_row[202])
                    * ((node_261) + (Bfe::from_raw_u64(21474836475ULL))));
            const auto node_801 = (((((current_main_row[195]) * (node_558))
                + ((current_main_row[196])
                    * ((node_261) + (Bfe::from_raw_u64(18446744060824649731ULL)))))
                + ((current_main_row[198])
                    * ((node_261) + (Bfe::from_raw_u64(18446744056529682436ULL)))))
                + ((current_main_row[200])
                    * ((node_261) + (Bfe::from_raw_u64(18446744052234715141ULL)))))
                + ((current_main_row[202])
                    * ((node_261) + (Bfe::from_raw_u64(18446744047939747846ULL))));
            const auto node_455 = (node_261) + (Bfe::from_raw_u64(21474836475ULL));
            const auto node_2032 = (next_main_row[31]) + (node_552);
            const auto node_2033 = (next_main_row[32]) + (node_554);
            const auto node_484 = ((((current_aux_row[73]) + (current_aux_row[74]))
                + (current_aux_row[75])) + (current_aux_row[76]))
                + ((current_main_row[202])
                    * ((next_aux_row[6])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[58]))));
            const auto node_809 = ((((current_aux_row[77]) + (current_aux_row[78]))
                + (current_aux_row[79])) + (current_aux_row[80]))
                + (current_aux_row[81]);
            const auto node_468 = (next_aux_row[6])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[58]));
            const auto node_2034 = (next_main_row[33]) + (node_556);
            const auto node_2035 = (next_main_row[34]) + (node_1349);
            const auto node_372 = (node_261) + (Bfe::from_raw_u64(12884901885ULL));
            const auto node_385 = (next_aux_row[6])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((current_aux_row[6]) * (current_aux_row[51])));
            const auto node_4551 = (next_main_row[18])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4696 = ((next_aux_row[11])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[6]) * (current_aux_row[11]))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[31]) * (current_main_row[10])));
            const auto node_4742 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * ((challenges[57]) * (current_main_row[10]));
            const auto node_4789 = ((node_4785)
                * (((((challenges[10])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[55]) * (current_main_row[22]))))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[56]) * (current_main_row[23]))))
                    + (node_4742))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[58]) * (next_main_row[22])))))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4746 = (challenges[10])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[55]) * (current_main_row[22])));
            const auto node_4839 = ((next_main_row[48])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[48])))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4838 = (next_main_row[48])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[48]));
            const auto node_4845 = (next_main_row[47])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_4871 = (next_aux_row[15])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[15]));
            const auto node_4908 = (next_main_row[51])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_4991 = (next_aux_row[21])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[21]));
            const auto node_5020 = ((next_main_row[59])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[59])))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_5019 = (next_main_row[59])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[59]));
            const auto node_5027 = ((node_5020)
                * ((current_main_row[58])
                    + (Bfe::from_raw_u64(18446744000695107601ULL))))
                * ((current_main_row[58])
                    + (Bfe::from_raw_u64(18446743931975630881ULL)));
            const auto node_5930 = (current_main_row[62])
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_5994 = (next_main_row[64])
                + (Bfe::from_raw_u64(18446744047939747846ULL));
            const auto node_6045 = (next_main_row[63])
                + (Bfe::from_raw_u64(18446743992105173011ULL));
            const auto node_6047 = (next_main_row[63])
                + (Bfe::from_raw_u64(18446743923385696291ULL));
            const auto node_6164 = (next_aux_row[28])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[28]));
            const auto node_6185 = (next_aux_row[29])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[29]));
            const auto node_6202 = (next_aux_row[30])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[30]));
            const auto node_6219 = (next_aux_row[31])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[31]));
            const auto node_6236 = (next_aux_row[32])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[32]));
            const auto node_6253 = (next_aux_row[33])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[33]));
            const auto node_6270 = (next_aux_row[34])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[34]));
            const auto node_6287 = (next_aux_row[35])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[35]));
            const auto node_6304 = (next_aux_row[36])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[36]));
            const auto node_6321 = (next_aux_row[37])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[37]));
            const auto node_6338 = (next_aux_row[38])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[38]));
            const auto node_6355 = (next_aux_row[39])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[39]));
            const auto node_6372 = (next_aux_row[40])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[40]));
            const auto node_6389 = (next_aux_row[41])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[41]));
            const auto node_6406 = (next_aux_row[42])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[42]));
            const auto node_6423 = (next_aux_row[43])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[43]));
            const auto node_6449 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[129]));
            const auto node_6551 = (current_main_row[142])
                + (Bfe::from_raw_u64(18446743940565565471ULL));
            const auto node_6578 = (node_6557)
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6586 = (next_main_row[142])
                + (Bfe::from_raw_u64(18446744017874976781ULL));
            const auto node_5954 = (next_main_row[62])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_2600 = (current_main_row[40]) * (challenges[22]);
            const auto node_2607 = (current_main_row[41]) * (challenges[22]);
            const auto node_2614 = (current_main_row[42]) * (challenges[22]);
            const auto node_30 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[3]);
            const auto node_47 = (next_main_row[6])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_51 = (next_aux_row[0])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[0]));
            const auto node_31 = (Bfe::from_raw_u64(38654705655ULL)) + (node_30);
            const auto node_74 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (next_main_row[1]);
            const auto node_90 = (Bfe::from_raw_u64(38654705655ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[3]));
            const auto node_88 = (next_aux_row[2])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[2]));
            const auto node_918 = (challenges[32]) * (current_main_row[28]);
            const auto node_939 = (challenges[32]) * (current_main_row[29]);
            const auto node_960 = (challenges[32]) * (current_main_row[30]);
            const auto node_981 = (challenges[32]) * (current_main_row[31]);
            const auto node_1023 = (challenges[32]) * (current_main_row[33]);
            const auto node_1044 = (challenges[32]) * (current_main_row[34]);
            const auto node_1065 = (challenges[32]) * (current_main_row[35]);
            const auto node_1086 = (challenges[32]) * (current_main_row[36]);
            const auto node_1100 = (challenges[46]) * (current_main_row[35]);
            const auto node_1107 = (challenges[32]) * (current_main_row[37]);
            const auto node_231 = ((challenges[32]) * (current_main_row[23]))
                + ((challenges[33]) * (current_main_row[24]));
            const auto node_1130 = (challenges[34]) * (current_main_row[22]);
            const auto node_233 = (node_231)
                + ((challenges[34]) * (current_main_row[25]));
            const auto node_1149 = (challenges[35]) * (current_main_row[22]);
            const auto node_235 = (node_233)
                + ((challenges[35]) * (current_main_row[26]));
            const auto node_1167 = (challenges[36]) * (current_main_row[22]);
            const auto node_237 = (node_235)
                + ((challenges[36]) * (current_main_row[27]));
            const auto node_1184 = (challenges[37]) * (current_main_row[22]);
            const auto node_239 = (node_237)
                + ((challenges[37]) * (current_main_row[28]));
            const auto node_1200 = (challenges[38]) * (current_main_row[22]);
            const auto node_241 = (node_239)
                + ((challenges[38]) * (current_main_row[29]));
            const auto node_1215 = (challenges[39]) * (current_main_row[22]);
            const auto node_243 = (node_241)
                + ((challenges[39]) * (current_main_row[30]));
            const auto node_1229 = (challenges[40]) * (current_main_row[22]);
            const auto node_245 = (node_243)
                + ((challenges[40]) * (current_main_row[31]));
            const auto node_1242 = (challenges[41]) * (current_main_row[22]);
            const auto node_247 = (node_245)
                + ((challenges[41]) * (current_main_row[32]));
            const auto node_1254 = (challenges[42]) * (current_main_row[22]);
            const auto node_249 = (node_247)
                + ((challenges[42]) * (current_main_row[33]));
            const auto node_1265 = (challenges[43]) * (current_main_row[22]);
            const auto node_251 = (node_249)
                + ((challenges[43]) * (current_main_row[34]));
            const auto node_1275 = (challenges[44]) * (current_main_row[22]);
            const auto node_253 = (node_251)
                + ((challenges[44]) * (current_main_row[35]));
            const auto node_1284 = (challenges[45]) * (current_main_row[22]);
            const auto node_255 = (node_253)
                + ((challenges[45]) * (current_main_row[36]));
            const auto node_1292 = (challenges[46]) * (current_main_row[22]);
            const auto node_257 = (node_255)
                + ((challenges[46]) * (current_main_row[37]));
            const auto node_1299 = (challenges[47]) * (current_main_row[22]);
            const auto node_1622 = (next_main_row[10])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[10]));
            const auto node_1690 = (node_120) + (Bfe::from_raw_u64(4294967295ULL));
            const auto node_1692 = (next_main_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[21]));
            const auto node_2353 = (next_main_row[22])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((current_main_row[22]) * (current_main_row[23])));
            const auto node_2354 = (next_main_row[22]) * (current_main_row[22]);
            const auto node_2375 = (current_main_row[23]) * (next_main_row[23]);
            const auto node_2378 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (next_main_row[22]);
            const auto node_2413 = (current_main_row[22]) * (current_main_row[25]);
            const auto node_2424 = ((current_main_row[24]) * (current_main_row[26]))
                + ((current_main_row[23]) * (current_main_row[27]));
            const auto node_2438 = (current_main_row[24]) * (next_main_row[23]);
            const auto node_2441 = (current_main_row[23]) * (next_main_row[24]);
            const auto node_2012 = (node_1305)
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_1946 = (node_1305)
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_529 = (next_main_row[23]) + (node_528);
            const auto node_112 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[9]);
            const auto node_1691 = (next_main_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[20]));
            const auto node_1693 = (current_main_row[28]) + (node_538);
            const auto node_2356 = (current_main_row[23]) + (node_528);
            const auto node_2409 = (next_main_row[23]) + (node_530);
            const auto node_2567 = (((Bfe::from_raw_u64(8589934590ULL))
                * (next_main_row[27])) + (current_main_row[44])) + (node_538);
            const auto node_2651 = (node_2409)
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_292 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[40]);
            const auto node_144 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * ((Bfe::from_raw_u64(34359738360ULL))
                    * (current_main_row[42]));
            const auto node_2699 = ((current_main_row[41]) * (current_main_row[43]))
                + ((current_main_row[40]) * (current_main_row[44]));
            const auto node_2102 = (next_main_row[27]) + (node_548);
            const auto node_2709 = (next_main_row[25]) + (node_534);
            const auto node_2700 = (current_main_row[41]) * (current_main_row[44]);
            const auto node_2103 = (next_main_row[28]) + (node_550);
            const auto node_201 = ((((((challenges[32]) * (next_main_row[22]))
                + ((challenges[33]) * (next_main_row[23]))) + (node_191))
                + (node_194)) + (node_197)) + (node_200);
            const auto node_607 = ((((((challenges[32]) * (current_main_row[22]))
                + (node_598)) + (node_600)) + (node_602)) + (node_604)) + (node_606);
            const auto node_2712 = (next_main_row[26]) + (node_536);
            const auto node_2104 = (next_main_row[29]) + (node_552);
            const auto node_2105 = (next_main_row[30]) + (node_554);
            const auto node_2106 = (next_main_row[31]) + (node_556);
            const auto node_2107 = (next_main_row[32]) + (node_1349);
            const auto node_480 = (((((current_main_row[195])
                * (((((((((((node_201) + (node_203)) + (node_206)) + (node_209))
                    + (node_212)) + (node_215)) + (node_218)) + (node_221)) + (node_224))
                    + (node_227))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL)) * (node_257))))
                + ((current_main_row[196])
                    * ((((((((((node_201) + (node_203)) + (node_206)) + (node_209))
                        + (node_212)) + (node_215)) + (node_218)) + (node_221)) + (node_224))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((((((((((((((node_299)
                                + ((challenges[33]) * (current_main_row[25])))
                                + ((challenges[34]) * (current_main_row[26])))
                                + ((challenges[35]) * (current_main_row[27])))
                                + ((challenges[36]) * (current_main_row[28])))
                                + ((challenges[37]) * (current_main_row[29])))
                                + ((challenges[38]) * (current_main_row[30])))
                                + ((challenges[39]) * (current_main_row[31])))
                                + ((challenges[40]) * (current_main_row[32])))
                                + ((challenges[41]) * (current_main_row[33])))
                                + ((challenges[42]) * (current_main_row[34])))
                                + ((challenges[43]) * (current_main_row[35])))
                                + ((challenges[44]) * (current_main_row[36])))
                                + ((challenges[45]) * (current_main_row[37])))))))
                + ((current_main_row[198])
                    * (((((((((node_201) + (node_203)) + (node_206)) + (node_209))
                        + (node_212)) + (node_215)) + (node_218)) + (node_221))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((((((((((((node_346)
                                + ((challenges[33]) * (current_main_row[26])))
                                + ((challenges[34]) * (current_main_row[27])))
                                + ((challenges[35]) * (current_main_row[28])))
                                + ((challenges[36]) * (current_main_row[29])))
                                + ((challenges[37]) * (current_main_row[30])))
                                + ((challenges[38]) * (current_main_row[31])))
                                + ((challenges[39]) * (current_main_row[32])))
                                + ((challenges[40]) * (current_main_row[33])))
                                + ((challenges[41]) * (current_main_row[34])))
                                + ((challenges[42]) * (current_main_row[35])))
                                + ((challenges[43]) * (current_main_row[36])))
                                + ((challenges[44]) * (current_main_row[37])))))))
                + ((current_main_row[200])
                    * ((((((((node_201) + (node_203)) + (node_206)) + (node_209))
                        + (node_212)) + (node_215)) + (node_218))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((((((((((((node_390)
                                + ((challenges[33]) * (current_main_row[27])))
                                + ((challenges[34]) * (current_main_row[28])))
                                + ((challenges[35]) * (current_main_row[29])))
                                + ((challenges[36]) * (current_main_row[30])))
                                + ((challenges[37]) * (current_main_row[31])))
                                + ((challenges[38]) * (current_main_row[32])))
                                + ((challenges[39]) * (current_main_row[33])))
                                + ((challenges[40]) * (current_main_row[34])))
                                + ((challenges[41]) * (current_main_row[35])))
                                + ((challenges[42]) * (current_main_row[36])))
                                + ((challenges[43]) * (current_main_row[37])))))))
                + ((current_main_row[202])
                    * (((((((node_201) + (node_203)) + (node_206)) + (node_209))
                        + (node_212)) + (node_215))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((((((((((node_433)
                                + ((challenges[33]) * (current_main_row[28])))
                                + ((challenges[34]) * (current_main_row[29])))
                                + ((challenges[35]) * (current_main_row[30])))
                                + ((challenges[36]) * (current_main_row[31])))
                                + ((challenges[37]) * (current_main_row[32])))
                                + ((challenges[38]) * (current_main_row[33])))
                                + ((challenges[39]) * (current_main_row[34])))
                                + ((challenges[40]) * (current_main_row[35])))
                                + ((challenges[41]) * (current_main_row[36])))
                                + ((challenges[42]) * (current_main_row[37]))))));
            const auto node_805 = (((((current_main_row[195])
                * (((((((((((((((((challenges[32]) * (next_main_row[23]))
                    + ((challenges[33]) * (next_main_row[24])))
                    + ((challenges[34]) * (next_main_row[25])))
                    + ((challenges[35]) * (next_main_row[26])))
                    + ((challenges[36]) * (next_main_row[27])))
                    + ((challenges[37]) * (next_main_row[28])))
                    + ((challenges[38]) * (next_main_row[29])))
                    + ((challenges[39]) * (next_main_row[30])))
                    + ((challenges[40]) * (next_main_row[31])))
                    + ((challenges[41]) * (next_main_row[32])))
                    + ((challenges[42]) * (next_main_row[33])))
                    + ((challenges[43]) * (next_main_row[34])))
                    + ((challenges[44]) * (next_main_row[35])))
                    + ((challenges[45]) * (next_main_row[36])))
                    + ((challenges[46]) * (next_main_row[37])))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((((((((node_607) + (node_608)) + (node_610)) + (node_612))
                            + (node_614)) + (node_616)) + (node_618)) + (node_620))
                            + (node_622)) + (node_624)))))
                + ((current_main_row[196])
                    * ((((((((((((((((challenges[32]) * (next_main_row[24]))
                        + ((challenges[33]) * (next_main_row[25])))
                        + ((challenges[34]) * (next_main_row[26])))
                        + ((challenges[35]) * (next_main_row[27])))
                        + ((challenges[36]) * (next_main_row[28])))
                        + ((challenges[37]) * (next_main_row[29])))
                        + ((challenges[38]) * (next_main_row[30])))
                        + ((challenges[39]) * (next_main_row[31])))
                        + ((challenges[40]) * (next_main_row[32])))
                        + ((challenges[41]) * (next_main_row[33])))
                        + ((challenges[42]) * (next_main_row[34])))
                        + ((challenges[43]) * (next_main_row[35])))
                        + ((challenges[44]) * (next_main_row[36])))
                        + ((challenges[45]) * (next_main_row[37])))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((((((((node_607) + (node_608)) + (node_610)) + (node_612))
                                + (node_614)) + (node_616)) + (node_618)) + (node_620))
                                + (node_622))))))
                + ((current_main_row[198])
                    * (((((((((((((((challenges[32]) * (next_main_row[25]))
                        + ((challenges[33]) * (next_main_row[26])))
                        + ((challenges[34]) * (next_main_row[27])))
                        + ((challenges[35]) * (next_main_row[28])))
                        + ((challenges[36]) * (next_main_row[29])))
                        + ((challenges[37]) * (next_main_row[30])))
                        + ((challenges[38]) * (next_main_row[31])))
                        + ((challenges[39]) * (next_main_row[32])))
                        + ((challenges[40]) * (next_main_row[33])))
                        + ((challenges[41]) * (next_main_row[34])))
                        + ((challenges[42]) * (next_main_row[35])))
                        + ((challenges[43]) * (next_main_row[36])))
                        + ((challenges[44]) * (next_main_row[37])))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((((((((node_607) + (node_608)) + (node_610)) + (node_612))
                                + (node_614)) + (node_616)) + (node_618)) + (node_620))))))
                + ((current_main_row[200])
                    * ((((((((((((((challenges[32]) * (next_main_row[26]))
                        + ((challenges[33]) * (next_main_row[27])))
                        + ((challenges[34]) * (next_main_row[28])))
                        + ((challenges[35]) * (next_main_row[29])))
                        + ((challenges[36]) * (next_main_row[30])))
                        + ((challenges[37]) * (next_main_row[31])))
                        + ((challenges[38]) * (next_main_row[32])))
                        + ((challenges[39]) * (next_main_row[33])))
                        + ((challenges[40]) * (next_main_row[34])))
                        + ((challenges[41]) * (next_main_row[35])))
                        + ((challenges[42]) * (next_main_row[36])))
                        + ((challenges[43]) * (next_main_row[37])))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (((((((node_607) + (node_608)) + (node_610)) + (node_612))
                                + (node_614)) + (node_616)) + (node_618))))))
                + ((current_main_row[202])
                    * (((((((((((((challenges[32]) * (next_main_row[27]))
                        + ((challenges[33]) * (next_main_row[28])))
                        + ((challenges[34]) * (next_main_row[29])))
                        + ((challenges[35]) * (next_main_row[30])))
                        + ((challenges[36]) * (next_main_row[31])))
                        + ((challenges[37]) * (next_main_row[32])))
                        + ((challenges[38]) * (next_main_row[33])))
                        + ((challenges[39]) * (next_main_row[34])))
                        + ((challenges[40]) * (next_main_row[35])))
                        + ((challenges[41]) * (next_main_row[36])))
                        + ((challenges[42]) * (next_main_row[37])))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((((((node_607) + (node_608)) + (node_610)) + (node_612))
                                + (node_614)) + (node_616)))));
            const auto node_457 = ((((((node_201) + (node_203)) + (node_206)) + (node_209))
                + (node_212)) + (node_215))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((((((((((node_433)
                        + ((challenges[33]) * (current_main_row[28])))
                        + ((challenges[34]) * (current_main_row[29])))
                        + ((challenges[35]) * (current_main_row[30])))
                        + ((challenges[36]) * (current_main_row[31])))
                        + ((challenges[37]) * (current_main_row[32])))
                        + ((challenges[38]) * (current_main_row[33])))
                        + ((challenges[39]) * (current_main_row[34])))
                        + ((challenges[40]) * (current_main_row[35])))
                        + ((challenges[41]) * (current_main_row[36])))
                        + ((challenges[42]) * (current_main_row[37]))));
            const auto node_2537 = ((challenges[2]) * (current_aux_row[4]))
                + (current_main_row[22]);
            const auto node_2542 = ((challenges[2]) * (node_2537))
                + (current_main_row[23]);
            const auto node_2547 = ((challenges[2]) * (node_2542))
                + (current_main_row[24]);
            const auto node_2552 = ((challenges[2]) * (node_2547))
                + (current_main_row[25]);
            const auto node_4505 = (next_aux_row[5])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[5]));
            const auto node_4641 = (next_aux_row[9])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[4]) * (current_aux_row[9])));
            const auto node_4642 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (((((node_201) + (node_203)) + (node_206)) + (node_209)) + (node_212));
            const auto node_4647 = (node_4641)
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((((((((((challenges[32])
                        * ((current_main_row[336]) + (current_main_row[337])))
                        + ((challenges[33])
                            * ((current_main_row[338]) + (current_main_row[339]))))
                        + ((challenges[34])
                            * ((current_main_row[340]) + (current_main_row[341]))))
                        + ((challenges[35])
                            * ((current_main_row[342]) + (current_main_row[343]))))
                        + ((challenges[36])
                            * ((current_main_row[344]) + (current_main_row[345]))))
                        + ((challenges[37])
                            * ((current_main_row[346]) + (current_main_row[347]))))
                        + ((challenges[38])
                            * ((current_main_row[348]) + (current_main_row[349]))))
                        + ((challenges[39])
                            * ((current_main_row[350]) + (current_main_row[351]))))
                        + ((challenges[40])
                            * ((current_main_row[352]) + (current_main_row[353]))))
                        + ((challenges[41])
                            * ((current_main_row[354])
                                + (current_main_row[355])))));
            const auto node_198 = (((((challenges[32]) * (next_main_row[22]))
                + ((challenges[33]) * (next_main_row[23]))) + (node_191))
                + (node_194)) + (node_197);
            const auto node_615 = ((((node_607) + (node_608)) + (node_610)) + (node_612))
                + (node_614);
            const auto node_4692 = (next_aux_row[11])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[6]) * (current_aux_row[11])));
            const auto node_574 = ((((challenges[32]) * (next_main_row[23]))
                + ((challenges[33]) * (next_main_row[24])))
                + ((challenges[34]) * (next_main_row[25])))
                + ((challenges[35]) * (next_main_row[26]));
            const auto node_4735 = (challenges[10])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[55]) * (next_main_row[22])));
            const auto node_4738 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * ((challenges[56]) * (next_main_row[23]));
            const auto node_4749 = (node_4746)
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[56]) * (current_main_row[23])));
            const auto node_4793 = ((node_4785)
                * (((node_4746) + (node_4742))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[58]) * (next_main_row[22])))))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4772 = (((node_4735)
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[56]) * (current_main_row[23]))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[57])
                        * (Bfe::from_raw_u64(25769803770ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (challenges[58]));
            const auto node_4776 = ((node_4746) + (node_4738))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((challenges[57])
                        * (Bfe::from_raw_u64(17179869180ULL))));
            const auto node_4795 = ((node_4785)
                * ((((challenges[10])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[55]) * (current_main_row[27]))))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[56]) * (next_main_row[27]))))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((challenges[57])
                            * (Bfe::from_raw_u64(17179869180ULL))))))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4862 = (next_main_row[47])
                * ((next_main_row[47])
                    + (Bfe::from_raw_u64(18446744065119617026ULL)));
            const auto node_4931 = (challenges[12])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (next_main_row[52]));
            const auto node_4936 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_aux_row[16]);
            const auto node_4982 = ((next_main_row[51])
                + (Bfe::from_raw_u64(18446744065119617026ULL)))
                * (next_main_row[51]);
            const auto node_5056 = (next_aux_row[23])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[23]));
            const auto node_5035 = (next_main_row[57])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[57]));
            const auto node_5912 = (current_main_row[63])
                + (Bfe::from_raw_u64(18446743897615892521ULL));
            const auto node_5914 = (current_main_row[64])
                + (Bfe::from_raw_u64(18446744047939747846ULL));
            const auto node_6145 = (next_aux_row[24])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[24]));
            const auto node_5093 = (((((current_main_row[65])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[66])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[67])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[68])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5104 = (((((current_main_row[69])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[70])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[71])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[72])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5115 = (((((current_main_row[73])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[74])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[75])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[76])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5126 = (((((current_main_row[77])
                * (Bfe::from_raw_u64(18446744069414518785ULL)))
                + ((current_main_row[78])
                    * (Bfe::from_raw_u64(18446744069414584320ULL))))
                + ((current_main_row[79])
                    * (Bfe::from_raw_u64(281474976645120ULL))))
                + (current_main_row[80])) * (Bfe::from_raw_u64(1ULL));
            const auto node_5944 = (node_5914) * (node_5912);
            const auto node_5958 = ((current_main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL)))
                * ((current_main_row[62])
                    + (Bfe::from_raw_u64(18446744060824649731ULL)));
            const auto node_5976 = (challenges[42])
                * ((next_main_row[103])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[103])));
            const auto node_5977 = (challenges[43])
                * ((next_main_row[104])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[104])));
            const auto node_5979 = (challenges[44])
                * ((next_main_row[105])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[105])));
            const auto node_5981 = (challenges[45])
                * ((next_main_row[106])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[106])));
            const auto node_5983 = (challenges[46])
                * ((next_main_row[107])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[107])));
            const auto node_5985 = (challenges[47])
                * ((next_main_row[108])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[108])));
            const auto node_6075 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (((((((((((challenges[32]) * (node_5809))
                    + ((challenges[33]) * (node_5820)))
                    + ((challenges[34]) * (node_5831)))
                    + ((challenges[35]) * (node_5842)))
                    + ((challenges[36]) * (next_main_row[97])))
                    + ((challenges[37]) * (next_main_row[98])))
                    + ((challenges[38]) * (next_main_row[99])))
                    + ((challenges[39]) * (next_main_row[100])))
                    + ((challenges[40]) * (next_main_row[101])))
                    + ((challenges[41]) * (next_main_row[102])));
            const auto node_6052 = (next_aux_row[25])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[25]));
            const auto node_6061 = (((((challenges[32]) * (node_5809))
                + ((challenges[33]) * (node_5820)))
                + ((challenges[34]) * (node_5831)))
                + ((challenges[35]) * (node_5842)))
                + ((challenges[36]) * (next_main_row[97]));
            const auto node_6086 = (next_aux_row[26])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[26]));
            const auto node_6112 = (next_aux_row[27])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[27]));
            const auto node_6115 = (next_main_row[63])
                + (Bfe::from_raw_u64(18446743828896415801ULL));
            const auto node_6459 = (next_aux_row[44])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[44]));
            const auto node_6475 = (next_aux_row[45])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[45]));
            const auto node_6470 = ((challenges[52]) * (next_main_row[131]))
                + ((challenges[53]) * (next_main_row[133]));
            const auto node_6473 = ((challenges[52]) * (next_main_row[130]))
                + ((challenges[53]) * (next_main_row[132]));
            const auto node_6513 = (next_aux_row[46])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[46]));
            const auto node_6569 = ((next_main_row[140])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[140])))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6575 = (node_6554)
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6595 = (next_main_row[147])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6597 = (next_main_row[147])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_6600 = (current_main_row[323]) * (next_main_row[147]);
            const auto node_6602 = (current_main_row[147])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6567 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (current_main_row[140]);
            const auto node_6661 = (next_main_row[147]) * (next_main_row[147]);
            const auto node_6611 = (Bfe::from_raw_u64(18446744065119617026ULL))
                * (node_6554);
            const auto node_6677 = (next_aux_row[48])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_aux_row[48]));
            const auto node_2775 = (current_main_row[12])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_2754 = (current_main_row[13])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_288 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[42]));
            const auto node_290 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (current_main_row[41]));
            const auto node_5987 = (next_main_row[64])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_5988 = (next_main_row[64])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_5990 = (next_main_row[64])
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_133 = (current_main_row[40])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_6580 = (next_main_row[142])
                + (Bfe::from_raw_u64(18446744052234715141ULL));
            const auto node_6582 = (next_main_row[142])
                + (Bfe::from_raw_u64(18446744009285042191ULL));
            const auto node_5992 = (next_main_row[64])
                + (Bfe::from_raw_u64(18446744052234715141ULL));
            const auto node_4542 = (next_main_row[12])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4546 = (next_main_row[15])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_4557 = (next_main_row[16])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_5929 = (current_main_row[62])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_5951 = (current_main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_281 = (challenges[7])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((node_272) + ((challenges[18]) * (next_main_row[38])))
                        + ((challenges[19]) * (next_main_row[37]))));
            const auto node_564 = (challenges[7])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((node_272) + ((challenges[18]) * (current_main_row[38])))
                        + ((challenges[19]) * (current_main_row[37]))));
            const auto node_1724 = (challenges[8])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((node_1715)
                        + (((next_main_row[22])
                            + (Bfe::from_raw_u64(4294967295ULL)))
                            * (challenges[21])))
                        + ((next_main_row[23]) * (challenges[22]))));
            const auto node_1952 = (challenges[8])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((node_1713) + ((current_main_row[22]) * (challenges[21])))
                        + ((current_main_row[23]) * (challenges[22]))));
            const auto node_1974 = ((current_main_row[22])
                + (Bfe::from_raw_u64(4294967295ULL))) * (challenges[21]);
            const auto node_2014 = ((current_main_row[22])
                + (Bfe::from_raw_u64(8589934590ULL))) * (challenges[21]);
            const auto node_2594 = (current_main_row[39]) * (challenges[22]);
            const auto node_2657 = (challenges[8])
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((node_1715) + ((current_main_row[22]) * (challenges[21])))
                        + (node_2594)));
            const auto node_2669 = (node_1715)
                + ((current_main_row[23]) * (challenges[21]));
            const auto node_2675 = (node_1715)
                + (((current_main_row[23])
                    + (Bfe::from_raw_u64(4294967295ULL))) * (challenges[21]));
            const auto node_2681 = (node_1715)
                + (((current_main_row[23])
                    + (Bfe::from_raw_u64(8589934590ULL))) * (challenges[21]));
            const auto node_2621 = (current_main_row[43]) * (challenges[22]);
    Xfe acc = Xfe::zero();
    acc = acc + (((((((((((((((((((current_main_row[210]) * (node_567))
                    + ((current_main_row[211]) * (node_543)))
                    + ((current_main_row[218]) * (node_1675)))
                    + ((current_main_row[226]) * (node_1682)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_539))
                            + ((current_main_row[196])
                                * ((next_main_row[29]) + (node_538))))
                            + ((current_main_row[198])
                                * ((next_main_row[30]) + (node_538))))
                            + ((current_main_row[200])
                                * ((next_main_row[31]) + (node_538))))
                            + ((current_main_row[202])
                                * ((next_main_row[32]) + (node_538))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1675))
                            + ((current_main_row[196])
                                * ((next_main_row[27]) + (node_542))))
                            + ((current_main_row[198]) * (node_2028)))
                            + ((current_main_row[200])
                                * ((next_main_row[27]) + (node_546))))
                            + ((current_main_row[202]) * (node_2102)))))
                    + ((current_main_row[237]) * (node_1683)))
                    + ((current_main_row[238]) * (node_1683)))
                    + ((current_main_row[244]) * (node_1681)))
                    + ((current_main_row[245]) * (node_567)))
                    + ((current_main_row[242]) * (node_1684)))
                    + ((current_main_row[246]) * (node_1684)))
                    + ((current_main_row[248]) * (node_1684)))
                    + ((current_main_row[253]) * (node_1684)))
                    + ((current_main_row[272]) * (node_128)))
                    + ((current_main_row[275]) * (node_128)))
                    + ((current_main_row[296]) * (node_120))) * (node_4429)) * weights[475];
    acc = acc + (((((((((((((((((((current_main_row[210]) * (node_124))
                    + ((current_main_row[211]) * (node_547)))
                    + ((current_main_row[218]) * (node_1677)))
                    + ((current_main_row[226]) * (node_1684)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_543))
                            + ((current_main_row[196])
                                * ((next_main_row[31]) + (node_542))))
                            + ((current_main_row[198])
                                * ((next_main_row[32]) + (node_542))))
                            + ((current_main_row[200])
                                * ((next_main_row[33]) + (node_542))))
                            + ((current_main_row[202])
                                * ((next_main_row[34]) + (node_542))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1677))
                            + ((current_main_row[196])
                                * ((next_main_row[29]) + (node_546))))
                            + ((current_main_row[198]) * (node_2030)))
                            + ((current_main_row[200])
                                * ((next_main_row[29]) + (node_550))))
                            + ((current_main_row[202]) * (node_2104)))))
                    + ((current_main_row[237]) * (node_262)))
                    + ((current_main_row[238]) * (node_262)))
                    + ((current_main_row[244]) * (node_1683)))
                    + ((current_main_row[245]) * (node_124)))
                    + ((current_main_row[242]) * (node_286)))
                    + ((current_main_row[246]) * (node_286)))
                    + ((current_main_row[248]) * (node_286)))
                    + ((current_main_row[253]) * (node_286)))
                    + ((current_main_row[272]) * (node_516)))
                    + ((current_main_row[275]) * (node_516)))
                    + ((current_main_row[296]) * (node_128))) * (node_4429)) * weights[476];
    acc = acc + (((((((((((((((((((current_main_row[210]) * (node_128))
                    + ((current_main_row[211]) * (node_549)))
                    + ((current_main_row[218]) * (node_1678)))
                    + ((current_main_row[226]) * (node_262)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_545))
                            + ((current_main_row[196])
                                * ((next_main_row[32]) + (node_544))))
                            + ((current_main_row[198])
                                * ((next_main_row[33]) + (node_544))))
                            + ((current_main_row[200])
                                * ((next_main_row[34]) + (node_544))))
                            + ((current_main_row[202])
                                * ((next_main_row[35]) + (node_544))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1678))
                            + ((current_main_row[196])
                                * ((next_main_row[30]) + (node_548))))
                            + ((current_main_row[198]) * (node_2031)))
                            + ((current_main_row[200])
                                * ((next_main_row[30]) + (node_552))))
                            + ((current_main_row[202]) * (node_2105)))))
                    + ((current_main_row[237]) * (node_286)))
                    + ((current_main_row[238]) * (node_286)))
                    + ((current_main_row[244]) * (node_1684)))
                    + ((current_main_row[245]) * (node_128)))
                    + ((current_main_row[242]) * (node_516)))
                    + ((current_main_row[246]) * (node_516)))
                    + ((current_main_row[248]) * (node_516)))
                    + ((current_main_row[253]) * (node_516)))
                    + ((current_main_row[272]) * (node_520)))
                    + ((current_main_row[275]) * (node_520)))
                    + ((current_main_row[296]) * (node_1623))) * (node_4429)) * weights[477];
    acc = acc + (((((((((((((((((((current_main_row[210]) * (node_116))
                    + ((current_main_row[211]) * (node_551)))
                    + ((current_main_row[218]) * (node_1679)))
                    + ((current_main_row[226]) * (node_286)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_547))
                            + ((current_main_row[196])
                                * ((next_main_row[33]) + (node_546))))
                            + ((current_main_row[198])
                                * ((next_main_row[34]) + (node_546))))
                            + ((current_main_row[200])
                                * ((next_main_row[35]) + (node_546))))
                            + ((current_main_row[202])
                                * ((next_main_row[36]) + (node_546))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1679))
                            + ((current_main_row[196])
                                * ((next_main_row[31]) + (node_550))))
                            + ((current_main_row[198]) * (node_2032)))
                            + ((current_main_row[200])
                                * ((next_main_row[31]) + (node_554))))
                            + ((current_main_row[202]) * (node_2106)))))
                    + ((current_main_row[237]) * (node_516)))
                    + ((current_main_row[238]) * (node_516)))
                    + ((current_main_row[244]) * (node_262)))
                    + ((current_main_row[245]) * (node_1623)))
                    + ((current_main_row[242]) * (node_520)))
                    + ((current_main_row[246]) * (node_520)))
                    + ((current_main_row[248]) * (node_520)))
                    + ((current_main_row[253]) * (node_520)))
                    + ((current_main_row[272]) * (node_524)))
                    + ((current_main_row[275]) * (node_524)))
                    + ((current_main_row[296]) * (node_516))) * (node_4429)) * weights[478];
    acc = acc + (((((((((((((((((current_main_row[210]) * (node_516))
                    + ((current_main_row[211]) * (node_553)))
                    + ((current_main_row[218]) * (node_1680)))
                    + ((current_main_row[226]) * (node_516)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_549))
                            + ((current_main_row[196])
                                * ((next_main_row[34]) + (node_548))))
                            + ((current_main_row[198])
                                * ((next_main_row[35]) + (node_548))))
                            + ((current_main_row[200])
                                * ((next_main_row[36]) + (node_548))))
                            + ((current_main_row[202])
                                * ((next_main_row[37]) + (node_548))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1680))
                            + ((current_main_row[196])
                                * ((next_main_row[32]) + (node_552))))
                            + ((current_main_row[198]) * (node_2033)))
                            + ((current_main_row[200])
                                * ((next_main_row[32]) + (node_556))))
                            + ((current_main_row[202]) * (node_2107)))))
                    + ((current_main_row[237]) * (node_520)))
                    + ((current_main_row[238]) * (node_520)))
                    + ((current_main_row[244]) * (node_286)))
                    + ((current_main_row[245]) * (node_516)))
                    + ((current_main_row[242]) * (node_524)))
                    + ((current_main_row[246]) * (node_524)))
                    + ((current_main_row[248]) * (node_524)))
                    + ((current_main_row[253]) * (node_524)))
                    + ((current_main_row[296]) * (node_520))) * (node_4429)) * weights[479];
    acc = acc + (((((((((((((current_main_row[210]) * (node_520))
                    + ((current_main_row[211]) * (node_555)))
                    + ((current_main_row[218]) * (node_1681)))
                    + ((current_main_row[226]) * (node_520)))
                    + ((current_main_row[220])
                        * (((((current_main_row[195]) * (node_551))
                            + ((current_main_row[196])
                                * ((next_main_row[35]) + (node_550))))
                            + ((current_main_row[198])
                                * ((next_main_row[36]) + (node_550))))
                            + ((current_main_row[200])
                                * ((next_main_row[37]) + (node_550))))))
                    + ((current_main_row[228])
                        * (((((current_main_row[195]) * (node_1681))
                            + ((current_main_row[196])
                                * ((next_main_row[33]) + (node_554))))
                            + ((current_main_row[198]) * (node_2034)))
                            + ((current_main_row[200])
                                * ((next_main_row[33]) + (node_1349))))))
                    + ((current_main_row[237]) * (node_524)))
                    + ((current_main_row[238]) * (node_524)))
                    + ((current_main_row[244]) * (node_516)))
                    + ((current_main_row[245]) * (node_520)))
                    + ((current_main_row[296]) * (node_524))) * (node_4429)) * weights[480];
    acc = acc + ((((((((((current_main_row[210]) * (node_524))
                    + ((current_main_row[211]) * (node_557)))
                    + ((current_main_row[218]) * (node_1682)))
                    + ((current_main_row[226]) * (node_524)))
                    + ((current_main_row[220])
                        * ((((current_main_row[195]) * (node_553))
                            + ((current_main_row[196])
                                * ((next_main_row[36]) + (node_552))))
                            + ((current_main_row[198])
                                * ((next_main_row[37]) + (node_552))))))
                    + ((current_main_row[228])
                        * ((((current_main_row[195]) * (node_1682))
                            + ((current_main_row[196])
                                * ((next_main_row[34]) + (node_556))))
                            + ((current_main_row[198]) * (node_2035)))))
                    + ((current_main_row[244]) * (node_520)))
                    + ((current_main_row[245]) * (node_524))) * (node_4429)) * weights[481];
    acc = acc + (((((((current_main_row[211]) * (node_558))
                    + ((current_main_row[218]) * (node_1683)))
                    + ((current_main_row[220])
                        * (((current_main_row[195]) * (node_555))
                            + ((current_main_row[196])
                                * ((next_main_row[37]) + (node_554))))))
                    + ((current_main_row[228])
                        * (((current_main_row[195]) * (node_1683))
                            + ((current_main_row[196])
                                * ((next_main_row[35]) + (node_1349))))))
                    + ((current_main_row[244]) * (node_524))) * (node_4429)) * weights[482];
    acc = acc + ((((((current_main_row[211]) * (node_567))
                    + ((current_main_row[218]) * (node_1684)))
                    + ((current_main_row[220])
                        * ((current_main_row[195]) * (node_557))))
                    + ((current_main_row[228])
                        * ((current_main_row[195]) * (node_1684)))) * (node_4429)) * weights[483];
    acc = acc + ((((((current_main_row[211]) * (node_516))
                    + ((current_main_row[218]) * (node_262)))
                    + ((current_main_row[220]) * (node_512)))
                    + ((current_main_row[228]) * (node_512))) * (node_4429)) * weights[484];
    acc = acc + ((((((current_main_row[211]) * (node_520))
                    + ((current_main_row[218]) * (node_286)))
                    + ((current_main_row[220]) * (node_520)))
                    + ((current_main_row[228]) * (node_520))) * (node_4429)) * weights[485];
    acc = acc + ((((((current_main_row[211]) * (node_524))
                    + ((current_main_row[218]) * (node_516)))
                    + ((current_main_row[220]) * (node_524)))
                    + ((current_main_row[228]) * (node_524))) * (node_4429)) * weights[486];
    acc = acc + (((current_main_row[218]) * (node_520)) * (node_4429)) * weights[487];
    acc = acc + (((current_main_row[218]) * (node_524)) * (node_4429)) * weights[488];
    acc = acc + ((((next_aux_row[13])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_aux_row[13])))
                    * ((challenges[11])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[7]))))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (next_main_row[45]))) * weights[489];
    acc = acc + (((node_4429)
                    * (((node_4505)
                        * ((challenges[3])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * ((((challenges[13]) * (next_main_row[9]))
                                    + ((challenges[14]) * (next_main_row[10])))
                                    + ((challenges[15]) * (next_main_row[11]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((next_main_row[8]) * (node_4505))) * weights[490];
    acc = acc + ((next_aux_row[8])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[8])
                            * ((challenges[9])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((((((challenges[24]) * (next_main_row[7]))
                                        + ((challenges[25]) * (next_main_row[10])))
                                        + ((challenges[26]) * (next_main_row[19])))
                                        + ((challenges[27]) * (next_main_row[20])))
                                        + ((challenges[28]) * (next_main_row[21])))))))) * weights[491];
    acc = acc + ((((((((next_main_row[10])
                    + (Bfe::from_raw_u64(18446743992105173011ULL)))
                    * ((next_main_row[10])
                        + (Bfe::from_raw_u64(18446743914795761701ULL))))
                    * ((next_main_row[10])
                        + (Bfe::from_raw_u64(18446743880436023341ULL))))
                    * ((next_aux_row[9])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[9]))))
                    + ((current_main_row[356]) * ((node_4641) + (node_4642))))
                    + (((current_main_row[357]) * (node_4551)) * (node_4647)))
                    + (((current_main_row[359]) * (node_4551)) * (node_4647))) * weights[492];
    acc = acc + ((((((current_main_row[10])
                    + (Bfe::from_raw_u64(18446743992105173011ULL)))
                    * ((current_main_row[10])
                        + (Bfe::from_raw_u64(18446743914795761701ULL))))
                    * ((current_main_row[10])
                        + (Bfe::from_raw_u64(18446743880436023341ULL))))
                    * ((next_aux_row[10])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[10]))))
                    + ((((current_main_row[239]) + (current_main_row[302]))
                        + (current_main_row[301]))
                        * (((next_aux_row[10])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * ((challenges[5]) * (current_aux_row[10]))))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_198))))) * weights[493];
    acc = acc + ((((((current_main_row[334])
                    * ((next_aux_row[11])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[11]))))
                    + ((current_main_row[261]) * (node_4696)))
                    + ((current_main_row[262])
                        * ((node_4696)
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_615)))))
                    + ((current_main_row[264])
                        * (((node_4692)
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * ((challenges[31])
                                    * (Bfe::from_raw_u64(146028888030ULL)))))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((((((node_574)
                                    + ((challenges[36]) * (current_main_row[39])))
                                    + ((challenges[37]) * (current_main_row[40])))
                                    + ((challenges[38]) * (current_main_row[41])))
                                    + ((challenges[39]) * (current_main_row[42])))
                                    + ((challenges[40]) * (current_main_row[43])))
                                    + ((challenges[41]) * (current_main_row[44])))))))
                    + ((current_main_row[266]) * ((node_4696) + (node_4642)))) * weights[494];
    acc = acc + ((((((((((((current_main_row[245])
                    * (((node_4785) * (((node_4735) + (node_4738)) + (node_4742)))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((current_main_row[242]) * (node_4789)))
                    + ((current_main_row[246]) * (node_4789)))
                    + ((current_main_row[248])
                        * (((node_4785)
                            * (((node_4749)
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((challenges[57])
                                        * (Bfe::from_raw_u64(60129542130ULL)))))
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((challenges[58])
                                        * (((current_main_row[22])
                                            + (current_main_row[23])) + (node_2378)))
                                        * (Bfe::from_raw_u64(9223372036854775808ULL))))))
                            + (Bfe::from_raw_u64(18446744065119617026ULL)))))
                    + ((current_main_row[253]) * (node_4789)))
                    + ((current_main_row[273]) * (node_4793)))
                    + ((current_main_row[270])
                        * (((((node_4785) * (node_4772)) * (node_4776))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_4772)))
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_4776)))))
                    + ((current_main_row[277]) * (node_4793)))
                    + ((current_main_row[302]) * (node_4795)))
                    + ((current_main_row[301]) * (node_4795)))
                    + (((Bfe::from_raw_u64(4294967295ULL))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[14]))) * (node_4785))) * weights[495];
    acc = acc + ((((next_aux_row[14])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[14])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((((challenges[16]) * (next_main_row[46]))
                                        + ((challenges[17]) * (next_main_row[47])))
                                        + ((challenges[18]) * (next_main_row[48])))
                                        + ((challenges[19]) * (next_main_row[49]))))))))
                    * (node_4845))
                    + (((next_aux_row[14])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[14]))) * (node_4862))) * weights[496];
    acc = acc + (((((((node_4871)
                    * ((challenges[11])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((next_main_row[46])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[46]))))))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * (node_4839))
                    * (node_4845)) + ((node_4871) * (node_4838)))
                    + ((node_4871) * (node_4862))) * weights[497];
    acc = acc + (((node_4913)
                    * ((next_aux_row[16])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((current_aux_row[16]) * (node_4931)))))
                    + ((node_4916) * ((next_aux_row[16]) + (node_4936)))) * weights[498];
    acc = acc + (((node_4913)
                    * (((next_aux_row[17]) + (node_4936))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((node_4931) * (current_aux_row[17])))))
                    + ((node_4916)
                        * ((next_aux_row[17])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_aux_row[17]))))) * weights[499];
    acc = acc + (((node_4913)
                    * (((next_aux_row[18])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[12]) * (current_aux_row[18]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[55]))))
                    + ((node_4916)
                        * ((next_aux_row[18])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_aux_row[18]))))) * weights[500];
    acc = acc + (((node_4913)
                    * (((next_aux_row[19])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[12]) * (current_aux_row[19]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[56]))))
                    + ((node_4916)
                        * ((next_aux_row[19])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_aux_row[19]))))) * weights[501];
    acc = acc + ((((next_aux_row[20])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[20])
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((((next_main_row[50]) * (challenges[20]))
                                        + ((next_main_row[52]) * (challenges[21])))
                                        + ((next_main_row[53]) * (challenges[22])))
                                        + ((next_main_row[51]) * (challenges[23]))))))))
                    * (node_4908))
                    + (((next_aux_row[20])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_aux_row[20]))) * (node_4982))) * weights[502];
    acc = acc + ((((current_aux_row[83]) * (node_4908)) + ((node_4991) * (node_4913)))
                    + ((node_4991) * (node_4982))) * weights[503];
    acc = acc + ((next_aux_row[22])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[22])
                            * ((challenges[9])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((((((challenges[24]) * (next_main_row[57]))
                                        + ((challenges[25]) * (next_main_row[58])))
                                        + ((challenges[26]) * (next_main_row[59])))
                                        + ((challenges[27]) * (next_main_row[60])))
                                        + ((challenges[28]) * (next_main_row[61])))))))) * weights[504];
    acc = acc + (((node_5020)
                    * (((node_5056)
                        * ((challenges[11])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_5035))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_5019) * (node_5056))) * weights[505];
    acc = acc + ((((current_main_row[360])
                    * (((next_aux_row[24])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[30]) * (current_aux_row[24]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((((((((((((((((((((challenges[29]) + (node_5809))
                                * (challenges[29])) + (node_5820))
                                * (challenges[29])) + (node_5831))
                                * (challenges[29])) + (node_5842))
                                * (challenges[29])) + (next_main_row[97]))
                                * (challenges[29])) + (next_main_row[98]))
                                * (challenges[29])) + (next_main_row[99]))
                                * (challenges[29])) + (next_main_row[100]))
                                * (challenges[29])) + (next_main_row[101]))
                                * (challenges[29])) + (next_main_row[102])))))
                    + ((next_main_row[64]) * (node_6145)))
                    + ((node_5933) * (node_6145))) * weights[506];
    acc = acc + (((current_main_row[361]) * (node_5933))
                    * (((((((((((challenges[0]) + (node_5093)) * (challenges[0]))
                        + (node_5104)) * (challenges[0])) + (node_5115))
                        * (challenges[0])) + (node_5126)) * (challenges[0]))
                        + (current_main_row[97]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (challenges[62])))) * weights[507];
    acc = acc + ((current_main_row[375])
                    * ((((((node_5976) + (node_5977)) + (node_5979)) + (node_5981))
                        + (node_5983)) + (node_5985))) * weights[508];
    acc = acc + ((current_main_row[376])
                    * (((((((((((((((((challenges[32])
                        * ((node_5809)
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_5093))))
                        + ((challenges[33])
                            * ((node_5820)
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (node_5104)))))
                        + ((challenges[34])
                            * ((node_5831)
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (node_5115)))))
                        + ((challenges[35])
                            * ((node_5842)
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (node_5126)))))
                        + ((challenges[36])
                            * ((next_main_row[97])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[97])))))
                        + ((challenges[37])
                            * ((next_main_row[98])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[98])))))
                        + ((challenges[38])
                            * ((next_main_row[99])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[99])))))
                        + ((challenges[39])
                            * ((next_main_row[100])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[100])))))
                        + ((challenges[40])
                            * ((next_main_row[101])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[101])))))
                        + ((challenges[41])
                            * ((next_main_row[102])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (current_main_row[102]))))) + (node_5976))
                        + (node_5977)) + (node_5979)) + (node_5981)) + (node_5983))
                        + (node_5985))) * weights[509];
    acc = acc + (((((current_main_row[325]) * (current_main_row[330]))
                    * (((next_aux_row[25])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[4]) * (current_aux_row[25]))))
                        + (node_6075))) + ((next_main_row[64]) * (node_6052)))
                    + ((node_5938) * (node_6052))) * weights[510];
    acc = acc + (((((node_6095) * (current_main_row[330]))
                    * (((next_aux_row[26])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[5]) * (current_aux_row[26]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_6061)))) + ((node_5994) * (node_6086)))
                    + ((node_5938) * (node_6086))) * weights[511];
    acc = acc + (((((current_main_row[325]) * (node_6045))
                    * ((((next_aux_row[27])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[6]) * (current_aux_row[27]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((challenges[31]) * (next_main_row[63]))))
                        + (node_6075))) + ((next_main_row[64]) * (node_6112)))
                    + ((((node_5942) * (node_6047)) * (node_6115)) * (node_6112))) * weights[512];
    acc = acc + ((((current_main_row[300])
                    * (((node_6164)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[65]))
                                    + ((challenges[50]) * (next_main_row[81]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6164))) + ((node_6172) * (node_6164))) * weights[513];
    acc = acc + ((((current_main_row[300])
                    * (((node_6185)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[66]))
                                    + ((challenges[50]) * (next_main_row[82]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6185))) + ((node_6172) * (node_6185))) * weights[514];
    acc = acc + ((((current_main_row[300])
                    * (((node_6202)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[67]))
                                    + ((challenges[50]) * (next_main_row[83]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6202))) + ((node_6172) * (node_6202))) * weights[515];
    acc = acc + ((((current_main_row[300])
                    * (((node_6219)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[68]))
                                    + ((challenges[50]) * (next_main_row[84]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6219))) + ((node_6172) * (node_6219))) * weights[516];
    acc = acc + ((((current_main_row[300])
                    * (((node_6236)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[69]))
                                    + ((challenges[50]) * (next_main_row[85]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6236))) + ((node_6172) * (node_6236))) * weights[517];
    acc = acc + ((((current_main_row[300])
                    * (((node_6253)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[70]))
                                    + ((challenges[50]) * (next_main_row[86]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6253))) + ((node_6172) * (node_6253))) * weights[518];
    acc = acc + ((((current_main_row[300])
                    * (((node_6270)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[71]))
                                    + ((challenges[50]) * (next_main_row[87]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6270))) + ((node_6172) * (node_6270))) * weights[519];
    acc = acc + ((((current_main_row[300])
                    * (((node_6287)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[72]))
                                    + ((challenges[50]) * (next_main_row[88]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6287))) + ((node_6172) * (node_6287))) * weights[520];
    acc = acc + ((((current_main_row[300])
                    * (((node_6304)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[73]))
                                    + ((challenges[50]) * (next_main_row[89]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6304))) + ((node_6172) * (node_6304))) * weights[521];
    acc = acc + ((((current_main_row[300])
                    * (((node_6321)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[74]))
                                    + ((challenges[50]) * (next_main_row[90]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6321))) + ((node_6172) * (node_6321))) * weights[522];
    acc = acc + ((((current_main_row[300])
                    * (((node_6338)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[75]))
                                    + ((challenges[50]) * (next_main_row[91]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6338))) + ((node_6172) * (node_6338))) * weights[523];
    acc = acc + ((((current_main_row[300])
                    * (((node_6355)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[76]))
                                    + ((challenges[50]) * (next_main_row[92]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6355))) + ((node_6172) * (node_6355))) * weights[524];
    acc = acc + ((((current_main_row[300])
                    * (((node_6372)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[77]))
                                    + ((challenges[50]) * (next_main_row[93]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6372))) + ((node_6172) * (node_6372))) * weights[525];
    acc = acc + ((((current_main_row[300])
                    * (((node_6389)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[78]))
                                    + ((challenges[50]) * (next_main_row[94]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6389))) + ((node_6172) * (node_6389))) * weights[526];
    acc = acc + ((((current_main_row[300])
                    * (((node_6406)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[79]))
                                    + ((challenges[50]) * (next_main_row[95]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6406))) + ((node_6172) * (node_6406))) * weights[527];
    acc = acc + ((((current_main_row[300])
                    * (((node_6423)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49]) * (next_main_row[80]))
                                    + ((challenges[50]) * (next_main_row[96]))))))
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    + ((node_6095) * (node_6423))) + ((node_6172) * (node_6423))) * weights[528];
    acc = acc + (((node_6449)
                    * (((node_6459)
                        * ((challenges[48])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((challenges[49])
                                    * (((Bfe::from_raw_u64(1099511627520ULL))
                                        * (next_main_row[130])) + (next_main_row[131])))
                                    + ((challenges[50])
                                        * (((Bfe::from_raw_u64(1099511627520ULL))
                                            * (next_main_row[132]))
                                            + (next_main_row[133])))))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[134]))))
                    + ((next_main_row[129]) * (node_6459))) * weights[529];
    acc = acc + (((node_6449)
                    * ((((((node_6475)
                        * ((challenges[51])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_6470))))
                        * ((challenges[51])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (node_6473))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((Bfe::from_raw_u64(8589934590ULL))
                                * (challenges[51])))) + (node_6470)) + (node_6473)))
                    + ((next_main_row[129]) * (node_6475))) * weights[530];
    acc = acc + (((node_6501)
                    * (((node_6513)
                        * ((challenges[51])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((next_main_row[136]) * (challenges[52]))
                                    + ((next_main_row[137]) * (challenges[53]))))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[138]))))
                    + ((next_main_row[135]) * (node_6513))) * weights[531];
    acc = acc + (((node_6501)
                    * (((next_aux_row[47])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((current_aux_row[47]) * (challenges[54]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[137]))))
                    + ((next_main_row[135])
                        * ((next_aux_row[47])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_aux_row[47]))))) * weights[532];
    acc = acc + ((node_6561) * (node_6677)) * weights[533];
    acc = acc + ((next_main_row[139])
                    * (((node_6677)
                        * ((challenges[10])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((((challenges[57]) * (next_main_row[142]))
                                    + ((challenges[55]) * (next_main_row[143])))
                                    + ((challenges[56]) * (next_main_row[145])))
                                    + ((challenges[58]) * (next_main_row[147]))))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[148])))) * weights[534];
    acc = acc + ((current_aux_row[49])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_281)
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(4294967295ULL)))))
                                        + ((challenges[19]) * (next_main_row[36])))))))) * weights[535];
    acc = acc + ((current_aux_row[50])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_564)
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(4294967295ULL)))))
                                        + ((challenges[19])
                                            * (current_main_row[36])))))))) * weights[536];
    acc = acc + ((current_aux_row[51])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[49])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(8589934590ULL)))))
                                        + ((challenges[19]) * (next_main_row[35])))))))) * weights[537];
    acc = acc + ((current_aux_row[52])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[50])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(8589934590ULL)))))
                                        + ((challenges[19])
                                            * (current_main_row[35])))))))) * weights[538];
    acc = acc + ((current_aux_row[53])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[51])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(12884901885ULL)))))
                                        + ((challenges[19]) * (next_main_row[34])))))))) * weights[539];
    acc = acc + ((current_aux_row[54])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[52])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(12884901885ULL)))))
                                        + ((challenges[19])
                                            * (current_main_row[34])))))))) * weights[540];
    acc = acc + ((current_aux_row[55])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[53])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(17179869180ULL)))))
                                        + ((challenges[19]) * (next_main_row[33])))))))) * weights[541];
    acc = acc + ((current_aux_row[56])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_1724)
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((next_main_row[22])
                                            + (Bfe::from_raw_u64(8589934590ULL)))
                                            * (challenges[21])))
                                        + ((next_main_row[24]) * (challenges[22])))))))) * weights[542];
    acc = acc + ((current_aux_row[57])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_1952)
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1713) + (node_1974))
                                        + ((current_main_row[24])
                                            * (challenges[22])))))))) * weights[543];
    acc = acc + ((current_aux_row[58])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[6]) * (current_aux_row[55])))) * weights[544];
    acc = acc + ((current_aux_row[59])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[54])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(17179869180ULL)))))
                                        + ((challenges[19])
                                            * (current_main_row[33])))))))) * weights[545];
    acc = acc + ((current_aux_row[60])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[56])
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((next_main_row[22])
                                            + (Bfe::from_raw_u64(12884901885ULL)))
                                            * (challenges[21])))
                                        + ((next_main_row[25]) * (challenges[22])))))))) * weights[546];
    acc = acc + ((current_aux_row[61])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[57])
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1713) + (node_2014))
                                        + ((current_main_row[25])
                                            * (challenges[22])))))))) * weights[547];
    acc = acc + ((current_aux_row[62])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[60])
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((next_main_row[22])
                                            + (Bfe::from_raw_u64(17179869180ULL)))
                                            * (challenges[21])))
                                        + ((next_main_row[26]) * (challenges[22])))))))) * weights[548];
    acc = acc + ((current_aux_row[63])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[61])
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1713)
                                        + (((current_main_row[22])
                                            + (Bfe::from_raw_u64(12884901885ULL)))
                                            * (challenges[21])))
                                        + ((current_main_row[26])
                                            * (challenges[22])))))))) * weights[549];
    acc = acc + ((current_aux_row[64])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((current_aux_row[55])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(21474836475ULL)))))
                                        + ((challenges[19]) * (next_main_row[32]))))))
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(25769803770ULL)))))
                                        + ((challenges[19]) * (next_main_row[31]))))))
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((next_main_row[38])
                                                + (Bfe::from_raw_u64(30064771065ULL)))))
                                        + ((challenges[19]) * (next_main_row[30])))))))) * weights[550];
    acc = acc + ((current_aux_row[65])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((current_aux_row[59])
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(21474836475ULL)))))
                                        + ((challenges[19]) * (current_main_row[32]))))))
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(25769803770ULL)))))
                                        + ((challenges[19]) * (current_main_row[31]))))))
                            * ((challenges[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_272)
                                        + ((challenges[18])
                                            * ((current_main_row[38])
                                                + (Bfe::from_raw_u64(30064771065ULL)))))
                                        + ((challenges[19])
                                            * (current_main_row[30])))))))) * weights[551];
    acc = acc + ((current_aux_row[66])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[6])
                            * (((current_aux_row[64])
                                * ((challenges[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * (((node_272)
                                            + ((challenges[18])
                                                * ((next_main_row[38])
                                                    + (Bfe::from_raw_u64(34359738360ULL)))))
                                            + ((challenges[19]) * (next_main_row[29]))))))
                                * ((challenges[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * (((node_272)
                                            + ((challenges[18])
                                                * ((next_main_row[38])
                                                    + (Bfe::from_raw_u64(38654705655ULL)))))
                                            + ((challenges[19]) * (next_main_row[28]))))))))) * weights[552];
    acc = acc + ((current_aux_row[67])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[6])
                            * (((current_aux_row[65])
                                * ((challenges[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * (((node_272)
                                            + ((challenges[18])
                                                * ((current_main_row[38])
                                                    + (Bfe::from_raw_u64(34359738360ULL)))))
                                            + ((challenges[19]) * (current_main_row[29]))))))
                                * ((challenges[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * (((node_272)
                                            + ((challenges[18])
                                                * ((current_main_row[38])
                                                    + (Bfe::from_raw_u64(38654705655ULL)))))
                                            + ((challenges[19])
                                                * (current_main_row[28]))))))))) * weights[553];
    acc = acc + ((current_aux_row[68])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[202])
                            * ((next_aux_row[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[7])
                                        * ((current_aux_row[62])
                                            * ((challenges[8])
                                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                                    * (((node_1715)
                                                        + (((next_main_row[22])
                                                            + (Bfe::from_raw_u64(21474836475ULL)))
                                                            * (challenges[21])))
                                                        + ((next_main_row[27])
                                                            * (challenges[22])))))))))))) * weights[554];
    acc = acc + ((current_aux_row[69])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[202])
                            * ((next_aux_row[7])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[7])
                                        * ((current_aux_row[63])
                                            * ((challenges[8])
                                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                                    * (((node_1713)
                                                        + (((current_main_row[22])
                                                            + (Bfe::from_raw_u64(17179869180ULL)))
                                                            * (challenges[21])))
                                                        + ((current_main_row[27])
                                                            * (challenges[22])))))))))))) * weights[555];
    acc = acc + ((current_aux_row[70])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((((challenges[8])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (((node_1715)
                                    + ((current_main_row[29]) * (challenges[21])))
                                    + (node_2594))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((current_main_row[29])
                                            + (Bfe::from_raw_u64(4294967295ULL)))
                                            * (challenges[21]))) + (node_2600)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((current_main_row[29])
                                            + (Bfe::from_raw_u64(8589934590ULL)))
                                            * (challenges[21]))) + (node_2607)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715)
                                        + (((current_main_row[29])
                                            + (Bfe::from_raw_u64(12884901885ULL)))
                                            * (challenges[21]))) + (node_2614))))))) * weights[556];
    acc = acc + ((current_aux_row[71])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_2657)
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715) + (node_1974)) + (node_2600)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * (((node_1715) + (node_2014)) + (node_2607)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((node_2669) + (node_2614))))))) * weights[557];
    acc = acc + ((current_aux_row[72])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_2657)
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((node_2669) + (node_2600)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((node_2675) + (node_2607)))))
                            * ((challenges[8])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((node_2681) + (node_2614))))))) * weights[558];
    acc = acc + ((current_aux_row[73])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[195]) * (node_286)))) * weights[559];
    acc = acc + ((current_aux_row[74])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[196])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[49]))))))) * weights[560];
    acc = acc + ((current_aux_row[75])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[198]) * (node_385)))) * weights[561];
    acc = acc + ((current_aux_row[76])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[200])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[53]))))))) * weights[562];
    acc = acc + ((current_aux_row[77])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[195]) * (node_567)))) * weights[563];
    acc = acc + ((current_aux_row[78])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[196])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[50]))))))) * weights[564];
    acc = acc + ((current_aux_row[79])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[198])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[52]))))))) * weights[565];
    acc = acc + ((current_aux_row[80])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[200])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[54]))))))) * weights[566];
    acc = acc + ((current_aux_row[81])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[202])
                            * ((next_aux_row[6])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((current_aux_row[6])
                                        * (current_aux_row[59]))))))) * weights[567];
    acc = acc + ((current_aux_row[82])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_aux_row[7])
                            * (((current_aux_row[71])
                                * ((challenges[8])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * ((node_2675) + (node_2621)))))
                                * ((challenges[8])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * ((node_2681)
                                            + ((current_main_row[44])
                                                * (challenges[22]))))))))) * weights[568];
    acc = acc + ((current_aux_row[83])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_4991)
                            * ((challenges[11])
                                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                    * ((next_main_row[50])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * (current_main_row[50]))))))
                            + (Bfe::from_raw_u64(18446744065119617026ULL)))
                            * (node_4916)))) * weights[569];
    acc = acc + ((current_aux_row[84])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[301])
                            * (((current_aux_row[7])
                                * ((current_aux_row[70])
                                    * ((challenges[8])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * (((node_1715)
                                                + (((current_main_row[29])
                                                    + (Bfe::from_raw_u64(17179869180ULL)))
                                                    * (challenges[21]))) + (node_2621))))))
                                + (node_2627))))) * weights[570];
    acc = acc + ((current_aux_row[85])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[220])
                            * ((((((current_main_row[195])
                                * ((next_aux_row[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * ((current_aux_row[7]) * (node_1724)))))
                                + ((current_main_row[196])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[56]))))))
                                + ((current_main_row[198])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[60]))))))
                                + ((current_main_row[200])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[62]))))))
                                + (current_aux_row[68]))))) * weights[571];
    acc = acc + ((current_aux_row[86])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[228])
                            * ((((((current_main_row[195])
                                * ((next_aux_row[7])
                                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                        * ((current_aux_row[7]) * (node_1952)))))
                                + ((current_main_row[196])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[57]))))))
                                + ((current_main_row[198])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[61]))))))
                                + ((current_main_row[200])
                                    * ((next_aux_row[7])
                                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                            * ((current_aux_row[7])
                                                * (current_aux_row[63]))))))
                                + (current_aux_row[69]))))) * weights[572];
    return acc;
}
}}} // namespace triton_vm::gpu::quotient_gen

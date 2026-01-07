#pragma once
// Generated from src/quotient/constraint_evaluations.cpp
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py
//
// This file contains transition constraints (part 1/4).

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_transition_part0_weighted(const Bfe* current_main_row, const Xfe* current_aux_row, const Bfe* next_main_row, const Xfe* next_aux_row, const Xfe* challenges, const Xfe* weights) {
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
    acc = acc + Xfe((next_main_row[0])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[0])
                            + (Bfe::from_raw_u64(4294967295ULL))))) * weights[175];
    acc = acc + Xfe((current_main_row[6])
                    * ((next_main_row[6])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[6])))) * weights[176];
    acc = acc + Xfe(((node_34) * (next_main_row[3]))
                    + ((current_main_row[4])
                        * (((next_main_row[3]) + (node_30))
                            + (Bfe::from_raw_u64(18446744065119617026ULL))))) * weights[177];
    acc = acc + Xfe((current_main_row[5])
                    * ((next_main_row[5])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[178];
    acc = acc + Xfe((((current_main_row[5])
                    + (Bfe::from_raw_u64(18446744065119617026ULL)))
                    * (next_main_row[5]))
                    * ((next_main_row[1])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[179];
    acc = acc + Xfe((current_main_row[5]) * (next_main_row[1])) * weights[180];
    acc = acc + Xfe(((current_main_row[5]) * (node_34)) * (node_47)) * weights[181];
    acc = acc + Xfe(((next_main_row[7])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (current_main_row[7])))
                    + (Bfe::from_raw_u64(18446744065119617026ULL))) * weights[182];
    acc = acc + Xfe((current_main_row[8])
                    * ((next_main_row[8])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[8])))) * weights[183];
    acc = acc + Xfe(((((((((((((((((((current_main_row[210]) * (node_120))
                    + ((current_main_row[211]) * (node_545)))
                    + ((current_main_row[218]) * (node_1676)))
                    + ((current_main_row[226]) * (node_1683)))
                    + ((current_main_row[220])
                        * ((((((current_main_row[195]) * (node_541))
                            + ((current_main_row[196])
                                * ((next_main_row[30]) + (node_540))))
                            + ((current_main_row[198])
                                * ((next_main_row[31]) + (node_540))))
                            + ((current_main_row[200])
                                * ((next_main_row[32]) + (node_540))))
                            + ((current_main_row[202])
                                * ((next_main_row[33]) + (node_540))))))
                    + ((current_main_row[228])
                        * ((((((current_main_row[195]) * (node_1676))
                            + ((current_main_row[196])
                                * ((next_main_row[28]) + (node_544))))
                            + ((current_main_row[198]) * (node_2029)))
                            + ((current_main_row[200])
                                * ((next_main_row[28]) + (node_548))))
                            + ((current_main_row[202]) * (node_2103)))))
                    + ((current_main_row[237]) * (node_1684)))
                    + ((current_main_row[238]) * (node_1684)))
                    + ((current_main_row[244]) * (node_1682)))
                    + ((current_main_row[245]) * (node_120)))
                    + ((current_main_row[242]) * (node_262)))
                    + ((current_main_row[246]) * (node_262)))
                    + ((current_main_row[248]) * (node_262)))
                    + ((current_main_row[253]) * (node_262)))
                    + ((current_main_row[272]) * (node_1623)))
                    + ((current_main_row[275]) * (node_1623)))
                    + ((current_main_row[296]) * (node_124))) * (node_4429)) * weights[184];
    acc = acc + Xfe((node_4839) * (node_4838)) * weights[185];
    acc = acc + Xfe(((node_4839)
                    * ((next_main_row[49])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[49])))) * (next_main_row[47])) * weights[186];
    acc = acc + Xfe(((current_main_row[47])
                    * ((current_main_row[47])
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    * (node_4845)) * weights[187];
    acc = acc + Xfe((((current_main_row[51])
                    + (Bfe::from_raw_u64(18446744065119617026ULL)))
                    * (current_main_row[51])) * (node_4908)) * weights[188];
    acc = acc + Xfe((current_main_row[54]) * (node_4916)) * weights[189];
    acc = acc + Xfe((node_4913) * (node_4916)) * weights[190];
    acc = acc + Xfe(((node_4916)
                    * ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (next_main_row[51])))
                    * ((next_main_row[53])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[53])))) * weights[191];
    acc = acc + Xfe((node_4916)
                    * ((next_main_row[55])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[55])))) * weights[192];
    acc = acc + Xfe((node_4916)
                    * ((next_main_row[56])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[56])))) * weights[193];
    acc = acc + Xfe((node_5020) * (node_5019)) * weights[194];
    acc = acc + Xfe((node_5027)
                    * ((next_main_row[60])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[60])))) * weights[195];
    acc = acc + Xfe((node_5027)
                    * ((next_main_row[61])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[61])))) * weights[196];
    acc = acc + Xfe((current_main_row[372])
                    * ((current_main_row[58])
                        + (Bfe::from_raw_u64(18446743858961186866ULL)))) * weights[197];
    acc = acc + Xfe(((current_main_row[367])
                    * ((current_main_row[64])
                        + (Bfe::from_raw_u64(18446744052234715141ULL))))
                    * (next_main_row[64])) * weights[198];
    acc = acc + Xfe((((next_main_row[62]) * (node_5912)) * (node_5914))
                    * (((next_main_row[64])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[64])))
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[199];
    acc = acc + Xfe((current_main_row[373]) * (node_5942)) * weights[200];
    acc = acc + Xfe((node_5944)
                    * ((next_main_row[63])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[63])))) * weights[201];
    acc = acc + Xfe((node_5944)
                    * ((next_main_row[62])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[62])))) * weights[202];
    acc = acc + Xfe(((current_main_row[368]) * (node_5938)) * (next_main_row[62])) * weights[203];
    acc = acc + Xfe((current_main_row[374]) * (next_main_row[62])) * weights[204];
    acc = acc + Xfe(((node_5958) * (node_5930)) * (next_main_row[62])) * weights[205];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(263719581847590ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(76643691379275ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(115096533571410ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(256362302871255ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[322]))) + (current_main_row[113]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_5809)))) * weights[206];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(4758823762860ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(263719581847590ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(76643691379275ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(115096533571410ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[322]))) + (current_main_row[114]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_5820)))) * weights[207];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(123480309731250ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(4758823762860ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(263719581847590ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(76643691379275ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[322]))) + (current_main_row[115]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_5831)))) * weights[208];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(145268678818785ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(123480309731250ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(4758823762860ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(263719581847590ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[322]))) + (current_main_row[116]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_5842)))) * weights[209];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(32014686216930ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(145268678818785ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(123480309731250ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(4758823762860ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[322]))) + (current_main_row[117]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[97])))) * weights[210];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(185731565704980ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(32014686216930ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(145268678818785ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(123480309731250ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[322]))) + (current_main_row[118]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[98])))) * weights[211];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(231348413345175ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(185731565704980ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(32014686216930ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(145268678818785ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[322]))) + (current_main_row[119]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[99])))) * weights[212];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(51685636428030ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(231348413345175ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(185731565704980ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(32014686216930ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[322]))) + (current_main_row[120]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[100])))) * weights[213];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(244602682417545ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(51685636428030ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(231348413345175ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(185731565704980ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[322]))) + (current_main_row[121]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[101])))) * weights[214];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(118201794925695ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(244602682417545ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(51685636428030ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(231348413345175ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[322]))) + (current_main_row[122]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[102])))) * weights[215];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(177601192615545ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(118201794925695ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(244602682417545ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(51685636428030ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[322]))) + (current_main_row[123]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[103])))) * weights[216];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(175668457332795ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(177601192615545ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(118201794925695ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(244602682417545ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(51629801853195ULL))
                            * (current_main_row[322]))) + (current_main_row[124]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[104])))) * weights[217];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(51629801853195ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(175668457332795ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(177601192615545ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(118201794925695ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(256362302871255ULL))
                            * (current_main_row[322]))) + (current_main_row[125]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[105])))) * weights[218];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(256362302871255ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(51629801853195ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(175668457332795ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(177601192615545ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(115096533571410ULL))
                            * (current_main_row[322]))) + (current_main_row[126]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[106])))) * weights[219];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(115096533571410ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(256362302871255ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(51629801853195ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(175668457332795ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(76643691379275ULL))
                            * (current_main_row[322]))) + (current_main_row[127]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[107])))) * weights[220];
    acc = acc + Xfe((next_main_row[64])
                    * (((((((((((((((((((Bfe::from_raw_u64(76643691379275ULL))
                        * (node_5149))
                        + ((Bfe::from_raw_u64(115096533571410ULL)) * (node_5160)))
                        + ((Bfe::from_raw_u64(256362302871255ULL)) * (node_5171)))
                        + ((Bfe::from_raw_u64(51629801853195ULL)) * (node_5182)))
                        + ((Bfe::from_raw_u64(175668457332795ULL))
                            * (current_main_row[311])))
                        + ((Bfe::from_raw_u64(177601192615545ULL))
                            * (current_main_row[312])))
                        + ((Bfe::from_raw_u64(118201794925695ULL))
                            * (current_main_row[313])))
                        + ((Bfe::from_raw_u64(244602682417545ULL))
                            * (current_main_row[314])))
                        + ((Bfe::from_raw_u64(51685636428030ULL))
                            * (current_main_row[315])))
                        + ((Bfe::from_raw_u64(231348413345175ULL))
                            * (current_main_row[316])))
                        + ((Bfe::from_raw_u64(185731565704980ULL))
                            * (current_main_row[317])))
                        + ((Bfe::from_raw_u64(32014686216930ULL))
                            * (current_main_row[318])))
                        + ((Bfe::from_raw_u64(145268678818785ULL))
                            * (current_main_row[319])))
                        + ((Bfe::from_raw_u64(123480309731250ULL))
                            * (current_main_row[320])))
                        + ((Bfe::from_raw_u64(4758823762860ULL))
                            * (current_main_row[321])))
                        + ((Bfe::from_raw_u64(263719581847590ULL))
                            * (current_main_row[322]))) + (current_main_row[128]))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[108])))) * weights[221];
    acc = acc + Xfe((current_main_row[129]) * (node_6449)) * weights[222];
    acc = acc + Xfe((current_main_row[135]) * (node_6501)) * weights[223];
    acc = acc + Xfe(((next_main_row[135]) * (next_main_row[136]))
                    + ((node_6501)
                        * (((next_main_row[136])
                            + ((Bfe::from_raw_u64(18446744065119617026ULL))
                                * (current_main_row[136])))
                            + (Bfe::from_raw_u64(18446744065119617026ULL))))) * weights[224];
    acc = acc + Xfe(((next_main_row[139]) * (current_main_row[143])) * (node_6551)) * weights[225];
    acc = acc + Xfe((next_main_row[139]) * (current_main_row[145])) * weights[226];
    acc = acc + Xfe((node_6561)
                    * ((next_main_row[142])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[142])))) * weights[227];
    acc = acc + Xfe((((node_6561) * (current_main_row[143])) * (node_6551)) * (node_6569)) * weights[228];
    acc = acc + Xfe(((node_6561) * (current_main_row[145])) * (node_6569)) * weights[229];
    acc = acc + Xfe((((node_6561) * (node_6551)) * (node_6554)) * (node_6575)) * weights[230];
    acc = acc + Xfe(((node_6561) * (node_6557)) * (node_6578)) * weights[231];
    acc = acc + Xfe((((current_main_row[323]) * (node_6595)) * (node_6597))
                    * (current_main_row[147])) * weights[232];
    acc = acc + Xfe(((node_6600) * (node_6597)) * (node_6602)) * weights[233];
    acc = acc + Xfe((((current_main_row[328]) * (node_6575)) * (node_6557)) * (node_6602)) * weights[234];
    acc = acc + Xfe((((current_main_row[328]) * (node_6554)) * (node_6578))
                    * (current_main_row[147])) * weights[235];
    acc = acc + Xfe(((current_main_row[370])
                    * ((current_main_row[139])
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    * ((current_main_row[147])
                        + (Bfe::from_raw_u64(18446744060824649731ULL)))) * weights[236];
    acc = acc + Xfe(((current_main_row[370]) * (current_main_row[139]))
                    * (current_main_row[147])) * weights[237];
    acc = acc + Xfe(((node_6561) * (current_main_row[369]))
                    * (((current_main_row[147])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((Bfe::from_raw_u64(8589934590ULL))
                                * (next_main_row[147]))))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * ((node_6554) * (node_6557))))) * weights[238];
    acc = acc + Xfe((current_main_row[377]) * ((current_main_row[147]) + (node_6567))) * weights[239];
    acc = acc + Xfe(((current_main_row[335]) * (next_main_row[143]))
                    * ((next_main_row[147])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[147])))) * weights[240];
    acc = acc + Xfe((current_main_row[371])
                    * ((next_main_row[143])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[143])))) * weights[241];
    acc = acc + Xfe(((current_main_row[371]) * (node_6578))
                    * ((current_main_row[147])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (node_6661)))) * weights[242];
    acc = acc + Xfe(((current_main_row[371]) * (node_6557))
                    * ((current_main_row[147])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (current_main_row[378])))) * weights[243];
    acc = acc + Xfe(((node_6561) * ((current_main_row[333]) * (node_6588)))
                    * (((current_main_row[147])
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (next_main_row[147]))) + (node_6611))) * weights[244];
    acc = acc + Xfe((current_main_row[169])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_2775) * (current_main_row[13])))) * weights[245];
    acc = acc + Xfe((current_main_row[170])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((current_main_row[12]) * (node_2754)) * (node_2744)))) * weights[246];
    acc = acc + Xfe((current_main_row[171])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[169]) * (node_2744)))) * weights[247];
    acc = acc + Xfe((current_main_row[172])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_2775) * (node_2754)))) * weights[248];
    acc = acc + Xfe((current_main_row[173])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_288) * (node_290)))) * weights[249];
    acc = acc + Xfe((current_main_row[174])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[172]) * (node_2744)))) * weights[250];
    acc = acc + Xfe((current_main_row[175])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_288) * (current_main_row[41])))) * weights[251];
    acc = acc + Xfe((current_main_row[176])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[170]) * (node_2746)))) * weights[252];
    acc = acc + Xfe((current_main_row[177])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[170]) * (current_main_row[15])))) * weights[253];
    acc = acc + Xfe((current_main_row[178])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[171]) * (node_2746)))) * weights[254];
    acc = acc + Xfe((current_main_row[179])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[173]) * (current_main_row[40])))) * weights[255];
    acc = acc + Xfe((current_main_row[180])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[171]) * (current_main_row[15])))) * weights[256];
    acc = acc + Xfe((current_main_row[181])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[175]) * (node_293)))) * weights[257];
    acc = acc + Xfe((current_main_row[182])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[176]) * (node_2748)))) * weights[258];
    acc = acc + Xfe((current_main_row[183])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[174]) * (node_2746)))) * weights[259];
    acc = acc + Xfe((current_main_row[184])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[174]) * (current_main_row[15])))) * weights[260];
    acc = acc + Xfe((current_main_row[185])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[173]) * (node_293)))) * weights[261];
    acc = acc + Xfe((current_main_row[186])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((current_main_row[12]) * (current_main_row[13]))
                            * (node_2744)))) * weights[262];
    acc = acc + Xfe((current_main_row[187])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[177]) * (node_2748)))) * weights[263];
    acc = acc + Xfe((current_main_row[188])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[169]) * (current_main_row[14])))) * weights[264];
    acc = acc + Xfe((current_main_row[189])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[180]) * (node_2748)))) * weights[265];
    acc = acc + Xfe((current_main_row[190])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[178]) * (node_2748)))) * weights[266];
    acc = acc + Xfe((current_main_row[191])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[177]) * (current_main_row[16])))) * weights[267];
    acc = acc + Xfe((current_main_row[192])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[42]) * (node_290)))) * weights[268];
    acc = acc + Xfe((current_main_row[193])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[42]) * (current_main_row[41])))) * weights[269];
    acc = acc + Xfe((current_main_row[194])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[172]) * (current_main_row[14])))) * weights[270];
    acc = acc + Xfe((current_main_row[195])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[185]) * (current_main_row[39])))) * weights[271];
    acc = acc + Xfe((current_main_row[196])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[179]) * (node_341)))) * weights[272];
    acc = acc + Xfe((current_main_row[197])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[183]) * (node_2748)))) * weights[273];
    acc = acc + Xfe((current_main_row[198])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((current_main_row[179]) * (current_main_row[39])))) * weights[274];
    return acc;
}
}}} // namespace triton_vm::gpu::quotient_gen

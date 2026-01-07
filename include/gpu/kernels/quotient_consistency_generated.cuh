#pragma once
// Generated from src/quotient/constraint_evaluations.cpp
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py
//
// This file contains consistency constraints.

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_consistency_weighted(const Bfe* main_row, const Xfe* aux_row, const Xfe* challenges, const Xfe* weights) {
            const auto node_102 = (main_row[152])
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744047939747846ULL)));
            const auto node_221 = (main_row[153])
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744047939747846ULL)));
            const auto node_238 = ((main_row[154])
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744052234715141ULL))))
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744047939747846ULL)));
            const auto node_245 = ((main_row[154])
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744056529682436ULL))))
                * ((main_row[64])
                    + (Bfe::from_raw_u64(18446744047939747846ULL)));
            const auto node_655 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[157]));
            const auto node_114 = (((main_row[63])
                + (Bfe::from_raw_u64(18446743992105173011ULL)))
                * ((main_row[63])
                    + (Bfe::from_raw_u64(18446743923385696291ULL))))
                * ((main_row[63])
                    + (Bfe::from_raw_u64(18446743828896415801ULL)));
            const auto node_116 = (node_102) * (main_row[161]);
            const auto node_660 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[160]));
            const auto node_101 = (main_row[64])
                + (Bfe::from_raw_u64(18446744047939747846ULL));
            const auto node_678 = (main_row[142])
                + (Bfe::from_raw_u64(18446743949155500061ULL));
            const auto node_674 = (main_row[142])
                + (Bfe::from_raw_u64(18446743940565565471ULL));
            const auto node_94 = (main_row[64])
                + (Bfe::from_raw_u64(18446744056529682436ULL));
            const auto node_97 = (main_row[64])
                + (Bfe::from_raw_u64(18446744052234715141ULL));
            const auto node_153 = ((((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[65])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[66]))) * (main_row[109]))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_155 = ((((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[69])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[70]))) * (main_row[110]))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_157 = ((((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[73])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[74]))) * (main_row[111]))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_159 = ((((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[77])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[78]))) * (main_row[112]))
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_680 = (main_row[139])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_90 = (main_row[64])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_670 = (main_row[142])
                + (Bfe::from_raw_u64(18446744017874976781ULL));
            const auto node_11 = (Bfe::from_raw_u64(4294967295ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (((Bfe::from_raw_u64(38654705655ULL))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (main_row[3]))) * (main_row[4])));
            const auto node_8 = (Bfe::from_raw_u64(38654705655ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[3]));
            const auto node_104 = (((main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL)))
                * ((main_row[62])
                    + (Bfe::from_raw_u64(18446744060824649731ULL))))
                * ((main_row[62])
                    + (Bfe::from_raw_u64(18446744056529682436ULL)));
            const auto node_85 = (main_row[62])
                + (Bfe::from_raw_u64(18446744060824649731ULL));
            const auto node_73 = (main_row[63])
                + (Bfe::from_raw_u64(18446743992105173011ULL));
            const auto node_79 = (main_row[63])
                + (Bfe::from_raw_u64(18446743923385696291ULL));
            const auto node_82 = (main_row[63])
                + (Bfe::from_raw_u64(18446743828896415801ULL));
            const auto node_126 = ((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[65])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[66]));
            const auto node_133 = ((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[69])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[70]));
            const auto node_140 = ((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[73])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[74]));
            const auto node_147 = ((Bfe::from_raw_u64(18446744065119617025ULL))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * ((main_row[77])
                        * (Bfe::from_raw_u64(281474976645120ULL)))))
                + ((Bfe::from_raw_u64(18446744065119617026ULL))
                    * (main_row[78]));
            const auto node_89 = (main_row[64])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_663 = (main_row[142])
                + (Bfe::from_raw_u64(18446744052234715141ULL));
            const auto node_666 = (main_row[142])
                + (Bfe::from_raw_u64(18446744009285042191ULL));
            const auto node_86 = ((main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL))) * (node_85);
            const auto node_83 = (main_row[62])
                + (Bfe::from_raw_u64(18446744065119617026ULL));
            const auto node_103 = (main_row[62])
                + (Bfe::from_raw_u64(18446744056529682436ULL));
    Xfe acc = Xfe::zero();
    acc = acc + Xfe((node_11) * (main_row[4])) * weights[81];
    acc = acc + Xfe((node_11) * (node_8)) * weights[82];
    acc = acc + Xfe((main_row[5])
                    * ((main_row[5])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[83];
    acc = acc + Xfe((main_row[6])
                    * ((main_row[6])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[84];
    acc = acc + Xfe((main_row[12])
                    * ((main_row[12])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[85];
    acc = acc + Xfe((main_row[13])
                    * ((main_row[13])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[86];
    acc = acc + Xfe((main_row[14])
                    * ((main_row[14])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[87];
    acc = acc + Xfe((main_row[15])
                    * ((main_row[15])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[88];
    acc = acc + Xfe((main_row[16])
                    * ((main_row[16])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[89];
    acc = acc + Xfe((main_row[17])
                    * ((main_row[17])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[90];
    acc = acc + Xfe((main_row[18])
                    * ((main_row[18])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[91];
    acc = acc + Xfe((main_row[8])
                    * ((main_row[8])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[92];
    acc = acc + Xfe((main_row[10])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((((((main_row[12])
                            + ((Bfe::from_raw_u64(8589934590ULL))
                                * (main_row[13])))
                            + ((Bfe::from_raw_u64(17179869180ULL))
                                * (main_row[14])))
                            + ((Bfe::from_raw_u64(34359738360ULL))
                                * (main_row[15])))
                            + ((Bfe::from_raw_u64(68719476720ULL))
                                * (main_row[16])))
                            + ((Bfe::from_raw_u64(137438953440ULL))
                                * (main_row[17])))
                            + ((Bfe::from_raw_u64(274877906880ULL))
                                * (main_row[18]))))) * weights[93];
    acc = acc + Xfe(((main_row[8])
                    * ((main_row[7])
                        + (Bfe::from_raw_u64(18446744065119617026ULL))))
                    * (main_row[45])) * weights[94];
    acc = acc + Xfe((node_104) * (main_row[62])) * weights[95];
    acc = acc + Xfe((node_85) * (node_73)) * weights[96];
    acc = acc + Xfe(((main_row[165]) * (node_79)) * (node_82)) * weights[97];
    acc = acc + Xfe((node_104) * (main_row[64])) * weights[98];
    acc = acc + Xfe((node_114) * (main_row[64])) * weights[99];
    acc = acc + Xfe((node_153) * (main_row[109])) * weights[100];
    acc = acc + Xfe((node_155) * (main_row[110])) * weights[101];
    acc = acc + Xfe((node_157) * (main_row[111])) * weights[102];
    acc = acc + Xfe((node_159) * (main_row[112])) * weights[103];
    acc = acc + Xfe((node_153) * (node_126)) * weights[104];
    acc = acc + Xfe((node_155) * (node_133)) * weights[105];
    acc = acc + Xfe((node_157) * (node_140)) * weights[106];
    acc = acc + Xfe((node_159) * (node_147)) * weights[107];
    acc = acc + Xfe((node_153)
                    * (((main_row[67])
                        * (Bfe::from_raw_u64(281474976645120ULL)))
                        + (main_row[68]))) * weights[108];
    acc = acc + Xfe((node_155)
                    * (((main_row[71])
                        * (Bfe::from_raw_u64(281474976645120ULL)))
                        + (main_row[72]))) * weights[109];
    acc = acc + Xfe((node_157)
                    * (((main_row[75])
                        * (Bfe::from_raw_u64(281474976645120ULL)))
                        + (main_row[76]))) * weights[110];
    acc = acc + Xfe((node_159)
                    * (((main_row[79])
                        * (Bfe::from_raw_u64(281474976645120ULL)))
                        + (main_row[80]))) * weights[111];
    acc = acc + Xfe((node_114) * (main_row[103])) * weights[112];
    acc = acc + Xfe((node_114) * (main_row[104])) * weights[113];
    acc = acc + Xfe((node_114) * (main_row[105])) * weights[114];
    acc = acc + Xfe((node_114) * (main_row[106])) * weights[115];
    acc = acc + Xfe((node_114) * (main_row[107])) * weights[116];
    acc = acc + Xfe((node_114) * (main_row[108])) * weights[117];
    acc = acc + Xfe((node_116)
                    * ((main_row[103])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[118];
    acc = acc + Xfe((node_116)
                    * ((main_row[104])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[119];
    acc = acc + Xfe((node_116)
                    * ((main_row[105])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[120];
    acc = acc + Xfe((node_116)
                    * ((main_row[106])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[121];
    acc = acc + Xfe((node_116)
                    * ((main_row[107])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[122];
    acc = acc + Xfe((node_116)
                    * ((main_row[108])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[123];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[113])
                        + (Bfe::from_raw_u64(11408918724931329738ULL))))
                    + ((node_221)
                        * ((main_row[113])
                            + (Bfe::from_raw_u64(16073625066478178581ULL)))))
                    + ((main_row[155])
                        * ((main_row[113])
                            + (Bfe::from_raw_u64(12231462398569191607ULL)))))
                    + ((node_238)
                        * ((main_row[113])
                            + (Bfe::from_raw_u64(9408518518620565480ULL)))))
                    + ((node_245)
                        * ((main_row[113])
                            + (Bfe::from_raw_u64(11492978409391175103ULL))))) * weights[124];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[114])
                        + (Bfe::from_raw_u64(2786462832312611053ULL))))
                    + ((node_221)
                        * ((main_row[114])
                            + (Bfe::from_raw_u64(11837051899140380443ULL)))))
                    + ((main_row[155])
                        * ((main_row[114])
                            + (Bfe::from_raw_u64(11546487907579866869ULL)))))
                    + ((node_238)
                        * ((main_row[114])
                            + (Bfe::from_raw_u64(1785884128667671832ULL)))))
                    + ((node_245)
                        * ((main_row[114])
                            + (Bfe::from_raw_u64(17615222217495663839ULL))))) * weights[125];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[115])
                        + (Bfe::from_raw_u64(6782977121958050999ULL))))
                    + ((node_221)
                        * ((main_row[115])
                            + (Bfe::from_raw_u64(15625104599191418968ULL)))))
                    + ((main_row[155])
                        * ((main_row[115])
                            + (Bfe::from_raw_u64(14006427992450931468ULL)))))
                    + ((node_238)
                        * ((main_row[115])
                            + (Bfe::from_raw_u64(1188899344229954938ULL)))))
                    + ((node_245)
                        * ((main_row[115])
                            + (Bfe::from_raw_u64(5864349944556149748ULL))))) * weights[126];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[116])
                        + (Bfe::from_raw_u64(8688421733879975670ULL))))
                    + ((node_221)
                        * ((main_row[116])
                            + (Bfe::from_raw_u64(12819157612210448391ULL)))))
                    + ((main_row[155])
                        * ((main_row[116])
                            + (Bfe::from_raw_u64(11770003398407723041ULL)))))
                    + ((node_238)
                        * ((main_row[116])
                            + (Bfe::from_raw_u64(14740727267735052728ULL)))))
                    + ((node_245)
                        * ((main_row[116])
                            + (Bfe::from_raw_u64(2745609811140253793ULL))))) * weights[127];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[117])
                        + (Bfe::from_raw_u64(8602724563769480463ULL))))
                    + ((node_221)
                        * ((main_row[117])
                            + (Bfe::from_raw_u64(6235256903503367222ULL)))))
                    + ((main_row[155])
                        * ((main_row[117])
                            + (Bfe::from_raw_u64(15124190001489436038ULL)))))
                    + ((node_238)
                        * ((main_row[117])
                            + (Bfe::from_raw_u64(880257844992994007ULL)))))
                    + ((node_245)
                        * ((main_row[117])
                            + (Bfe::from_raw_u64(15189664869386394185ULL))))) * weights[128];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[118])
                        + (Bfe::from_raw_u64(13589155570211330507ULL))))
                    + ((node_221)
                        * ((main_row[118])
                            + (Bfe::from_raw_u64(11242082964257948320ULL)))))
                    + ((main_row[155])
                        * ((main_row[118])
                            + (Bfe::from_raw_u64(14834674155811570980ULL)))))
                    + ((node_238)
                        * ((main_row[118])
                            + (Bfe::from_raw_u64(10737952517017171197ULL)))))
                    + ((node_245)
                        * ((main_row[118])
                            + (Bfe::from_raw_u64(5192963426821415349ULL))))) * weights[129];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[119])
                        + (Bfe::from_raw_u64(10263462378312899510ULL))))
                    + ((node_221)
                        * ((main_row[119])
                            + (Bfe::from_raw_u64(5820425254787221108ULL)))))
                    + ((main_row[155])
                        * ((main_row[119])
                            + (Bfe::from_raw_u64(13004675752386552573ULL)))))
                    + ((node_238)
                        * ((main_row[119])
                            + (Bfe::from_raw_u64(15757222735741919824ULL)))))
                    + ((node_245)
                        * ((main_row[119])
                            + (Bfe::from_raw_u64(11971160388083607515ULL))))) * weights[130];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[120])
                        + (Bfe::from_raw_u64(3264875873073042616ULL))))
                    + ((node_221)
                        * ((main_row[120])
                            + (Bfe::from_raw_u64(12019227591549292608ULL)))))
                    + ((main_row[155])
                        * ((main_row[120])
                            + (Bfe::from_raw_u64(1475232519215872482ULL)))))
                    + ((node_238)
                        * ((main_row[120])
                            + (Bfe::from_raw_u64(14382578632612566479ULL)))))
                    + ((node_245)
                        * ((main_row[120])
                            + (Bfe::from_raw_u64(11608544217838050708ULL))))) * weights[131];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[121])
                        + (Bfe::from_raw_u64(3133435276616064683ULL))))
                    + ((node_221)
                        * ((main_row[121])
                            + (Bfe::from_raw_u64(4625353063880731092ULL)))))
                    + ((main_row[155])
                        * ((main_row[121])
                            + (Bfe::from_raw_u64(4883869161905122316ULL)))))
                    + ((node_238)
                        * ((main_row[121])
                            + (Bfe::from_raw_u64(3305272539067787726ULL)))))
                    + ((node_245)
                        * ((main_row[121])
                            + (Bfe::from_raw_u64(674972795234232729ULL))))) * weights[132];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[122])
                        + (Bfe::from_raw_u64(13508500531157332153ULL))))
                    + ((node_221)
                        * ((main_row[122])
                            + (Bfe::from_raw_u64(3723900760706330287ULL)))))
                    + ((main_row[155])
                        * ((main_row[122])
                            + (Bfe::from_raw_u64(12579737103870920763ULL)))))
                    + ((node_238)
                        * ((main_row[122])
                            + (Bfe::from_raw_u64(17082569335437832789ULL)))))
                    + ((node_245)
                        * ((main_row[122])
                            + (Bfe::from_raw_u64(14165256104883557753ULL))))) * weights[133];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[123])
                        + (Bfe::from_raw_u64(6968886508437513677ULL))))
                    + ((node_221)
                        * ((main_row[123])
                            + (Bfe::from_raw_u64(615596267195055952ULL)))))
                    + ((main_row[155])
                        * ((main_row[123])
                            + (Bfe::from_raw_u64(10119826060478909841ULL)))))
                    + ((node_238)
                        * ((main_row[123])
                            + (Bfe::from_raw_u64(229051680548583225ULL)))))
                    + ((node_245)
                        * ((main_row[123])
                            + (Bfe::from_raw_u64(15283356519694111298ULL))))) * weights[134];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[124])
                        + (Bfe::from_raw_u64(9713264609690967820ULL))))
                    + ((node_221)
                        * ((main_row[124])
                            + (Bfe::from_raw_u64(18227830850447556704ULL)))))
                    + ((main_row[155])
                        * ((main_row[124])
                            + (Bfe::from_raw_u64(1528714547662620921ULL)))))
                    + ((node_238)
                        * ((main_row[124])
                            + (Bfe::from_raw_u64(2943254981416254648ULL)))))
                    + ((node_245)
                        * ((main_row[124])
                            + (Bfe::from_raw_u64(2306049938060341466ULL))))) * weights[135];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[125])
                        + (Bfe::from_raw_u64(12482374976099749513ULL))))
                    + ((node_221)
                        * ((main_row[125])
                            + (Bfe::from_raw_u64(15609691041895848348ULL)))))
                    + ((main_row[155])
                        * ((main_row[125])
                            + (Bfe::from_raw_u64(12972275929555275935ULL)))))
                    + ((node_238)
                        * ((main_row[125])
                            + (Bfe::from_raw_u64(5767629304344025219ULL)))))
                    + ((node_245)
                        * ((main_row[125])
                            + (Bfe::from_raw_u64(11578793764462375094ULL))))) * weights[136];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[126])
                        + (Bfe::from_raw_u64(13209711277645656680ULL))))
                    + ((node_221)
                        * ((main_row[126])
                            + (Bfe::from_raw_u64(15235800289984546486ULL)))))
                    + ((main_row[155])
                        * ((main_row[126])
                            + (Bfe::from_raw_u64(15992731669612695172ULL)))))
                    + ((node_238)
                        * ((main_row[126])
                            + (Bfe::from_raw_u64(16721422493821450473ULL)))))
                    + ((node_245)
                        * ((main_row[126])
                            + (Bfe::from_raw_u64(7511767364422267184ULL))))) * weights[137];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[127])
                        + (Bfe::from_raw_u64(87705059284758253ULL))))
                    + ((node_221)
                        * ((main_row[127])
                            + (Bfe::from_raw_u64(11392407538241985753ULL)))))
                    + ((main_row[155])
                        * ((main_row[127])
                            + (Bfe::from_raw_u64(17877154195438905917ULL)))))
                    + ((node_238)
                        * ((main_row[127])
                            + (Bfe::from_raw_u64(5753720429376839714ULL)))))
                    + ((node_245)
                        * ((main_row[127])
                            + (Bfe::from_raw_u64(16999805755930336630ULL))))) * weights[138];
    acc = acc + Xfe((((((node_102)
                    * ((main_row[128])
                        + (Bfe::from_raw_u64(330155256278907084ULL))))
                    + ((node_221)
                        * ((main_row[128])
                            + (Bfe::from_raw_u64(11776128816341368822ULL)))))
                    + ((main_row[155])
                        * ((main_row[128])
                            + (Bfe::from_raw_u64(939319986782105612ULL)))))
                    + ((node_238)
                        * ((main_row[128])
                            + (Bfe::from_raw_u64(2063756830275051942ULL)))))
                    + ((node_245)
                        * ((main_row[128])
                            + (Bfe::from_raw_u64(940614108343834936ULL))))) * weights[139];
    acc = acc + Xfe((main_row[129])
                    * ((Bfe::from_raw_u64(4294967295ULL))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (main_row[129])))) * weights[140];
    acc = acc + Xfe((main_row[135])
                    * ((Bfe::from_raw_u64(4294967295ULL))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (main_row[135])))) * weights[141];
    acc = acc + Xfe((main_row[139])
                    * ((Bfe::from_raw_u64(4294967295ULL))
                        + ((Bfe::from_raw_u64(18446744065119617026ULL))
                            * (main_row[139])))) * weights[142];
    acc = acc + Xfe((main_row[139]) * (main_row[140])) * weights[143];
    acc = acc + Xfe((Bfe::from_raw_u64(4294967295ULL))
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[141])
                            * ((main_row[140])
                                + (Bfe::from_raw_u64(18446743927680663586ULL)))))) * weights[144];
    acc = acc + Xfe((main_row[144]) * (node_655)) * weights[145];
    acc = acc + Xfe((main_row[143]) * (node_655)) * weights[146];
    acc = acc + Xfe((main_row[146]) * (node_660)) * weights[147];
    acc = acc + Xfe((main_row[145]) * (node_660)) * weights[148];
    acc = acc + Xfe((main_row[167])
                    * ((main_row[147])
                        + (Bfe::from_raw_u64(18446744060824649731ULL)))) * weights[149];
    acc = acc + Xfe((main_row[168]) * (main_row[147])) * weights[150];
    acc = acc + Xfe((((main_row[163]) * (node_655)) * (node_660)) * (main_row[147])) * weights[151];
    acc = acc + Xfe((((main_row[166]) * (node_678)) * (node_660))
                    * ((main_row[147])
                        + (Bfe::from_raw_u64(18446744065119617026ULL)))) * weights[152];
    acc = acc + Xfe((((main_row[164]) * (node_680)) * (node_655))
                    * ((main_row[147]) + (Bfe::from_raw_u64(4294967295ULL)))) * weights[153];
    acc = acc + Xfe((((main_row[166]) * (node_674)) * (node_655)) * (main_row[147])) * weights[154];
    acc = acc + Xfe(((main_row[164]) * (main_row[139])) * (node_655)) * weights[155];
    acc = acc + Xfe((node_680) * (main_row[148])) * weights[156];
    acc = acc + Xfe((main_row[151])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[64]) * (node_89)))) * weights[157];
    acc = acc + Xfe((main_row[152])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_89) * (node_90)) * (node_94)) * (node_97)))) * weights[158];
    acc = acc + Xfe((main_row[153])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[64]) * (node_90)) * (node_94)) * (node_97)))) * weights[159];
    acc = acc + Xfe((main_row[154])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[151]) * (node_90)))) * weights[160];
    acc = acc + Xfe((main_row[155])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[151]) * (node_94)) * (node_97)) * (node_101)))) * weights[161];
    acc = acc + Xfe((main_row[156])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_663)
                            * ((main_row[142])
                                + (Bfe::from_raw_u64(18446744043644780551ULL)))))) * weights[162];
    acc = acc + Xfe((main_row[157])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[143]) * (main_row[144])))) * weights[163];
    acc = acc + Xfe((main_row[158])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_663) * (node_666)) * (node_670)) * (node_674)))) * weights[164];
    acc = acc + Xfe((main_row[159])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[156]) * (node_666)))) * weights[165];
    acc = acc + Xfe((main_row[160])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[145]) * (main_row[146])))) * weights[166];
    acc = acc + Xfe((main_row[161])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((node_86) * (main_row[62])))) * weights[167];
    acc = acc + Xfe((main_row[162])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[158]) * (node_678)))) * weights[168];
    acc = acc + Xfe((main_row[163])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[156]) * (node_670)) * (node_674)) * (node_678)))) * weights[169];
    acc = acc + Xfe((main_row[164])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * (((main_row[159]) * (node_674)) * (node_678)))) * weights[170];
    acc = acc + Xfe((main_row[165])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((node_83) * (node_103)) * (main_row[62]))
                            * ((main_row[63])
                                + (Bfe::from_raw_u64(18446743897615892521ULL)))))) * weights[171];
    acc = acc + Xfe((main_row[166])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((main_row[159]) * (node_670)))) * weights[172];
    acc = acc + Xfe((main_row[167])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[162]) * (node_680)) * (node_655)) * (node_660)))) * weights[173];
    acc = acc + Xfe((main_row[168])
                    + ((Bfe::from_raw_u64(18446744065119617026ULL))
                        * ((((main_row[162]) * (main_row[139])) * (node_655))
                            * (node_660)))) * weights[174];
    return acc;
}
}}} // namespace triton_vm::gpu::quotient_gen

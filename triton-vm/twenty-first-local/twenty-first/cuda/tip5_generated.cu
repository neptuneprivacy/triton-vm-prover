__device__ void generated_function(unsigned long long *input,
    unsigned long long *output) {
unsigned long long in0 = input[0], in1 = input[1], in2 = input[2],
in3 = input[3];
unsigned long long in4 = input[4], in5 = input[5], in6 = input[6],
in7 = input[7];
unsigned long long in8 = input[8], in9 = input[9], in10 = input[10],
in11 = input[11];
unsigned long long in12 = input[12], in13 = input[13], in14 = input[14],
in15 = input[15];

unsigned long long node_34 = in0 + in8;
unsigned long long node_35 = in1 + in9;
unsigned long long node_36 = in2 + in10;
unsigned long long node_37 = in3 + in11;
unsigned long long node_38 = in4 + in12;
unsigned long long node_39 = in5 + in13;
unsigned long long node_40 = in6 + in14;
unsigned long long node_41 = in7 + in15;

unsigned long long node_160 = in0 - in8;
unsigned long long node_161 = in1 - in9;
unsigned long long node_162 = in2 - in10;
unsigned long long node_163 = in3 - in11;
unsigned long long node_164 = in4 - in12;
unsigned long long node_165 = in5 - in13;
unsigned long long node_166 = in6 - in14;
unsigned long long node_167 = in7 - in15;

unsigned long long node_34_plus_38 = node_34 + node_38;
unsigned long long node_36_plus_40 = node_36 + node_40;
unsigned long long node_51 = node_35 + node_39;
unsigned long long node_53 = node_37 + node_41;

unsigned long long node_58 = node_34_plus_38 + node_36_plus_40;
unsigned long long node_59 = node_51 + node_53;
unsigned long long node_71 = node_34_plus_38 - node_36_plus_40;
unsigned long long node_72 = node_51 - node_53;

unsigned long long node_90 = node_34 - node_38;
unsigned long long node_91 = node_35 - node_39;
unsigned long long node_92 = node_36 - node_40;
unsigned long long node_93 = node_37 - node_41;
unsigned long long node_98 = node_90 + node_92;
unsigned long long node_99 = node_91 + node_93;

unsigned long long node_176 = node_160 + node_164;
unsigned long long node_177 = node_161 + node_165;
unsigned long long node_178 = node_162 + node_166;
unsigned long long node_179 = node_163 + node_167;
unsigned long long node_184 = node_160 + node_162;
unsigned long long node_185 = node_161 + node_163;
unsigned long long node_227 = node_164 + node_166;
unsigned long long node_228 = node_165 + node_167;
unsigned long long node_270 = node_176 + node_178;
unsigned long long node_271 = node_177 + node_179;

unsigned long long node_64, node_67;
asm volatile("mul.lo.u64 %0, %1, 524757;"
: "=l"(node_64)
: "l"(node_58 + node_59));
asm volatile("mul.lo.u64 %0, %1, 52427;"
: "=l"(node_67)
: "l"(node_58 - node_59));
unsigned long long node_69 = node_64 + node_67;
unsigned long long node_70 = node_64 - node_67;

unsigned long long node_397, node_702;
asm volatile("mul.lo.u64 %0, %2, 18446744073709525744;\n\t"
"mul.lo.u64 %1, %2, 53918;"
: "=l"(node_397), "=l"(node_702)
: "l"(node_71));

unsigned long long temp;
asm volatile("mul.lo.u64 %0, %1, 53918;" : "=l"(temp) : "l"(node_72));
node_397 -= temp;
asm volatile("mul.lo.u64 %0, %1, 18446744073709525744;"
: "=l"(temp)
: "l"(node_72));
node_702 += temp;

unsigned long long node_1857, node_1961, node_1865, node_1963;
asm volatile("mul.lo.u64 %0, %2, 395512;\n\t"
"mul.lo.u64 %1, %2, 18446744073709254400;"
: "=l"(node_1857), "=l"(node_1961)
: "l"(node_90));
asm volatile("mul.lo.u64 %0, %2, 18446744073709254400;\n\t"
"mul.lo.u64 %1, %2, 395512;"
: "=l"(node_1865), "=l"(node_1963)
: "l"(node_91));

unsigned long long node_1873, node_1965, node_1869, node_1967;
asm volatile("mul.lo.u64 %0, %2, 18446744073709509368;\n\t"
"mul.lo.u64 %1, %2, 179380;"
: "=l"(node_1873), "=l"(node_1965)
: "l"(node_92));
asm volatile("mul.lo.u64 %0, %2, 179380;\n\t"
"mul.lo.u64 %1, %2, 18446744073709509368;"
: "=l"(node_1869), "=l"(node_1967)
: "l"(node_93));

unsigned long long node_98_mult1, node_98_mult2, node_99_mult1, node_99_mult2;
asm volatile("mul.lo.u64 %0, %2, 353264;\n\t"
"mul.lo.u64 %1, %2, 18446744073709433780;"
: "=l"(node_98_mult1), "=l"(node_98_mult2)
: "l"(node_98));
asm volatile("mul.lo.u64 %0, %2, 18446744073709433780;\n\t"
"mul.lo.u64 %1, %2, 353264;"
: "=l"(node_99_mult1), "=l"(node_99_mult2)
: "l"(node_99));

unsigned long long node_1879, node_1970, node_1915, node_1973;
asm volatile("mul.lo.u64 %0, %2, 35608;\n\t"
"mul.lo.u64 %1, %2, 18446744073709340312;"
: "=l"(node_1879), "=l"(node_1970)
: "l"(node_160));
asm volatile("mul.lo.u64 %0, %2, 18446744073709340312;\n\t"
"mul.lo.u64 %1, %2, 35608;"
: "=l"(node_1915), "=l"(node_1973)
: "l"(node_161));

unsigned long long node_1927, node_1982, node_1921, node_1985;
asm volatile("mul.lo.u64 %0, %2, 18446744073709450808;\n\t"
"mul.lo.u64 %1, %2, 18446744073709494992;"
: "=l"(node_1927), "=l"(node_1982)
: "l"(node_162));
asm volatile("mul.lo.u64 %0, %2, 18446744073709494992;\n\t"
"mul.lo.u64 %1, %2, 18446744073709450808;"
: "=l"(node_1921), "=l"(node_1985)
: "l"(node_163));

unsigned long long node_1957, node_1994, node_1939, node_1997;
asm volatile("mul.lo.u64 %0, %2, 18446744073709515080;\n\t"
"mul.lo.u64 %1, %2, 18446744073709420056;"
: "=l"(node_1957), "=l"(node_1994)
: "l"(node_164));
asm volatile("mul.lo.u64 %0, %2, 18446744073709420056;\n\t"
"mul.lo.u64 %1, %2, 18446744073709515080;"
: "=l"(node_1939), "=l"(node_1997)
: "l"(node_165));

unsigned long long node_1951, node_1988, node_1945, node_1991;
asm volatile("mul.lo.u64 %0, %2, 216536;\n\t"
"mul.lo.u64 %1, %2, 18446744073709505128;"
: "=l"(node_1951), "=l"(node_1988)
: "l"(node_166));
asm volatile("mul.lo.u64 %0, %2, 18446744073709505128;\n\t"
"mul.lo.u64 %1, %2, 216536;"
: "=l"(node_1945), "=l"(node_1991)
: "l"(node_167));

unsigned long long node_2035, node_2038, node_1891, node_2041;
asm volatile("mul.lo.u64 %0, %2, 18446744073709550688;\n\t"
"mul.lo.u64 %1, %2, 18446744073709208752;"
: "=l"(node_2035), "=l"(node_2038)
: "l"(node_176));
asm volatile("mul.lo.u64 %0, %2, 18446744073709208752;\n\t"
"mul.lo.u64 %1, %2, 18446744073709550688;"
: "=l"(node_1891), "=l"(node_2041)
: "l"(node_177));

unsigned long long node_1903, node_1976, node_1897, node_1979;
asm volatile("mul.lo.u64 %0, %2, 115728;\n\t"
"mul.lo.u64 %1, %2, 18446744073709448504;"
: "=l"(node_1903), "=l"(node_1976)
: "l"(node_178));
asm volatile("mul.lo.u64 %0, %2, 18446744073709448504;\n\t"
"mul.lo.u64 %1, %2, 115728;"
: "=l"(node_1897), "=l"(node_1979)
: "l"(node_179));

unsigned long long node_2007, node_2020, node_1909, node_2023;
asm volatile("mul.lo.u64 %0, %2, 18446744073709486416;\n\t"
"mul.lo.u64 %1, %2, 18446744073709283688;"
: "=l"(node_2007), "=l"(node_2020)
: "l"(node_184));
asm volatile("mul.lo.u64 %0, %2, 18446744073709283688;\n\t"
"mul.lo.u64 %1, %2, 18446744073709486416;"
: "=l"(node_1909), "=l"(node_2023)
: "l"(node_185));

unsigned long long node_2013, node_2026, node_1933, node_2029;
asm volatile("mul.lo.u64 %0, %2, 180000;\n\t"
"mul.lo.u64 %1, %2, 18446744073709373568;"
: "=l"(node_2013), "=l"(node_2026)
: "l"(node_227));
asm volatile("mul.lo.u64 %0, %2, 18446744073709373568;\n\t"
"mul.lo.u64 %1, %2, 180000;"
: "=l"(node_1933), "=l"(node_2029)
: "l"(node_228));

unsigned long long node_270_mult1, node_270_mult2, node_271_mult1,
node_271_mult2;
asm volatile("mul.lo.u64 %0, %2, 114800;\n\t"
"mul.lo.u64 %1, %2, 18446744073709105640;"
: "=l"(node_270_mult1), "=l"(node_270_mult2)
: "l"(node_270));
asm volatile("mul.lo.u64 %0, %2, 18446744073709105640;\n\t"
"mul.lo.u64 %1, %2, 114800;"
: "=l"(node_271_mult1), "=l"(node_271_mult2)
: "l"(node_271));

unsigned long long node_1965_plus_1967 = node_1965 + node_1967;
unsigned long long node_1961_plus_1963 = node_1961 + node_1963;
unsigned long long node_1970_plus_1973 = node_1970 + node_1973;
unsigned long long node_1982_plus_1985 = node_1982 + node_1985;
unsigned long long node_1976_plus_1979 = node_1976 + node_1979;
unsigned long long node_1988_plus_1991 = node_1988 + node_1991;
unsigned long long node_1994_plus_1997 = node_1994 + node_1997;
unsigned long long node_2038_plus_2041 = node_2038 + node_2041;

unsigned long long node_403 =
node_1857 - (node_99_mult1 - node_1865 - node_1869 + node_1873);
unsigned long long node_708 = node_1961_plus_1963 - node_1965_plus_1967;
unsigned long long node_897 =
node_1865 + node_98_mult1 - node_1857 - node_1873 - node_1869;
unsigned long long node_1077 =
node_98_mult2 + node_99_mult2 - node_1961_plus_1963 - node_1965_plus_1967;

unsigned long long node_1909_minus_terms =
node_1909 - node_1915 - node_1921 + node_1927;
unsigned long long node_1933_minus_terms =
node_1933 - node_1939 - node_1945 + node_1951;

unsigned long long node_412 =
node_1879 - (node_271_mult1 - node_1891 - node_1897 + node_1903 -
node_1909_minus_terms - node_1933_minus_terms + node_1957);

unsigned long long node_717 =
node_1970_plus_1973 - (node_1976_plus_1979 - node_1982_plus_1985 -
node_1988_plus_1991 + node_1994_plus_1997);

unsigned long long node_1939_plus_terms =
node_1939 + node_2013 - node_1957 - node_1951;
unsigned long long node_1897_minus_terms =
node_1897 - node_1921 - node_1945 + node_1939_plus_terms;

unsigned long long node_906 =
node_1915 + node_2007 - node_1879 - node_1927 - node_1897_minus_terms;

unsigned long long node_2026_plus_terms =
node_2026 + node_2029 - node_1994_plus_1997 - node_1988_plus_1991;

unsigned long long node_1086 = node_2020 + node_2023 - node_1970_plus_1973 -
  node_1982_plus_1985 - node_2026_plus_terms;

unsigned long long node_1237 = node_1909_minus_terms + node_2035 - node_1879 -
  node_1957 - node_1933_minus_terms;

unsigned long long node_1375 = node_1982_plus_1985 + node_2038_plus_2041 -
  node_1970_plus_1973 - node_1994_plus_1997 -
  node_1988_plus_1991;

unsigned long long node_1891_plus_terms =
node_1891 + node_270_mult1 - node_2035 - node_1903;

unsigned long long node_1492 =
node_1921 + node_1891_plus_terms -
(node_1915 + node_2007 - node_1879 - node_1927) - node_1939_plus_terms -
node_1945;

unsigned long long node_2020_plus_terms =
node_2020 + node_2023 - node_1970_plus_1973 - node_1982_plus_1985;

unsigned long long node_1657 = node_270_mult2 + node_271_mult2 -
  node_2038_plus_2041 - node_1976_plus_1979 -
  node_2020_plus_terms - node_2026_plus_terms;

unsigned long long node_86 = node_69 + node_397;
unsigned long long node_88 = node_70 + node_702;
unsigned long long node_87 = node_69 - node_397;
unsigned long long node_89 = node_70 - node_702;

unsigned long long node_152 = node_86 + node_403;
unsigned long long node_154 = node_88 + node_708;
unsigned long long node_156 = node_87 + node_897;
unsigned long long node_158 = node_89 + node_1077;
unsigned long long node_153 = node_86 - node_403;
unsigned long long node_155 = node_88 - node_708;
unsigned long long node_157 = node_87 - node_897;
unsigned long long node_159 = node_89 - node_1077;

output[0] = node_152 + node_412;
output[1] = node_154 + node_717;
output[2] = node_156 + node_906;
output[3] = node_158 + node_1086;
output[4] = node_153 + node_1237;
output[5] = node_155 + node_1375;
output[6] = node_157 + node_1492;
output[7] = node_159 + node_1657;
output[8] = node_152 - node_412;
output[9] = node_154 - node_717;
output[10] = node_156 - node_906;
output[11] = node_158 - node_1086;
output[12] = node_153 - node_1237;
output[13] = node_155 - node_1375;
output[14] = node_157 - node_1492;
output[15] = node_159 - node_1657;
}

#include "table/master_table.hpp"
#include "stark/challenges.hpp"
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include <vector>
#include <iostream>

namespace triton_vm {

// Pure C++ implementation of degree lowering aux columns computation
// Direct translation from generated Rust code
void evaluate_degree_lowering_aux_columns_cpp(
    const std::vector<std::vector<BFieldElement>>& main_data,
    std::vector<std::vector<XFieldElement>>& aux_data,
    const Challenges& challenges) {

    const size_t num_rows = main_data.size();
    if (num_rows < 2) return;

    // Degree lowering columns: 49 to 86 (38 columns)
    // Dual-row substitution (tran constraints)
    // Iterate over rows 0 to num_rows-2

    for (size_t current_row_index = 0; current_row_index < num_rows - 1; ++current_row_index) {
        const size_t next_row_index = current_row_index + 1;
        const auto& current_main_row = main_data[current_row_index];
        const auto& next_main_row = main_data[next_row_index];
        
        // Debug: Print first row computation details for column 49
        const bool debug_first_row = (current_row_index == 0);
        
        // In Rust, current_aux_row starts with columns 0-48, then 49-86 are pushed incrementally
        // In C++, we need to match this exactly: start with 0-48, then push 49-86 incrementally
        std::vector<XFieldElement> current_aux_row;
        current_aux_row.reserve(88);
        
        // Start with columns 0-48 (matching Rust's original_part.row(current_row_index))
        for (size_t i = 0; i < 49 && i < aux_data[current_row_index].size(); ++i) {
            current_aux_row.push_back(aux_data[current_row_index][i]);
        }
        
        // In Rust, next_aux_row is from original_part (columns 0-48 only)
        // Create a view of only columns 0-48 for next_aux_row to match Rust
        std::vector<XFieldElement> next_aux_row_view;
        next_aux_row_view.reserve(49);
        for (size_t i = 0; i < 49 && i < aux_data[next_row_index].size(); ++i) {
            next_aux_row_view.push_back(aux_data[next_row_index][i]);
        }
        const auto& next_aux_row = next_aux_row_view;

        // Constants - use from_raw_u64 to match Rust exactly
        const BFieldElement NEG_ONE = BFieldElement::from_raw_u64(18446744065119617026ULL);
        const BFieldElement CONST_1 = BFieldElement::from_raw_u64(4294967295ULL);
        const BFieldElement CONST_2 = BFieldElement::from_raw_u64(8589934590ULL);
        const BFieldElement CONST_3 = BFieldElement::from_raw_u64(12884901885ULL);
        const BFieldElement CONST_4 = BFieldElement::from_raw_u64(17179869180ULL);
        const BFieldElement CONST_5 = BFieldElement::from_raw_u64(21474836475ULL);
        const BFieldElement CONST_6 = BFieldElement::from_raw_u64(25769803770ULL);
        const BFieldElement CONST_7 = BFieldElement::from_raw_u64(30064771065ULL);
        const BFieldElement CONST_8 = BFieldElement::from_raw_u64(34359738360ULL);
        const BFieldElement CONST_9 = BFieldElement::from_raw_u64(38654705655ULL);

        // section_row[0] -> becomes current_aux_row[49]
        XFieldElement section_row_0 = ((challenges[7])
            + ((NEG_ONE)
                * (((((challenges[16]) * (current_main_row[7]))
                    + ((challenges[17]) * (current_main_row[13])))
                    + ((challenges[18]) * (next_main_row[38])))
                    + ((challenges[19]) * (next_main_row[37])))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_1))))
                        + ((challenges[19]) * (next_main_row[36])))));
        current_aux_row.push_back(section_row_0);

        // section_row[1] -> becomes current_aux_row[50]
        XFieldElement section_row_1 = ((challenges[7])
            + ((NEG_ONE)
                * (((((challenges[16]) * (current_main_row[7]))
                    + ((challenges[17]) * (current_main_row[13])))
                    + ((challenges[18]) * (current_main_row[38])))
                    + ((challenges[19]) * (current_main_row[37])))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_1))))
                        + ((challenges[19]) * (current_main_row[36])))));
        current_aux_row.push_back(section_row_1);

        // section_row[2] -> becomes current_aux_row[51]
        // Uses current_aux_row[49] which is section_row_0
        XFieldElement section_row_2 = (current_aux_row[49])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_2))))
                        + ((challenges[19]) * (next_main_row[35])))));
        current_aux_row.push_back(section_row_2);

        // section_row[3] -> becomes current_aux_row[52]
        // Uses current_aux_row[50] which is section_row_1
        XFieldElement section_row_3 = (current_aux_row[50])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_2))))
                        + ((challenges[19]) * (current_main_row[35])))));
        current_aux_row.push_back(section_row_3);

        // section_row[4] -> becomes current_aux_row[53]
        // Uses current_aux_row[51] which is section_row_2
        XFieldElement section_row_4 = (current_aux_row[51])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_3))))
                        + ((challenges[19]) * (next_main_row[34])))));
        current_aux_row.push_back(section_row_4);

        // section_row[5] -> becomes current_aux_row[54]
        // Uses current_aux_row[52] which is section_row_3
        XFieldElement section_row_5 = (current_aux_row[52])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_3))))
                        + ((challenges[19]) * (current_main_row[34])))));
        current_aux_row.push_back(section_row_5);

        // section_row[6] -> becomes current_aux_row[55]
        // Uses current_aux_row[53] which is section_row_4
        XFieldElement section_row_6 = (current_aux_row[53])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_4))))
                        + ((challenges[19]) * (next_main_row[33])))));
        current_aux_row.push_back(section_row_6);

        // section_row[7] -> becomes current_aux_row[56]
        XFieldElement section_row_7 = ((challenges[8])
            + ((NEG_ONE)
                * (((((current_main_row[7]) * (challenges[20]))
                    + (challenges[23]))
                    + (((next_main_row[22]) + (CONST_1))
                        * (challenges[21])))
                    + ((next_main_row[23]) * (challenges[22])))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((next_main_row[22]) + (CONST_2))
                            * (challenges[21])))
                        + ((next_main_row[24]) * (challenges[22])))));
        current_aux_row.push_back(section_row_7);

        // section_row[8] -> becomes current_aux_row[57]
        XFieldElement section_row_8 = ((challenges[8])
            + ((NEG_ONE)
                * ((((current_main_row[7]) * (challenges[20]))
                    + ((current_main_row[22]) * (challenges[21])))
                    + ((current_main_row[23]) * (challenges[22])))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * ((((current_main_row[7]) * (challenges[20]))
                        + (((current_main_row[22]) + (CONST_1))
                            * (challenges[21])))
                        + ((current_main_row[24]) * (challenges[22])))));
        current_aux_row.push_back(section_row_8);

        // section_row[9] -> becomes current_aux_row[58]
        // Uses current_aux_row[6] (original) and current_aux_row[55] (section_row_6)
        XFieldElement section_row_9 = (current_aux_row[6]) * (current_aux_row[55]);
        current_aux_row.push_back(section_row_9);

        // section_row[10] -> becomes current_aux_row[59]
        // Uses current_aux_row[54] which is section_row_5
        XFieldElement section_row_10 = (current_aux_row[54])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_4))))
                        + ((challenges[19]) * (current_main_row[33])))));
        current_aux_row.push_back(section_row_10);

        // section_row[11] -> becomes current_aux_row[60]
        // Uses current_aux_row[56] which is section_row_7
        XFieldElement section_row_11 = (current_aux_row[56])
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((next_main_row[22]) + (CONST_3))
                            * (challenges[21])))
                        + ((next_main_row[25]) * (challenges[22])))));
        current_aux_row.push_back(section_row_11);

        // section_row[12] -> becomes current_aux_row[61]
        // Uses current_aux_row[57] which is section_row_8
        XFieldElement section_row_12 = (current_aux_row[57])
            * ((challenges[8])
                + ((NEG_ONE)
                    * ((((current_main_row[7]) * (challenges[20]))
                        + (((current_main_row[22]) + (CONST_2))
                            * (challenges[21])))
                        + ((current_main_row[25]) * (challenges[22])))));
        current_aux_row.push_back(section_row_12);

        // section_row[13] -> becomes current_aux_row[62]
        // Uses current_aux_row[60] which is section_row_11
        XFieldElement section_row_13 = (current_aux_row[60])
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((next_main_row[22]) + (CONST_4))
                            * (challenges[21])))
                        + ((next_main_row[26]) * (challenges[22])))));
        current_aux_row.push_back(section_row_13);

        // section_row[14] -> becomes current_aux_row[63]
        // Uses current_aux_row[61] which is section_row_12
        XFieldElement section_row_14 = (current_aux_row[61])
            * ((challenges[8])
                + ((NEG_ONE)
                    * ((((current_main_row[7]) * (challenges[20]))
                        + (((current_main_row[22]) + (CONST_3))
                            * (challenges[21])))
                        + ((current_main_row[26]) * (challenges[22])))));
        current_aux_row.push_back(section_row_14);

        // section_row[15] -> becomes current_aux_row[64]
        // Uses current_aux_row[55] which is section_row_6
        XFieldElement section_row_15 = (((current_aux_row[55])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_5))))
                        + ((challenges[19]) * (next_main_row[32]))))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_6))))
                        + ((challenges[19]) * (next_main_row[31]))))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((next_main_row[38]) + (CONST_7))))
                        + ((challenges[19]) * (next_main_row[30])))));
        current_aux_row.push_back(section_row_15);

        // section_row[16] -> becomes current_aux_row[65]
        // Uses current_aux_row[59] which is section_row_10
        XFieldElement section_row_16 = (((current_aux_row[59])
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_5))))
                        + ((challenges[19]) * (current_main_row[32]))))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_6))))
                        + ((challenges[19]) * (current_main_row[31]))))))
            * ((challenges[7])
                + ((NEG_ONE)
                    * (((((challenges[16]) * (current_main_row[7]))
                        + ((challenges[17]) * (current_main_row[13])))
                        + ((challenges[18])
                            * ((current_main_row[38]) + (CONST_7))))
                        + ((challenges[19]) * (current_main_row[30])))));
        current_aux_row.push_back(section_row_16);

        // section_row[17] -> becomes current_aux_row[66]
        // Uses current_aux_row[6] (original) and current_aux_row[64] (section_row_15)
        XFieldElement section_row_17 = (current_aux_row[6])
            * (((current_aux_row[64])
                * ((challenges[7])
                    + ((NEG_ONE)
                        * (((((challenges[16]) * (current_main_row[7]))
                            + ((challenges[17]) * (current_main_row[13])))
                            + ((challenges[18])
                                * ((next_main_row[38]) + (CONST_8))))
                            + ((challenges[19]) * (next_main_row[29]))))))
                * ((challenges[7])
                    + ((NEG_ONE)
                        * (((((challenges[16]) * (current_main_row[7]))
                            + ((challenges[17]) * (current_main_row[13])))
                            + ((challenges[18])
                                * ((next_main_row[38]) + (CONST_9))))
                            + ((challenges[19]) * (next_main_row[28]))))));
        current_aux_row.push_back(section_row_17);

        // section_row[18] -> becomes current_aux_row[67]
        // Uses current_aux_row[6] (original) and current_aux_row[65] (section_row_16)
        XFieldElement section_row_18 = (current_aux_row[6])
            * (((current_aux_row[65])
                * ((challenges[7])
                    + ((NEG_ONE)
                        * (((((challenges[16]) * (current_main_row[7]))
                            + ((challenges[17]) * (current_main_row[13])))
                            + ((challenges[18])
                                * ((current_main_row[38]) + (CONST_8))))
                            + ((challenges[19]) * (current_main_row[29]))))))
                * ((challenges[7])
                    + ((NEG_ONE)
                        * (((((challenges[16]) * (current_main_row[7]))
                            + ((challenges[17]) * (current_main_row[13])))
                            + ((challenges[18])
                                * ((current_main_row[38]) + (CONST_9))))
                            + ((challenges[19]) * (current_main_row[28]))))));
        current_aux_row.push_back(section_row_18);

        // section_row[19] -> becomes current_aux_row[68]
        // Uses current_aux_row[7] (original), current_aux_row[62] (section_row_13), next_aux_row[7]
        XFieldElement section_row_19 = (current_main_row[202])
            * ((next_aux_row[7])
                + ((NEG_ONE)
                    * ((current_aux_row[7])
                        * ((current_aux_row[62])
                            * ((challenges[8])
                                + ((NEG_ONE)
                                    * (((((current_main_row[7]) * (challenges[20]))
                                        + (challenges[23]))
                                        + (((next_main_row[22]) + (CONST_5))
                                            * (challenges[21])))
                                        + ((next_main_row[27]) * (challenges[22])))))))));
        current_aux_row.push_back(section_row_19);

        // section_row[20] -> becomes current_aux_row[69]
        // Uses current_aux_row[7] (original), current_aux_row[63] (section_row_14), next_aux_row[7]
        XFieldElement section_row_20 = (current_main_row[202])
            * ((next_aux_row[7])
                + ((NEG_ONE)
                    * ((current_aux_row[7])
                        * ((current_aux_row[63])
                            * ((challenges[8])
                                + ((NEG_ONE)
                                    * ((((current_main_row[7]) * (challenges[20]))
                                        + (((current_main_row[22]) + (CONST_4))
                                            * (challenges[21])))
                                        + ((current_main_row[27]) * (challenges[22])))))))));
        current_aux_row.push_back(section_row_20);

        // section_row[21] -> becomes current_aux_row[70]
        XFieldElement section_row_21 = ((((challenges[8])
            + ((NEG_ONE)
                * (((((current_main_row[7]) * (challenges[20]))
                    + (challenges[23]))
                    + ((current_main_row[29]) * (challenges[21])))
                    + ((current_main_row[39]) * (challenges[22])))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[29]) + (CONST_1))
                            * (challenges[21])))
                        + ((current_main_row[40]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[29]) + (CONST_2))
                            * (challenges[21])))
                        + ((current_main_row[41]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[29]) + (CONST_3))
                            * (challenges[21])))
                        + ((current_main_row[42]) * (challenges[22])))));
        current_aux_row.push_back(section_row_21);

        // section_row[22] -> becomes current_aux_row[71]
        XFieldElement section_row_22 = ((((challenges[8])
            + ((NEG_ONE)
                * (((((current_main_row[7]) * (challenges[20]))
                    + (challenges[23]))
                    + ((current_main_row[22]) * (challenges[21])))
                    + ((current_main_row[39]) * (challenges[22])))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[22]) + (CONST_1))
                            * (challenges[21])))
                        + ((current_main_row[40]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[22]) + (CONST_2))
                            * (challenges[21])))
                        + ((current_main_row[41]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + ((current_main_row[23]) * (challenges[21])))
                        + ((current_main_row[42]) * (challenges[22])))));
        current_aux_row.push_back(section_row_22);

        // section_row[23] -> becomes current_aux_row[72]
        XFieldElement section_row_23 = ((((challenges[8])
            + ((NEG_ONE)
                * (((((current_main_row[7]) * (challenges[20]))
                    + (challenges[23]))
                    + ((current_main_row[22]) * (challenges[21])))
                    + ((current_main_row[39]) * (challenges[22])))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + ((current_main_row[23]) * (challenges[21])))
                        + ((current_main_row[40]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[23]) + (CONST_1))
                            * (challenges[21])))
                        + ((current_main_row[41]) * (challenges[22]))))))
            * ((challenges[8])
                + ((NEG_ONE)
                    * (((((current_main_row[7]) * (challenges[20]))
                        + (challenges[23]))
                        + (((current_main_row[23]) + (CONST_2))
                            * (challenges[21])))
                        + ((current_main_row[42]) * (challenges[22])))));
        current_aux_row.push_back(section_row_23);

        // section_row[24] -> becomes current_aux_row[73]
        // Uses current_aux_row[6] (original), next_aux_row[6]
        XFieldElement section_row_24 = (current_main_row[195])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6])
                        * ((challenges[7])
                            + ((NEG_ONE)
                                * (((((challenges[16]) * (current_main_row[7]))
                                    + ((challenges[17]) * (current_main_row[13])))
                                    + ((challenges[18]) * (next_main_row[38])))
                                    + ((challenges[19]) * (next_main_row[37]))))))));
        current_aux_row.push_back(section_row_24);

        // section_row[25] -> becomes current_aux_row[74]
        // Uses current_aux_row[6] (original), current_aux_row[49] (section_row_0), next_aux_row[6]
        XFieldElement section_row_25 = (current_main_row[196])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[49]))));
        current_aux_row.push_back(section_row_25);

        // section_row[26] -> becomes current_aux_row[75]
        // Uses current_aux_row[6] (original), current_aux_row[51] (section_row_2), next_aux_row[6]
        XFieldElement section_row_26 = (current_main_row[198])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[51]))));
        current_aux_row.push_back(section_row_26);

        // section_row[27] -> becomes current_aux_row[76]
        // Uses current_aux_row[6] (original), current_aux_row[53] (section_row_4), next_aux_row[6]
        XFieldElement section_row_27 = (current_main_row[200])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[53]))));
        current_aux_row.push_back(section_row_27);

        // section_row[28] -> becomes current_aux_row[77]
        // Uses current_aux_row[6] (original), next_aux_row[6]
        XFieldElement section_row_28 = (current_main_row[195])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6])
                        * ((challenges[7])
                            + ((NEG_ONE)
                                * (((((challenges[16]) * (current_main_row[7]))
                                    + ((challenges[17]) * (current_main_row[13])))
                                    + ((challenges[18]) * (current_main_row[38])))
                                    + ((challenges[19]) * (current_main_row[37]))))))));
        current_aux_row.push_back(section_row_28);

        // section_row[29] -> becomes current_aux_row[78]
        // Uses current_aux_row[6] (original), current_aux_row[50] (section_row_1), next_aux_row[6]
        XFieldElement section_row_29 = (current_main_row[196])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[50]))));
        current_aux_row.push_back(section_row_29);

        // section_row[30] -> becomes current_aux_row[79]
        // Uses current_aux_row[6] (original), current_aux_row[52] (section_row_3), next_aux_row[6]
        XFieldElement section_row_30 = (current_main_row[198])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[52]))));
        current_aux_row.push_back(section_row_30);

        // section_row[31] -> becomes current_aux_row[80]
        // Uses current_aux_row[6] (original), current_aux_row[54] (section_row_5), next_aux_row[6]
        XFieldElement section_row_31 = (current_main_row[200])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[54]))));
        current_aux_row.push_back(section_row_31);

        // section_row[32] -> becomes current_aux_row[81]
        // Uses current_aux_row[6] (original), current_aux_row[59] (section_row_10), next_aux_row[6]
        XFieldElement section_row_32 = (current_main_row[202])
            * ((next_aux_row[6])
                + ((NEG_ONE)
                    * ((current_aux_row[6]) * (current_aux_row[59]))));
        current_aux_row.push_back(section_row_32);

        // section_row[33] -> becomes current_aux_row[82]
        // Uses current_aux_row[7] (original), current_aux_row[71] (section_row_22)
        XFieldElement section_row_33 = (current_aux_row[7])
            * (((current_aux_row[71])
                * ((challenges[8])
                    + ((NEG_ONE)
                        * (((((current_main_row[7]) * (challenges[20]))
                            + (challenges[23]))
                            + (((current_main_row[23]) + (CONST_1))
                                * (challenges[21])))
                            + ((current_main_row[43]) * (challenges[22]))))))
                * ((challenges[8])
                    + ((NEG_ONE)
                        * (((((current_main_row[7]) * (challenges[20]))
                            + (challenges[23]))
                            + (((current_main_row[23]) + (CONST_2))
                                * (challenges[21])))
                            + ((current_main_row[44]) * (challenges[22]))))));
        current_aux_row.push_back(section_row_33);

        // section_row[34] -> becomes current_aux_row[83]
        // Uses next_aux_row[21], current_aux_row[21] (original)
        XFieldElement section_row_34 = ((((next_aux_row[21])
            + ((NEG_ONE) * (current_aux_row[21])))
            * ((challenges[11])
                + ((NEG_ONE)
                    * ((next_main_row[50])
                        + ((NEG_ONE) * (current_main_row[50]))))))
            + (NEG_ONE))
            * ((CONST_1)
                + ((NEG_ONE)
                    * (((next_main_row[52])
                        + ((NEG_ONE) * (current_main_row[52])))
                        * (current_main_row[54]))));
        current_aux_row.push_back(section_row_34);

        // section_row[35] -> becomes current_aux_row[84]
        // Uses current_aux_row[7] (original), current_aux_row[70] (section_row_21), next_aux_row[7]
        XFieldElement section_row_35 = (current_main_row[301])
            * (((current_aux_row[7])
                * ((current_aux_row[70])
                    * ((challenges[8])
                        + ((NEG_ONE)
                            * (((((current_main_row[7]) * (challenges[20]))
                                + (challenges[23]))
                                + (((current_main_row[29]) + (CONST_4))
                                    * (challenges[21])))
                                + ((current_main_row[43]) * (challenges[22])))))))
                + ((NEG_ONE) * (next_aux_row[7])));
        current_aux_row.push_back(section_row_35);

        // section_row[36] -> becomes current_aux_row[85]
        // Uses current_aux_row[7] (original), current_aux_row[56,60,62] (section_row 7,11,13), 
        // current_aux_row[68] (section_row_19), next_aux_row[7]
        XFieldElement section_row_36 = (current_main_row[220])
            * ((((((current_main_row[195])
                * ((next_aux_row[7])
                    + ((NEG_ONE)
                        * ((current_aux_row[7])
                            * ((challenges[8])
                                + ((NEG_ONE)
                                    * (((((current_main_row[7]) * (challenges[20]))
                                        + (challenges[23]))
                                        + (((next_main_row[22]) + (CONST_1))
                                            * (challenges[21])))
                                        + ((next_main_row[23]) * (challenges[22])))))))))
                + ((current_main_row[196])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[56]))))))
                + ((current_main_row[198])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[60]))))))
                + ((current_main_row[200])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[62]))))))
                + (current_aux_row[68]));
        current_aux_row.push_back(section_row_36);

        // section_row[37] -> becomes current_aux_row[86]
        // Uses current_aux_row[7] (original), current_aux_row[57,61,63] (section_row 8,12,14),
        // current_aux_row[69] (section_row_20), next_aux_row[7]
        XFieldElement section_row_37 = (current_main_row[228])
            * ((((((current_main_row[195])
                * ((next_aux_row[7])
                    + ((NEG_ONE)
                        * ((current_aux_row[7])
                            * ((challenges[8])
                                + ((NEG_ONE)
                                    * ((((current_main_row[7]) * (challenges[20]))
                                        + ((current_main_row[22]) * (challenges[21])))
                                        + ((current_main_row[23]) * (challenges[22])))))))))
                + ((current_main_row[196])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[57]))))))
                + ((current_main_row[198])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[61]))))))
                + ((current_main_row[200])
                    * ((next_aux_row[7])
                        + ((NEG_ONE)
                            * ((current_aux_row[7]) * (current_aux_row[63]))))))
                + (current_aux_row[69]));
        current_aux_row.push_back(section_row_37);

        // Update aux_data with computed row
        aux_data[current_row_index] = current_aux_row;
    }
}

} // namespace triton_vm
#include "test_data_loader.hpp"
#include <fstream>
#include <stdexcept>

namespace triton_vm {

TestDataLoader::TestDataLoader(const std::string& test_data_dir)
    : test_data_dir_(test_data_dir) {}

std::string TestDataLoader::file_path(const std::string& filename) const {
    return test_data_dir_ + "/" + filename;
}

nlohmann::json TestDataLoader::load_json(const std::string& filename) const {
    std::ifstream file(file_path(filename));
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path(filename));
    }
    return nlohmann::json::parse(file);
}

nlohmann::json TestDataLoader::load_step(int step_num, const std::string& name) const {
    std::string filename = (step_num < 10 ? "0" : "") + std::to_string(step_num) + "_" + name + ".json";
    return load_json(filename);
}

TestDataLoader::TraceExecutionData TestDataLoader::load_trace_execution() const {
    auto json = load_step(1, "trace_execution");
    
    TraceExecutionData data;
    data.processor_trace_height = json["processor_trace_height"];
    data.processor_trace_width = json["processor_trace_width"];
    data.padded_height = json["padded_height"];
    
    if (json.contains("public_output")) {
        data.public_output = json["public_output"].get<std::vector<uint64_t>>();
    }
    
    return data;
}

TestDataLoader::ParametersData TestDataLoader::load_parameters() const {
    auto json = load_step(2, "parameters");
    
    ParametersData data;
    data.padded_height = json["padded_height"];
    data.log2_padded_height = json["log2_padded_height"];
    data.fri_domain_length = json["fri_domain_length"];
    data.trace_domain_length = json["trace_domain_length"];
    data.randomized_trace_domain_length = json["randomized_trace_domain_length"];
    data.quotient_domain_length = json["quotient_domain_length"];
    
    return data;
}

TestDataLoader::MainTableCreateData TestDataLoader::load_main_table_create() const {
    auto json = load_step(3, "main_tables_create");
    
    MainTableCreateData data;
    data.num_columns = json["num_columns"];
    
    auto shape = json["trace_table_shape"];
    data.trace_table_shape = {shape[0], shape[1]};
    
    if (json.contains("first_row")) {
        data.first_row = json["first_row"].get<std::vector<uint64_t>>();
    }
    
    return data;
}

TestDataLoader::MerkleRootData TestDataLoader::load_main_tables_merkle() const {
    auto json = load_step(6, "main_tables_merkle");
    MerkleRootData data;
    data.merkle_root_hex = json["merkle_root"];
    return data;
}

TestDataLoader::MerkleRootData TestDataLoader::load_aux_tables_merkle() const {
    auto json = load_step(9, "aux_tables_merkle");
    MerkleRootData data;
    data.merkle_root_hex = json["aux_merkle_root"];
    return data;
}

TestDataLoader::MerkleRootData TestDataLoader::load_quotient_merkle() const {
    auto json = load_step(13, "quotient_merkle");
    MerkleRootData data;
    data.merkle_root_hex = json["quotient_merkle_root"];
    return data;
}

TestDataLoader::ChallengesData TestDataLoader::load_fiat_shamir_challenges() const {
    auto json = load_step(7, "fiat_shamir_challenges");
    
    ChallengesData data;
    data.num_challenges = json["challenges_sample_count"];
    
    // XFieldElements are stored as strings like "(coeff2·x² + coeff1·x + coeff0)"
    // We need to parse them
    if (json.contains("challenge_values")) {
        for (const auto& challenge_str : json["challenge_values"]) {
            std::string s = challenge_str.get<std::string>();
            
            // Extract coefficients using simple parsing
            // Format: "(coeff2·x² + coeff1·x + coeff0)"
            std::array<std::string, 3> xfe;
            
            // Find all numbers in the string
            std::vector<std::string> nums;
            std::string current_num;
            for (char c : s) {
                if (c >= '0' && c <= '9') {
                    current_num += c;
                } else if (!current_num.empty()) {
                    nums.push_back(current_num);
                    current_num.clear();
                }
            }
            if (!current_num.empty()) {
                nums.push_back(current_num);
            }
            
            // Format is: coeff2·x² + coeff1·x + coeff0
            if (nums.size() >= 3) {
                xfe[0] = nums[2];  // constant (coeff0)
                xfe[1] = nums[1];  // x coeff (coeff1)
                xfe[2] = nums[0];  // x² coeff (coeff2)
            }
            
            data.challenges.push_back(xfe);
        }
    }
    
    return data;
}

TestDataLoader::AuxTableCreateData TestDataLoader::load_aux_table_create() const {
    auto json = load_step(7, "aux_tables_create");
    
    AuxTableCreateData data;
    auto shape = json["aux_table_shape"];
    data.aux_table_shape = {shape[0], shape[1]};
    data.num_columns = json["num_columns"];
    
    return data;
}

TestDataLoader::QuotientCalculationData TestDataLoader::load_quotient_calculation() const {
    auto json = load_step(10, "quotient_calculation");
    
    QuotientCalculationData data;
    data.cached = json.value("cached", false);
    if (json.contains("note")) {
        data.note = json["note"].get<std::string>();
    }
    
    return data;
}

TestDataLoader::QuotientHashRowsData TestDataLoader::load_quotient_hash_rows() const {
    auto json = load_step(12, "quotient_hash_rows");
    
    QuotientHashRowsData data;
    data.num_quotient_segment_digests = json["num_quotient_segment_digests"];
    
    return data;
}

TestDataLoader::OutOfDomainRowsData TestDataLoader::load_out_of_domain_rows() const {
    auto json = load_step(14, "out_of_domain_rows");
    
    OutOfDomainRowsData data;
    data.out_of_domain_point_curr_row = json["out_of_domain_point_curr_row"].get<std::string>();
    data.out_of_domain_point_next_row = json["out_of_domain_point_next_row"].get<std::string>();
    
    return data;
}

TestDataLoader::LinearCombinationData TestDataLoader::load_linear_combination() const {
    auto json = load_step(15, "linear_combination");
    
    LinearCombinationData data;
    data.combination_codeword_length = json["combination_codeword_length"];
    
    return data;
}

TestDataLoader::DeepData TestDataLoader::load_deep() const {
    auto json = load_step(16, "deep");
    
    DeepData data;
    data.deep_codeword_length = json["deep_codeword_length"];
    
    return data;
}

TestDataLoader::CombinedDeepPolynomialData TestDataLoader::load_combined_deep_polynomial() const {
    auto json = load_step(17, "combined_deep_polynomial");
    
    CombinedDeepPolynomialData data;
    data.fri_combination_codeword_length = json["fri_combination_codeword_length"];
    
    return data;
}

TestDataLoader::FriData TestDataLoader::load_fri() const {
    auto json = load_step(18, "fri");
    
    FriData data;
    data.num_revealed_indices = json["num_revealed_indices"];
    
    return data;
}

TestDataLoader::OpenTraceLeafsData TestDataLoader::load_open_trace_leafs() const {
    auto json = load_step(19, "open_trace_leafs");
    
    OpenTraceLeafsData data;
    data.num_revealed_main_rows = json["num_revealed_main_rows"];
    data.num_revealed_aux_rows = json["num_revealed_aux_rows"];
    data.num_revealed_quotient_rows = json["num_revealed_quotient_rows"];
    
    return data;
}

TestDataLoader::TraceExecutionSampleData TestDataLoader::load_trace_execution_sample() const {
    auto json = load_step(1, "trace_execution_sample");
    
    TraceExecutionSampleData data;
    if (json.contains("first_row")) {
        data.first_row = json["first_row"].get<std::vector<uint64_t>>();
    }
    if (json.contains("last_row")) {
        data.last_row = json["last_row"].get<std::vector<uint64_t>>();
    }
    
    return data;
}

TestDataLoader::MainTableLdeData TestDataLoader::load_main_table_lde_metadata() const {
    auto json = load_step(5, "main_tables_lde");
    
    MainTableLdeData data;
    if (json.contains("lde_table_shape")) {
        auto shape = json["lde_table_shape"];
        data.lde_table_shape = {shape[0], shape[1]};
    }
    
    return data;
}

TestDataLoader::AuxTableLdeData TestDataLoader::load_aux_table_lde_metadata() const {
    auto json = load_step(8, "aux_tables_lde");
    
    AuxTableLdeData data;
    if (json.contains("aux_lde_table_shape")) {
        auto shape = json["aux_lde_table_shape"];
        data.aux_lde_table_shape = {shape[0], shape[1]};
    }
    
    return data;
}

TestDataLoader::QuotientLdeData TestDataLoader::load_quotient_lde_metadata() const {
    auto json = load_step(11, "quotient_lde");
    
    QuotientLdeData data;
    if (json.contains("quotient_segments_shape")) {
        auto shape = json["quotient_segments_shape"];
        data.quotient_segments_shape = {shape[0], shape[1]};
    }
    
    return data;
}

TestDataLoader::MerkleRootWithLeafsData TestDataLoader::load_main_tables_merkle_full() const {
    auto json = load_step(6, "main_tables_merkle");
    
    MerkleRootWithLeafsData data;
    data.merkle_root_hex = json["merkle_root"];
    data.num_leafs = json["num_leafs"];
    
    return data;
}

TestDataLoader::MerkleRootWithLeafsData TestDataLoader::load_aux_tables_merkle_full() const {
    auto json = load_step(9, "aux_tables_merkle");
    
    MerkleRootWithLeafsData data;
    data.merkle_root_hex = json["aux_merkle_root"];
    data.num_leafs = json["num_leafs"];
    
    return data;
}

TestDataLoader::MerkleRootWithLeafsData TestDataLoader::load_quotient_merkle_full() const {
    auto json = load_step(13, "quotient_merkle");
    
    MerkleRootWithLeafsData data;
    data.merkle_root_hex = json["quotient_merkle_root"];
    data.num_leafs = json["num_leafs"];
    
    return data;
}

} // namespace triton_vm


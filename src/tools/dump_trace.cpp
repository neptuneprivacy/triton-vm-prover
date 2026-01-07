/**
 * Trace Dump Utility
 * 
 * Dumps execution trace (AET) and related data for comparison with Rust
 * This allows verification of early components even for large inputs
 * that require too much memory for full GPU proof generation.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include "vm/vm.hpp"
#include "vm/aet.hpp"
#include "stark.hpp"
#include "types/b_field_element.hpp"
#include "stark/domains.hpp"
#include <sstream>

using namespace triton_vm;

void dump_trace_data(
    const std::string& output_dir,
    const Program& program,
    const std::vector<BFieldElement>& public_input,
    const VM::TraceResult& trace_result
) {
    const AlgebraicExecutionTrace& aet = trace_result.aet;
    
    // Create output directory
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());
    
    // 1. Dump trace execution summary
    {
        std::ofstream f(output_dir + "/01_trace_execution.json");
        f << "{\n";
        f << "  \"processor_trace_height\": " << aet.processor_trace_height() << ",\n";
        f << "  \"processor_trace_width\": " << aet.processor_trace()[0].size() << ",\n";
        f << "  \"public_output\": [";
        for (size_t i = 0; i < trace_result.output.size(); ++i) {
            if (i > 0) f << ", ";
            f << trace_result.output[i].value();
        }
        f << "],\n";
        f << "  \"padded_height\": " << aet.padded_height() << "\n";
        f << "}\n";
    }
    
    // 2. Dump sample processor trace rows
    {
        std::ofstream f(output_dir + "/01_trace_execution_sample.json");
        f << "{\n";
        f << "  \"first_row\": [";
        if (aet.processor_trace_height() > 0) {
            const auto& row = aet.processor_trace()[0];
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) f << ", ";
                f << row[i].value();
            }
        }
        f << "],\n";
        f << "  \"last_row\": [";
        if (aet.processor_trace_height() > 1) {
            const auto& row = aet.processor_trace()[aet.processor_trace_height() - 1];
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) f << ", ";
                f << row[i].value();
            }
        }
        f << "]\n";
        f << "}\n";
    }
    
    // 3. Dump domain setup
    {
        Stark stark = Stark::default_stark();
        const size_t padded_height = aet.padded_height();
        const size_t rand_trace_len = stark.randomized_trace_len(padded_height);
        const size_t fri_domain_length = stark.fri_expansion_factor() * rand_trace_len;
        ArithmeticDomain fri_domain = ArithmeticDomain::of_length(fri_domain_length)
            .with_offset(BFieldElement::generator());
        
        ProverDomains domains = ProverDomains::derive(
            padded_height,
            stark.num_trace_randomizers(),
            fri_domain,
            stark.max_degree(padded_height)
        );
        
        std::ofstream f(output_dir + "/02_domains.json");
        f << "{\n";
        f << "  \"padded_height\": " << padded_height << ",\n";
        f << "  \"trace_domain\": {\n";
        f << "    \"length\": " << domains.trace.length << ",\n";
        f << "    \"offset\": " << domains.trace.offset.value() << ",\n";
        f << "    \"generator\": " << domains.trace.generator.value() << "\n";
        f << "  },\n";
        f << "  \"quotient_domain\": {\n";
        f << "    \"length\": " << domains.quotient.length << ",\n";
        f << "    \"offset\": " << domains.quotient.offset.value() << ",\n";
        f << "    \"generator\": " << domains.quotient.generator.value() << "\n";
        f << "  },\n";
        f << "  \"fri_domain\": {\n";
        f << "    \"length\": " << domains.fri.length << ",\n";
        f << "    \"offset\": " << domains.fri.offset.value() << ",\n";
        f << "    \"generator\": " << domains.fri.generator.value() << "\n";
        f << "  }\n";
        f << "}\n";
    }
    
    // 4. Dump table lengths
    {
        std::ofstream f(output_dir + "/03_table_lengths.json");
        f << "{\n";
        f << "  \"table_lengths\": [\n";
        for (size_t i = 0; i < 9; ++i) {
            if (i > 0) f << ",\n";
            f << "    " << aet.height_of_table(i);
        }
        f << "\n  ]\n";
        f << "}\n";
    }
    
    std::cout << "Trace data dumped to: " << output_dir << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <program.tasm> <input> <output_dir>\n";
        std::cerr << "Example: " << argv[0] << " spin_input21.tasm 21 /tmp/cpp_trace_21\n";
        return 1;
    }
    
    std::string program_path = argv[1];
    std::string input_str = argv[2];
    std::string output_dir = argv[3];
    
    // Parse input
    std::vector<uint64_t> input;
    std::istringstream iss(input_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        input.push_back(std::stoull(token));
    }
    
    // Convert to BFieldElement
    std::vector<BFieldElement> public_input;
    for (uint64_t val : input) {
        public_input.push_back(BFieldElement(val));
    }
    
    // Load program
    Program program = Program::from_file(program_path);
    
    // Execute and generate trace
    std::cout << "Generating trace for input: " << input_str << std::endl;
    auto trace_result = VM::trace_execution(program, public_input);
    
    std::cout << "Processor trace: " << trace_result.aet.processor_trace_height() 
              << " rows x " << trace_result.aet.processor_trace()[0].size() << " cols" << std::endl;
    
    // Dump trace data
    dump_trace_data(output_dir, program, public_input, trace_result);
    
    return 0;
}


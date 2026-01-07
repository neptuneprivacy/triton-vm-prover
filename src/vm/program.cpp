#include "vm/program.hpp"
#include "hash/tip5.hpp"
#include "table/extend_helpers.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <iostream>

namespace triton_vm {

Program Program::from_code(const std::string& code) {
    Program program;
    program.parse_code(code);
    return program;
}

Program Program::from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_code(buffer.str());
}

Program Program::from_bwords(const uint64_t* bwords_data, size_t bwords_len) {
    Program program;
    program.bwords_.reserve(bwords_len);
    for (size_t i = 0; i < bwords_len; ++i) {
        program.bwords_.push_back(BFieldElement(bwords_data[i]));
    }
    // Note: We don't populate program_ (TritonProgram) or label_bword_map_
    // because from_bwords is only used for hashing, not for execution
    return program;
}

void Program::parse_code(const std::string& code) {
    // Two-pass parser: first pass collects all labels, second pass resolves calls
    // IMPORTANT: TASM allows multiple instructions on a single line
    // We need to parse all instructions on each line, not just the first one
    
    // First pass: collect all labels and count bwords (opcode + optional arg) to get label addresses in bword space.
    std::istringstream iss1(code);
    std::string line1;
    size_t bword_count = 0;
    std::unordered_map<std::string, size_t> label_map;
    
    // Opcodes that occupy 2 bwords in the program encoding.
    const std::unordered_set<std::string> two_word_opcodes = {
        "pop","push","divine","pick","place","dup","swap","call",
        "read_mem","write_mem","addi","read_io","write_io"
    };
    const std::unordered_set<std::string> one_word_opcodes = {
        "halt","nop","skiz","return","recurse","recurse_or_return","assert",
        "add","mul","invert","eq","split","lt","and","xor","log_2_floor","pow","div_mod","pop_count",
        "hash","assert_vector","sponge_init","sponge_absorb","sponge_absorb_mem","sponge_squeeze",
        "xx_add","xx_mul","x_invert","xb_mul","merkle_step","merkle_step_mem","xx_dot_step","xb_dot_step",
        "write_mem" // write_mem itself is 2-word in ISA; kept in set above; harmless if duplicated
    };
    
    auto count_bwords_on_line = [&](std::string line) -> size_t {
        // Strip comments (both `//` and `#` styles).
        if (auto pos = line.find("//"); pos != std::string::npos) line = line.substr(0, pos);
        if (auto pos = line.find('#'); pos != std::string::npos) line = line.substr(0, pos);
        // Trim
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty()) return 0;
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) return 0;

        std::istringstream line_stream(line);
        std::string token;
        size_t count = 0;
        bool skipping_hint = false;
        while (line_stream >> token) {
            std::string lower = token;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "hint" || lower == "error_id") {
                skipping_hint = true;
                continue;
            }
            if (skipping_hint) {
                continue;
            }
            if (!lower.empty() && lower.back() == ':') {
                continue;
            }
            if (two_word_opcodes.count(lower)) {
                count += 2;
            } else if (one_word_opcodes.count(lower)) {
                count += 1;
            }
        }
        return count;
    };
    
    while (std::getline(iss1, line1)) {
        // Strip comments
        if (auto pos = line1.find("//"); pos != std::string::npos) line1 = line1.substr(0, pos);
        if (auto pos = line1.find('#'); pos != std::string::npos) line1 = line1.substr(0, pos);
        line1.erase(0, line1.find_first_not_of(" \t"));
        if (line1.empty()) continue;
        line1.erase(line1.find_last_not_of(" \t") + 1);
        if (line1.empty()) continue;
        if (line1.back() == ':') {
            std::string label = line1.substr(0, line1.size() - 1);
            label_map[label] = bword_count; // Label points to next bword address
        } else {
            bword_count += count_bwords_on_line(line1);
        }
    }
    
    // Second pass: parse all instructions on each line
    std::istringstream iss(code);
    std::string line;
    
    // Helper function to parse a single instruction from a stream
    auto parse_single_instruction = [&](std::istringstream& line_stream) -> std::optional<TritonInstruction> {
        auto read_optional_size_t = [&](size_t default_value) -> size_t {
            std::streampos pos = line_stream.tellg();
            size_t v = default_value;
            if (line_stream >> v) {
                return v;
            }
            // Failed to parse a number: restore stream state/position so the next token can be parsed as opcode.
            line_stream.clear();
            line_stream.seekg(pos);
            return default_value;
        };

        std::string opcode;
        if (!(line_stream >> opcode)) {
            return std::nullopt; // No more tokens
        }
        
        // Convert to lowercase
        std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::tolower);
        
        // Skip hints and error_id tokens - these are metadata, not instructions
        if (opcode == "hint" || opcode == "error_id") {
            // Skip rest of hint/error_id tokens until we find another instruction or end of line
            std::string dummy;
            while (line_stream >> dummy) {
                // Check if this might be an instruction
                std::string lower_dummy = dummy;
                std::transform(lower_dummy.begin(), lower_dummy.end(), lower_dummy.begin(), ::tolower);
                if (lower_dummy == "push" || lower_dummy == "pop" || lower_dummy == "add" || 
                    lower_dummy == "mul" || lower_dummy == "halt" || lower_dummy == "nop" ||
                    lower_dummy == "dup" || lower_dummy == "swap" || lower_dummy == "read_io" ||
                    lower_dummy == "write_io" || lower_dummy == "call" || lower_dummy == "return" ||
                    lower_dummy == "assert" || lower_dummy == "lt" || lower_dummy == "eq" ||
                    lower_dummy == "addi" || lower_dummy == "pow" || lower_dummy == "sponge_init" ||
                    lower_dummy == "sponge_squeeze" || lower_dummy == "write_mem" ||
                    lower_dummy == "split" || lower_dummy == "skiz" || lower_dummy == "recurse") {
                    // Found an instruction, parse it
                    opcode = lower_dummy;
                    break; // Exit the skip loop and continue with instruction parsing
                }
                if (dummy.find('#') != std::string::npos) break; // Hit a comment
            }
            // If we didn't find an instruction, return nullopt
            if (opcode == "hint" || opcode == "error_id") {
                return std::nullopt;
            }
            // Otherwise, continue with the instruction we found
        }
        
        TritonInstruction instr;
        
        if (opcode == "push") {
            int64_t value;
            if (line_stream >> value) {
                instr.type = AnInstruction::Push;
                if (value >= 0) {
                    instr.bfield_arg = BFieldElement(static_cast<uint64_t>(value));
                } else {
                    uint64_t abs = static_cast<uint64_t>(-value);
                    uint64_t mod = abs % BFieldElement::MODULUS;
                    instr.bfield_arg = (mod == 0) ? BFieldElement::zero()
                                                  : BFieldElement(BFieldElement::MODULUS - mod);
                }
            } else {
                return std::nullopt; // Invalid push
            }
        }
        else if (opcode == "pop") {
            size_t n = read_optional_size_t(1);
            instr.type = AnInstruction::Pop;
            switch (n) {
                case 1: instr.num_words_arg = NumberOfWords::N1; break;
                case 2: instr.num_words_arg = NumberOfWords::N2; break;
                case 3: instr.num_words_arg = NumberOfWords::N3; break;
                case 4: instr.num_words_arg = NumberOfWords::N4; break;
                case 5: instr.num_words_arg = NumberOfWords::N5; break;
                default: instr.num_words_arg = NumberOfWords::N1; break;
            }
        }
        else if (opcode == "add") {
            instr.type = AnInstruction::Add;
        }
        else if (opcode == "mul") {
            instr.type = AnInstruction::Mul;
        }
        else if (opcode == "halt") {
            instr.type = AnInstruction::Halt;
        }
        else if (opcode == "nop") {
            instr.type = AnInstruction::Nop;
        }
        else if (opcode == "dup") {
            size_t depth = read_optional_size_t(0);
            instr.type = AnInstruction::Dup;
            if (depth <= 15) {
                instr.op_stack_arg = static_cast<OpStackElement>(depth);
            } else {
                instr.op_stack_arg = OpStackElement::ST0;
            }
        }
        else if (opcode == "swap") {
            size_t depth = read_optional_size_t(1);
            instr.type = AnInstruction::Swap;
            if (depth <= 15) {
                instr.op_stack_arg = static_cast<OpStackElement>(depth);
            } else {
                instr.op_stack_arg = OpStackElement::ST1;
            }
        }
        else if (opcode == "read_io") {
            size_t n = read_optional_size_t(1);
            instr.type = AnInstruction::ReadIo;
            switch (n) {
                case 1: instr.num_words_arg = NumberOfWords::N1; break;
                case 2: instr.num_words_arg = NumberOfWords::N2; break;
                case 3: instr.num_words_arg = NumberOfWords::N3; break;
                case 4: instr.num_words_arg = NumberOfWords::N4; break;
                case 5: instr.num_words_arg = NumberOfWords::N5; break;
                default: instr.num_words_arg = NumberOfWords::N1; break;
            }
        }
        else if (opcode == "write_io") {
            size_t n = read_optional_size_t(1);
            instr.type = AnInstruction::WriteIo;
            switch (n) {
                case 1: instr.num_words_arg = NumberOfWords::N1; break;
                case 2: instr.num_words_arg = NumberOfWords::N2; break;
                case 3: instr.num_words_arg = NumberOfWords::N3; break;
                case 4: instr.num_words_arg = NumberOfWords::N4; break;
                case 5: instr.num_words_arg = NumberOfWords::N5; break;
                default: instr.num_words_arg = NumberOfWords::N1; break;
            }
        }
        else if (opcode == "call") {
            std::string arg;
            line_stream >> arg;
            uint64_t addr = 0;
            
            if (std::istringstream(arg) >> addr) {
                instr.type = AnInstruction::Call;
                instr.bfield_arg = BFieldElement(addr);
            } else {
                auto it = label_map.find(arg);
                if (it != label_map.end()) {
                    instr.type = AnInstruction::Call;
                    instr.bfield_arg = BFieldElement(it->second);
                } else {
                    auto label_addr_opt = program_.find_label(arg);
                    if (label_addr_opt.has_value()) {
                        instr.type = AnInstruction::Call;
                        instr.bfield_arg = BFieldElement(label_addr_opt.value());
                    } else {
                        throw std::runtime_error("Label not found: " + arg);
                    }
                }
            }
        }
        else if (opcode == "return") {
            instr.type = AnInstruction::Return;
        }
        else if (opcode == "assert") {
            instr.type = AnInstruction::Assert;
            std::string token;
            if (line_stream >> token && token == "error_id") {
                uint64_t error_id = 0;
                if (line_stream >> error_id) {
                    instr.bfield_arg = BFieldElement(error_id);
                } else {
                    instr.bfield_arg = BFieldElement(0);
                }
            } else {
                // Put token back if it wasn't "error_id"
                // We can't easily put it back, so just set error_id to 0
                instr.bfield_arg = BFieldElement(0);
            }
        }
        else if (opcode == "lt") {
            instr.type = AnInstruction::Lt;
        }
        else if (opcode == "eq") {
            instr.type = AnInstruction::Eq;
        }
        else if (opcode == "addi") {
            int64_t value;
            if (line_stream >> value) {
                instr.type = AnInstruction::AddI;
                if (value >= 0) {
                    instr.bfield_arg = BFieldElement(static_cast<uint64_t>(value));
                } else {
                    uint64_t abs = static_cast<uint64_t>(-value);
                    uint64_t mod = abs % BFieldElement::MODULUS;
                    instr.bfield_arg = (mod == 0) ? BFieldElement::zero()
                                                  : BFieldElement(BFieldElement::MODULUS - mod);
                }
            } else {
                return std::nullopt;
            }
        }
        else if (opcode == "pow") {
            instr.type = AnInstruction::Pow;
        }
        else if (opcode == "sponge_init") {
            instr.type = AnInstruction::SpongeInit;
        }
        else if (opcode == "sponge_squeeze") {
            instr.type = AnInstruction::SpongeSqueeze;
        }
        else if (opcode == "write_mem") {
            size_t n = read_optional_size_t(1);
            instr.type = AnInstruction::WriteMem;
            switch (n) {
                case 1: instr.num_words_arg = NumberOfWords::N1; break;
                case 2: instr.num_words_arg = NumberOfWords::N2; break;
                case 3: instr.num_words_arg = NumberOfWords::N3; break;
                case 4: instr.num_words_arg = NumberOfWords::N4; break;
                case 5: instr.num_words_arg = NumberOfWords::N5; break;
                default: instr.num_words_arg = NumberOfWords::N1; break;
            }
        }
        else if (opcode == "split") {
            instr.type = AnInstruction::Split;
        }
        else if (opcode == "skiz") {
            instr.type = AnInstruction::Skiz;
        }
        else if (opcode == "recurse") {
            instr.type = AnInstruction::Recurse;
        }
        else {
            // Unknown instruction - return nullopt to skip
            return std::nullopt;
        }
        
        return instr;
    };
    
    while (std::getline(iss, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Handle labels - they're already in the map, just skip them
        if (line.back() == ':') {
            continue;
        }
        
        // Parse all instructions on this line
        std::istringstream line_stream(line);
        while (true) {
            auto instr_opt = parse_single_instruction(line_stream);
            if (!instr_opt.has_value()) {
                break; // No more instructions on this line
            }
            program_.add_instruction(instr_opt.value());
        }
    }

    // Cache label map in bword space
    label_bword_map_ = label_map;

    // Build cached bword representation
    bwords_.clear();
    bwords_.reserve(program_.size() * 2);
    for (size_t i = 0; i < program_.size(); ++i) {
        auto instr_opt = program_.instruction_at(i);
        if (!instr_opt.has_value()) continue;
        const auto& instr = instr_opt.value();
        bwords_.push_back(BFieldElement(static_cast<uint64_t>(instr.opcode())));
        if (auto arg = instr.arg(); arg.has_value()) {
            bwords_.push_back(arg.value());
        }
    }

    if (const char* env = std::getenv("TVM_DEBUG_PROGRAM")) {
        (void)env;
        std::cout << "DEBUG program: bwords.len=" << bwords_.size() << std::endl;
        for (const auto& [name, addr] : label_bword_map_) {
            std::cout << "DEBUG program: label " << name << " -> ip " << addr << std::endl;
        }
        size_t limit = std::min<size_t>(bwords_.size(), 80);
        for (size_t ip = 0; ip < limit; ++ip) {
            std::cout << "DEBUG program: bword[" << ip << "]=" << bwords_[ip].value() << std::endl;
        }
    }
}

Digest Program::hash() const {
    return Tip5::hash_varlen(bwords_);
}

size_t Program::len_bwords() const {
    return bwords_.size();
}

std::optional<TritonInstruction> Program::instruction_at(size_t index) const {
    if (index >= bwords_.size()) return std::nullopt;
    BFieldElement ci = bwords_[index];
    BFieldElement nia = (index + 1 < bwords_.size()) ? bwords_[index + 1] : BFieldElement::zero();
    return decode_instruction(ci, nia);
}

std::optional<size_t> Program::find_label(const std::string& name) const {
    auto it = label_bword_map_.find(name);
    if (it == label_bword_map_.end()) return std::nullopt;
    return it->second;
}

std::vector<BFieldElement> Program::to_bwords() const {
    return bwords_;
}

} // namespace triton_vm

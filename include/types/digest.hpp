#pragma once

#include "types/b_field_element.hpp"
#include <array>
#include <string>
#include <vector>

namespace triton_vm {

/**
 * Digest - 5-element hash digest (Tip5 hash output)
 * 
 * A digest consists of 5 BFieldElements, matching the Triton VM
 * Tip5 hash function output.
 */
class Digest {
public:
    static constexpr size_t LEN = 5;
    
    // Constructors
    Digest() : elements_{} {}
    
    explicit Digest(const std::array<BFieldElement, LEN>& elements)
        : elements_(elements) {}
    
    Digest(BFieldElement e0, BFieldElement e1, BFieldElement e2, 
           BFieldElement e3, BFieldElement e4)
        : elements_{e0, e1, e2, e3, e4} {}
    
    // Factory methods
    static Digest zero() { return Digest(); }
    
    // Accessors
    const std::array<BFieldElement, LEN>& elements() const { return elements_; }
    BFieldElement operator[](size_t i) const { return elements_[i]; }
    BFieldElement& operator[](size_t i) { return elements_[i]; }
    
    // Convert to vector of BFieldElements
    std::vector<BFieldElement> to_b_field_elements() const {
        return std::vector<BFieldElement>(elements_.begin(), elements_.end());
    }
    
    // Comparison
    bool operator==(const Digest& rhs) const;
    bool operator!=(const Digest& rhs) const;
    
    // Hex representation
    std::string to_hex() const;
    static Digest from_hex(const std::string& hex);
    
    friend std::ostream& operator<<(std::ostream& os, const Digest& digest);

private:
    std::array<BFieldElement, LEN> elements_;
};

} // namespace triton_vm


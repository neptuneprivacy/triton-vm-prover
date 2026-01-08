use triton_vm::twenty_first::prelude::*;

fn main() {
    let montgomery_value = 18446744065119617026u64;
    println!("Testing BFieldElement conversion:");
    println!("Montgomery value: {}", montgomery_value);
    println!("Montgomery value (hex): 0x{:016x}", montgomery_value);
    
    // Create BFieldElement from raw (Montgomery) value
    let bfe = BFieldElement::from_raw_u64(montgomery_value);
    println!("BFieldElement created from raw_u64");
    
    // Get canonical representation
    let canonical = bfe.value();
    println!("Canonical value: {}", canonical);
    println!("Canonical value (hex): 0x{:016x}", canonical);
    
    // Expected value
    let expected = 18446744065119617023u64;
    println!("Expected value: {}", expected);
    println!("Expected value (hex): 0x{:016x}", expected);
    println!("Match: {}", canonical == expected);
    
    // Also test raw_u64 to see what's stored internally
    println!("Internal (Montgomery) value: {}", bfe.raw_u64());
    println!("Internal (Montgomery) value (hex): 0x{:016x}", bfe.raw_u64());
    
    // Test montyred directly - but it's not public, so we'll test via value()
    println!("\nDirect conversion test:");
    let bfe2 = BFieldElement::from_raw_u64(montgomery_value);
    let canonical2 = bfe2.value();
    println!("from_raw_u64({}) -> value() = {}", montgomery_value, canonical2);
}

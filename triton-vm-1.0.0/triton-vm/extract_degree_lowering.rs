
fn main() {
    // Try to include and print the generated code
    // But this won't work because include! happens at compile time
    
    // Instead, let's check if we can find the file in the build directory
    use std::env;
    use std::fs;
    use std::path::Path;
    
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let file_path = Path::new(&out_dir).join("degree_lowering_table.rs");
        if file_path.exists() {
            println!("Found: {}", file_path.display());
            if let Ok(contents) = fs::read_to_string(&file_path) {
                println!("File size: {} bytes", contents.len());
                // Find fill_derived_aux_columns
                if let Some(idx) = contents.find("pub fn fill_derived_aux_columns") {
                    println!("Function found at position {}", idx);
                    // Extract it
                    let mut brace_count = 0;
                    let mut i = contents[idx..].find('{').unwrap() + idx;
                    let start = idx;
                    while i < contents.len() {
                        match contents.chars().nth(i) {
                            Some('{') => brace_count += 1,
                            Some('}') => {
                                brace_count -= 1;
                                if brace_count == 0 {
                                    let func = &contents[start..=i];
                                    println!("Function length: {} chars", func.len());
                                    fs::write("degree_lowering_extracted.rs", func).unwrap();
                                    println!("Saved to degree_lowering_extracted.rs");
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
            }
        } else {
            println!("File not found at: {}", file_path.display());
        }
    } else {
        println!("OUT_DIR not set");
    }
}

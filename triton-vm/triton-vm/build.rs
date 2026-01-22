use std::path::Path;

use constraint_builder::Constraints;
use constraint_builder::codegen::Codegen;
use constraint_builder::codegen::RustBackend;
use constraint_builder::codegen::TasmBackend;
use constraint_builder::cuda_backend::CudaBackend;
use proc_macro2::TokenStream;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let mut constraints = Constraints::all();
    let degree_lowering_info = Constraints::default_degree_lowering_info();
    let substitutions =
        constraints.lower_to_target_degree_through_substitutions(degree_lowering_info);
    let deg_low_table = substitutions.generate_degree_lowering_table_code();

    let constraints = constraints.combine_with_substitution_induced_constraints(substitutions);
    let rust = RustBackend::constraint_evaluation_code(&constraints);
    let tasm = TasmBackend::constraint_evaluation_code(&constraints);

    // Generate CUDA code for GPU quotient evaluation
    let cuda_code = CudaBackend::constraint_evaluation_code(&constraints);

    write_code_to_file(deg_low_table, "degree_lowering_table.rs");
    write_code_to_file(rust, "evaluate_constraints.rs");
    write_code_to_file(tasm, "tasm_constraints.rs");

    // Write CUDA code directly as string
    write_string_to_file(&cuda_code, "quotient_constraints.cuh");

    // Compile quotient evaluation kernel only if GPU feature is enabled
    #[cfg(feature = "gpu")]
    compile_quotient_kernel();
}

fn write_code_to_file(code: TokenStream, file_name: &str) {
    let syntax_tree = syn::parse2(code).unwrap();
    let code = prettyplease::unparse(&syntax_tree);

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join(file_name);
    std::fs::write(file_path, code).unwrap();
}

fn write_string_to_file(code: &str, file_name: &str) {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join(file_name);
    std::fs::write(&file_path, code).unwrap();
    println!("cargo::warning=Generated CUDA quotient constraints at {:?}", file_path);
}

#[cfg(feature = "gpu")]
fn compile_quotient_kernel() {
    use std::process::Command;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir_path = Path::new(&out_dir);

    // Path to the CUDA source files
    let cuda_src_dir = Path::new("../twenty-first-local/twenty-first/cuda");
    let quotient_cu = cuda_src_dir.join("quotient_evaluation.cu");

    // Output PTX file
    let ptx_output = out_dir_path.join("quotient_evaluation.ptx");

    println!("cargo::rerun-if-changed={}", quotient_cu.display());
    println!("cargo::rerun-if-changed={}", cuda_src_dir.join("field_arithmetic.cuh").display());

    // Compile with nvcc
    // -I for include paths: OUT_DIR (for generated .cuh) and cuda_src_dir (for field_arithmetic.cuh)
    // Use   (Ampere) which works on both RTX 3080 and B200 via JIT
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let arch_second = "sm_89".to_string();

    let status = Command::new("nvcc")
        .arg("--ptx")                           // Compile to PTX
        .arg("-O3")                              // Optimization level 3
        .arg("--maxrregcount=128")               // Limit registers to improve occupancy (critical for complex kernels)
        .arg(format!("-arch={}", arch))          // Target architecture (sm_90a for B200, sm_80 for Ampere)
        .arg(format!("-arch={}", arch_second))
        .arg(format!("-I{}", out_dir))          // Include OUT_DIR for generated quotient_constraints.cuh
        .arg(format!("-I{}", cuda_src_dir.display()))  // Include for field_arithmetic.cuh
        .arg("-o")
        .arg(&ptx_output)
        .arg(&quotient_cu)
        .status()
        .expect("Failed to run nvcc - ensure CUDA toolkit is installed");

    if !status.success() {
        panic!("nvcc failed to compile quotient_evaluation.cu");
    }

    println!("cargo::warning=Compiled quotient evaluation kernel to {:?}", ptx_output);
}

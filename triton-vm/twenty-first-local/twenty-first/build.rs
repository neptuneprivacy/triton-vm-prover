use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Check if source file is newer than output file
fn needs_recompilation(source: &Path, output: &Path) -> bool {
    println!("cargo:warning={:?} {:?}",source,output);
    if !output.exists() {
        return true;
    }

    let src_time = fs::metadata(source).and_then(|m| m.modified()).ok();
    let out_time = fs::metadata(output).and_then(|m| m.modified()).ok();

    match (src_time, out_time) {
        (Some(src), Some(out)) => src > out,
        _ => true, // If we can't get timestamps, recompile to be safe
    }
}

fn main() {
    // Check if GPU feature is enabled
    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    if !gpu_enabled {
        println!("cargo:warning=GPU feature not enabled, skipping CUDA compilation");
        return;
    }

    // Tell Cargo to only rerun this build script if build.rs itself changes
    // This prevents unnecessary reruns when other Rust source files change
    println!("cargo:rerun-if-changed=build.rs");

    // Get CUDA installation path
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let cuda_lib_path = format!("{}/lib64", cuda_path);

    println!("cargo:warning=xxx {} {} {}",cuda_path,nvcc,cuda_lib_path);

    // Tell cargo to link against CUDA runtime
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_dir = Path::new("cuda");

    // Compile BFieldElement NTT kernel
    let bfield_cu = cuda_dir.join("bfield_ntt.cu");
    let bfield_ptx = Path::new(&out_dir).join("bfield_ntt.ptx");
    println!("cargo:rerun-if-changed={}", bfield_cu.display());

    if needs_recompilation(&bfield_cu, &bfield_ptx) {
        println!("cargo:warning=Compiling BField NTT CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86", // Blackwell B200 (sm_90), forward-compatible with Ampere
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                bfield_ptx.to_str().unwrap(),
                bfield_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for bfield_ntt.cu");

        if !status.success() {
            panic!("Failed to compile bfield_ntt.cu");
        }
        println!(
            "cargo:warning=BField NTT kernel compiled: {}",
            bfield_ptx.display()
        );
    } else {
        println!("cargo:warning=BField NTT kernel up-to-date, skipping compilation");
    }

    // Compile XFieldElement NTT kernel
    let xfield_cu = cuda_dir.join("xfield_ntt.cu");
    let xfield_ptx = Path::new(&out_dir).join("xfield_ntt.ptx");

    println!("cargo:rerun-if-changed={}", xfield_cu.display());

    if needs_recompilation(&xfield_cu, &xfield_ptx) {
        println!("cargo:warning=Compiling XField NTT CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86",
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                xfield_ptx.to_str().unwrap(),
                xfield_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for xfield_ntt.cu");

        if !status.success() {
            panic!("Failed to compile xfield_ntt.cu");
        }
        println!(
            "cargo:warning=XField NTT kernel compiled: {}",
            xfield_ptx.display()
        );
    } else {
        println!("cargo:warning=XField NTT kernel up-to-date, skipping compilation");
    }

    // Compile BFieldElement Coset Scaling kernel
    let coset_scale_cu = cuda_dir.join("coset_scale.cu");
    let coset_scale_ptx = Path::new(&out_dir).join("coset_scale.ptx");

    println!("cargo:rerun-if-changed={}", coset_scale_cu.display());

    if needs_recompilation(&coset_scale_cu, &coset_scale_ptx) {
        println!("cargo:warning=Compiling BField Coset Scaling CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86",
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                coset_scale_ptx.to_str().unwrap(),
                coset_scale_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for coset_scale.cu");

        if !status.success() {
            panic!("Failed to compile coset_scale.cu");
        }
        println!(
            "cargo:warning=BField Coset Scaling kernel compiled: {}",
            coset_scale_ptx.display()
        );
    } else {
        println!("cargo:warning=BField Coset Scaling kernel up-to-date, skipping compilation");
    }

    // Compile Auxiliary Extension kernel (for running evaluations)
    let aux_extend_cu = cuda_dir.join("aux_extend.cu");
    let aux_extend_ptx = Path::new(&out_dir).join("aux_extend.ptx");

    println!("cargo:rerun-if-changed={}", aux_extend_cu.display());

    if needs_recompilation(&aux_extend_cu, &aux_extend_ptx) {
        println!("cargo:warning=Compiling Auxiliary Extension CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86",
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                aux_extend_ptx.to_str().unwrap(),
                aux_extend_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for aux_extend.cu");

        if !status.success() {
            panic!("Failed to compile aux_extend.cu");
        }
        println!(
            "cargo:warning=Auxiliary Extension kernel compiled: {}",
            aux_extend_ptx.display()
        );
    } else {
        println!("cargo:warning=Auxiliary Extension kernel up-to-date, skipping compilation");
    }

    // Compile Tip5 Hash kernel
    let tip5_hash_cu = cuda_dir.join("tip5_hash.cu");
    let tip5_hash_ptx = Path::new(&out_dir).join("tip5_hash.ptx");

    println!("cargo:rerun-if-changed={}", tip5_hash_cu.display());

    if needs_recompilation(&tip5_hash_cu, &tip5_hash_ptx) {
        println!("cargo:warning=Compiling Tip5 Hash CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86",
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                tip5_hash_ptx.to_str().unwrap(),
                tip5_hash_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for tip5_hash.cu");

        if !status.success() {
            panic!("Failed to compile tip5_hash.cu");
        }
        println!(
            "cargo:warning=Tip5 Hash kernel compiled: {}",
            tip5_hash_ptx.display()
        );
    } else {
        println!("cargo:warning=Tip5 Hash kernel up-to-date, skipping compilation");
    }

    // Compile Extract Rows kernel (for partial row download optimization)
    let extract_rows_cu = cuda_dir.join("extract_rows.cu");
    let extract_rows_ptx = Path::new(&out_dir).join("extract_rows.ptx");

    println!("cargo:rerun-if-changed={}", extract_rows_cu.display());

    if needs_recompilation(&extract_rows_cu, &extract_rows_ptx) {
        println!("cargo:warning=Compiling Extract Rows CUDA kernel...");
        let status = Command::new(&nvcc)
            .args(&[
                "-ptx",
                "-O3",
                "--gpu-architecture=sm_86",
                "--gpu-architecture=sm_89",
                "--use_fast_math",
                "-o",
                extract_rows_ptx.to_str().unwrap(),
                extract_rows_cu.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc for extract_rows.cu");

        if !status.success() {
            panic!("Failed to compile extract_rows.cu");
        }
        println!(
            "cargo:warning=Extract Rows kernel compiled: {}",
            extract_rows_ptx.display()
        );
    } else {
        println!("cargo:warning=Extract Rows kernel up-to-date, skipping compilation");
    }
}

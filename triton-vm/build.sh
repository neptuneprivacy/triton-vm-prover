export CUDA_PATH=(your cuda path)
export PATH="$CUDA_PATH/bin:$PATH"

echo ">>> Using CUDA toolkit at: $CUDA_PATH"
"$CUDA_PATH/bin/nvcc" --version

cargo build --release --features gpu --bin tester
//! Low-level CUDA Driver API wrappers
//!
//! This module provides thin wrappers around the CUDA Driver API (`cudarc::driver::sys`)
//! for direct control over GPU resources and operations.

use cudarc::driver::sys as cuda;
use std::ffi::CString;
use std::sync::Arc;

/// Check CUDA result and return error if failed
#[inline]
pub fn check_cuda(result: cuda::CUresult, msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    if result != cuda::CUresult::CUDA_SUCCESS {
        return Err(format!("CUDA Error at {}: {:?}", msg, result).into());
    }
    Ok(())
}

/// CUDA device handle
#[derive(Clone)]
pub struct CudaDevice {
    pub device: cuda::CUdevice,
    pub context: Arc<CudaContextInner>,
}

/// CUDA context (wrapped in Arc for sharing)
pub struct CudaContextInner {
    pub context: cuda::CUcontext,
}

// SAFETY: CUDA contexts are thread-safe and can be shared across threads
unsafe impl Send for CudaContextInner {}
unsafe impl Sync for CudaContextInner {}

impl Drop for CudaContextInner {
    fn drop(&mut self) {
        unsafe {
            // Best effort cleanup - ignore errors during drop
            let _ = cuda::cuCtxDestroy_v2(self.context);
        }
    }
}

impl CudaDevice {
    /// Create a new CUDA device and context
    pub fn new(device_id: i32) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            // Initialize CUDA Driver API (safe to call multiple times)
            check_cuda(cuda::cuInit(0), "cuInit")?;

            // Get device handle
            let mut device: cuda::CUdevice = 0;
            check_cuda(
                cuda::cuDeviceGet(&mut device, device_id),
                &format!("cuDeviceGet({})", device_id),
            )?;

            // Create context
            let mut context: cuda::CUcontext = std::ptr::null_mut();
            check_cuda(
                cuda::cuCtxCreate_v2(&mut context, 0, device),
                &format!("cuCtxCreate({})", device_id),
            )?;

            Ok(Self {
                device,
                context: Arc::new(CudaContextInner { context }),
            })
        }
    }

    /// Set this context as current
    pub fn set_current(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            check_cuda(
                cuda::cuCtxSetCurrent(self.context.context),
                "cuCtxSetCurrent",
            )
        }
    }

    /// Get the default stream (null stream)
    pub fn default_stream(&self) -> CudaStream {
        CudaStream {
            stream: std::ptr::null_mut(),
            context: self.context.clone(),
        }
    }

    /// Create a new non-blocking stream
    pub fn create_stream(&self) -> Result<CudaStream, Box<dyn std::error::Error>> {
        unsafe {
            self.set_current()?;
            let mut stream: cuda::CUstream = std::ptr::null_mut();
            check_cuda(
                cuda::cuStreamCreate(
                    &mut stream,
                    cuda::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                ),
                "cuStreamCreate",
            )?;
            Ok(CudaStream {
                stream,
                context: self.context.clone(),
            })
        }
    }

    /// Load a PTX/CUBIN module
    pub fn load_module(&self, module_data: &[u8]) -> Result<CudaModule, Box<dyn std::error::Error>> {
        unsafe {
            self.set_current()?;
            let mut module: cuda::CUmodule = std::ptr::null_mut();
            check_cuda(
                cuda::cuModuleLoadData(&mut module, module_data.as_ptr() as *const _),
                "cuModuleLoadData",
            )?;
            Ok(CudaModule { module })
        }
    }

    /// Check if this device can access peer device's memory
    pub fn can_access_peer(&self, peer_device: &CudaDevice) -> Result<bool, Box<dyn std::error::Error>> {
        unsafe {
            let mut can_access: i32 = 0;
            check_cuda(
                cuda::cuDeviceCanAccessPeer(&mut can_access, self.device, peer_device.device),
                "cuDeviceCanAccessPeer",
            )?;
            Ok(can_access != 0)
        }
    }

    /// Enable peer access from this device to peer device
    pub fn enable_peer_access(&self, peer_device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.set_current()?;
            check_cuda(
                cuda::cuCtxEnablePeerAccess(peer_device.context.context, 0),
                "cuCtxEnablePeerAccess",
            )
        }
    }
}

/// CUDA stream
pub struct CudaStream {
    pub stream: cuda::CUstream,
    pub context: Arc<CudaContextInner>,
}

// SAFETY: CUDA streams are thread-safe and can be used from multiple threads
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // Don't destroy null stream (default stream)
        if !self.stream.is_null() {
            unsafe {
                let _ = cuda::cuStreamDestroy_v2(self.stream);
            }
        }
    }
}

impl CudaStream {
    /// Set this stream's context as current for the calling thread
    #[inline]
    fn set_context_current(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            check_cuda(
                cuda::cuCtxSetCurrent(self.context.context),
                "cuCtxSetCurrent",
            )
        }
    }

    /// Allocate device memory without initialization
    pub fn alloc<T>(&self, count: usize) -> Result<DeviceBuffer, Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            let size_bytes = count * std::mem::size_of::<T>();

            // Allocate device memory
            let mut d_ptr: cuda::CUdeviceptr = 0;
            check_cuda(
                cuda::cuMemAlloc_v2(&mut d_ptr, size_bytes),
                "cuMemAlloc",
            )?;

            Ok(DeviceBuffer {
                ptr: d_ptr,
                size_bytes,
            })
        }
    }

    /// Upload data from host to device
    pub fn memcpy_htod<T>(&self, host_data: &[T]) -> Result<DeviceBuffer, Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            let size_bytes = host_data.len() * std::mem::size_of::<T>();

            // Allocate device memory
            let mut d_ptr: cuda::CUdeviceptr = 0;
            check_cuda(
                cuda::cuMemAlloc_v2(&mut d_ptr, size_bytes),
                "cuMemAlloc",
            )?;

            // Copy data
            check_cuda(
                cuda::cuMemcpyHtoDAsync_v2(
                    d_ptr,
                    host_data.as_ptr() as *const _,
                    size_bytes,
                    self.stream,
                ),
                "cuMemcpyHtoDAsync",
            )?;

            Ok(DeviceBuffer {
                ptr: d_ptr,
                size_bytes,
            })
        }
    }

    /// Upload data from host to a specific device buffer slice
    pub fn memcpy_htod_to_slice<T>(&self, host_data: &[T], device_buf: &DeviceBuffer, offset: usize) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            let size_bytes = host_data.len() * std::mem::size_of::<T>();
            let offset_bytes = offset * std::mem::size_of::<T>();

            if offset_bytes + size_bytes > device_buf.size_bytes {
                return Err("Host data too large for device buffer slice".into());
            }

            check_cuda(
                cuda::cuMemcpyHtoDAsync_v2(
                    device_buf.ptr + offset_bytes as u64,
                    host_data.as_ptr() as *const _,
                    size_bytes,
                    self.stream,
                ),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    /// Download data from device to host
    pub fn memcpy_dtoh<T>(&self, device_buf: &DeviceBuffer, host_data: &mut [T]) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            let size_bytes = host_data.len() * std::mem::size_of::<T>();
            if size_bytes > device_buf.size_bytes {
                return Err("Host buffer too large for device buffer".into());
            }

            check_cuda(
                cuda::cuMemcpyDtoHAsync_v2(
                    host_data.as_mut_ptr() as *mut _,
                    device_buf.ptr,
                    size_bytes,
                    self.stream,
                ),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    /// Download data from a device buffer slice to host
    pub fn memcpy_dtoh_from_slice<T>(&self, device_buf: &DeviceBuffer, offset: usize, host_data: &mut [T]) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            let size_bytes = host_data.len() * std::mem::size_of::<T>();
            let offset_bytes = offset * std::mem::size_of::<T>();

            if offset_bytes + size_bytes > device_buf.size_bytes {
                return Err("Host buffer too large for device buffer slice".into());
            }

            check_cuda(
                cuda::cuMemcpyDtoHAsync_v2(
                    host_data.as_mut_ptr() as *mut _,
                    device_buf.ptr + offset_bytes as u64,
                    size_bytes,
                    self.stream,
                ),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    /// Copy data between two GPU devices using P2P (peer-to-peer) transfer
    /// NOTE: Peer access must be enabled first using CudaDevice::enable_peer_access()
    pub fn memcpy_peer_async(
        &self,
        dst_device_buf: &DeviceBuffer,
        dst_context: &Arc<CudaContextInner>,
        src_device_buf: &DeviceBuffer,
        src_context: &Arc<CudaContextInner>,
        size_bytes: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            if size_bytes > src_device_buf.size_bytes || size_bytes > dst_device_buf.size_bytes {
                return Err("Size exceeds buffer capacity".into());
            }

            check_cuda(
                cuda::cuMemcpyPeerAsync(
                    dst_device_buf.ptr,
                    dst_context.context,
                    src_device_buf.ptr,
                    src_context.context,
                    size_bytes,
                    self.stream,
                ),
                "cuMemcpyPeerAsync",
            )
        }
    }

    /// Synchronize stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        self.set_context_current()?;

        unsafe {
            check_cuda(cuda::cuStreamSynchronize(self.stream), "cuStreamSynchronize")
        }
    }
}

/// Device memory buffer
pub struct DeviceBuffer {
    pub ptr: cuda::CUdeviceptr,
    pub size_bytes: usize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda::cuMemFree_v2(self.ptr);
        }
    }
}

impl DeviceBuffer {
    /// Get the device pointer
    pub fn as_ptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size_bytes
    }
}

/// CUDA module (loaded PTX/CUBIN)
pub struct CudaModule {
    pub module: cuda::CUmodule,
}

impl CudaModule {
    /// Load a function from the module
    pub fn get_function(&self, name: &str) -> Result<CudaFunction, Box<dyn std::error::Error>> {
        unsafe {
            let name_cstr = CString::new(name)?;
            let mut function: cuda::CUfunction = std::ptr::null_mut();
            check_cuda(
                cuda::cuModuleGetFunction(&mut function, self.module, name_cstr.as_ptr()),
                &format!("cuModuleGetFunction({})", name),
            )?;
            Ok(CudaFunction { function })
        }
    }
}

/// CUDA kernel function
#[derive(Clone)]
pub struct CudaFunction {
    pub function: cuda::CUfunction,
}

// SAFETY: CUDA functions are thread-safe and can be shared across threads
unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    /// Launch the kernel
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &CudaStream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        stream.set_context_current()?;

        unsafe {
            check_cuda(
                cuda::cuLaunchKernel(
                    self.function,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    shared_mem,
                    stream.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    /// Launch a cooperative kernel (required for kernels using cooperative groups)
    pub fn launch_cooperative(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &CudaStream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // CRITICAL: Set context as current for this thread before CUDA operations
        stream.set_context_current()?;

        unsafe {
            check_cuda(
                cuda::cuLaunchCooperativeKernel(
                    self.function,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    shared_mem,
                    stream.stream,
                    args.as_mut_ptr(),
                ),
                "cuLaunchCooperativeKernel",
            )
        }
    }
}

/// Helper to build kernel arguments
pub struct KernelArgs {
    args: Vec<*mut std::ffi::c_void>,
    // Keep owned data alive
    _owned: Vec<Box<dyn std::any::Any>>,
}

impl KernelArgs {
    pub fn new() -> Self {
        Self {
            args: Vec::new(),
            _owned: Vec::new(),
        }
    }

    /// Add a device pointer argument
    pub fn push_ptr(&mut self, ptr: &cuda::CUdeviceptr) -> &mut Self {
        self.args.push(ptr as *const _ as *mut _);
        self
    }

    /// Add a mutable device pointer argument
    pub fn push_mut_ptr(&mut self, ptr: &mut cuda::CUdeviceptr) -> &mut Self {
        self.args.push(ptr as *mut _ as *mut _);
        self
    }

    /// Add a value argument (copies it)
    pub fn push_value<T: 'static>(&mut self, value: T) -> &mut Self {
        let boxed = Box::new(value);
        let ptr = &*boxed as *const T as *mut std::ffi::c_void;
        self.args.push(ptr);
        self._owned.push(boxed);
        self
    }

    /// Get mutable slice of argument pointers
    pub fn as_mut_slice(&mut self) -> &mut [*mut std::ffi::c_void] {
        &mut self.args
    }
}

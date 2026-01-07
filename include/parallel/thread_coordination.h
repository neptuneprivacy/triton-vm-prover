#pragma once

/**
 * Thread Coordination Utility
 * 
 * Coordinates thread counts between OpenMP, TBB, and Taskflow to avoid oversubscription.
 * All three libraries share the same OS thread pool, so we need to coordinate their usage.
 * 
 * Strategy:
 * - OpenMP: Used for loop-level parallelism (data parallelism)
 * - TBB: Used for task-based parallelism and parallel STL algorithms
 * - Taskflow: Used for dependency graph-based task parallelism
 * 
 * Thread allocation:
 * - If OMP_NUM_THREADS is set, use that as the base
 * - Otherwise, use physical core count (not logical threads)
 * - TBB and Taskflow will share the same thread pool
 */

#include <cstdlib>
#include <thread>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TVM_USE_TBB
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#endif

#ifdef TVM_USE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

namespace triton_vm::parallel {

/**
 * Get the optimal thread count for parallel execution
 * Respects OMP_NUM_THREADS if set, otherwise uses physical core count
 */
inline int get_optimal_thread_count() {
    // Check environment variable first
    const char* omp_threads = std::getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        int count = std::atoi(omp_threads);
        if (count > 0) {
            return count;
        }
    }
    
    // Use physical core count (not logical threads)
    // This prevents oversubscription on systems with SMT/Hyperthreading
    unsigned int hw_threads = std::thread::hardware_concurrency();
    // For systems with SMT, use physical cores (typically half of logical cores)
    // Threadripper 9995WX: 96 cores, 192 threads -> use 96
    return std::max(1, static_cast<int>(hw_threads / 2));
}

/**
 * Initialize thread coordination for all parallel libraries
 * Call this once at program startup
 */
inline void initialize_thread_coordination() {
    int thread_count = get_optimal_thread_count();
    
#ifdef _OPENMP
    // OpenMP is already configured via OMP_NUM_THREADS environment variable
    // But we can also set it programmatically
    omp_set_num_threads(thread_count);
#endif

#ifdef TVM_USE_TBB
    // TBB: Set global thread limit
    // TBB will automatically share threads with OpenMP if both are active
    // We set it to the same count to avoid oversubscription
    static tbb::global_control tbb_control(
        tbb::global_control::max_allowed_parallelism,
        thread_count
    );
#endif

#ifdef TVM_USE_TASKFLOW
    // Taskflow: Thread count is set when creating the executor
    // We'll use the same count for consistency
    // (Taskflow executor is created per-use, not globally)
#endif
}

/**
 * Get current thread count being used
 */
inline int get_current_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return get_optimal_thread_count();
#endif
}

/**
 * Coordination strategy:
 * 
 * 1. OpenMP: Best for regular loops with independent iterations
 *    - Use #pragma omp parallel for
 *    - Automatic load balancing
 *    - Low overhead
 * 
 * 2. TBB: Best for task-based parallelism and parallel STL
 *    - Use tbb::parallel_for, tbb::parallel_invoke
 *    - Better for irregular workloads
 *    - Automatic work stealing
 * 
 * 3. Taskflow: Best for dependency graphs
 *    - Use when tasks have dependencies
 *    - Better for complex workflows
 *    - Automatic scheduling
 * 
 * When to use each:
 * - OpenMP: Simple loops, data parallelism
 * - TBB: Task parallelism, parallel STL algorithms, work stealing needed
 * - Taskflow: Dependency graphs, complex workflows
 * 
 * They can all be used together, but be careful about:
 * - Thread oversubscription (coordinate via this utility)
 * - Memory bandwidth saturation
 * - Cache contention
 */

} // namespace triton_vm::parallel


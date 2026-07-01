import time
import numpy as np
import custom_matrix_ops

def python_matmul(A: list, B: list) -> list:
    """Pure Python nested loop matrix multiplication."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    out = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out

def run_benchmark():
    print("--- Starting Matrix Multiplication Execution Speedup Audit ---")
    
    # 1. Warm-up and test correctness on small shapes (32 x 32)
    size_small = 32
    A_small = np.random.randn(size_small, size_small).tolist()
    B_small = np.random.randn(size_small, size_small).tolist()
    
    py_res = python_matmul(A_small, B_small)
    
    a_raw_small = custom_matrix_ops.RawMatrix(size_small, size_small, np.array(A_small).flatten().astype(np.float32).tolist())
    b_raw_small = custom_matrix_ops.RawMatrix(size_small, size_small, np.array(B_small).flatten().astype(np.float32).tolist())
    cpp_res_raw = custom_matrix_ops.matmul(a_raw_small, b_raw_small)
    cpp_res = np.array(cpp_res_raw.data).reshape(size_small, size_small).tolist()
    
    # Assert correctness
    np.testing.assert_array_almost_equal(py_res, cpp_res, decimal=4)
    print("✅ Correctness check passed: C++ results match Python nested loops.")
    
    # 2. Benchmark on 128 x 128 matrices to avoid hanging Python execution
    size_bench = 128
    print(f"\nBenchmarking on {size_bench} x {size_bench} matrices:")
    
    A_np = np.random.randn(size_bench, size_bench)
    B_np = np.random.randn(size_bench, size_bench)
    A_list = A_np.tolist()
    B_list = B_np.tolist()
    
    # Benchmark Python
    start_time = time.time()
    _ = python_matmul(A_list, B_list)
    py_duration = time.time() - start_time
    print(f"| Pure Python Matmul: {py_duration * 1000:.2f} ms")
    
    # Benchmark C++
    a_raw = custom_matrix_ops.RawMatrix(size_bench, size_bench, A_np.flatten().astype(np.float32).tolist())
    b_raw = custom_matrix_ops.RawMatrix(size_bench, size_bench, B_np.flatten().astype(np.float32).tolist())
    
    start_time = time.time()
    _ = custom_matrix_ops.matmul(a_raw, b_raw)
    cpp_duration = time.time() - start_time
    print(f"| Optimized C++ Matmul: {cpp_duration * 1000:.2f} ms")
    
    speedup = py_duration / cpp_duration
    print(f"--> Speedup Factor: {speedup:.2f}x faster execution in C++")

    # 3. High Performance Computing (HPC) demonstration for large matrices (512 x 512)
    size_large = 512
    print(f"\nDemonstrating C++ HPC capability on large {size_large} x {size_large} matrices:")
    A_large = np.random.randn(size_large, size_large)
    B_large = np.random.randn(size_large, size_large)
    
    a_raw_large = custom_matrix_ops.RawMatrix(size_large, size_large, A_large.flatten().astype(np.float32).tolist())
    b_raw_large = custom_matrix_ops.RawMatrix(size_large, size_large, B_large.flatten().astype(np.float32).tolist())
    
    start_time = time.time()
    _ = custom_matrix_ops.matmul(a_raw_large, b_raw_large)
    cpp_large_duration = time.time() - start_time
    print(f"| C++ Matmul ({size_large}x{size_large}): {cpp_large_duration * 1000:.2f} ms")
    
    # Estimated pure Python time based on O(N^3) complexity
    estimated_py = py_duration * ((size_large / size_bench) ** 3)
    print(f"| Estimated Pure Python Time: {estimated_py:.2f} s")
    print(f"--> Projected Speedup Factor: {estimated_py / cpp_large_duration:.2f}x faster execution")

if __name__ == "__main__":
    run_benchmark()

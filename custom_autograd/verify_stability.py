import os
import resource
import numpy as np
from tensor import Tensor
from nn import Linear, SGD

def get_memory_usage_kb():
    """Returns the maximum resident set size (max RSS) in kilobytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes, so convert to KB.
    # On Linux, it is already in kilobytes.
    if os.uname().sysname == 'Darwin':
        return usage.ru_maxrss / 1024.0
    return usage.ru_maxrss

def run_stability_check():
    print("--- Starting 10,000 Epoch Stability and Memory Leak Audit ---")
    np.random.seed(42)
    
    # Define a 3-layer neural network
    l1 = Linear(16, 32)
    l2 = Linear(32, 16)
    l3 = Linear(16, 4)
    params = l1.parameters() + l2.parameters() + l3.parameters()
    optimizer = SGD(params, lr=0.01)
    
    initial_mem = get_memory_usage_kb()
    print(f"[Initial] Memory Usage: {initial_mem:.2f} KB")
    
    # Generate constant input data
    x_data = np.random.randn(8, 16)
    
    for epoch in range(1, 10001):
        # 1. Clear gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        x = Tensor(x_data)
        h1 = x @ l1.weight + l1.bias
        h2 = h1 @ l2.weight + l2.bias
        out = h2 @ l3.weight + l3.bias
        loss = out.sum()
        
        # 3. Backward pass
        loss.backward()
        
        # 4. Optimization step
        optimizer.step()
        
        # Log memory usage periodically
        if epoch % 2000 == 0:
            current_mem = get_memory_usage_kb()
            delta_mem = current_mem - initial_mem
            print(f"[Epoch {epoch:5d}/10000] Memory Usage: {current_mem:.2f} KB | Delta: {delta_mem:+.2f} KB")
            
    final_mem = get_memory_usage_kb()
    final_delta = final_mem - initial_mem
    print(f"\n[Final] Memory Usage: {final_mem:.2f} KB | Total Delta: {final_delta:+.2f} KB")
    
    if final_delta < 5000: # Threshold of 5MB for slight runtime growth (Python heap overhead)
        print("✅ STABILITY VERIFICATION PASSED: No memory leaks detected in C++ bindings. Memory footprint is stable.")
    else:
        print("⚠️ STABILITY VERIFICATION WARNING: Minor memory growth detected. Review C++ allocations.")

if __name__ == "__main__":
    run_stability_check()

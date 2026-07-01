# C++ Accelerated Custom Autograd Engine

A lightweight, high-performance automatic differentiation library built in Python with a native C++ extension layer to accelerate heavy matrix operations. The engine mirrors PyTorch's API design and behaves identically during both forward and backward propagation.

---

## Architecture

The framework splits execution boundaries between dynamic graph building in Python and optimized linear algebra processing in C++:

```text
               +----------------------------------------+
               |             Python User API            |
               |       Tensor, nn.Linear, nn.SGD        |
               +----------------------------------------+
                                   |
                                   |  Data Flattening & Casts
                                   v
               +----------------------------------------+
               |         pybind11 Binding Layer         |
               |        (custom_matrix_ops.so)          |
               +----------------------------------------+
                                   |
                                   |  Fast Raw Pointers
                                   v
               +----------------------------------------+
               |            Native C++ Engine           |
               |   RawMatrix, matmul, transpose, add    |
               +----------------------------------------+
```

### 1. Python Core ([`tensor.py`](file:///Users/navaneethvurimalla/Downloads/projects/custom_autograd/tensor.py))
* **Graph Auto-Differentiation**: Builds a dynamic computational graph track (`_prev` and `_op`) during forward passes.
* **Topological Sort Backpropagation**: Resolves execution dependencies by building a topological sort of the graph and evaluating gradient backpropagation hooks in reverse topological order.
* **Broadcasting Gradients**: Automatically reduces gradients over expanded axes to support broadcasting element-wise operations and matrix alignments.

### 2. C++ Acceleration Module ([`matrix_ops.cpp`](file:///Users/navaneethvurimalla/Downloads/projects/custom_autograd/matrix_ops.cpp))
* **Cache Locality Optimization**: Standard matrix multiplication transposes the second operand to align memory layout, performing inner-loop dot products along contiguous memory rows (stride-1 cache line utilization).
* **Boundaries Check**: Strictly enforces size checking at compilation boundaries to prevent memory leaks or segmentation faults.

### 3. pybind11 Bridge ([`bindings.cpp`](file:///Users/navaneethvurimalla/Downloads/projects/custom_autograd/bindings.cpp))
* Connects the Python `Tensor` structures with C++ native structs and functions, converting standard Python sequences to STL vector wrappers automatically.

---

## File Directory

```text
projects/custom_autograd/
├── tensor.py            # Autograd Tensor class and broadcasting logic
├── nn.py                # High-level layers (Linear) and optimizer (SGD)
├── matrix_ops.hpp       # C++ matrix structure and operations declarations
├── matrix_ops.cpp       # Optimized C++ matrix multiplication and additions
├── bindings.cpp         # pybind11 Python bindings
├── setup.py             # Compiler configuration script (-O3 compile flags)
├── requirements.txt     # Dependency prerequisites
├── test_tensor.py       # Core addition/subtraction unit tests
├── test_nn.py           # Linear layer and SGD optimizer unit tests
├── test_engine.py       # Validation test matching custom engine vs. PyTorch
└── README.md            # Documentation
```

---

## Installation & Build

### 1. Prerequisites
Ensure you have a C++ compiler supporting C++11 (such as Apple Clang) and Python 3.8+ installed.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Compile the C++ Extension Natively
Build the high-performance C++ matrix operations library directly in-place:
```bash
python3 setup.py build_ext --inplace
```
This produces a shared library file (e.g., `custom_matrix_ops.cpython-...so`) in the project directory that Python can import immediately.

---

## Running Verification Tests

### Run PyTorch Alignment Tests
This harness initializes identical random inputs and weight configurations, runs multi-layer perceptron forward passes, registers intermediate activations, triggers `.backward()`, and compares outputs and gradients against official PyTorch:
```bash
python3 test_engine.py
```
Outputs and gradients match PyTorch to a relative tolerance of $10^{-5}$ (`rtol=1e-5`, `atol=1e-5`).

### Run Core Engine Unit Tests
```bash
python3 -m unittest discover -p "test_*.py"
```
Verification covers element-wise operations, scalar broadcasting, shape broadcasting, multi-path gradient accumulations, and training parameter updates.

import numpy as np
import custom_matrix_ops
from typing import Set, Tuple, Union, Callable

def _sum_broadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sums gradient dimensions that were expanded during broadcasting to match target_shape.
    This ensures element-wise math and broadcasted operations propagate gradients correctly.
    """
    if grad.shape == target_shape:
        return grad
        
    # If target is a scalar (represented as an empty tuple shape)
    if not target_shape:
        return np.sum(grad)
        
    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    
    # 1. Track dims added to the left (e.g. target shape (3,), grad shape (2, 3))
    axes_to_sum = list(range(grad_ndim - target_ndim))
    
    # 2. Track dims that were 1 in target but expanded in grad (e.g. target (1, 3), grad (2, 3))
    for i in range(target_ndim):
        grad_axis = grad_ndim - target_ndim + i
        if target_shape[i] == 1 and grad.shape[grad_axis] != 1:
            axes_to_sum.append(grad_axis)
            
    if axes_to_sum:
        summed = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
        # Squeeze out added left-most dimensions
        if grad_ndim > target_ndim:
            summed = np.squeeze(summed, axis=tuple(range(grad_ndim - target_ndim)))
        return summed.reshape(target_shape)
        
    return grad

class Tensor:
    """
    A custom PyTorch-like Tensor class supporting automatic differentiation (autograd).
    Wraps a 2D (or arbitrary dimensional) NumPy array and tracks the computational graph.
    """
    def __init__(self, data: Union[int, float, list, np.ndarray], _children: Tuple['Tensor', ...] = (), _op: str = ""):
        # Normalize incoming data to a Float64 NumPy array
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            raise TypeError(f"Unsupported data type for Tensor initialization: {type(data)}")
            
        # Initialize gradients to match the shape of the data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        
        # Graph tracking properties
        self._prev: Set['Tensor'] = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def zero_grad(self) -> None:
        """Resets the gradient accumulator for this tensor to zeros."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self) -> None:
        """
        Runs a topological sort on the computational graph starting from this node,
        and executes backpropagation to calculate gradients for all dependent nodes.
        """
        topo: list['Tensor'] = []
        visited: Set['Tensor'] = set()
        
        def build_topo(v: 'Tensor') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        # Build topological sort list
        build_topo(self)
        
        # Reset all gradients in the graph except for this output node, which starts at 1.0
        # In PyTorch, calling backward() accumulates gradients, but the base seed is 1.0
        self.grad = np.ones_like(self.data, dtype=np.float64)
        
        # Process nodes in reverse topological order
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += _sum_broadcast(out.grad, self.data.shape)
            other.grad += _sum_broadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        return out

    def __radd__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        return self + other

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other), _op="-")

        def _backward():
            self.grad += _sum_broadcast(out.grad, self.data.shape)
            other.grad += _sum_broadcast(-out.grad, other.data.shape)
            
        out._backward = _backward
        return out

    def __rsub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication (A @ B) intercepted and executed in C++.
        Gradients are propagated using C++ operations: dA = dY @ B^T, dB = A^T @ dY.
        """
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires both operands to be Tensors.")
            
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(f"Matrix multiplication @ is only supported for 2D Tensors. Got shapes {self.data.shape} and {other.data.shape}")
            
        # Convert inputs to C++ RawMatrix format
        a_raw = custom_matrix_ops.RawMatrix(self.data.shape[0], self.data.shape[1], self.data.flatten().astype(np.float32).tolist())
        b_raw = custom_matrix_ops.RawMatrix(other.data.shape[0], other.data.shape[1], other.data.flatten().astype(np.float32).tolist())
        
        # Execute in C++
        out_raw = custom_matrix_ops.matmul(a_raw, b_raw)
        
        # Reshape output back to 2D NumPy array
        out_data = np.array(out_raw.data, dtype=np.float64).reshape((out_raw.rows, out_raw.cols))
        out = Tensor(out_data, _children=(self, other), _op="@")

        def _backward():
            # Convert gradient out.grad to C++ RawMatrix
            dy_raw = custom_matrix_ops.RawMatrix(out.grad.shape[0], out.grad.shape[1], out.grad.flatten().astype(np.float32).tolist())
            
            # dA = dY @ B^T
            b_t_raw = custom_matrix_ops.transpose(b_raw)
            da_raw = custom_matrix_ops.matmul(dy_raw, b_t_raw)
            da_data = np.array(da_raw.data, dtype=np.float64).reshape((da_raw.rows, da_raw.cols))
            self.grad += da_data
            
            # dB = A^T @ dY
            a_t_raw = custom_matrix_ops.transpose(a_raw)
            db_raw = custom_matrix_ops.matmul(a_t_raw, dy_raw)
            db_data = np.array(db_raw.data, dtype=np.float64).reshape((db_raw.rows, db_raw.cols))
            other.grad += db_data
            
        out._backward = _backward
        return out

    def sum(self) -> 'Tensor':
        """Computes the sum of all elements in this tensor."""
        out = Tensor(np.sum(self.data), _children=(self,), _op="sum")
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
            
        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor({self.data.tolist()}, grad={self.grad.tolist()})"

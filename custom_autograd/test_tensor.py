import unittest
import numpy as np
from tensor import Tensor

class TestTensorAutograd(unittest.TestCase):
    def test_elementwise_add(self):
        """Test simple element-wise addition of two Tensors of the same shape."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a + b
        
        # Verify forward pass
        np.testing.assert_array_equal(c.data, np.array([[6.0, 8.0], [10.0, 12.0]]))
        
        # Run backward pass
        c.backward()
        
        # Verify backward pass
        np.testing.assert_array_equal(a.grad, np.ones((2, 2)))
        np.testing.assert_array_equal(b.grad, np.ones((2, 2)))

    def test_elementwise_sub(self):
        """Test simple element-wise subtraction of two Tensors of the same shape."""
        a = Tensor([[10.0, 20.0]])
        b = Tensor([[3.0, 15.0]])
        c = a - b
        
        # Verify forward pass
        np.testing.assert_array_equal(c.data, np.array([[7.0, 5.0]]))
        
        # Run backward pass
        c.backward()
        
        # Verify backward pass
        np.testing.assert_array_equal(a.grad, np.array([[1.0, 1.0]]))
        np.testing.assert_array_equal(b.grad, np.array([[-1.0, -1.0]]))

    def test_scalar_broadcasting(self):
        """Test adding/subtracting a scalar (int/float) to a 2D Tensor."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Add scalar
        c = a + 5.0
        np.testing.assert_array_equal(c.data, np.array([[6.0, 7.0], [8.0, 9.0]]))
        
        c.backward()
        np.testing.assert_array_equal(a.grad, np.ones((2, 2)))
        
        # Reset and subtract scalar (reverse)
        a.zero_grad()
        d = 10.0 - a
        np.testing.assert_array_equal(d.data, np.array([[9.0, 8.0], [7.0, 6.0]]))
        
        d.backward()
        np.testing.assert_array_equal(a.grad, np.array([[-1.0, -1.0], [-1.0, -1.0]]))

    def test_gradient_accumulation_multi_path(self):
        """Test that gradients accumulate (+=) correctly when a node is reused in the graph."""
        x = Tensor([[2.0, 3.0]])
        
        # Y = X + X  (dY/dX = 2)
        y = x + x
        # Z = Y - X  (dZ/dX = dZ/dY * dY/dX + dZ/dX = 1 * 2 - 1 = 1)
        z = y - x
        
        # Forward pass verification
        np.testing.assert_array_equal(z.data, np.array([[2.0, 3.0]]))
        
        # Backward pass verification
        z.backward()
        np.testing.assert_array_equal(x.grad, np.array([[1.0, 1.0]]))

    def test_matrix_broadcasting_shapes(self):
        """Test matrix broadcasting with shape (1, 2) and (2, 2)."""
        a = Tensor([[1.0, 2.0]])  # Shape (1, 2)
        b = Tensor([[3.0, 4.0], [5.0, 6.0]])  # Shape (2, 2)
        
        c = a + b  # a broadcasts along row axis
        np.testing.assert_array_equal(c.data, np.array([[4.0, 6.0], [6.0, 8.0]]))
        
        c.backward()
        # a should accumulate gradients from both rows: [[1, 1] + [1, 1]] = [[2, 2]]
        np.testing.assert_array_equal(a.grad, np.array([[2.0, 2.0]]))
        np.testing.assert_array_equal(b.grad, np.ones((2, 2)))

if __name__ == "__main__":
    unittest.main()

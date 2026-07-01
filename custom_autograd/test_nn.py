import unittest
import numpy as np
from tensor import Tensor
from nn import Linear, SGD

class TestNeuralNetworkAPI(unittest.TestCase):
    def test_matrix_multiplication_backward(self):
        """Test matrix multiplication forward and backward using C++ execution."""
        # A: (2, 3), B: (3, 2)
        a_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b_data = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        
        a = Tensor(a_data)
        b = Tensor(b_data)
        
        # Forward pass: A @ B
        c = a @ b
        expected_forward = np.matmul(np.array(a_data), np.array(b_data))
        np.testing.assert_array_almost_equal(c.data, expected_forward, decimal=5)
        
        # Backpropagate standard loss sum(c)
        c.backward()
        
        # dA = dY @ B^T. Under sum loss, dY = ones(2, 2).
        # dA = [[1, 1], [1, 1]] @ [[7, 9, 11], [8, 10, 12]]
        #    = [[15, 19, 23], [15, 19, 23]]
        expected_da = np.matmul(np.ones((2, 2)), np.array(b_data).T)
        np.testing.assert_array_almost_equal(a.grad, expected_da, decimal=5)
        
        # dB = A^T @ dY
        # dB = [[1, 4], [2, 5], [3, 6]] @ [[1, 1], [1, 1]]
        #    = [[5, 5], [7, 7], [9, 9]]
        expected_db = np.matmul(np.array(a_data).T, np.ones((2, 2)))
        np.testing.assert_array_almost_equal(b.grad, expected_db, decimal=5)

    def test_linear_layer_and_sgd(self):
        """Test high-level Linear layer forwarding, gradient backprop, and SGD optimizer updates."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create layer
        layer = Linear(in_features=2, out_features=2)
        
        # Override weights and biases with specific values to verify analytical math
        layer.weight.data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
        layer.bias.data = np.array([[0.1, 0.2]], dtype=np.float64)
        
        # Input tensor
        x = Tensor([[1.0, 2.0]])  # Shape (1, 2)
        
        # Forward pass: Y = X @ W + b
        # Expected: [[1.0*0.5 + 2.0*2.5 + 0.1,  1.0*1.5 + 2.0*3.5 + 0.2]]
        #          = [[0.5 + 5.0 + 0.1,  1.5 + 7.0 + 0.2]]
        #          = [[5.6, 8.7]]
        y = layer(x)
        np.testing.assert_array_almost_equal(y.data, np.array([[5.6, 8.7]]), decimal=5)
        
        # Calculate loss (sum of output elements)
        loss = y + 0.0  # simple identity, loss = y_1 + y_2
        
        # Run backpropagation
        loss.backward()
        
        # Verify gradients:
        # dY = [[1.0, 1.0]]
        # dW = X^T @ dY = [[1.0], [2.0]] @ [[1.0, 1.0]] = [[1.0, 1.0], [2.0, 2.0]]
        np.testing.assert_array_almost_equal(layer.weight.grad, np.array([[1.0, 1.0], [2.0, 2.0]]), decimal=5)
        
        # db = sum_broadcast(dY) = [[1.0, 1.0]]
        np.testing.assert_array_almost_equal(layer.bias.grad, np.array([[1.0, 1.0]]), decimal=5)
        
        # Run optimizer step with learning rate 0.1
        optimizer = SGD(layer.parameters(), lr=0.1)
        optimizer.step()
        
        # Verify weight updates:
        # W_new = [[0.5, 1.5], [2.5, 3.5]] - 0.1 * [[1, 1], [2, 2]]
        #       = [[0.4, 1.4], [2.3, 3.3]]
        np.testing.assert_array_almost_equal(
            layer.weight.data, 
            np.array([[0.4, 1.4], [2.3, 3.3]]), 
            decimal=5
        )
        
        # Verify bias updates:
        # b_new = [[0.1, 0.2]] - 0.1 * [[1, 1]] = [[0.0, 0.1]]
        np.testing.assert_array_almost_equal(
            layer.bias.data, 
            np.array([[0.0, 0.1]]), 
            decimal=5
        )
        
        # Test zero_grad
        optimizer.zero_grad()
        np.testing.assert_array_equal(layer.weight.grad, np.zeros((2, 2)))
        np.testing.assert_array_equal(layer.bias.grad, np.zeros((1, 2)))

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import torch
from tensor import Tensor

class TestAutogradEngineAgainstPyTorch(unittest.TestCase):
    def test_multi_layer_perceptron_alignment(self):
        """
        Verify that a multi-layer forward pass and subsequent backward pass
        computes exactly the same data values and gradients as PyTorch.
        """
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Dimensions
        batch_size = 3
        in_features = 5
        hidden_1 = 4
        hidden_2 = 3
        out_features = 2
        
        # Initialize raw random values
        x_raw = np.random.randn(batch_size, in_features)
        
        w1_raw = np.random.randn(in_features, hidden_1)
        b1_raw = np.random.randn(1, hidden_1)
        
        w2_raw = np.random.randn(hidden_1, hidden_2)
        b2_raw = np.random.randn(1, hidden_2)
        
        w3_raw = np.random.randn(hidden_2, out_features)
        b3_raw = np.random.randn(1, out_features)
        
        # --- PYTORCH GRAPH SETUP ---
        x_pt = torch.tensor(x_raw, requires_grad=True, dtype=torch.float64)
        w1_pt = torch.tensor(w1_raw, requires_grad=True, dtype=torch.float64)
        b1_pt = torch.tensor(b1_raw, requires_grad=True, dtype=torch.float64)
        w2_pt = torch.tensor(w2_raw, requires_grad=True, dtype=torch.float64)
        b2_pt = torch.tensor(b2_raw, requires_grad=True, dtype=torch.float64)
        w3_pt = torch.tensor(w3_raw, requires_grad=True, dtype=torch.float64)
        b3_pt = torch.tensor(b3_raw, requires_grad=True, dtype=torch.float64)
        
        # --- CUSTOM ENGINE GRAPH SETUP ---
        x_custom = Tensor(x_raw)
        w1_custom = Tensor(w1_raw)
        b1_custom = Tensor(b1_raw)
        w2_custom = Tensor(w2_raw)
        b2_custom = Tensor(b2_raw)
        w3_custom = Tensor(w3_raw)
        b3_custom = Tensor(b3_raw)
        
        # ====================================================
        # FORWARD PASS
        # ====================================================
        
        # PyTorch Forward
        y1_pt = torch.matmul(x_pt, w1_pt) + b1_pt
        y1_pt.retain_grad()  # PyTorch discards intermediate grads unless retained
        
        y2_pt = torch.matmul(y1_pt, w2_pt) + b2_pt
        y2_pt.retain_grad()
        
        y3_pt = torch.matmul(y2_pt, w3_pt) + b3_pt
        y3_pt.retain_grad()
        
        loss_pt = y3_pt.sum()
        
        # Custom Engine Forward
        y1_custom = x_custom @ w1_custom + b1_custom
        y2_custom = y1_custom @ w2_custom + b2_custom
        y3_custom = y2_custom @ w3_custom + b3_custom
        loss_custom = y3_custom.sum()
        
        # --- VERIFY FORWARD ACTIVATIONS (Tolerance: 1e-5) ---
        np.testing.assert_allclose(y1_custom.data, y1_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y2_custom.data, y2_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y3_custom.data, y3_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(loss_custom.data, loss_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
        
        # ====================================================
        # BACKWARD PASS
        # ====================================================
        
        # Trigger backpropagation
        loss_pt.backward()
        loss_custom.backward()
        
        # --- VERIFY GRADIENTS (Tolerance: 1e-5) ---
        # 1. Parameter Weights & Biases
        np.testing.assert_allclose(w3_custom.grad, w3_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b3_custom.grad, b3_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(w2_custom.grad, w2_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b2_custom.grad, b2_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(w1_custom.grad, w1_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b1_custom.grad, b1_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        
        # 2. Input features
        np.testing.assert_allclose(x_custom.grad, x_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        
        # 3. Intermediate Layer Activations
        np.testing.assert_allclose(y3_custom.grad, y3_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y2_custom.grad, y2_pt.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y1_custom.grad, y1_pt.grad.numpy(), rtol=1e-5, atol=1e-5)

        print("[Verification] Custom Autograd outputs and gradients matched PyTorch perfectly!")

if __name__ == "__main__":
    unittest.main()

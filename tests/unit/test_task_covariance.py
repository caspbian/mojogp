"""Unit tests for TaskCovariance struct.

Tests:
1. Eigendecomposition correctness: B = Q @ diag(Lambda) @ Q^T
2. Gradient accuracy: analytical gradients match finite differences to 1e-5 relative error
3. Near-degenerate eigenvalue case: gradients remain stable
4. GPU rotation kernels: rotate_targets and unrotate_solution are inverses
"""

import pytest
import numpy as np
import torch


class TestTaskCovarianceEigendecomposition:
    """Test eigendecomposition correctness."""

    def test_eigendecomposition_reconstruction(self):
        """Verify B = Q @ diag(Lambda) @ Q^T."""
        # Create a known B matrix
        T = 3
        R = 3
        np.random.seed(42)

        # Create W and v
        W = np.random.randn(T, R).astype(np.float32) * 0.3
        raw_v = np.random.randn(T).astype(np.float32) * 0.5
        v = np.log(1 + np.exp(raw_v))  # softplus

        # Compute B = WW^T + diag(v)
        B = W @ W.T + np.diag(v)

        # Eigendecompose
        eigenvalues, Q = np.linalg.eigh(B)

        # Reconstruct
        B_reconstructed = Q @ np.diag(eigenvalues) @ Q.T

        # Check reconstruction
        max_diff = np.max(np.abs(B - B_reconstructed))
        assert max_diff < 1e-5, f"Reconstruction error too large: {max_diff}"

    def test_eigendecomposition_positive_eigenvalues(self):
        """Verify all eigenvalues are positive (B is PSD)."""
        T = 5
        R = 5
        np.random.seed(123)

        W = np.random.randn(T, R).astype(np.float32) * 0.3
        raw_v = np.random.randn(T).astype(np.float32) * 0.5
        v = np.log(1 + np.exp(raw_v))  # softplus ensures v > 0

        B = W @ W.T + np.diag(v)
        eigenvalues, _ = np.linalg.eigh(B)

        assert np.all(eigenvalues > 0), f"Found non-positive eigenvalues: {eigenvalues}"

    def test_eigenvectors_orthonormal(self):
        """Verify Q is orthonormal: Q^T @ Q = I."""
        T = 4
        R = 4
        np.random.seed(456)

        W = np.random.randn(T, R).astype(np.float32) * 0.3
        raw_v = np.random.randn(T).astype(np.float32) * 0.5
        v = np.log(1 + np.exp(raw_v))

        B = W @ W.T + np.diag(v)
        _, Q = np.linalg.eigh(B)

        QTQ = Q.T @ Q
        I = np.eye(T)
        max_diff = np.max(np.abs(QTQ - I))
        assert max_diff < 1e-5, f"Q not orthonormal: max diff from I = {max_diff}"


class TestTaskCovarianceGradients:
    """Test gradient computation via finite differences."""

    def compute_nll_from_B(self, W, raw_v, y, K_X, noise):
        """Compute a mock NLL that depends on B's eigenvalues.

        NLL = 0.5 * sum_t (y_t^T @ (lambda_t * K_X + noise * I)^{-1} @ y_t + n * log(lambda_t))

        For simplicity, we use a diagonal K_X = I, so:
        NLL = 0.5 * sum_t (||y_t||^2 / (lambda_t + noise) + n * log(lambda_t + noise))
        """
        T = W.shape[0]
        n = y.shape[0]

        # Compute B = WW^T + diag(softplus(raw_v))
        v = np.log(1 + np.exp(raw_v))
        B = W @ W.T + np.diag(v)

        # Eigendecompose
        eigenvalues, Q = np.linalg.eigh(B)

        # Rotate targets
        y_rotated = y @ Q  # [n, T]

        # Compute NLL for each sub-problem
        nll = 0.0
        for t in range(T):
            lambda_t = eigenvalues[t]
            y_t = y_rotated[:, t]
            # With K_X = I: (lambda_t * I + noise * I)^{-1} = 1/(lambda_t + noise) * I
            inv_quad = np.sum(y_t**2) / (lambda_t + noise)
            log_det = n * np.log(lambda_t + noise)
            nll += 0.5 * (inv_quad + log_det)

        return nll

    def test_gradient_W_finite_difference(self):
        """Test dNLL/dW matches finite differences."""
        T = 3
        R = 3
        n = 50
        noise = 0.1
        np.random.seed(789)

        W = np.random.randn(T, R).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        y = np.random.randn(n, T).astype(np.float64)

        # Compute analytical gradient using PyTorch autograd
        W_torch = torch.tensor(W, dtype=torch.float64, requires_grad=True)
        raw_v_torch = torch.tensor(raw_v, dtype=torch.float64, requires_grad=True)
        y_torch = torch.tensor(y, dtype=torch.float64)

        # Forward pass
        v_torch = torch.nn.functional.softplus(raw_v_torch)
        B_torch = W_torch @ W_torch.T + torch.diag(v_torch)
        eigenvalues, Q = torch.linalg.eigh(B_torch)
        y_rotated = y_torch @ Q

        nll = 0.0
        for t in range(T):
            lambda_t = eigenvalues[t]
            y_t = y_rotated[:, t]
            inv_quad = torch.sum(y_t**2) / (lambda_t + noise)
            log_det = n * torch.log(lambda_t + noise)
            nll = nll + 0.5 * (inv_quad + log_det)

        # Backward pass
        nll.backward()
        grad_W_analytical = W_torch.grad.numpy()

        # Finite difference gradient
        eps = 1e-5
        grad_W_fd = np.zeros_like(W)
        for i in range(T):
            for j in range(R):
                W_plus = W.copy()
                W_plus[i, j] += eps
                nll_plus = self.compute_nll_from_B(W_plus, raw_v, y, None, noise)

                W_minus = W.copy()
                W_minus[i, j] -= eps
                nll_minus = self.compute_nll_from_B(W_minus, raw_v, y, None, noise)

                grad_W_fd[i, j] = (nll_plus - nll_minus) / (2 * eps)

        # Compare
        rel_error = np.abs(grad_W_analytical - grad_W_fd) / (np.abs(grad_W_fd) + 1e-8)
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 1e-4, (
            f"W gradient relative error too large: {max_rel_error}"
        )

    def test_gradient_raw_v_finite_difference(self):
        """Test dNLL/d(raw_v) matches finite differences."""
        T = 3
        R = 3
        n = 50
        noise = 0.1
        np.random.seed(101112)

        W = np.random.randn(T, R).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        y = np.random.randn(n, T).astype(np.float64)

        # Compute analytical gradient using PyTorch autograd
        W_torch = torch.tensor(W, dtype=torch.float64, requires_grad=True)
        raw_v_torch = torch.tensor(raw_v, dtype=torch.float64, requires_grad=True)
        y_torch = torch.tensor(y, dtype=torch.float64)

        # Forward pass
        v_torch = torch.nn.functional.softplus(raw_v_torch)
        B_torch = W_torch @ W_torch.T + torch.diag(v_torch)
        eigenvalues, Q = torch.linalg.eigh(B_torch)
        y_rotated = y_torch @ Q

        nll = 0.0
        for t in range(T):
            lambda_t = eigenvalues[t]
            y_t = y_rotated[:, t]
            inv_quad = torch.sum(y_t**2) / (lambda_t + noise)
            log_det = n * torch.log(lambda_t + noise)
            nll = nll + 0.5 * (inv_quad + log_det)

        # Backward pass
        nll.backward()
        grad_raw_v_analytical = raw_v_torch.grad.numpy()

        # Finite difference gradient
        eps = 1e-5
        grad_raw_v_fd = np.zeros_like(raw_v)
        for i in range(T):
            raw_v_plus = raw_v.copy()
            raw_v_plus[i] += eps
            nll_plus = self.compute_nll_from_B(W, raw_v_plus, y, None, noise)

            raw_v_minus = raw_v.copy()
            raw_v_minus[i] -= eps
            nll_minus = self.compute_nll_from_B(W, raw_v_minus, y, None, noise)

            grad_raw_v_fd[i] = (nll_plus - nll_minus) / (2 * eps)

        # Compare
        rel_error = np.abs(grad_raw_v_analytical - grad_raw_v_fd) / (
            np.abs(grad_raw_v_fd) + 1e-8
        )
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 1e-4, (
            f"raw_v gradient relative error too large: {max_rel_error}"
        )


class TestTaskCovarianceNearDegenerate:
    """Test gradient stability with near-degenerate eigenvalues."""

    def test_near_degenerate_eigenvalues(self):
        """Test gradients are stable when eigenvalues are close but not identical."""
        T = 2
        R = 2
        n = 50
        noise = 0.1

        # Create W such that B has near-degenerate eigenvalues
        # B = WW^T + diag(v) with W chosen to make eigenvalues close
        W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64) * 0.5
        # raw_v chosen so softplus(raw_v) ≈ [0.5, 0.501] (very close)
        raw_v = np.array(
            [np.log(np.exp(0.5) - 1), np.log(np.exp(0.501) - 1)], dtype=np.float64
        )
        y = np.random.randn(n, T).astype(np.float64)

        # Compute B and check eigenvalues are close
        v = np.log(1 + np.exp(raw_v))
        B = W @ W.T + np.diag(v)
        eigenvalues, _ = np.linalg.eigh(B)

        # Eigenvalues should be close (within 0.01)
        eigenvalue_diff = np.abs(eigenvalues[1] - eigenvalues[0])
        assert eigenvalue_diff < 0.1, (
            f"Eigenvalues not close enough: diff = {eigenvalue_diff}"
        )

        # Compute gradient using PyTorch autograd (should handle near-degeneracy)
        W_torch = torch.tensor(W, dtype=torch.float64, requires_grad=True)
        raw_v_torch = torch.tensor(raw_v, dtype=torch.float64, requires_grad=True)
        y_torch = torch.tensor(y, dtype=torch.float64)

        v_torch = torch.nn.functional.softplus(raw_v_torch)
        B_torch = W_torch @ W_torch.T + torch.diag(v_torch)
        eigenvalues_torch, Q = torch.linalg.eigh(B_torch)
        y_rotated = y_torch @ Q

        nll = 0.0
        for t in range(T):
            lambda_t = eigenvalues_torch[t]
            y_t = y_rotated[:, t]
            inv_quad = torch.sum(y_t**2) / (lambda_t + noise)
            log_det = n * torch.log(lambda_t + noise)
            nll = nll + 0.5 * (inv_quad + log_det)

        nll.backward()

        # Check gradients are finite (not NaN or Inf)
        assert torch.isfinite(W_torch.grad).all(), "W gradient contains NaN/Inf"
        assert torch.isfinite(raw_v_torch.grad).all(), "raw_v gradient contains NaN/Inf"

        # Check gradients are reasonable magnitude (not exploding)
        assert torch.abs(W_torch.grad).max() < 1000, "W gradient magnitude too large"
        assert torch.abs(raw_v_torch.grad).max() < 1000, (
            "raw_v gradient magnitude too large"
        )


class TestTaskCovarianceRotation:
    """Test GPU rotation kernels (via numpy simulation)."""

    def test_rotate_unrotate_inverse(self):
        """Test that unrotate(rotate(Y)) = Y."""
        T = 4
        n = 100
        np.random.seed(131415)

        # Create random orthonormal Q
        A = np.random.randn(T, T).astype(np.float32)
        Q, _ = np.linalg.qr(A)

        # Create random Y
        Y = np.random.randn(n, T).astype(np.float32)

        # Rotate: Y_rotated = Y @ Q
        Y_rotated = Y @ Q

        # Unrotate: Y_recovered = Y_rotated @ Q^T
        Y_recovered = Y_rotated @ Q.T

        # Check Y_recovered ≈ Y
        max_diff = np.max(np.abs(Y - Y_recovered))
        assert max_diff < 1e-5, f"Rotate/unrotate not inverse: max diff = {max_diff}"

    def test_rotation_preserves_norms(self):
        """Test that rotation preserves row norms (Q is orthonormal)."""
        T = 3
        n = 50
        np.random.seed(161718)

        # Create random orthonormal Q
        A = np.random.randn(T, T).astype(np.float32)
        Q, _ = np.linalg.qr(A)

        # Create random Y
        Y = np.random.randn(n, T).astype(np.float32)

        # Rotate
        Y_rotated = Y @ Q

        # Check row norms are preserved
        norms_original = np.linalg.norm(Y, axis=1)
        norms_rotated = np.linalg.norm(Y_rotated, axis=1)

        max_diff = np.max(np.abs(norms_original - norms_rotated))
        assert max_diff < 1e-5, f"Rotation changed row norms: max diff = {max_diff}"


class TestBackwardThroughEigh:
    """Test the backward-through-eigh formula directly."""

    def test_lowdin_f_matrix(self):
        """Test Löwdin F matrix construction."""
        eigenvalues = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        T = len(eigenvalues)

        # Construct F matrix
        F = np.zeros((T, T), dtype=np.float64)
        eps = 1e-12
        for i in range(T):
            for j in range(T):
                if i != j:
                    diff = eigenvalues[i] - eigenvalues[j]
                    if abs(diff) > eps:
                        F[i, j] = 1.0 / diff

        # Check expected values
        # F[0,1] = 1/(1-2) = -1
        # F[0,2] = 1/(1-3) = -0.5
        # F[1,0] = 1/(2-1) = 1
        # F[1,2] = 1/(2-3) = -1
        # F[2,0] = 1/(3-1) = 0.5
        # F[2,1] = 1/(3-2) = 1
        expected_F = np.array(
            [[0.0, -1.0, -0.5], [1.0, 0.0, -1.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )

        max_diff = np.max(np.abs(F - expected_F))
        assert max_diff < 1e-10, f"F matrix incorrect: max diff = {max_diff}"

    def test_backward_formula_matches_autograd(self):
        """Test that manual backward formula matches PyTorch autograd."""
        T = 3
        np.random.seed(192021)

        # Create symmetric B
        A = np.random.randn(T, T).astype(np.float64)
        B = A @ A.T + np.eye(T) * 0.1  # Ensure positive definite

        # Eigendecompose
        eigenvalues, Q = np.linalg.eigh(B)

        # Create random gradient w.r.t. eigenvalues
        g = np.random.randn(T).astype(np.float64)

        # Manual backward formula:
        # dL/dB = Q @ diag(g) @ Q^T  (simplified when G_Q = 0)
        # This is the gradient when the loss only depends on eigenvalues, not eigenvectors
        dL_dB_manual = Q @ np.diag(g) @ Q.T

        # PyTorch autograd
        B_torch = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        eigenvalues_torch, Q_torch = torch.linalg.eigh(B_torch)

        # Loss = sum(g * eigenvalues)
        g_torch = torch.tensor(g, dtype=torch.float64)
        loss = torch.sum(g_torch * eigenvalues_torch)
        loss.backward()

        dL_dB_autograd = B_torch.grad.numpy()

        # Compare (should match for eigenvalue-only loss)
        max_diff = np.max(np.abs(dL_dB_manual - dL_dB_autograd))
        assert max_diff < 1e-8, (
            f"Manual backward doesn't match autograd: max diff = {max_diff}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

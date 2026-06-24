"""Unit tests for per-task noise (Rakitsch symmetrization).

Tests:
1. Symmetrization correctness: NLL via symmetrized path matches full system
2. Gradient accuracy: analytical gradients match finite differences
3. Condition number behavior: symmetrization doesn't degrade conditioning excessively
4. Edge cases: homoscedastic noise, extreme noise ratios
"""

import pytest
import numpy as np
import torch


def softplus(x):
    """Softplus function: log(1 + exp(x))."""
    return np.log(1 + np.exp(x))


def softplus_derivative(x):
    """Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0):
    """Compute RBF kernel matrix."""
    X = torch.tensor(X, dtype=torch.float64) if not isinstance(X, torch.Tensor) else X
    sq_dist = torch.cdist(X, X, p=2) ** 2
    return outputscale * torch.exp(-0.5 * sq_dist / lengthscale**2)


def make_B_from_W_v(W, raw_v):
    """Construct B = WW^T + diag(softplus(raw_v))."""
    v = softplus(raw_v)
    return W @ W.T + np.diag(v)


class TestRakitschSymmetrization:
    """Test Rakitsch symmetrization for per-task noise."""

    def compute_nll_full_system(self, K_X, B, y, task_noise):
        """Compute NLL with per-task noise via full Kronecker system.

        Full covariance: K_X ⊗ B + D_full
        where D_full = I_n ⊗ diag(task_noise)
        """
        n, T = y.shape

        # D_task = diag(task_noise) repeated n times
        D_full = torch.kron(torch.eye(n, dtype=K_X.dtype), torch.diag(task_noise))

        # Full covariance: K_X ⊗ B + D_full
        K_full = torch.kron(K_X, B) + D_full

        y_flat = y.flatten()
        L = torch.linalg.cholesky(K_full)
        alpha = torch.cholesky_solve(y_flat.unsqueeze(1), L).squeeze()
        log_det = 2 * torch.sum(torch.log(torch.diag(L))).item()
        inv_quad = (y_flat @ alpha).item()

        nll = 0.5 * (inv_quad + log_det + n * T * np.log(2 * np.pi))
        return nll

    def compute_nll_symmetrized(self, K_X, B, y, task_noise):
        """Compute NLL via Rakitsch symmetrization.

        1. D^{-1/2} = diag(1/sqrt(task_noise))
        2. B_sym = D^{-1/2} @ B @ D^{-1/2}
        3. Eigendecompose B_sym
        4. Transform targets: y_tilde = y @ D^{-1/2} @ Q_sym
        5. Sub-problems have noise = 1.0
        6. Add log|D| correction
        """
        n, T = y.shape

        # D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))

        # Symmetrized B: B_sym = D^{-1/2} B D^{-1/2}
        B_sym = D_inv_sqrt @ B @ D_inv_sqrt

        # Eigendecompose B_sym
        Lambda_sym, Q_sym = torch.linalg.eigh(B_sym)

        # Transform y: y_tilde = y @ D^{-1/2} @ Q_sym
        transform = D_inv_sqrt @ Q_sym
        y_transformed = y @ transform

        inv_quad_total = 0.0
        log_det_total = 0.0

        for t in range(T):
            lambda_t = Lambda_sym[t].item()
            y_t = y_transformed[:, t]

            # K_t = lambda_t * K_X + I (noise is 1.0 after symmetrization)
            K_t = lambda_t * K_X + torch.eye(n, dtype=K_X.dtype)

            L_t = torch.linalg.cholesky(K_t)
            alpha_t = torch.cholesky_solve(y_t.unsqueeze(1), L_t).squeeze()
            log_det_t = 2 * torch.sum(torch.log(torch.diag(L_t))).item()

            inv_quad_total += (y_t @ alpha_t).item()
            log_det_total += log_det_t

        # Add log|D_task| correction (n copies of the T x T diagonal)
        log_det_D = n * torch.sum(torch.log(task_noise)).item()

        nll = 0.5 * (
            inv_quad_total + log_det_total + log_det_D + n * T * np.log(2 * np.pi)
        )
        return nll, {
            "log_det_sub": log_det_total,
            "log_det_D": log_det_D,
            "inv_quad": inv_quad_total,
            "Lambda_sym": Lambda_sym.numpy(),
            "cond_B": torch.linalg.cond(B).item(),
            "cond_B_sym": torch.linalg.cond(B_sym).item(),
        }

    def test_symmetrization_homoscedastic(self):
        """Test symmetrization with equal noise across tasks."""
        np.random.seed(42)
        torch.manual_seed(42)

        n, T, d = 50, 3, 5

        # Create B matrix
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        # Create K_X
        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)

        # Homoscedastic noise
        task_noise = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        # Generate data
        D_full = torch.kron(torch.eye(n, dtype=torch.float64), torch.diag(task_noise))
        K_full = torch.kron(K_X, B_torch) + D_full
        L_full = torch.linalg.cholesky(K_full)
        y = (L_full @ torch.randn(n * T, dtype=torch.float64)).reshape(n, T)

        # Compute NLL both ways
        nll_full = self.compute_nll_full_system(K_X, B_torch, y, task_noise)
        nll_sym, info = self.compute_nll_symmetrized(K_X, B_torch, y, task_noise)

        rel_err = abs(nll_full - nll_sym) / abs(nll_full)
        assert rel_err < 1e-5, f"Homoscedastic NLL mismatch: rel_err={rel_err:.2e}"

    def test_symmetrization_heteroscedastic(self):
        """Test symmetrization with different noise per task."""
        np.random.seed(123)
        torch.manual_seed(123)

        n, T, d = 50, 3, 5

        # Create B matrix
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        # Create K_X
        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)

        # Heteroscedastic noise (10x ratio)
        task_noise = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)

        # Generate data
        D_full = torch.kron(torch.eye(n, dtype=torch.float64), torch.diag(task_noise))
        K_full = torch.kron(K_X, B_torch) + D_full
        L_full = torch.linalg.cholesky(K_full)
        y = (L_full @ torch.randn(n * T, dtype=torch.float64)).reshape(n, T)

        # Compute NLL both ways
        nll_full = self.compute_nll_full_system(K_X, B_torch, y, task_noise)
        nll_sym, info = self.compute_nll_symmetrized(K_X, B_torch, y, task_noise)

        rel_err = abs(nll_full - nll_sym) / abs(nll_full)
        assert rel_err < 1e-5, f"Heteroscedastic NLL mismatch: rel_err={rel_err:.2e}"

    def test_symmetrization_extreme_noise_ratio(self):
        """Test symmetrization with extreme noise ratio (1000x)."""
        np.random.seed(456)
        torch.manual_seed(456)

        n, T, d = 30, 3, 5

        # Create B matrix
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        # Create K_X
        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)

        # Extreme noise ratio (1000x)
        task_noise = torch.tensor([0.001, 0.1, 1.0], dtype=torch.float64)

        # Generate data
        D_full = torch.kron(torch.eye(n, dtype=torch.float64), torch.diag(task_noise))
        K_full = torch.kron(K_X, B_torch) + D_full
        L_full = torch.linalg.cholesky(K_full)
        y = (L_full @ torch.randn(n * T, dtype=torch.float64)).reshape(n, T)

        # Compute NLL both ways
        nll_full = self.compute_nll_full_system(K_X, B_torch, y, task_noise)
        nll_sym, info = self.compute_nll_symmetrized(K_X, B_torch, y, task_noise)

        rel_err = abs(nll_full - nll_sym) / abs(nll_full)
        # Allow slightly larger error for extreme ratios
        assert rel_err < 1e-4, f"Extreme ratio NLL mismatch: rel_err={rel_err:.2e}"


class TestPerTaskNoiseGradients:
    """Test gradient computation for per-task noise."""

    def compute_nll_with_per_task_noise(self, K_X, W, raw_v, raw_noise_per_task, y):
        """Compute NLL with per-task noise using PyTorch autograd.

        This is the reference implementation for gradient testing.
        """
        n, T = y.shape

        # Compute B = WW^T + diag(softplus(raw_v))
        v = torch.nn.functional.softplus(raw_v)
        B = W @ W.T + torch.diag(v)

        # Compute task_noise = softplus(raw_noise_per_task)
        task_noise = torch.nn.functional.softplus(raw_noise_per_task)

        # D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))

        # Symmetrized B
        B_sym = D_inv_sqrt @ B @ D_inv_sqrt

        # Eigendecompose
        Lambda_sym, Q_sym = torch.linalg.eigh(B_sym)

        # Transform targets
        transform = D_inv_sqrt @ Q_sym
        y_transformed = y @ transform

        # Compute NLL for each sub-problem
        nll = torch.tensor(0.0, dtype=torch.float64)
        for t in range(T):
            lambda_t = Lambda_sym[t]
            y_t = y_transformed[:, t]

            # K_t = lambda_t * K_X + I
            K_t = lambda_t * K_X + torch.eye(n, dtype=K_X.dtype)

            L_t = torch.linalg.cholesky(K_t)
            alpha_t = torch.cholesky_solve(y_t.unsqueeze(1), L_t).squeeze()
            log_det_t = 2 * torch.sum(torch.log(torch.diag(L_t)))

            nll = nll + 0.5 * (y_t @ alpha_t + log_det_t)

        # Add log|D| correction
        log_det_D = n * torch.sum(torch.log(task_noise))
        nll = nll + 0.5 * log_det_D

        # Add constant
        nll = nll + 0.5 * n * T * np.log(2 * np.pi)

        return nll

    def test_gradient_raw_noise_per_task_finite_difference(self):
        """Test dNLL/d(raw_noise_per_task) matches finite differences."""
        np.random.seed(789)
        torch.manual_seed(789)

        n, T, d = 30, 3, 5

        # Create data
        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)
        y = torch.randn(n, T, dtype=torch.float64)

        # Parameters - use leaf tensors
        W_data = torch.randn(T, T, dtype=torch.float64) * 0.3
        raw_v_data = torch.randn(T, dtype=torch.float64) * 0.5
        raw_noise_data = torch.randn(T, dtype=torch.float64) * 0.5

        W = W_data.clone().requires_grad_(True)
        raw_v = raw_v_data.clone().requires_grad_(True)
        raw_noise_per_task = raw_noise_data.clone().requires_grad_(True)

        # Compute analytical gradient
        nll = self.compute_nll_with_per_task_noise(K_X, W, raw_v, raw_noise_per_task, y)
        nll.backward()
        grad_analytical = raw_noise_per_task.grad.numpy().copy()

        # Compute finite difference gradient
        eps = 1e-5
        grad_fd = np.zeros(T, dtype=np.float64)

        for t in range(T):
            # Plus
            raw_noise_plus = raw_noise_per_task.detach().clone()
            raw_noise_plus[t] += eps
            nll_plus = self.compute_nll_with_per_task_noise(
                K_X, W.detach(), raw_v.detach(), raw_noise_plus, y
            )

            # Minus
            raw_noise_minus = raw_noise_per_task.detach().clone()
            raw_noise_minus[t] -= eps
            nll_minus = self.compute_nll_with_per_task_noise(
                K_X, W.detach(), raw_v.detach(), raw_noise_minus, y
            )

            grad_fd[t] = (nll_plus.item() - nll_minus.item()) / (2 * eps)

        # Compare
        rel_error = np.abs(grad_analytical - grad_fd) / (np.abs(grad_fd) + 1e-8)
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 1e-4, (
            f"raw_noise_per_task gradient relative error too large: {max_rel_error}\n"
            f"Analytical: {grad_analytical}\n"
            f"Finite diff: {grad_fd}"
        )

    def test_gradient_W_with_per_task_noise(self):
        """Test dNLL/dW matches finite differences with per-task noise."""
        np.random.seed(101112)
        torch.manual_seed(101112)

        n, T, d = 30, 3, 5
        R = T  # Full rank

        # Create data
        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)
        y = torch.randn(n, T, dtype=torch.float64)

        # Parameters - use leaf tensors
        W_data = torch.randn(T, R, dtype=torch.float64) * 0.3
        raw_v_data = torch.randn(T, dtype=torch.float64) * 0.5

        W = W_data.clone().requires_grad_(True)
        raw_v = raw_v_data.clone().requires_grad_(True)
        raw_noise_per_task = torch.tensor(
            [0.0, 0.5, 1.0], dtype=torch.float64
        )  # Fixed noise

        # Compute analytical gradient
        nll = self.compute_nll_with_per_task_noise(K_X, W, raw_v, raw_noise_per_task, y)
        nll.backward()
        grad_W_analytical = W.grad.numpy().copy()

        # Compute finite difference gradient
        eps = 1e-5
        grad_W_fd = np.zeros((T, R), dtype=np.float64)

        for i in range(T):
            for j in range(R):
                # Plus
                W_plus = W.detach().clone()
                W_plus[i, j] += eps
                nll_plus = self.compute_nll_with_per_task_noise(
                    K_X, W_plus, raw_v.detach(), raw_noise_per_task, y
                )

                # Minus
                W_minus = W.detach().clone()
                W_minus[i, j] -= eps
                nll_minus = self.compute_nll_with_per_task_noise(
                    K_X, W_minus, raw_v.detach(), raw_noise_per_task, y
                )

                grad_W_fd[i, j] = (nll_plus.item() - nll_minus.item()) / (2 * eps)

        # Compare
        rel_error = np.abs(grad_W_analytical - grad_W_fd) / (np.abs(grad_W_fd) + 1e-8)
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 1e-4, (
            f"W gradient relative error too large: {max_rel_error}"
        )


class TestConditionNumberBehavior:
    """Test condition number behavior under symmetrization."""

    def test_condition_number_not_degraded_significantly(self):
        """Test that symmetrization doesn't degrade condition number excessively."""
        np.random.seed(131415)

        T = 4

        # Create well-conditioned B
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        # Moderate noise ratio (10x)
        task_noise = torch.tensor([0.1, 0.3, 0.5, 1.0], dtype=torch.float64)

        # Symmetrize
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))
        B_sym = D_inv_sqrt @ B_torch @ D_inv_sqrt

        cond_B = torch.linalg.cond(B_torch).item()
        cond_B_sym = torch.linalg.cond(B_sym).item()

        # Condition number shouldn't increase by more than 10x for moderate noise ratios
        cond_ratio = cond_B_sym / cond_B
        assert cond_ratio < 10, (
            f"Condition number degraded too much: {cond_B:.2f} -> {cond_B_sym:.2f} ({cond_ratio:.2f}x)"
        )

    def test_extreme_noise_ratio_warning(self):
        """Test that extreme noise ratios lead to high condition numbers."""
        np.random.seed(161718)

        T = 3

        # Create B
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        # Extreme noise ratio (10000x)
        task_noise = torch.tensor([0.0001, 0.1, 1.0], dtype=torch.float64)

        # Symmetrize
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))
        B_sym = D_inv_sqrt @ B_torch @ D_inv_sqrt

        cond_B_sym = torch.linalg.cond(B_sym).item()

        # With 10000x noise ratio, condition number should be high
        # This is expected behavior - just documenting it
        assert cond_B_sym > 100, (
            f"Expected high condition number for extreme noise ratio, got {cond_B_sym:.2f}"
        )


class TestLogDetDCorrection:
    """Test the log|D| correction term."""

    def test_log_det_D_computation(self):
        """Test that log|D| is computed correctly."""
        n = 50
        T = 3

        task_noise = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)

        # log|D| = n * sum(log(task_noise))
        # Because D = I_n ⊗ diag(task_noise), so |D| = prod(task_noise)^n
        expected_log_det_D = n * torch.sum(torch.log(task_noise)).item()

        # Verify by computing determinant directly for small n
        D_full = torch.kron(torch.eye(n, dtype=torch.float64), torch.diag(task_noise))
        actual_log_det_D = torch.logdet(D_full).item()

        rel_err = abs(expected_log_det_D - actual_log_det_D) / abs(actual_log_det_D)
        assert rel_err < 1e-10, f"log|D| computation error: rel_err={rel_err:.2e}"

    def test_log_det_D_with_unit_noise(self):
        """Test that log|D| = 0 when all noise = 1."""
        n = 50
        T = 3

        task_noise = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        log_det_D = n * torch.sum(torch.log(task_noise)).item()

        assert abs(log_det_D) < 1e-10, (
            f"log|D| should be 0 for unit noise, got {log_det_D}"
        )


class TestTargetTransformation:
    """Test the target transformation y_tilde = y @ D^{-1/2} @ Q_sym."""

    def test_transformation_invertible(self):
        """Test that target transformation is invertible."""
        np.random.seed(192021)
        torch.manual_seed(192021)

        n, T = 50, 3

        # Create B and task_noise
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        task_noise = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)

        # Symmetrize and eigendecompose
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))
        B_sym = D_inv_sqrt @ B_torch @ D_inv_sqrt
        Lambda_sym, Q_sym = torch.linalg.eigh(B_sym)

        # Create random y
        y = torch.randn(n, T, dtype=torch.float64)

        # Transform
        transform = D_inv_sqrt @ Q_sym
        y_transformed = y @ transform

        # Inverse transform: y = y_transformed @ Q_sym^T @ D^{1/2}
        D_sqrt = torch.diag(torch.sqrt(task_noise))
        y_recovered = y_transformed @ Q_sym.T @ D_sqrt

        max_diff = torch.max(torch.abs(y - y_recovered)).item()
        assert max_diff < 1e-10, (
            f"Target transformation not invertible: max_diff={max_diff}"
        )

    def test_transformation_preserves_structure(self):
        """Test that transformation preserves the Kronecker structure."""
        np.random.seed(222324)
        torch.manual_seed(222324)

        n, T, d = 30, 3, 5

        # Create B and K_X
        W = np.random.randn(T, T).astype(np.float64) * 0.3
        raw_v = np.random.randn(T).astype(np.float64) * 0.5
        B = make_B_from_W_v(W, raw_v)
        B_torch = torch.tensor(B, dtype=torch.float64)

        X = torch.randn(n, d, dtype=torch.float64)
        K_X = rbf_kernel_matrix(X, lengthscale=1.0, outputscale=1.0)

        task_noise = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)

        # Symmetrize
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(task_noise))
        B_sym = D_inv_sqrt @ B_torch @ D_inv_sqrt
        Lambda_sym, Q_sym = torch.linalg.eigh(B_sym)

        # The key property: after transformation, the covariance becomes
        # K_X ⊗ diag(Lambda_sym) + I
        # which decomposes into T independent problems

        # Verify eigenvalues are positive
        assert torch.all(Lambda_sym > 0), f"Non-positive eigenvalues: {Lambda_sym}"

        # Verify Q_sym is orthonormal
        QTQ = Q_sym.T @ Q_sym
        I = torch.eye(T, dtype=torch.float64)
        max_diff = torch.max(torch.abs(QTQ - I)).item()
        assert max_diff < 1e-10, f"Q_sym not orthonormal: max_diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

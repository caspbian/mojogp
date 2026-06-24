"""
Tests for kernel gradient computation.

Compares analytical gradients against finite differences and PyTorch autograd.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist


class TestKernelGradientsFiniteDiff:
    """Test kernel gradients via finite differences."""

    def _finite_diff_gradient(self, func, x, eps=1e-5):
        """Compute gradient via central finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad

    def test_rbf_gradient_lengthscale_finite_diff(self, random_seed):
        """Test RBF gradient w.r.t. lengthscale via finite differences."""
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def rbf_kernel(X, ls, os):
            dist_sq = cdist(X, X, metric='sqeuclidean')
            return os * np.exp(-dist_sq / (2 * ls ** 2))

        # Finite difference gradient
        K_plus = rbf_kernel(X, lengthscale + eps, outputscale)
        K_minus = rbf_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = K * r^2 / l^3
        K = rbf_kernel(X, lengthscale, outputscale)
        dist_sq = cdist(X, X, metric='sqeuclidean')
        grad_analytical = K * dist_sq / (lengthscale ** 3)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_rbf_gradient_outputscale_finite_diff(self, random_seed):
        """Test RBF gradient w.r.t. outputscale via finite differences."""
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def rbf_kernel(X, ls, os):
            dist_sq = cdist(X, X, metric='sqeuclidean')
            return os * np.exp(-dist_sq / (2 * ls ** 2))

        # Finite difference gradient
        K_plus = rbf_kernel(X, lengthscale, outputscale + eps)
        K_minus = rbf_kernel(X, lengthscale, outputscale - eps)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/d(sigma^2) = K / sigma^2
        K = rbf_kernel(X, lengthscale, outputscale)
        grad_analytical = K / outputscale

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_matern32_gradient_lengthscale_finite_diff(self, random_seed):
        """Test Matern 3/2 gradient w.r.t. lengthscale via finite differences."""
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern32_kernel(X, ls, os):
            dist = cdist(X, X, metric='euclidean')
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / ls
            return os * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

        # Finite difference gradient
        K_plus = matern32_kernel(X, lengthscale + eps, outputscale)
        K_minus = matern32_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = outputscale * 3 * r^2 / l^3 * exp(-sqrt(3)*r/l)
        dist = cdist(X, X, metric='euclidean')
        dist_sq = dist ** 2
        sqrt3 = np.sqrt(3.0)
        sqrt3_r_l = sqrt3 * dist / lengthscale
        grad_analytical = outputscale * 3.0 * dist_sq / (lengthscale ** 3) * np.exp(-sqrt3_r_l)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_matern52_gradient_lengthscale_finite_diff(self, random_seed):
        """Test Matern 5/2 gradient w.r.t. lengthscale via finite differences."""
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern52_kernel(X, ls, os):
            dist = cdist(X, X, metric='euclidean')
            dist_sq = dist ** 2
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / ls
            r_sq_term = (5.0 / 3.0) * dist_sq / (ls ** 2)
            return os * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        # Finite difference gradient
        K_plus = matern52_kernel(X, lengthscale + eps, outputscale)
        K_minus = matern52_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = outputscale * (5*r^2/3l^3) * (1 + sqrt(5)*r/l) * exp(-sqrt(5)*r/l)
        dist = cdist(X, X, metric='euclidean')
        dist_sq = dist ** 2
        sqrt5 = np.sqrt(5.0)
        sqrt5_r_l = sqrt5 * dist / lengthscale
        coeff = (5.0 / 3.0) * dist_sq / (lengthscale ** 3)
        grad_analytical = outputscale * coeff * (1.0 + sqrt5_r_l) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)


class TestKernelGradientsAutograd:
    """Test kernel gradients against PyTorch autograd."""

    @pytest.mark.parametrize("kernel_type", ["rbf", "matern32", "matern52"])
    def test_lengthscale_gradient_vs_autograd(self, kernel_type, random_seed):
        """Test dK/dl matches PyTorch autograd."""
        n = 20
        X = np.random.randn(n, 3).astype(np.float32)
        lengthscale = 1.5
        outputscale = 2.0

        # PyTorch autograd reference
        X_torch = torch.tensor(X, requires_grad=False)
        ls_torch = torch.tensor(lengthscale, requires_grad=True)

        if kernel_type == "rbf":
            dist_sq = torch.cdist(X_torch, X_torch).pow(2)
            K = outputscale * torch.exp(-dist_sq / (2 * ls_torch**2))
        elif kernel_type == "matern32":
            dist = torch.cdist(X_torch, X_torch)
            sqrt3_r_l = np.sqrt(3) * dist / ls_torch
            K = outputscale * (1 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)
        elif kernel_type == "matern52":
            dist = torch.cdist(X_torch, X_torch)
            sqrt5_r_l = np.sqrt(5) * dist / ls_torch
            r_sq_term = (5/3) * dist.pow(2) / ls_torch**2
            K = outputscale * (1 + sqrt5_r_l + r_sq_term) * torch.exp(-sqrt5_r_l)

        # Sum to get scalar for autograd
        loss = K.sum()
        loss.backward()
        grad_autograd = ls_torch.grad.item()

        # Analytical gradient (sum of dK/dl)
        X_np = X.astype(np.float64)
        dist = cdist(X_np, X_np, metric='euclidean')
        dist_sq = dist ** 2

        if kernel_type == "rbf":
            K_np = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
            dK_dl = K_np * dist_sq / (lengthscale ** 3)
        elif kernel_type == "matern32":
            sqrt3_r_l = np.sqrt(3) * dist / lengthscale
            dK_dl = outputscale * 3.0 * dist_sq / (lengthscale ** 3) * np.exp(-sqrt3_r_l)
        elif kernel_type == "matern52":
            sqrt5_r_l = np.sqrt(5) * dist / lengthscale
            coeff = (5.0 / 3.0) * dist_sq / (lengthscale ** 3)
            dK_dl = outputscale * coeff * (1.0 + sqrt5_r_l) * np.exp(-sqrt5_r_l)

        grad_analytical = dK_dl.sum()

        # Compare
        rel_error = abs(grad_analytical - grad_autograd) / (abs(grad_autograd) + 1e-8)
        assert rel_error < 0.01, f"Gradient error {rel_error:.2%} for {kernel_type}"

    def test_outputscale_gradient_vs_autograd(self, random_seed):
        """Test dK/d(sigma^2) matches PyTorch autograd."""
        n = 20
        X = np.random.randn(n, 3).astype(np.float32)
        lengthscale = 1.0
        outputscale = 2.0

        # PyTorch autograd reference
        X_torch = torch.tensor(X, requires_grad=False)
        os_torch = torch.tensor(outputscale, requires_grad=True)

        dist_sq = torch.cdist(X_torch, X_torch).pow(2)
        K = os_torch * torch.exp(-dist_sq / (2 * lengthscale**2))

        loss = K.sum()
        loss.backward()
        grad_autograd = os_torch.grad.item()

        # Analytical gradient: dK/d(sigma^2) = K/sigma^2
        X_np = X.astype(np.float64)
        dist_sq_np = cdist(X_np, X_np, metric='sqeuclidean')
        K_np = outputscale * np.exp(-dist_sq_np / (2 * lengthscale**2))
        dK_dos = K_np / outputscale
        grad_analytical = dK_dos.sum()

        rel_error = abs(grad_analytical - grad_autograd) / (abs(grad_autograd) + 1e-8)
        assert rel_error < 0.01, f"Outputscale gradient error {rel_error:.2%}"


class TestGradientProperties:
    """Test gradient properties."""

    def test_gradient_symmetry(self, random_seed):
        """Test gradient matrix is symmetric."""
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric='sqeuclidean')
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        dK_dl = K * dist_sq / (lengthscale ** 3)

        np.testing.assert_allclose(dK_dl, dK_dl.T, rtol=1e-10)

    def test_gradient_zero_at_diagonal(self, random_seed):
        """Test gradient is zero on diagonal (r=0)."""
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric='sqeuclidean')
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        dK_dl = K * dist_sq / (lengthscale ** 3)

        # Diagonal should be zero (r=0 means dist_sq=0)
        np.testing.assert_allclose(np.diag(dK_dl), np.zeros(20), atol=1e-10)

    def test_gradient_sign(self, random_seed):
        """Test gradient has correct sign."""
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric='sqeuclidean')
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        dK_dl = K * dist_sq / (lengthscale ** 3)

        # dK/dl should be non-negative (increasing lengthscale increases K for r>0)
        assert np.all(dK_dl >= -1e-10), "dK/dl should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

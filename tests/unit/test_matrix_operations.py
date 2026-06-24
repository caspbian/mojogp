"""
Tests for matrix operations: inversion, Cholesky factorization.

These tests validate the small matrix operations used in the preconditioner.
"""
import numpy as np
import pytest


class TestMatrixInversion:
    """Test small matrix inversion (used in preconditioner)."""

    @pytest.mark.parametrize("rank", [5, 10, 15, 20])
    def test_inverse_accuracy_spd(self, rank, random_seed):
        """Test (L^T L + noise I)^{-1} accuracy for SPD matrices."""
        n = 100
        L = np.random.randn(n, rank).astype(np.float64)
        noise = 0.1

        # Build LTL + noise*I
        LTL = L.T @ L
        A = LTL + noise * np.eye(rank)

        # NumPy reference
        A_inv_numpy = np.linalg.inv(A)

        # Verify A @ A^{-1} = I
        product = A @ A_inv_numpy
        np.testing.assert_allclose(product, np.eye(rank), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("rank", [5, 10, 15])
    def test_inverse_times_original(self, rank, random_seed):
        """Test A @ A^{-1} = I for random SPD matrix."""
        A = np.random.randn(rank, rank).astype(np.float64)
        A = A @ A.T + 0.1 * np.eye(rank)  # Make SPD

        A_inv = np.linalg.inv(A)

        product = A @ A_inv
        np.testing.assert_allclose(product, np.eye(rank), rtol=1e-10, atol=1e-10)

    def test_ill_conditioned_matrix(self, random_seed):
        """Test handling of ill-conditioned matrix."""
        rank = 10
        # Create ill-conditioned matrix
        A = np.eye(rank, dtype=np.float64)
        A[0, 0] = 1e6
        A[1, 1] = 1e-6

        # Should still compute (may have larger error)
        A_inv = np.linalg.inv(A)

        # Just check it's finite
        assert np.all(np.isfinite(A_inv)), "Inverse should be finite"

        # Check approximate inverse
        product = A @ A_inv
        # Looser tolerance for ill-conditioned
        np.testing.assert_allclose(product, np.eye(rank), rtol=1e-4, atol=1e-4)


class TestCholeskyFactorization:
    """Test Cholesky-based operations."""

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_cholesky_solve(self, n, random_seed):
        """Test solving Ax = b via Cholesky."""
        A = np.random.randn(n, n).astype(np.float64)
        A = A @ A.T + 0.1 * np.eye(n)  # SPD
        b = np.random.randn(n).astype(np.float64)

        # Cholesky solve
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        x_chol = np.linalg.solve(L.T, y)

        # Direct solve reference
        x_direct = np.linalg.solve(A, b)

        np.testing.assert_allclose(x_chol, x_direct, rtol=1e-10)

    @pytest.mark.parametrize("n", [50, 100])
    def test_cholesky_logdet(self, n, random_seed):
        """Test log-det via Cholesky: log|A| = 2 * sum(log(diag(L)))."""
        A = np.random.randn(n, n).astype(np.float64)
        A = A @ A.T + 0.1 * np.eye(n)  # SPD

        # Cholesky log-det
        L = np.linalg.cholesky(A)
        logdet_chol = 2 * np.sum(np.log(np.diag(L)))

        # NumPy reference
        sign, logdet_numpy = np.linalg.slogdet(A)

        np.testing.assert_allclose(logdet_chol, logdet_numpy, rtol=1e-10)

    def test_cholesky_fails_non_spd(self):
        """Test Cholesky fails for non-SPD matrix."""
        n = 10
        A = np.random.randn(n, n).astype(np.float64)
        # Make non-SPD by having negative eigenvalue
        A = A @ A.T
        A[0, 0] = -1.0  # Force negative eigenvalue

        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(A)


class TestWoodburyFormula:
    """Test Woodbury matrix identity for preconditioner."""

    def test_woodbury_inverse(self, random_seed):
        """Test (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}."""
        n = 50
        k = 10

        # A = sigma^2 * I (diagonal)
        sigma_sq = 0.1
        A = sigma_sq * np.eye(n)
        A_inv = (1.0 / sigma_sq) * np.eye(n)

        # U, V = L, L^T for low-rank update
        L = np.random.randn(n, k).astype(np.float64)
        U = L
        V = L.T
        C = np.eye(k)

        # Direct inverse
        M = A + U @ C @ V
        M_inv_direct = np.linalg.inv(M)

        # Woodbury formula
        C_inv = np.eye(k)
        inner = C_inv + V @ A_inv @ U
        inner_inv = np.linalg.inv(inner)
        M_inv_woodbury = A_inv - A_inv @ U @ inner_inv @ V @ A_inv

        np.testing.assert_allclose(M_inv_woodbury, M_inv_direct, rtol=1e-10)

    def test_woodbury_matvec(self, random_seed):
        """Test Woodbury formula for matrix-vector product."""
        n = 100
        k = 15

        sigma_sq = 0.1
        L = np.random.randn(n, k).astype(np.float64)
        v = np.random.randn(n).astype(np.float64)

        # Build full matrix and compute inverse
        M = L @ L.T + sigma_sq * np.eye(n)
        M_inv = np.linalg.inv(M)
        result_direct = M_inv @ v

        # Woodbury: (LL^T + sigma^2 I)^{-1} v
        # = (1/sigma^2) * (v - L @ (L^T L + sigma^2 I)^{-1} @ L^T @ v)
        LTL = L.T @ L
        inner = LTL + sigma_sq * np.eye(k)
        inner_inv = np.linalg.inv(inner)

        result_woodbury = (1.0 / sigma_sq) * (v - L @ inner_inv @ L.T @ v)

        np.testing.assert_allclose(result_woodbury, result_direct, rtol=1e-10)


class TestQRDecomposition:
    """Test QR decomposition for log|P| computation."""

    def test_qr_logdet(self, random_seed):
        """Test log-det via QR: log|A| = sum(log|diag(R)|) for square A."""
        n = 50
        A = np.random.randn(n, n).astype(np.float64)
        A = A @ A.T + 0.1 * np.eye(n)  # SPD

        # QR of Cholesky factor gives same result
        L = np.linalg.cholesky(A)
        Q, R = np.linalg.qr(L)

        # log|L| = sum(log|diag(R)|) since Q is orthogonal
        logdet_qr = np.sum(np.log(np.abs(np.diag(R))))
        logdet_chol = np.sum(np.log(np.diag(L)))

        np.testing.assert_allclose(logdet_qr, logdet_chol, rtol=1e-10)

    def test_qr_precond_logdet(self, random_seed):
        """Test QR-based log|P| computation (GPyTorch approach)."""
        n = 100
        k = 15
        noise = 0.1

        L = np.random.randn(n, k).astype(np.float64)

        # GPyTorch approach: QR of [L; sqrt(noise)*I_k]
        sqrt_noise = np.sqrt(noise)
        stacked = np.vstack([L, sqrt_noise * np.eye(k)])
        Q, R = np.linalg.qr(stacked)

        # log|P| = 2 * sum(log|diag(R)|) + (n-k) * log(noise)
        logdet_qr = 2 * np.sum(np.log(np.abs(np.diag(R)))) + (n - k) * np.log(noise)

        # Direct computation: P = LL^T + noise*I
        P = L @ L.T + noise * np.eye(n)
        sign, logdet_direct = np.linalg.slogdet(P)

        np.testing.assert_allclose(logdet_qr, logdet_direct, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for probe vector generation.

Tests Rademacher and Gaussian probe vectors for SLQ estimation.
"""
import numpy as np
import pytest


class TestRademacherProbes:
    """Test Rademacher probe vector generation."""

    def test_rademacher_values(self, random_seed):
        """Test Rademacher probes are ±1."""
        n = 1000
        num_probes = 10

        # Generate Rademacher probes
        probes = np.sign(np.random.randn(n, num_probes))
        probes[probes == 0] = 1  # Handle rare case of exactly 0

        # Check values are ±1
        assert np.all((probes == 1) | (probes == -1)), "Rademacher probes should be ±1"

    def test_rademacher_mean_zero(self, random_seed):
        """Test Rademacher probes have approximately zero mean."""
        n = 10000
        num_probes = 100

        probes = np.sign(np.random.randn(n, num_probes))
        probes[probes == 0] = 1

        # Mean should be close to 0
        mean = np.mean(probes)
        assert abs(mean) < 0.05, f"Rademacher mean {mean} should be ~0"

    def test_rademacher_covariance(self, random_seed):
        """Test E[zz^T] ≈ I for Rademacher probes."""
        n = 100
        num_probes = 1000  # Many probes for statistical test

        probes = np.sign(np.random.randn(n, num_probes))
        probes[probes == 0] = 1

        # Sample covariance
        cov = (probes @ probes.T) / num_probes

        # Should be close to identity
        diag_mean = np.diag(cov).mean()
        off_diag_mean = np.abs(cov[~np.eye(n, dtype=bool)]).mean()

        assert 0.9 < diag_mean < 1.1, f"Diagonal mean {diag_mean} should be ~1"
        assert off_diag_mean < 0.1, f"Off-diagonal mean {off_diag_mean} should be ~0"


class TestGaussianProbes:
    """Test Gaussian probe vector generation."""

    def test_gaussian_mean_zero(self, random_seed):
        """Test Gaussian probes have approximately zero mean."""
        n = 10000
        num_probes = 100

        probes = np.random.randn(n, num_probes)

        # Mean should be close to 0
        mean = np.mean(probes)
        assert abs(mean) < 0.05, f"Gaussian mean {mean} should be ~0"

    def test_gaussian_variance_one(self, random_seed):
        """Test Gaussian probes have approximately unit variance."""
        n = 10000
        num_probes = 100

        probes = np.random.randn(n, num_probes)

        # Variance should be close to 1
        var = np.var(probes)
        assert 0.95 < var < 1.05, f"Gaussian variance {var} should be ~1"

    def test_gaussian_covariance(self, random_seed):
        """Test E[zz^T] ≈ I for Gaussian probes."""
        n = 100
        num_probes = 1000

        probes = np.random.randn(n, num_probes)

        # Sample covariance
        cov = (probes @ probes.T) / num_probes

        # Should be close to identity
        diag_mean = np.diag(cov).mean()
        off_diag_mean = np.abs(cov[~np.eye(n, dtype=bool)]).mean()

        assert 0.9 < diag_mean < 1.1, f"Diagonal mean {diag_mean} should be ~1"
        assert off_diag_mean < 0.15, f"Off-diagonal mean {off_diag_mean} should be ~0"


class TestProbeNormalization:
    """Test probe vector normalization."""

    def test_normalized_probes_unit_norm(self, random_seed):
        """Test normalized probes have unit norm."""
        n = 100
        num_probes = 10

        probes = np.random.randn(n, num_probes)
        norms = np.linalg.norm(probes, axis=0)
        probes_normalized = probes / norms

        # Check unit norm
        norms_after = np.linalg.norm(probes_normalized, axis=0)
        np.testing.assert_allclose(norms_after, np.ones(num_probes), rtol=1e-10)

    def test_normalization_preserves_direction(self, random_seed):
        """Test normalization preserves direction."""
        n = 100
        num_probes = 10

        probes = np.random.randn(n, num_probes)
        norms = np.linalg.norm(probes, axis=0)
        probes_normalized = probes / norms

        # Dot product with original should be positive
        for i in range(num_probes):
            dot = np.dot(probes[:, i], probes_normalized[:, i])
            assert dot > 0, "Normalization should preserve direction"


class TestPreconditionerProbes:
    """Test probe vectors sampled from preconditioner N(0, P)."""

    def test_precond_probes_covariance(self, random_seed):
        """Test probes from N(0, P) have covariance ≈ P."""
        n = 50
        k = 10
        num_probes = 1000
        noise = 0.1

        # Create preconditioner P = LL^T + noise*I
        L = np.random.randn(n, k).astype(np.float64)
        P = L @ L.T + noise * np.eye(n)

        # Sample from N(0, P) using Cholesky
        P_chol = np.linalg.cholesky(P)
        z = np.random.randn(n, num_probes)
        probes = P_chol @ z  # probes ~ N(0, P)

        # Sample covariance
        cov = (probes @ probes.T) / num_probes

        # Should be close to P
        rel_error = np.linalg.norm(cov - P, 'fro') / np.linalg.norm(P, 'fro')
        assert rel_error < 0.2, f"Covariance error {rel_error:.2%} too high"

    def test_precond_probes_via_woodbury(self, random_seed):
        """Test probes from N(0, P) using Woodbury formula."""
        n = 50
        k = 10
        num_probes = 100
        noise = 0.1

        # Create preconditioner P = LL^T + noise*I
        L = np.random.randn(n, k).astype(np.float64)

        # Sample using Woodbury: z = sqrt(noise)*e + L*v where e~N(0,I), v~N(0,I)
        e = np.random.randn(n, num_probes)
        v = np.random.randn(k, num_probes)
        probes = np.sqrt(noise) * e + L @ v

        # Sample covariance
        cov = (probes @ probes.T) / num_probes

        # Expected covariance: P = LL^T + noise*I
        P = L @ L.T + noise * np.eye(n)

        # Should be close to P (looser tolerance for 100 probes)
        rel_error = np.linalg.norm(cov - P, 'fro') / np.linalg.norm(P, 'fro')
        assert rel_error < 0.4, f"Woodbury covariance error {rel_error:.2%} too high"


class TestProbeVectorStatistics:
    """Test statistical properties of probe vectors."""

    def test_slq_variance_decreases_with_probes(self, random_seed):
        """Test SLQ variance decreases with more probes."""
        n = 50

        # Create SPD matrix
        A = np.random.randn(n, n).astype(np.float64)
        A = A @ A.T + 0.1 * np.eye(n)

        # True log-det
        logdet_true = np.linalg.slogdet(A)[1]

        # Estimate with different probe counts
        probe_counts = [5, 10, 20, 50]
        variances = []

        for num_probes in probe_counts:
            estimates = []
            for _ in range(20):  # Multiple runs
                # Simple SLQ estimate using eigendecomposition
                probes = np.random.randn(n, num_probes)
                probes = probes / np.linalg.norm(probes, axis=0)

                # Compute z^T log(A) z for each probe
                eigs, V = np.linalg.eigh(A)
                log_eigs = np.log(eigs)

                estimate = 0.0
                for i in range(num_probes):
                    z = probes[:, i]
                    Vz = V.T @ z
                    estimate += n * np.sum(Vz**2 * log_eigs)
                estimate /= num_probes
                estimates.append(estimate)

            variances.append(np.var(estimates))

        # Variance should generally decrease with more probes
        # (not strictly monotonic due to randomness, but trend should be clear)
        assert variances[-1] < variances[0], \
            f"Variance should decrease: {probe_counts[0]} probes var={variances[0]:.4f}, " \
            f"{probe_counts[-1]} probes var={variances[-1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Public support-boundary tests for MultiOutputLMCGP."""

import numpy as np
import pytest

from mojogp import Kernel, MultiOutputLMCGP
from mojogp.multi_output_gp import (
    _build_lmc_initial_params_from_lengthscales,
    _summarize_lmc_kernel_params,
)


def _tiny_multi_output_data(n: int = 8, d: int = 2, tasks: int = 2):
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, tasks)).astype(np.float32)
    return X, Y


class TestLMCPublicSupportBoundaries:
    def test_pure_categorical_latent_fails_before_backend_training(self):
        X, Y = _tiny_multi_output_data(d=1)
        X[:, 0] = np.arange(X.shape[0]) % 3
        gp = MultiOutputLMCGP(kernels=[Kernel.ehh(levels=3, active_dims=[0])])

        with pytest.raises(ValueError, match="Pure categorical latent kernels"):
            gp.fit(X, Y, max_iterations=1)

    def test_grouped_noise_placeholder_raises_before_backend(self):
        X, Y = _tiny_multi_output_data()
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])

        with pytest.raises(NotImplementedError, match="Grouped noise"):
            gp.fit(X, Y, grouped_noise=np.zeros((2, 2), dtype=np.float32))

    def test_fixed_sample_task_noise_shape_is_validated(self):
        X, Y = _tiny_multi_output_data()
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])

        with pytest.raises(ValueError, match="fixed_observation_noise must have shape"):
            gp.fit(
                X,
                Y,
                fixed_observation_noise=np.ones((X.shape[0], 1), dtype=np.float32),
            )

    def test_fixed_sample_task_noise_rejects_negative_entries(self):
        X, Y = _tiny_multi_output_data()
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])
        fixed_noise = np.ones_like(Y, dtype=np.float32) * 0.01
        fixed_noise[0, 0] = -0.1

        with pytest.raises(ValueError, match="entries must be >= 0"):
            gp.fit(X, Y, fixed_observation_noise=fixed_noise)

    def test_fixed_sample_task_noise_is_explicitly_continuous_lmc_only(self):
        X, Y = _tiny_multi_output_data(d=2)
        X[:, 1] = np.arange(X.shape[0]) % 3
        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=3, active_dims=[1])]
        )

        with pytest.raises(NotImplementedError, match="Fixed per-sample-per-task noise"):
            gp.fit(X, Y, fixed_observation_noise=np.ones_like(Y, dtype=np.float32))

    def test_lmc_fit_rejects_ambiguous_initial_parameter_inputs(self):
        X, Y = _tiny_multi_output_data()
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])

        with pytest.raises(ValueError, match="not both"):
            gp.fit(
                X,
                Y,
                initial_params=np.ones(2, dtype=np.float32),
                initial_lengthscales=np.ones(1, dtype=np.float32),
            )

    def test_lmc_fit_validates_initial_lengthscale_shape_before_backend_training(self):
        X, Y = _tiny_multi_output_data(d=3)
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()], ard=True)

        with pytest.raises(ValueError, match="initial_lengthscales"):
            gp.fit(X, Y, initial_lengthscales=np.ones((2, 2), dtype=np.float32))


class TestLMCARDConfiguration:
    def test_ard_is_applied_once_for_continuous_latents(self):
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()], ard=True)
        gp._configure_latent_kernels_for_fit(total_dim=3)

        assert gp._latent_compiled_kernels is not None
        assert [k.num_params() for k in gp._latent_compiled_kernels] == [4, 4]

    def test_ard_uses_active_dim_count_per_latent(self):
        kernels = [
            Kernel.rbf(active_dims=[0, 2]),
            Kernel.matern52(active_dims=[1, 3, 4]),
        ]
        gp = MultiOutputLMCGP(kernels=kernels, ard=True)
        gp._configure_latent_kernels_for_fit(total_dim=5)

        assert gp._latent_compiled_kernels is not None
        assert [k.num_params() for k in gp._latent_compiled_kernels] == [3, 4]

    def test_mixed_ard_uses_continuous_dimension_count_after_categorical_split(self):
        kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(
            levels=3, active_dims=[2]
        )
        gp = MultiOutputLMCGP(kernels=[kernel], ard=True)
        gp._configure_latent_kernels_for_fit(total_dim=3)

        assert gp._latent_compiled_kernels is not None
        assert gp._latent_cont_dims == [2]
        # RBF contributes 2 ARD lengthscales + outputscale; EHH(levels=3)
        # contributes 3 categorical angle params.
        assert gp._latent_compiled_kernels[0].num_params() == 6

    def test_lmc_param_summary_does_not_mislabel_extra_kernel_params(self):
        kernels = [Kernel.rq(), Kernel.periodic(), Kernel.linear(), Kernel.polynomial()]
        params = [
            np.array([1.2, 0.7, 2.0], dtype=np.float32),
            np.array([0.8, 1.5, 1.1], dtype=np.float32),
            np.array([3.0, 0.9], dtype=np.float32),
            np.array([2.0, 1.0, 0.6], dtype=np.float32),
        ]

        lengthscales, outputscales, lengthscales_per_dim = _summarize_lmc_kernel_params(
            kernels, params, use_ard=False
        )

        np.testing.assert_allclose(lengthscales, np.array([1.2, 0.8], dtype=np.float32))
        np.testing.assert_allclose(
            outputscales, np.array([2.0, 1.1, 0.9, 0.6], dtype=np.float32)
        )
        assert lengthscales_per_dim is None

    def test_lmc_initial_lengthscales_populate_isotropic_latents(self):
        kernels = [Kernel.rbf(), Kernel.matern52()]

        params = _build_lmc_initial_params_from_lengthscales(
            kernels, np.array([0.4, 1.7], dtype=np.float32)
        )

        np.testing.assert_allclose(
            params, np.array([0.4, 1.0, 1.7, 1.0], dtype=np.float32)
        )

    def test_lmc_initial_lengthscales_populate_heterogeneous_ard_latents(self):
        kernels = [
            Kernel.rbf(active_dims=[0, 2]),
            Kernel.matern52(active_dims=[1, 3, 4]),
        ]
        gp = MultiOutputLMCGP(kernels=kernels, ard=True)
        gp._configure_latent_kernels_for_fit(total_dim=5)

        params = _build_lmc_initial_params_from_lengthscales(
            gp._latent_compiled_kernels,
            np.array([0.2, 0.3, 1.1, 1.2, 1.3], dtype=np.float32),
        )

        np.testing.assert_allclose(
            params,
            np.array([0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 1.0], dtype=np.float32),
        )

    def test_lmc_initial_lengthscales_reject_bad_shape_for_heterogeneous_ard(self):
        kernels = [
            Kernel.rbf(active_dims=[0, 2]),
            Kernel.matern52(active_dims=[1, 3, 4]),
        ]
        gp = MultiOutputLMCGP(kernels=kernels, ard=True)
        gp._configure_latent_kernels_for_fit(total_dim=5)

        with pytest.raises(ValueError, match="per-latent lengthscale counts"):
            _build_lmc_initial_params_from_lengthscales(
                gp._latent_compiled_kernels,
                np.ones((2, 2), dtype=np.float32),
            )

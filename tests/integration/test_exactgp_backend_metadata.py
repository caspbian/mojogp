"""Integration tests for ExactGP JIT backend training metadata.

These tests exercise the real JIT engine binding to verify the expanded train
argument plumbing reports the active backend route and preconditioner
configuration.
"""

import numpy as np

from mojogp import SingleOutputGP
from mojogp.gp import (
    _DEFAULT_EXACT_PREDICT_CG_TOL,
    _DEFAULT_EXACT_PREDICT_MAX_CG_ITER,
    _DEFAULT_PREDICT_CG_TOL,
    _DEFAULT_PREDICT_MAX_CG_ITER,
)
from mojogp.loader import load_engine, load_kernel_module_engine
from mojogp.kernel import RBF
from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.integration.run_exactgp_backend_metadata_case"


def _run_backend_metadata_case(case: str, **payload) -> dict[str, object]:
    return run_isolated_case(
        module=MODULE,
        payload={"case": case, **payload},
        timeout=300,
        description=f"Runs ExactGP backend metadata case {case}",
    )


def _make_data(n=2000, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(n)).astype(np.float32)
    return X, y


def _make_provider(force_materialized: bool = False):
    X, y = _make_data()
    params = np.array([1.0, 1.0], dtype=np.float32)
    kernel_module = load_kernel_module_engine(
        RBF(), dim=X.shape[1], force_recompile=True, verbose=False
    )
    engine = load_engine(verbose=False)
    info = kernel_module.init_provider(X, params, 0.1)
    if force_materialized:
        kernel_module.materialize(info)
    return engine, info, y, params


def _make_ard_provider(force_materialized: bool):
    X, y = _make_data()
    params = np.ones(X.shape[1] + 1, dtype=np.float32)
    kernel_module = load_kernel_module_engine(
        RBF(ard=True), dim=X.shape[1], force_recompile=True, verbose=False
    )
    engine = load_engine(verbose=False)
    info = kernel_module.init_provider(X, params, 0.1)
    if force_materialized:
        kernel_module.materialize(info)
    return engine, info, y, params


class TestExactGPBackendMetadata:
    def test_materialized_ard_matches_matrix_free_one_step_nll(self):
        matrix_free = _make_ard_provider(force_materialized=False)
        materialized = _make_ard_provider(force_materialized=True)

        mf_engine, mf_info, y, params = matrix_free
        mat_engine, mat_info, _, _ = materialized
        kwargs = (
            0.1,
            1,
            0.0,
            10,
            100,
            1e-2,
            15,
            False,
            True,
            True,
            30,
            0.5,
            0,
            float(np.mean(y)),
        )

        mf_result = mf_engine.train(mf_info, y, params, *kwargs)
        mat_result = mat_engine.train(mat_info, y, params, *kwargs)

        assert mf_result["training_route"] == "matrix_free"
        assert mf_result["materialization_mode"] == 0
        assert mat_result["training_route"] == "materialized"
        assert mat_result["materialization_mode"] == 1
        assert np.isclose(
            float(mat_result["final_nll"]),
            float(mf_result["final_nll"]),
            rtol=0.0,
            atol=5e-5,
        )

    def test_train_reports_explicit_default_metadata(self):
        engine, info, y, params = _make_provider(force_materialized=False)

        result = engine.train(
            info,
            y,
            params,
            0.1,
            1,
            0.01,
            5,
            25,
            1e-2,
            8,
            False,
            True,
            True,
            30,
            0.5,
            2,
            float(np.mean(y)),
        )

        assert result["training_route"] == "matrix_free"
        assert result["materialization_mode"] == 0
        assert result["precond_method"] == 2
        assert result["max_tridiag_iter"] == 30
        assert np.isclose(float(result["precond_rebuild_threshold"]), 0.5)
        assert result["use_preconditioner"] is True

    def test_train_reports_explicit_route_and_preconditioner_metadata(self):
        engine, info, y, params = _make_provider(force_materialized=True)

        result = engine.train(
            info,
            y,
            params,
            0.1,
            1,
            0.01,
            5,
            25,
            1e-2,
            8,
            False,
            True,
            True,
            7,
            0.125,
            0,
            float(np.mean(y)),
        )

        assert result["training_route"] == "materialized"
        assert result["materialization_mode"] == 1
        assert result["precond_method"] == 0
        assert result["precond_rank"] == 8
        assert result["max_tridiag_iter"] == 7
        assert np.isclose(float(result["precond_rebuild_threshold"]), 0.125)
        assert result["use_preconditioner"] is True

    def test_train_reports_preconditioner_diagnostics(self):
        engine, info, y, params = _make_provider(force_materialized=False)

        result = engine.train(
            info,
            y,
            params,
            0.1,
            1,
            0.01,
            5,
            25,
            1e-2,
            8,
            False,
            True,
            True,
            30,
            0.5,
            0,
            float(np.mean(y)),
        )

        assert result["precond_build_count"] >= 1
        assert result["precond_rebuild_count"] == 0
        assert float(result["precond_build_total_ms"]) >= 0.0
        rank_history = list(result["precond_rank_history"])
        rebuild_steps = list(result["precond_rebuild_steps"])
        assert len(rank_history) == int(result["iterations"])
        assert rebuild_steps == []
        assert int(result["actual_precond_rank"]) == rank_history[-1]
        assert 0 < int(result["actual_precond_rank"]) <= int(result["precond_rank"])

    def test_wrapper_backend_train_info_includes_preconditioner_diagnostics(self):
        X, y = _make_data()
        gp = SingleOutputGP(kernel=RBF(), verbose=False)
        gp.fit(
            X,
            y,
            max_iterations=2,
            learning_rate=0.01,
            method="matrix_free",
            num_probes=2,
            max_cg_iterations=15,
            preconditioner_rank=6,
            max_tridiag_iterations=7,
            preconditioner="greedy",
            verbose=False,
        )

        info = gp.backend_train_info
        assert info is not None
        assert info["precond_build_count"] >= 1
        assert info["precond_rebuild_count"] >= 0
        assert info["precond_build_total_ms"] >= 0.0
        assert len(info["precond_rank_history"]) == gp.training_result.iterations
        assert info["actual_precond_rank"] == info["precond_rank_history"][-1]

    def test_train_reports_explicit_early_stopping_metadata(self):
        engine, info, y, params = _make_provider(force_materialized=False)

        result = engine.train(
            info,
            y,
            params,
            0.1,
            10,
            0.01,
            5,
            25,
            1e-2,
            8,
            False,
            True,
            True,
            30,
            0.5,
            2,
            float(np.mean(y)),
            True,
            1,
            1e6,
        )

        assert result["enable_early_stopping"] is True
        assert result["early_stop_patience"] == 1
        assert np.isclose(float(result["early_stop_tol"]), 1e6)
        assert int(result["iterations"]) < 10
        assert result["converged"] is True

    def test_predict_reports_route_and_preconditioner_metadata(self):
        payload = _run_backend_metadata_case("predict_reports_route")
        assert payload["shape"] == [8]
        info = payload["info"]
        assert info["requested_method"] == "materialized"
        assert info["training_route"] == "materialized"
        assert info["actual_prediction_route"] == "predict"
        assert info["actual_variance_route"] == "predict"
        assert info["backend_prediction_used"] is True
        assert info["backend_variance_used"] is True
        assert info["fallback_used"] is False
        assert info["precond_rank"] == 6
        assert info["precond_method"] == 0
        assert info["max_cg_iterations"] == _DEFAULT_EXACT_PREDICT_MAX_CG_ITER
        assert np.isclose(info["cg_tolerance"], _DEFAULT_EXACT_PREDICT_CG_TOL)
        assert info["telemetry_quality"] == "observed"
        assert info["configured_for_cg"] is True
        assert info["observed_cg_calls"] is True
        assert info["exact_block_cols"] == 8
        assert info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_predict_can_override_route_independently_of_training(self):
        payload = _run_backend_metadata_case("predict_override_route")
        assert payload["shape"] == [12]
        info = payload["info"]
        assert info["requested_method"] == "matrix_free"
        assert info["prediction_method"] == "matrix_free"
        assert info["training_route"] == "materialized"
        assert info["actual_prediction_route"] == "predict"
        assert info["backend_prediction_used"] is True
        assert info["fallback_used"] is False
        assert info["exact_block_cols"] == 12
        assert info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_matrix_free_exact_prediction_reports_no_materialized_fallback(self):
        payload = _run_backend_metadata_case(
            "matrix_free_exact_prediction_no_fallback"
        )
        assert payload["shape"] == [10]
        info = payload["info"]
        assert info["requested_method"] == "matrix_free"
        assert info["prediction_method"] == "matrix_free"
        assert info["training_route"] == "matrix_free"
        assert info["actual_prediction_route"] == "predict"
        assert info["actual_variance_route"] == "predict"
        assert info["backend_prediction_used"] is True
        assert info["backend_variance_used"] is True
        assert info["fallback_used"] is False
        assert info["telemetry_quality"] == "observed"
        assert info["configured_for_cg"] is True
        assert info["observed_cg_calls"] is True
        assert info["exact_block_cols"] == 10
        assert info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_materialized_repeated_predict_reuses_provider_state(self):
        payload = _run_backend_metadata_case(
            "materialized_repeated_predict_reuses_provider_state"
        )
        assert payload["shape"] == [10]
        assert payload["mean_match"] is True
        assert payload["var_match"] is True
        first_info = payload["first_info"]
        info = payload["info"]
        assert first_info["alpha_cache_used"] is False
        assert info["alpha_cache_used"] is True
        assert info["requested_method"] == "materialized"
        assert info["training_route"] == "materialized"
        assert info["provider_state_update_skipped"] is True
        assert first_info["love_cross_chunk_width"] == 0
        assert info["love_cross_chunk_width"] == 0
        assert info["exact_block_cols"] == 10
        assert info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_predict_uses_method_specific_default_rhs_width(self):
        expected_by_method = {"materialized": 600, "matrix_free": 512}

        for method, expected_block_cols in expected_by_method.items():
            payload = _run_backend_metadata_case(
                "predict_default_block_cap", method=method
            )
            assert payload["shape"] == [600]
            info = payload["info"]
            assert info["requested_method"] == method
            assert info["prediction_method"] == method
            assert info["exact_block_cols"] == expected_block_cols
            assert info["exact_block_cols_requested"] == 0
            assert info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_materialized_repeated_love_predict_reuses_root_state(self):
        payload = _run_backend_metadata_case(
            "materialized_repeated_love_predict_reuses_root_state"
        )
        assert payload["shape"] == [10]
        assert payload["mean_match"] is True
        assert payload["var_match"] is True
        first_info = payload["first_info"]
        info = payload["info"]
        assert first_info["alpha_cache_used"] is False
        assert first_info["love_root_cache_used"] is False
        assert info["alpha_cache_used"] is True
        assert info["love_root_cache_used"] is True
        assert info["requested_method"] == "materialized"
        assert info["training_route"] == "materialized"
        assert info["provider_state_update_skipped"] is True

    def test_love_first_prediction_uses_exact_alpha_defaults(self):
        payload = _run_backend_metadata_case(
            "love_first_predict_uses_exact_alpha_defaults"
        )
        assert payload["shape"] == [16]
        info = payload["info"]
        assert info["variance_method"] == "love"
        assert info["alpha_cache_used"] is False
        assert info["max_cg_iterations"] == _DEFAULT_EXACT_PREDICT_MAX_CG_ITER
        assert info["cg_tolerance"] == _DEFAULT_EXACT_PREDICT_CG_TOL
        assert _DEFAULT_PREDICT_MAX_CG_ITER == _DEFAULT_EXACT_PREDICT_MAX_CG_ITER
        assert _DEFAULT_PREDICT_CG_TOL == _DEFAULT_EXACT_PREDICT_CG_TOL

    def test_prepared_prediction_cache_reuses_device_state(self):
        payload = _run_backend_metadata_case(
            "prepared_prediction_cache_reuses_device_state"
        )
        assert payload["shape"] == [12]
        assert payload["mean_match"] is True
        assert payload["var_match"] is True

        cache_info = payload["cache_info"]
        assert cache_info["prediction_cache_handle"] != 0
        assert cache_info["prediction_cache_rank"] == 9
        assert cache_info["prediction_cache_has_love_root"] is True
        assert cache_info["prediction_cache_prepare_time_s"] > 0.0

        first_info = payload["first_info"]
        info = payload["info"]
        assert first_info["prediction_cache_used"] is True
        assert info["prediction_cache_used"] is True
        assert first_info["alpha_cache_used"] is True
        assert first_info["love_root_cache_used"] is True
        assert info["prediction_alpha_time_s"] == 0.0
        assert info["prediction_cache_rank"] == 9
        assert info["prediction_cache_has_love_root"] is True

    def test_fit_can_prepare_prediction_cache(self):
        payload = _run_backend_metadata_case("fit_prepares_prediction_cache")
        assert payload["shape"] == [12]
        info = payload["info"]
        assert info["prediction_cache_used"] is True
        assert info["prediction_cache_rank"] == 9
        assert info["prediction_cache_has_love_root"] is True
        assert info["prediction_alpha_time_s"] == 0.0
        assert info["love_root_cache_used"] is True

    def test_exact_prediction_parity_between_materialized_and_matrix_free(self):
        payload_mat = _run_backend_metadata_case(
            "exact_prediction_parity", method="materialized"
        )
        payload_mf = _run_backend_metadata_case(
            "exact_prediction_parity", method="matrix_free"
        )

        np.testing.assert_allclose(
            np.asarray(payload_mat["mean"], dtype=np.float32),
            np.asarray(payload_mf["mean"], dtype=np.float32),
            atol=2e-2,
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(payload_mat["var"], dtype=np.float32),
            np.asarray(payload_mf["var"], dtype=np.float32),
            atol=5e-2,
            rtol=5e-2,
        )

        mat_info = payload_mat["info"]
        mf_info = payload_mf["info"]
        assert mat_info["training_route"] == "materialized"
        assert mf_info["training_route"] == "matrix_free"
        assert mat_info["actual_prediction_route"] == "predict"
        assert mf_info["actual_prediction_route"] == "predict"
        assert mat_info["actual_variance_route"] == "predict"
        assert mf_info["actual_variance_route"] == "predict"
        assert mat_info["fallback_used"] is False
        assert mf_info["fallback_used"] is False
        assert mat_info["telemetry_quality"] == "observed"
        assert mf_info["telemetry_quality"] == "observed"
        assert mat_info["exact_block_cols"] == 16
        assert mf_info["exact_block_cols"] == 16
        assert mat_info["exact_cross_mode"] == "direct_fill_cross_covariance"
        assert mf_info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_matrix_free_blocked_blas_exact_prediction_matches_default(self):
        payload = _run_backend_metadata_case(
            "blocked_blas_exact_prediction_parity"
        )

        np.testing.assert_allclose(
            np.asarray(payload["baseline_mean"], dtype=np.float32),
            np.asarray(payload["blocked_mean"], dtype=np.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(payload["baseline_var"], dtype=np.float32),
            np.asarray(payload["blocked_var"], dtype=np.float32),
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(payload["default_mean"], dtype=np.float32),
            np.asarray(payload["blocked_mean"], dtype=np.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(payload["default_var"], dtype=np.float32),
            np.asarray(payload["blocked_var"], dtype=np.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        assert payload["mean_max_abs_diff"] <= 1e-5
        assert payload["variance_max_abs_diff"] <= 1e-3
        assert payload["default_blocked_mean_max_abs_diff"] <= 1e-5
        assert payload["default_blocked_variance_max_abs_diff"] <= 1e-5

        baseline_info = payload["baseline_info"]
        blocked_info = payload["blocked_info"]
        default_info = payload["default_info"]
        assert payload["shape"] == [128]
        assert baseline_info["training_route"] == "matrix_free"
        assert blocked_info["training_route"] == "matrix_free"
        assert default_info["training_route"] == "matrix_free"
        assert baseline_info["prediction_method"] == "matrix_free"
        assert blocked_info["prediction_method"] == "matrix_free"
        assert default_info["prediction_method"] == "matrix_free"
        assert baseline_info["exact_block_cols"] == 128
        assert blocked_info["exact_block_cols"] == 128
        assert default_info["exact_block_cols"] == 128
        assert baseline_info["exact_cg_block_count"] == 1
        assert blocked_info["exact_cg_block_count"] == 1
        assert default_info["exact_cg_block_count"] == 1
        assert baseline_info["exact_cross_mode"] == "direct_fill_cross_covariance"
        assert blocked_info["exact_cross_mode"] == "direct_fill_cross_covariance"
        assert default_info["exact_cross_mode"] == "direct_fill_cross_covariance"

    def test_matrix_free_blocked_blas_repeated_predict_and_save_load(self):
        payload = _run_backend_metadata_case("blocked_blas_repeated_and_save_load")

        assert payload["shape"] == [128]
        assert payload["repeat_mean_max_abs_diff"] <= 1e-5
        assert payload["repeat_variance_max_abs_diff"] <= 1e-3
        assert payload["loaded_mean_max_abs_diff"] <= 1e-5
        assert payload["loaded_variance_max_abs_diff"] <= 1e-3

        first_info = payload["first_info"]
        second_info = payload["second_info"]
        loaded_info = payload["loaded_info"]
        assert first_info["alpha_cache_used"] is False
        assert second_info["alpha_cache_used"] is True
        assert loaded_info["prediction_method"] == "matrix_free"
        assert second_info["exact_block_cols"] == 128
        assert loaded_info["exact_block_cols"] == 128
        assert second_info["exact_cg_block_count"] == 1
        assert loaded_info["exact_cg_block_count"] == 1

    def test_fit_can_enable_and_disable_early_stopping_from_python(self):
        X, y = _make_data(n=2000, d=3, seed=19)

        gp_disabled = SingleOutputGP(kernel=RBF(), verbose=False)
        result_disabled = gp_disabled.fit(
            X,
            y,
            max_iterations=10,
            learning_rate=0.01,
            method="materialized",
            num_probes=4,
            max_cg_iterations=25,
            preconditioner_rank=8,
            max_tridiag_iterations=10,
            preconditioner="greedy",
            enable_early_stopping=False,
            early_stop_patience=1,
            early_stop_tol=1e6,
            verbose=False,
        )

        gp_enabled = SingleOutputGP(kernel=RBF(), verbose=False)
        result_enabled = gp_enabled.fit(
            X,
            y,
            max_iterations=10,
            learning_rate=0.01,
            method="materialized",
            num_probes=4,
            max_cg_iterations=25,
            preconditioner_rank=8,
            max_tridiag_iterations=10,
            preconditioner="greedy",
            enable_early_stopping=True,
            early_stop_patience=1,
            early_stop_tol=1e6,
            verbose=False,
        )

        assert int(result_disabled.iterations) == 10
        assert bool(result_disabled.converged) is False
        assert int(result_enabled.iterations) < 10
        assert bool(result_enabled.converged) is True

        disabled_info = gp_disabled.backend_train_info
        enabled_info = gp_enabled.backend_train_info
        assert disabled_info["enable_early_stopping"] is False
        assert enabled_info["enable_early_stopping"] is True
        assert enabled_info["early_stop_patience"] == 1
        assert np.isclose(enabled_info["early_stop_tol"], 1e6)

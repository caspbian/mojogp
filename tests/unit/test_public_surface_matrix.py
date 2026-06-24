"""Unit tests for the live public package surface.

These tests pin the exported JIT-first API so stale docs/examples cannot drift
back toward the removed `MojoGP` surface.
"""

import inspect
from pathlib import Path

import mojogp


ROOT = Path(__file__).resolve().parents[2]


def test_primary_public_classes_are_exported():
    assert "SingleOutputGP" in mojogp.__all__
    assert "ExactGP" not in mojogp.__all__
    assert "MultiOutputGP" in mojogp.__all__
    assert "MultiOutputLMCGP" in mojogp.__all__


def test_dead_mojogp_class_is_not_exported():
    assert not hasattr(mojogp, "MojoGP")
    assert "MojoGP" not in mojogp.__all__


def test_single_output_predict_signature_matches_live_surface():
    sig = inspect.signature(mojogp.SingleOutputGP.predict)
    assert "return_full" not in sig.parameters
    assert "variance_method" in sig.parameters
    assert "return_var" in sig.parameters
    assert "return_std" in sig.parameters


def test_multi_output_predict_signature_matches_live_surface():
    sig = inspect.signature(mojogp.MultiOutputGP.predict)
    assert "return_var" in sig.parameters
    assert "return_std" in sig.parameters
    assert "variance_method" in sig.parameters


def test_multi_output_fit_exposes_current_noise_surface():
    sig = inspect.signature(mojogp.MultiOutputGP.fit)
    assert "fixed_observation_noise" not in sig.parameters
    assert "observation_noise" in sig.parameters
    assert "input_dependent_noise" in sig.parameters
    assert "grouped_noise" in sig.parameters


def test_lmc_fit_exposes_noise_placeholders():
    sig = inspect.signature(mojogp.MultiOutputLMCGP.fit)
    assert "fixed_observation_noise" in sig.parameters
    assert "input_dependent_noise" in sig.parameters
    assert "grouped_noise" in sig.parameters


def test_single_output_fit_uses_max_iterations_not_num_iterations():
    sig = inspect.signature(mojogp.SingleOutputGP.fit)
    assert "max_iterations" in sig.parameters
    assert "num_iterations" not in sig.parameters
    assert "observation_noise" in sig.parameters
    assert "observation_noise_fn" in sig.parameters
    assert "noise_model" in sig.parameters
    assert "noise_group_train" in sig.parameters
    assert "group_noise" in sig.parameters


def test_public_solver_controls_use_canonical_names():
    expected = {
        "max_cg_iterations",
        "cg_tolerance",
        "max_tridiag_iterations",
        "preconditioner_rank",
        "preconditioner",
    }
    stale = {
        "max_cg_iter",
        "cg_tol",
        "max_tridiag_iter",
        "precond_rank",
        "precond",
    }

    checked = [
        inspect.signature(mojogp.SingleOutputGP.fit),
        inspect.signature(mojogp.SingleOutputGP.predict),
        inspect.signature(mojogp.SingleOutputGP.prepare_prediction_cache),
        inspect.signature(mojogp.MultiOutputGP),
        inspect.signature(mojogp.MultiOutputLMCGP),
    ]

    for sig in checked:
        names = set(sig.parameters)
        assert names.isdisjoint(stale)

    assert expected <= set(inspect.signature(mojogp.SingleOutputGP.fit).parameters)
    assert {"preconditioner_rank"} <= set(
        inspect.signature(mojogp.SingleOutputGP.predict).parameters
    )
    assert {"preconditioner_rank"} <= set(
        inspect.signature(mojogp.SingleOutputGP.prepare_prediction_cache).parameters
    )
    assert expected <= set(inspect.signature(mojogp.MultiOutputGP).parameters)
    assert expected <= set(inspect.signature(mojogp.MultiOutputLMCGP).parameters)


def test_current_kernel_shortcuts_are_exported():
    for name in [
        "RBF",
        "Matern12",
        "Matern32",
        "Matern52",
        "RQ",
        "Periodic",
        "Linear",
        "Polynomial",
        "GD",
        "CR",
        "EHH",
        "HH",
        "FE",
    ]:
        assert name in mojogp.__all__


def test_examples_and_notebooks_avoid_removed_mojogp_api():
    checked_files = list((ROOT / "examples").glob("*.py")) + list(
        (ROOT / "notebooks" / "examples").glob("*.py")
    )

    for path in checked_files:
        text = path.read_text()
        assert "from mojogp import MojoGP" not in text, path
        assert "MojoGP(" not in text, path


def test_canonical_docs_do_not_repeat_stale_multi_output_limitations():
    readme = (ROOT / "README.md").read_text()
    package_doc = (ROOT / "mojogp" / "__init__.py").read_text()

    stale_phrases = [
        "Multi-output only supports 'materialized' method",
        "LMC variance is a CPU-side diagonal approximation",
        "Multi-output categorical kernels are not yet supported",
    ]

    for phrase in stale_phrases:
        assert phrase not in readme
        assert phrase not in package_doc


def test_canonical_docs_mark_lmc_ard_available_on_documented_scope():
    api_doc = (ROOT / "docs" / "API.md").read_text()
    package_doc = (ROOT / "mojogp" / "__init__.py").read_text()

    assert "`ard=True` | Alpha on tested continuous scope" in api_doc
    assert "ARD applies once per continuous latent after active-dim/categorical remapping" in api_doc
    assert "MultiOutputLMCGP(ard=True)` is available on the documented continuous LMC scope" in package_doc
    assert "single-relevant-dimension recovery gate" in api_doc
    assert "Arbitrary relevance recovery remains workflow-dependent" in api_doc


def test_canonical_docs_include_lmc_support_matrix_and_variance_contract():
    api_doc = (ROOT / "docs" / "API.md").read_text()
    feature_matrix = (ROOT / "docs" / "FEATURE_MATRIX.md").read_text()
    package_doc = (ROOT / "mojogp" / "__init__.py").read_text()

    assert "### `MultiOutputLMCGP`" in api_doc
    assert "#### LMC Support Matrix" in api_doc
    assert "sum_s kron(K_s, A_s) + I_n kron(D_task)" in api_doc
    assert 'backend_predict_info["actual_prediction_route"]' in api_doc
    assert "Pure categorical LMC | Unsupported" in api_doc
    assert "Fixed per-sample-per-task noise `[n, T]` | Alpha on targeted continuous LMC scope" in api_doc
    assert "lmc_variance_exactness" in api_doc
    assert "exact_full_lmc_covariance" in api_doc
    assert "scalar_latent_approximation" in api_doc
    assert "Mixed LMC Supported Patterns" in api_doc
    assert "Feature Matrix" in feature_matrix
    assert "SingleOutput: Continuous" in feature_matrix
    assert "Fixed per-sample noise `[n]` | alpha | in-dev" in feature_matrix
    assert "Continuous `*` categorical subtrees" in api_doc
    assert "marginal predictive observation variance" in api_doc
    assert "fixed per-sample-per-task LMC observation noise `[n, T]` is available" in package_doc


def test_canonical_docs_include_single_output_continuous_alpha_evidence():
    api_doc = (ROOT / "docs" / "API.md").read_text()
    feature_matrix = (ROOT / "docs" / "FEATURE_MATRIX.md").read_text()

    assert "#### SingleOutput Continuous Support Notes" in api_doc
    assert "Exact and LOVE variance | Alpha on continuous SingleOutput scope" in api_doc
    assert "Polynomial pathwise sampling | Alpha on fixed-positive-integer SingleOutput scope" in api_doc
    assert "| SingleOutput: Continuous | alpha | alpha | alpha | alpha |" in feature_matrix
    assert "| SingleOutput: Continuous | alpha | alpha | alpha | unsupported |" in feature_matrix


def test_single_output_doc_examples_use_live_prediction_surface():
    package_doc = (ROOT / "mojogp" / "__init__.py").read_text()

    assert "mean, std = gp.predict(X_test, return_std=True)" in package_doc
    assert 'pred = gp.predict(X_test, variance_method="exact")' in package_doc

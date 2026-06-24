"""Unit tests for normalized route metadata helpers."""

from mojogp._multi_output_backend import (
    build_backend_predict_info,
    build_backend_train_info,
    resolve_preconditioner_settings,
)


def test_build_backend_train_info_fills_route_defaults():
    info = build_backend_train_info(raw={}, method="materialized")

    assert info == {"training_route": "materialized", "materialization_mode": 1}


def test_build_backend_train_info_preserves_preconditioner_fields():
    raw = {
        "training_route": "matrix_free",
        "materialization_mode": 0,
        "precond_rank": 8,
        "precond_method": 2,
        "precond_rebuild_count": 3,
        "precond_rebuild_threshold": 0.25,
        "max_tridiag_iter": 15,
        "use_preconditioner": True,
    }

    info = build_backend_train_info(raw=raw, method="matrix_free")

    assert info["training_route"] == "matrix_free"
    assert info["materialization_mode"] == 0
    assert info["precond_rank"] == 8
    assert info["precond_method"] == 2
    assert info["precond_rebuild_count"] == 3
    assert info["precond_rebuild_threshold"] == 0.25
    assert info["max_tridiag_iter"] == 15
    assert info["use_preconditioner"] is True


def test_build_backend_predict_info_carries_optional_fields():
    info = build_backend_predict_info(
        requested_method="matrix_free",
        actual_prediction_route="predict_multi_output",
        backend_prediction_used=True,
        backend_variance_used=False,
        variance_method="love",
        fallback_used=True,
        backend_error="non-finite LOVE variance",
        actual_variance_route="predict_multi_output_exact_retry",
        training_route="matrix_free",
        precond_rank=12,
        precond_method=1,
        precond_rebuild_count=4,
    )

    assert info["requested_method"] == "matrix_free"
    assert info["actual_prediction_route"] == "predict_multi_output"
    assert info["backend_prediction_used"] is True
    assert info["backend_variance_used"] is False
    assert info["variance_method"] == "love"
    assert info["fallback_used"] is True
    assert info["backend_error"] == "non-finite LOVE variance"
    assert info["actual_variance_route"] == "predict_multi_output_exact_retry"
    assert info["training_route"] == "matrix_free"
    assert info["precond_rank"] == 12
    assert info["precond_method"] == 1
    assert info["precond_rebuild_count"] == 4


def test_resolve_preconditioner_settings_maps_threshold_and_method():
    resolved = resolve_preconditioner_settings(
        {}, precond_rank=6, precond_rebuild_threshold=0.25, precond="rpcholesky"
    )

    assert resolved["precond_rank"] == 6
    assert resolved["precond_rebuild_threshold"] == 0.25
    assert resolved["precond"] == "rpcholesky"
    assert resolved["precond_method"] == 1
    assert resolved["use_preconditioner"] is True


def test_resolve_preconditioner_settings_disables_rank_when_requested():
    resolved = resolve_preconditioner_settings(
        {"precond_rank": 12}, use_preconditioner=False
    )

    assert resolved["use_preconditioner"] is False
    assert resolved["precond_rank"] == 0

"""Tests for the central feature-support registry and generated matrix."""

import warnings

import pytest

import mojogp
from mojogp.feature_support import (
    ExperimentalFeatureWarning,
    FeatureStatus,
    InDevelopmentFeatureWarning,
    MojoGPFeatureWarning,
    SURFACE_ICM_CONTINUOUS,
    TABLE_MAIN,
    TABLE_PREDICTION,
    assert_registry_complete,
    check_feature_support,
    get_feature_warnings_enabled,
    set_feature_warnings_enabled,
    feature_warnings_suppressed,
    render_feature_matrix_markdown,
    surface_for_icm,
    surface_for_single_output,
    warn_surface_status,
)


def test_registry_covers_every_matrix_surface():
    assert_registry_complete()


def test_generated_feature_matrix_uses_explicit_not_started_marker():
    rendered = render_feature_matrix_markdown()

    assert "| -- | not started / no implementation yet |" in rendered
    assert "Learned input-dependent heteroskedasticity | alpha | in-dev | -- | --" in rendered
    assert "|  |" not in rendered


def test_not_started_feature_raises_not_implemented():
    surface = surface_for_icm(is_mixed=False)

    with pytest.raises(NotImplementedError, match="Learned input-dependent heteroskedasticity"):
        check_feature_support(TABLE_MAIN, surface, "learned_input_dependent_noise")


def test_experimental_feature_warns():
    surface = surface_for_single_output(is_mixed=True)

    with pytest.warns(ExperimentalFeatureWarning, match="LOVE Variance"):
        entry = check_feature_support(TABLE_PREDICTION, surface, "love_variance")

    assert entry.status == FeatureStatus.EXP


def test_feature_warning_global_setting_suppresses_maturity_warnings():
    surface = surface_for_single_output(is_mixed=True)
    previous = get_feature_warnings_enabled()

    try:
        set_feature_warnings_enabled(False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = check_feature_support(TABLE_PREDICTION, surface, "love_variance")

        assert entry.status == FeatureStatus.EXP
        assert not [w for w in caught if issubclass(w.category, MojoGPFeatureWarning)]
    finally:
        set_feature_warnings_enabled(previous)


def test_feature_warning_context_suppresses_and_restores():
    surface = surface_for_single_output(is_mixed=True)
    set_feature_warnings_enabled(True)

    with feature_warnings_suppressed():
        assert not get_feature_warnings_enabled()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_feature_support(TABLE_PREDICTION, surface, "love_variance")

    assert get_feature_warnings_enabled()
    assert not [w for w in caught if issubclass(w.category, MojoGPFeatureWarning)]


def test_surface_status_warning_uses_feature_warning_setting():
    previous = get_feature_warnings_enabled()

    try:
        set_feature_warnings_enabled(False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = warn_surface_status(SURFACE_ICM_CONTINUOUS)

        assert entry.status == FeatureStatus.EXP
        assert not [w for w in caught if issubclass(w.category, MojoGPFeatureWarning)]
    finally:
        set_feature_warnings_enabled(previous)


def test_in_development_feature_still_raises_when_requested_under_suppression():
    surface = surface_for_single_output(is_mixed=True)

    with feature_warnings_suppressed():
        with pytest.raises(NotImplementedError, match="Fixed per-sample noise"):
            check_feature_support(
                TABLE_MAIN,
                surface,
                "fixed_per_sample_noise",
                fail_on_in_dev=True,
            )


def test_in_development_feature_warning_remains_enabled_by_default():
    surface = surface_for_single_output(is_mixed=True)

    with pytest.warns(InDevelopmentFeatureWarning, match="Fixed per-sample noise"):
        entry = check_feature_support(TABLE_MAIN, surface, "fixed_per_sample_noise")

    assert entry.status == FeatureStatus.IN_DEV


def test_public_settings_api_controls_feature_warning_setting():
    previous = mojogp.settings.get_feature_warnings_enabled()

    try:
        mojogp.set_feature_warnings_enabled(False)
        assert not mojogp.get_feature_warnings_enabled()
        assert not mojogp.settings.get_feature_warnings_enabled()

        mojogp.settings.set_feature_warnings_enabled(True)
        assert mojogp.get_feature_warnings_enabled()
    finally:
        mojogp.settings.set_feature_warnings_enabled(previous)


def test_alpha_feature_does_not_emit_maturity_warning():
    surface = surface_for_single_output(is_mixed=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entry = check_feature_support(TABLE_PREDICTION, surface, "love_variance")

    assert entry.status == FeatureStatus.ALPHA
    assert not [w for w in caught if issubclass(w.category, MojoGPFeatureWarning)]

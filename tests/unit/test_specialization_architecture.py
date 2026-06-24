from __future__ import annotations

from mojogp import RBF
from mojogp.specialization import (
    EXISTING_LOW_D_STATIONARY_LEAF_PROFILE,
    SpecializationProfile,
    SpecializationRequest,
    apply_specialization_to_case_id,
    build_single_output_descriptor,
    default_specialization_registry,
    existing_materialized_predict_ncols_profile,
    manual_benchmark_override_profile,
    translate_compile_inputs,
)


def test_registry_returns_disabled_default_decision():
    descriptor = build_single_output_descriptor(
        kernel=RBF(),
        dim=5,
        training_method="matrix_free",
        n_train=2000,
    )

    decision = default_specialization_registry().resolve(
        descriptor,
        SpecializationRequest.disabled(),
    )

    assert decision.mode == "disabled"
    assert decision.applied is False
    assert decision.profile.specialization_key == "default"
    assert decision.default_equivalent is True


def test_explicit_profile_translates_to_compile_inputs():
    descriptor = build_single_output_descriptor(
        kernel=RBF(),
        dim=5,
        training_method="matrix_free",
        n_train=5000,
    )
    profile = SpecializationProfile(
        specialization_key="rbf_tm1_probe",
        source="benchmark",
        schedule_overrides={
            "tm": 1,
            "use_shmem": True,
            "j_unroll": 1,
            "ncols": [6, 1],
            "block_size": 128,
            "max_registers": 200,
            "precompute_inv_ls": False,
        },
        ncols_hint=(6, 1),
        module_suffix="rbf_tm1_probe",
    )
    decision = default_specialization_registry().resolve(
        descriptor,
        SpecializationRequest(mode="applied", profile=profile),
    )

    translation = translate_compile_inputs(decision)

    assert decision.applied is True
    assert translation.schedule_overrides is not None
    assert translation.schedule_overrides.tm == 1
    assert translation.schedule_overrides.block_size == 128
    assert translation.ncols_hint == [6, 1]
    assert translation.module_suffix == "rbf_tm1_probe"


def test_case_id_suffix_only_for_applied_non_default_specialization():
    base_case_id = "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"

    assert apply_specialization_to_case_id(base_case_id, None) == base_case_id
    assert (
        apply_specialization_to_case_id(
            base_case_id,
            {"specialization_mode": "shadow", "specialization_key": "shadow_probe"},
        )
        == base_case_id
    )
    assert (
        apply_specialization_to_case_id(
            base_case_id,
            {"specialization_mode": "applied", "specialization_key": "default"},
        )
        == base_case_id
    )
    assert (
        apply_specialization_to_case_id(
            base_case_id,
            {
                "specialization_mode": "applied",
                "specialization_key": "rbf_tm1_probe",
            },
        )
        == f"{base_case_id}.spec.rbf_tm1_probe"
    )


def test_builtin_profile_catalog_exposes_versioned_profiles():
    manual = manual_benchmark_override_profile(
        "rbf_tm1_probe",
        schedule_overrides={"tm": 1},
        ncols_hint=(6, 1),
    )
    materialized = existing_materialized_predict_ncols_profile((11, 6, 1))

    assert EXISTING_LOW_D_STATIONARY_LEAF_PROFILE.policy_version == "v1"
    assert materialized.specialization_key == "existing_materialized_predict_ncols"
    assert materialized.ncols_hint == (11, 6, 1)
    assert manual.source == "benchmark"
    assert manual.policy_version == "v1"

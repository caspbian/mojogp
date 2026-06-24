from __future__ import annotations

from tests.benchmarks.single_output_scaling import _single_output_scaling_suite_name


def test_single_output_scaling_suite_name_tracks_specific_benchmark_family():
    assert (
        _single_output_scaling_suite_name("single_output_preset_sweep_matrix_free")
        == "single_output_preset_sweep"
    )
    assert (
        _single_output_scaling_suite_name("single_output_extensive_scaling_materialized")
        == "single_output_extensive_scaling"
    )
    assert (
        _single_output_scaling_suite_name("scaling_certification_extensive_materialized")
        == "scaling_certification"
    )
    assert _single_output_scaling_suite_name("ad_hoc_single_output") == "single_output_scaling"

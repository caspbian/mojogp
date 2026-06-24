from __future__ import annotations

from pathlib import Path

from tests.benchmarks.dataset_manifest import DatasetSpec, ensure_dataset, load_dataset_artifact
from tests.shared.benchmarking.data_generators import generate_structured_function_data


def test_dataset_id_is_deterministic_for_same_spec():
    spec_a = DatasetSpec(
        generator_name="single_output_structured_function",
        config={"n_train": 2000, "n_test": 200, "d": 5, "function_type": "smooth", "noise_level": "medium", "seed": 42, "mean_offset": 0.0},
    )
    spec_b = DatasetSpec(
        generator_name="single_output_structured_function",
        config={"n_train": 2000, "n_test": 200, "d": 5, "function_type": "smooth", "noise_level": "medium", "seed": 42, "mean_offset": 0.0},
    )
    assert spec_a.dataset_id == spec_b.dataset_id


def test_ensure_dataset_reuses_saved_dataset(tmp_path: Path):
    spec = DatasetSpec(
        generator_name="single_output_structured_function",
        config={"n_train": 2000, "n_test": 200, "d": 5, "function_type": "smooth", "noise_level": "medium", "seed": 42, "mean_offset": 0.0},
    )
    dataset_id_a, npz_path_a, meta_path_a = ensure_dataset(spec, generate_structured_function_data, dataset_root=tmp_path)
    dataset_id_b, npz_path_b, meta_path_b = ensure_dataset(spec, generate_structured_function_data, dataset_root=tmp_path)
    assert dataset_id_a == dataset_id_b
    assert npz_path_a == npz_path_b
    assert meta_path_a == meta_path_b

    loaded = load_dataset_artifact(npz_path_a)
    assert loaded["X_train"].shape == (2000, 5)
    assert loaded["X_test"].shape == (200, 5)

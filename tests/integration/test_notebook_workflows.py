"""Notebook-facing workflow checks for the public marimo examples."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_public_exactgp_multioutput_prediction_workflow_notebooks_pass_marimo_check():
    """Public ExactGP, multi-output, prediction, persistence, and sampling notebooks should remain valid marimo apps."""

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "notebooks",
            "marimo",
            "check",
            "notebooks/examples/01_hello_gp.py",
            "notebooks/examples/04_multi_output.py",
            "notebooks/examples/10_predictive_uncertainty_and_love_variance.py",
            "notebooks/examples/17_model_persistence_roundtrip.py",
            "notebooks/examples/18_posterior_sampling_methods.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_public_exactgp_multioutput_prediction_workflow_notebooks_export_to_html():
    """Public ExactGP, multi-output, prediction, persistence, and sampling notebooks should export cleanly to HTML."""

    notebooks = [
        "notebooks/examples/01_hello_gp.py",
        "notebooks/examples/04_multi_output.py",
        "notebooks/examples/10_predictive_uncertainty_and_love_variance.py",
        "notebooks/examples/17_model_persistence_roundtrip.py",
        "notebooks/examples/18_posterior_sampling_methods.py",
    ]
    with tempfile.TemporaryDirectory(prefix="mojogp_notebook_export_") as tmp_dir:
        for notebook in notebooks:
            output_path = Path(tmp_dir) / (Path(notebook).stem + ".html")
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "--extra",
                    "notebooks",
                    "marimo",
                    "export",
                    "html",
                    notebook,
                    "-o",
                    str(output_path),
                    "-f",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=1200,
            )
            assert completed.returncode == 0, completed.stderr or completed.stdout
            assert output_path.exists(), f"Expected export output for {notebook}"

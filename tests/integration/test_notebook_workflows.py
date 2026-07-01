"""Notebook-facing workflow checks for the public marimo examples."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks" / "examples"
RENDER_DIR = NOTEBOOK_DIR / "__marimo__" / "ipynb"
HTML_DIR = NOTEBOOK_DIR / "__marimo__" / "html"
MARIMO_EMBEDDED_IMAGE_PATTERN = re.compile(
    r"<marimo-mime-renderer\b[^>]*\bdata-data=.*?image/png",
    re.DOTALL,
)


def public_notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.py"))


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


def test_public_marimo_notebooks_have_current_github_rendered_ipynb():
    """Every public marimo notebook should have a current GitHub-renderable ipynb artifact."""

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "notebooks",
            "--with",
            "marimo==0.23.1",
            "--with",
            "nbformat",
            "python",
            "tools/render_marimo_notebooks.py",
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    for notebook in public_notebook_paths():
        rendered = RENDER_DIR / f"{notebook.stem}.ipynb"
        rendered_html = HTML_DIR / f"{notebook.stem}.html"
        assert rendered.exists(), f"Missing rendered artifact for {notebook.name}"
        assert rendered_html.exists(), f"Missing rendered HTML for {notebook.name}"
        with rendered.open(encoding="utf-8") as file:
            document = json.load(file)
        assert document.get("nbformat") == 4
        assert document.get("cells"), f"Rendered notebook has no cells: {rendered}"
        output_count = sum(len(cell.get("outputs", [])) for cell in document["cells"])
        assert output_count > 0, f"Rendered notebook has no outputs: {rendered}"
        for cell in document["cells"]:
            outputs = cell.get("outputs", [])
            embedded_image_count = sum(
                len(MARIMO_EMBEDDED_IMAGE_PATTERN.findall(str(output.get("data", {}).get("text/html", ""))))
                for output in outputs
            )
            if embedded_image_count:
                assert any("image/png" in output.get("data", {}) for output in outputs)
        html = rendered_html.read_text(encoding="utf-8")
        assert "<!doctype html>" in html[:100].lower()

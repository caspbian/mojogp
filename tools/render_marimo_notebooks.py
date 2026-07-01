#!/usr/bin/env python3
"""Render public marimo examples to output-bearing GitHub artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "examples"
RENDER_DIR = NOTEBOOK_DIR / "__marimo__" / "ipynb"
HTML_DIR = NOTEBOOK_DIR / "__marimo__" / "html"
MANIFEST_PATH = NOTEBOOK_DIR / "__marimo__" / "render_manifest.json"
MANIFEST_VERSION = 2
MARIMO_MIME_BUNDLE_PATTERN = re.compile(
    r"<marimo-mime-renderer\b[^>]*\bdata-data=(['\"])(?P<data>.*?)\1",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate committed ipynb render artifacts in place.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed ipynb render artifacts are missing or stale.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream marimo export output instead of only showing failures.",
    )
    return parser.parse_args()


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.py"))


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def rendered_path(render_dir: Path, notebook: Path) -> Path:
    return render_dir / f"{notebook.stem}.ipynb"


def html_path(html_dir: Path, notebook: Path) -> Path:
    return html_dir / f"{notebook.stem}.html"


def run_command(command: list[str], *, stream: bool) -> None:
    if stream:
        print("$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True)
    else:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        if not stream:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
        raise RuntimeError("Command failed: " + " ".join(command))


def run_ipynb_export(notebook: Path, output_path: Path, *, stream: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "ipynb",
        relative_to_root(notebook),
        "-o",
        str(output_path),
        "--sort=top-down",
        "--include-outputs",
        "-f",
    ]
    run_command(command, stream=stream)
    expose_marimo_embedded_images(output_path)


def text_mime_to_string(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def extract_marimo_embedded_pngs(text_html: object) -> list[dict[str, object]]:
    text = text_mime_to_string(text_html)
    images: list[dict[str, object]] = []
    for match in MARIMO_MIME_BUNDLE_PATTERN.finditer(text):
        raw_attribute = html.unescape(match.group("data"))
        try:
            encoded_bundle = json.loads(raw_attribute)
            bundle = json.loads(encoded_bundle)
        except json.JSONDecodeError:
            continue

        image_data = bundle.get("image/png")
        if not isinstance(image_data, str):
            continue
        prefix = "data:image/png;base64,"
        if image_data.startswith(prefix):
            image_data = image_data[len(prefix) :]
        image: dict[str, object] = {
            "output_type": "display_data",
            "metadata": bundle.get("__metadata__", {}),
            "data": {"image/png": image_data},
        }
        images.append(image)
    return images


def expose_marimo_embedded_images(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in document.get("cells", []):
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        updated_outputs: list[dict[str, object]] = []
        for output in outputs:
            if not isinstance(output, dict):
                updated_outputs.append(output)
                continue
            data = output.get("data")
            if isinstance(data, dict) and "text/html" in data:
                images = extract_marimo_embedded_pngs(data["text/html"])
                if images:
                    updated_outputs.extend(images)
                    changed = True
            updated_outputs.append(output)
        cell["outputs"] = updated_outputs

    if changed:
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def run_html_export(notebook: Path, output_path: Path, *, stream: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        relative_to_root(notebook),
        "-o",
        str(output_path),
        "-f",
    ]
    run_command(command, stream=stream)
    normalize_html(output_path)


def normalize_html(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def build_manifest(notebooks: list[Path]) -> dict[str, object]:
    try:
        marimo_version = importlib.metadata.version("marimo")
    except importlib.metadata.PackageNotFoundError:
        marimo_version = "unknown"

    return {
        "version": MANIFEST_VERSION,
        "marimo_version": marimo_version,
        "artifacts": [
            {
                "source": relative_to_root(notebook),
                "source_sha256": file_sha256(notebook),
                "ipynb": relative_to_root(rendered_path(RENDER_DIR, notebook)),
                "html": relative_to_root(html_path(HTML_DIR, notebook)),
            }
            for notebook in notebooks
        ],
    }


def write_manifest(notebooks: list[Path]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(build_manifest(notebooks), indent=2) + "\n", encoding="utf-8")


def remove_stale_files(directory: Path, expected_names: set[str], pattern: str) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.glob(pattern)):
        if path.name not in expected_names:
            path.unlink()


def render_all(ipynb_dir: Path, html_dir: Path, notebooks: list[Path], *, stream: bool) -> None:
    expected_ipynb = {rendered_path(ipynb_dir, notebook).name for notebook in notebooks}
    expected_html = {html_path(html_dir, notebook).name for notebook in notebooks}
    remove_stale_files(ipynb_dir, expected_ipynb, "*.ipynb")
    remove_stale_files(html_dir, expected_html, "*.html")
    for notebook in notebooks:
        ipynb_destination = rendered_path(ipynb_dir, notebook)
        html_destination = html_path(html_dir, notebook)
        print(
            f"Rendering {relative_to_root(notebook)} -> "
            f"{relative_to_root(ipynb_destination)}, {relative_to_root(html_destination)}",
            flush=True,
        )
        run_ipynb_export(notebook, ipynb_destination, stream=stream)
        run_html_export(notebook, html_destination, stream=stream)


def find_file_differences(actual_dir: Path, expected_names: set[str], pattern: str) -> list[str]:
    differences: list[str] = []
    for name in sorted(expected_names):
        actual = actual_dir / name
        if not actual.exists():
            differences.append(f"missing {relative_to_root(actual)}")

    if actual_dir.exists():
        for path in sorted(actual_dir.glob(pattern)):
            if path.name not in expected_names:
                differences.append(f"stale extra {relative_to_root(path)}")

    return differences


def find_render_differences(notebooks: list[Path]) -> list[str]:
    expected_ipynb = {rendered_path(RENDER_DIR, notebook).name for notebook in notebooks}
    expected_html = {html_path(HTML_DIR, notebook).name for notebook in notebooks}
    differences = find_file_differences(RENDER_DIR, expected_ipynb, "*.ipynb")
    differences.extend(find_file_differences(HTML_DIR, expected_html, "*.html"))

    if not MANIFEST_PATH.exists():
        differences.append(f"missing {relative_to_root(MANIFEST_PATH)}")
        return differences

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        differences.append(f"invalid {relative_to_root(MANIFEST_PATH)}: {exc}")
        return differences

    expected_manifest = build_manifest(notebooks)
    if manifest != expected_manifest:
        differences.append(f"stale {relative_to_root(MANIFEST_PATH)}")

    return differences


def check_rendered(notebooks: list[Path], *, stream: bool) -> int:
    differences = find_render_differences(notebooks)

    if not differences:
        print(f"Rendered notebooks are current: {len(notebooks)} checked")
        return 0

    print("Rendered notebooks are stale or incomplete:", file=sys.stderr)
    for difference in differences:
        print(f"  - {difference}", file=sys.stderr)
    print("Run `task notebooks:render` and commit the updated artifacts.", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    notebooks = notebook_paths()
    if not notebooks:
        print("No public marimo notebooks found under notebooks/examples.", file=sys.stderr)
        return 1

    if args.write:
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        render_all(RENDER_DIR, HTML_DIR, notebooks, stream=args.stream)
        write_manifest(notebooks)
        print(
            f"Rendered notebooks: {len(notebooks)} written to "
            f"{relative_to_root(RENDER_DIR)} and {relative_to_root(HTML_DIR)}"
        )
        return 0

    return check_rendered(notebooks, stream=args.stream)


if __name__ == "__main__":
    raise SystemExit(main())

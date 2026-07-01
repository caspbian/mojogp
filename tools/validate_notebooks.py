#!/usr/bin/env python3
"""Validate and export public marimo notebook examples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "examples"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for validation artifacts.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream subprocess logs while still writing validation artifacts.",
    )
    return parser.parse_args()


def run_command(
    command: list[str],
    cwd: Path,
    *,
    stream: bool,
) -> subprocess.CompletedProcess[str]:
    if not stream:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True)

    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(output_lines),
        stderr="",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.py"))


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    notebooks = notebook_paths()
    if not notebooks:
        print("No marimo notebooks found under notebooks/examples.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (RESULTS_DIR / f"notebook_validation_{timestamp}")
    output_dir = output_dir.resolve()
    html_dir = output_dir / "html"
    logs_dir = output_dir / "logs"
    html_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    notebook_args = [relative_to_root(path) for path in notebooks]
    check_command = [
        "uv",
        "run",
        "--extra",
        "notebooks",
        "marimo",
        "check",
        *notebook_args,
    ]
    print(f"Checking {len(notebooks)} marimo notebooks...", flush=True)
    check_result = run_command(check_command, cwd=ROOT, stream=args.stream)
    write_text(logs_dir / "marimo_check.stdout.txt", check_result.stdout)
    write_text(logs_dir / "marimo_check.stderr.txt", check_result.stderr)

    exports: list[dict[str, object]] = []
    export_failures = 0
    for notebook in notebooks:
        print(f"Exporting {relative_to_root(notebook)}...", flush=True)
        html_output = html_dir / f"{notebook.stem}.html"
        export_command = [
            "uv",
            "run",
            "--extra",
            "notebooks",
            "marimo",
            "export",
            "html",
            relative_to_root(notebook),
            "-o",
            str(html_output),
            "-f",
        ]
        export_result = run_command(export_command, cwd=ROOT, stream=args.stream)
        write_text(logs_dir / f"{notebook.stem}.stdout.txt", export_result.stdout)
        write_text(logs_dir / f"{notebook.stem}.stderr.txt", export_result.stderr)
        if export_result.returncode != 0:
            export_failures += 1
        exports.append(
            {
                "notebook": relative_to_root(notebook),
                "html": relative_to_root(html_output),
                "returncode": export_result.returncode,
                "ok": export_result.returncode == 0,
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": relative_to_root(output_dir),
        "check": {
            "command": check_command,
            "returncode": check_result.returncode,
            "ok": check_result.returncode == 0,
            "stdout_log": relative_to_root(logs_dir / "marimo_check.stdout.txt"),
            "stderr_log": relative_to_root(logs_dir / "marimo_check.stderr.txt"),
        },
        "exports": exports,
    }
    manifest_path = output_dir / "manifest.json"
    write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    print(f"Notebook validation output: {relative_to_root(output_dir)}")
    print(f"marimo check: {'PASS' if check_result.returncode == 0 else 'FAIL'}")
    print(f"HTML exports: {len(notebooks) - export_failures}/{len(notebooks)} succeeded")
    print(f"Manifest: {relative_to_root(manifest_path)}")

    if check_result.returncode != 0 or export_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

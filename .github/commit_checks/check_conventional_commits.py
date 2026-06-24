#!/usr/bin/env python3
"""Validate commit messages against the MojoGP commit policy."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ALLOWED_TYPES = {
    "feat",
    "fix",
    "perf",
    "refactor",
    "test",
    "doc",
    "docs",
    "ci",
    "build",
    "chore",
    "revert",
}

HEADER_RE = re.compile(
    r"^(?P<type>[a-z][a-z0-9-]*)(?:\((?P<scope>[a-z0-9._/-]+)\))?(?P<breaking>!)?: (?P<description>\S.*)$"
)

AI_VENDOR_RE = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)

GENERIC_DESCRIPTIONS = {
    "change",
    "changes",
    "cleanup",
    "fix",
    "fixes",
    "fix stuff",
    "misc",
    "stuff",
    "temp",
    "test",
    "update",
    "updates",
    "update code",
    "wip",
    "work in progress",
}

MAX_HEADER_LENGTH = 100
MAX_SENTENCES = 3


@dataclass(frozen=True)
class CommitMessage:
    ref: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    @property
    def non_comment_lines(self) -> list[str]:
        return [line for line in self.lines if not line.startswith("#")]


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def read_message_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def commits_from_range(commit_range: str, exclude_remotes: bool = False) -> list[CommitMessage]:
    rev_list_args = ["rev-list", "--no-merges", commit_range]
    if exclude_remotes:
        rev_list_args.extend(["--not", "--remotes"])
    commits = [line for line in run_git(rev_list_args).splitlines() if line]
    messages: list[CommitMessage] = []
    for commit in commits:
        text = run_git(["log", "--format=%B", "-n", "1", commit])
        messages.append(CommitMessage(ref=commit, text=text))
    return messages


def first_non_empty_line(lines: list[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def validate_message(message: CommitMessage) -> list[str]:
    errors: list[str] = []
    lines = message.non_comment_lines
    header = first_non_empty_line(lines)

    if not header:
        return ["commit message is empty"]

    if AI_VENDOR_RE.search(message.text):
        errors.append("commit message must not mention Claude or Anthropic")

    if len(header) > MAX_HEADER_LENGTH:
        errors.append(f"header must be {MAX_HEADER_LENGTH} characters or fewer")

    match = HEADER_RE.match(header)
    if not match:
        errors.append("header must match '<type>[optional scope][!]: <description>'")
        return errors

    commit_type = match.group("type")
    description = match.group("description").strip()

    if commit_type not in ALLOWED_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TYPES))
        errors.append(f"type '{commit_type}' is not allowed; use one of: {allowed}")

    if description.endswith("."):
        errors.append("header description should not end with a period")

    normalized_description = re.sub(r"\s+", " ", description.lower()).strip()
    if normalized_description in GENERIC_DESCRIPTIONS:
        errors.append("header description is too vague")

    if len(lines) > 1 and lines[1].strip():
        errors.append("body must be separated from the header by a blank line")

    prose = " ".join(line.strip() for line in lines if line.strip())
    if sentence_count(prose) > MAX_SENTENCES:
        errors.append(f"commit message should be at most {MAX_SENTENCES} sentences")

    return errors


def print_errors(messages: list[tuple[str, list[str]]]) -> None:
    for ref, errors in messages:
        print(f"ERROR: invalid commit message for {ref}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message-file", type=Path, help="Commit message file to validate")
    source.add_argument("--range", dest="commit_range", help="Git revision range to validate")
    parser.add_argument("--allow-empty-range", action="store_true", help="Treat an empty range as valid")
    parser.add_argument("--exclude-remotes", action="store_true", help="Exclude commits already present on any remote")
    args = parser.parse_args()

    if args.message_file is not None:
        messages = [CommitMessage(ref=str(args.message_file), text=read_message_file(args.message_file))]
    else:
        messages = commits_from_range(args.commit_range, exclude_remotes=args.exclude_remotes)
        if not messages and not args.allow_empty_range:
            print(f"ERROR: revision range has no commits: {args.commit_range}", file=sys.stderr)
            return 1

    invalid: list[tuple[str, list[str]]] = []
    for message in messages:
        errors = validate_message(message)
        if errors:
            invalid.append((message.ref, errors))

    if invalid:
        print_errors(invalid)
        return 1

    print(f"Conventional commit check passed for {len(messages)} commit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

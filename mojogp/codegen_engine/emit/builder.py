"""MojoBuilder: indentation-safe programmatic string builder for Mojo code.

All GPU kernel scaffolding and module assembly uses this builder
instead of f-strings or templates. Guarantees correct indentation.
"""

from contextlib import contextmanager


class MojoBuilder:
    """Builds Mojo source code with managed indentation.

    Usage:
        b = MojoBuilder()
        b.line("fn foo(x: Int) -> Int:")
        with b.block():
            b.line("return x * 2")
        print(b.build())
    """

    def __init__(self, indent_str: str = "    "):
        self._parts: list[str] = []
        self._indent = 0
        self._indent_str = indent_str

    def line(self, text: str = ""):
        """Add a line with current indentation."""
        if text:
            self._parts.append(self._indent_str * self._indent + text)
        else:
            self._parts.append("")

    def raw(self, text: str):
        """Add raw text without indentation (for pre-formatted blocks)."""
        self._parts.append(text)

    def comment(self, text: str):
        """Add a comment line."""
        self.line(f"# {text}")

    def blank(self):
        """Add a blank line."""
        self._parts.append("")

    def indent(self):
        """Increase indentation by one level."""
        self._indent += 1

    def dedent(self):
        """Decrease indentation by one level."""
        self._indent = max(0, self._indent - 1)

    @contextmanager
    def block(self, header: str = ""):
        """Context manager: emit header, indent body, dedent on exit.

        Usage:
            with b.block("if x > 0:"):
                b.line("return x")
        """
        if header:
            self.line(header)
        self.indent()
        try:
            yield
        finally:
            self.dedent()

    def build(self) -> str:
        """Return the complete source as a single string."""
        return "\n".join(self._parts)

    def __len__(self) -> int:
        """Number of lines."""
        return len(self._parts)

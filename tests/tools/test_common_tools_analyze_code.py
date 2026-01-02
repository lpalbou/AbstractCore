from __future__ import annotations

from pathlib import Path

from abstractcore.tools.common_tools import analyze_code


def test_analyze_code_python_outlines_imports_classes_functions_and_attrs(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "import os\n"
        "\n"
        "class Foo:\n"
        "    def bar(self, x: int) -> str:\n"
        "        self.y = x\n"
        "        return str(x)\n"
        "\n"
        "def baz(a, b=1):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    out = analyze_code(file_path=str(path))
    abs_path = str(path.absolute())

    assert out.startswith(f"Code Analysis: {abs_path} (language=python, lines=9)")
    assert "imports:" in out
    assert "1: import os" in out

    assert "classes:" in out
    assert "Foo (lines 3-6)" in out
    assert "methods:" in out
    assert "4-6: bar(self, x: int) -> str" in out
    assert "self_attributes_set: y" in out

    assert "functions:" in out
    assert "8-9: baz(a, b=1)" in out


def test_analyze_code_javascript_outlines_imports_classes_functions_and_refs(tmp_path: Path) -> None:
    (tmp_path / "x.js").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "y.js").write_text("module.exports = {};\n", encoding="utf-8")

    path = tmp_path / "demo.js"
    path.write_text(
        "import x from './x.js';\n"
        "const y = require('./y');\n"
        "\n"
        "class Foo extends Bar {\n"
        "  method(a) { return a; }\n"
        "}\n"
        "\n"
        "export function baz(q) { return q; }\n"
        "const qux = (z) => { return z; };\n",
        encoding="utf-8",
    )

    out = analyze_code(file_path=str(path))
    abs_path = str(path.absolute())

    assert out.startswith(f"Code Analysis: {abs_path} (language=javascript, lines=9)")
    assert "imports:" in out
    assert "1: import ./x.js" in out
    assert "2: require ./y" in out

    assert "classes:" in out
    assert "Foo (lines 4-6) extends Bar" in out

    assert "functions:" in out
    assert "8: baz(q)" in out
    assert "9: qux(z) =>" in out

    assert "references:" in out
    assert f"./x.js -> {tmp_path.joinpath('x.js').absolute()} (exists)" in out


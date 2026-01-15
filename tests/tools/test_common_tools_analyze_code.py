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


def test_analyze_code_html_outlines_ids_assets_and_lint(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text("console.log('x');\n", encoding="utf-8")
    (tmp_path / "style.css").write_text("body { color: red; }\n", encoding="utf-8")

    path = tmp_path / "demo.html"
    content = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <title>Demo</title>\n"
        "  <link rel=\"stylesheet\" href=\"./style.css\">\n"
        "</head>\n"
        "<body>\n"
        "  <img src=\"./img.png\">\n"
        "  <a href=\"https://example.com\" target=\"_blank\">x</a>\n"
        "  <div id=\"app\"></div>\n"
        "  <div id=\"app\"></div>\n"
        "  <script src=\"./app.js\"></script>\n"
        "</body>\n"
        "</html>\n"
    )
    path.write_text(content, encoding="utf-8")

    out = analyze_code(file_path=str(path))
    abs_path = str(path.absolute())

    assert out.startswith(f"Code Analysis: {abs_path} (language=html, lines={len(content.splitlines())})")
    assert "language: html" in out
    assert "scripts:" in out
    assert "src=./app.js" in out
    assert "links:" in out
    assert "href=./style.css" in out
    assert "duplicate_id" in out
    assert "img_missing_alt" in out
    assert "target_blank_missing_noopener" in out
    assert f"script ./app.js -> {tmp_path.joinpath('app.js').absolute()} (exists)" in out


def test_analyze_code_r_outlines_libraries_sources_and_functions(tmp_path: Path) -> None:
    (tmp_path / "utils.R").write_text("x <- 1\n", encoding="utf-8")

    path = tmp_path / "demo.R"
    content = (
        "library(ggplot2)\n"
        "source(\"utils.R\")\n"
        "\n"
        "foo <- function(x) {\n"
        "  y <- x + 1\n"
        "  y\n"
        "}\n"
        "\n"
        "bar <- function(z) z * 2\n"
    )
    path.write_text(content, encoding="utf-8")

    out = analyze_code(file_path=str(path))
    abs_path = str(path.absolute())

    assert out.startswith(f"Code Analysis: {abs_path} (language=r, lines={len(content.splitlines())})")
    assert "language: r" in out
    assert "libraries:" in out
    assert "1: ggplot2" in out
    assert "sources:" in out
    assert f"source utils.R -> {tmp_path.joinpath('utils.R').absolute()} (exists)" in out
    assert "functions:" in out
    assert "foo(x)" in out
    assert "bar(z)" in out

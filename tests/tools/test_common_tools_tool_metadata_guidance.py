from __future__ import annotations


def test_edit_file_and_write_file_metadata_emphasize_surgical_vs_overwrite() -> None:
    from abstractcore.tools.common_tools import edit_file, write_file

    assert hasattr(edit_file, "_tool_definition")
    assert hasattr(write_file, "_tool_definition")

    edit_def = edit_file._tool_definition
    write_def = write_file._tool_definition

    assert "surgical" in (edit_def.description or "").lower()
    assert "small" in (edit_def.when_to_use or "").lower()
    assert "write_file" in (edit_def.when_to_use or "")

    assert "overwrite" in (write_def.description or "").lower()
    assert "edit_file" in (write_def.description or "").lower()
    assert "overwrite" in (write_def.when_to_use or "").lower()


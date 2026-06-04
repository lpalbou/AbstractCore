import sys
import types


def test_abstractcore_serve_dispatches_to_server_runner(monkeypatch) -> None:
    from abstractcore.config import main as core_cli

    called = {}

    def fake_serve(argv):
        called["argv"] = list(argv)
        return 0

    monkeypatch.setattr(core_cli, "_handle_serve_subcommand", fake_serve)

    rc = core_cli.main(["serve", "--host", "127.0.0.1", "--port", "9999"])

    assert rc == 0
    assert called["argv"] == ["--host", "127.0.0.1", "--port", "9999"]


def test_server_runner_accepts_cli_args_and_invokes_uvicorn(monkeypatch) -> None:
    from abstractcore.server import app as server_app

    captured = {}
    uvicorn = types.ModuleType("uvicorn")

    def fake_run(**kwargs):
        captured.update(kwargs)

    uvicorn.run = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    server_app.run_server_with_args(["--host", "127.0.0.1", "--port", "9999", "--reload"])

    assert captured["app"] == "abstractcore.server.app:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9999
    assert captured["reload"] is True


def test_python_module_server_entrypoint_delegates_to_runner(monkeypatch) -> None:
    from abstractcore.server import __main__ as server_main

    called = {}

    def fake_run(argv, *, prog=None):
        called["argv"] = list(argv)
        called["prog"] = prog

    monkeypatch.setattr(server_main, "run_server_with_args", fake_run)

    rc = server_main.main(["--host", "127.0.0.1"])

    assert rc == 0
    assert called == {"argv": ["--host", "127.0.0.1"], "prog": "python -m abstractcore.server"}

"""Opt-in cached-model default/override test. Cool at least 60s between models."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.getenv("ABSTRACTCORE_MTP_DEFAULTS_LIVE_MODEL"), reason="explicit local checkpoint required")
def test_fresh_default_and_runtime_overrides(monkeypatch, tmp_path):
    model = os.environ["ABSTRACTCORE_MTP_DEFAULTS_LIVE_MODEL"]
    assert Path(model).is_dir()
    for key, value in {
        "ABSTRACTCORE_CONFIG_FILE": str(tmp_path / "core.json"),
        "ABSTRACTFRAMEWORK_DATA_REGISTRY": str(tmp_path / "registry.json"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    import abstractcore.config.manager as configuration
    monkeypatch.setattr(configuration, "_config_manager", None)
    manager = configuration.ConfigurationManager(apply_env=False)
    manager._save_config()
    from abstractcore.core.retry import RetryConfig
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
    client = LocalAbstractCoreLLMClient(
        provider="mlx", model=model, core_config_file=manager.config_file,
        llm_kwargs={"mlx_batching": True, "mlx_max_batch_size": 4,
                    "mlx_batch_wait_ms": 100, "enable_tracing": False,
                    "retry_config": RetryConfig(max_attempts=1), "max_output_tokens": 24},
    )
    llm = client._llm
    rows = []

    def generate(control=None, stream=False):
        params = {"thinking": False, "temperature": 0, "max_output_tokens": 24, "stream": stream}
        if control is not None:
            params["speculation"] = control
        result = client.generate(prompt="Write the integers from 1 to 30, one per line.", params=params)
        assert result["content"]
        return {"request": control, "speculation": result["metadata"]["speculation"], "content": result["content"]}

    try:
        assert llm._mtp_active, "fresh depth2 default must prepare the cached head without an explicit constructor request"
        first = generate()
        assert first["speculation"]["used"] is True
        assert first["speculation"]["num_draft_tokens"] == 2
        rows.append(first)
        with ThreadPoolExecutor(max_workers=2) as pool:
            off = pool.submit(generate, False)
            depth = pool.submit(generate, {"num_draft_tokens": 3, "require_acceleration": True}, True)
            off, depth = off.result(timeout=180), depth.result(timeout=180)
        assert off["speculation"]["used"] is False
        assert depth["speculation"]["used"] is True and depth["speculation"]["num_draft_tokens"] == 3
        rows.extend([off, depth])
        manager.update_capability_default("output.text", options={"speculation": {"mode": "native_mtp", "num_draft_tokens": 4}})
        changed = generate()
        assert changed["speculation"]["used"] is True and changed["speculation"]["num_draft_tokens"] == 4
        rows.append(changed)
        manager.update_capability_default("output.text", options={"speculation": False})
        changed_off = generate()
        assert changed_off["speculation"]["used"] is False
        rows.append(changed_off)
        receipt = tmp_path / "mtp-defaults-live.json"
        receipt.write_text(json.dumps({"model": model, "rows": rows}, indent=2))
        print(f"MTP_DEFAULTS_EVIDENCE={receipt}")
    finally:
        llm.unload_model(model)

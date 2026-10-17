from pathlib import Path

from src.config import Settings


def test_settings_load_from_env_file(tmp_path: Path, monkeypatch) -> None:
    for key in [
        "QWEN_BASE_URL",
        "QWEN_API_KEY",
        "QWEN_MODEL",
        "QWEN_MAX_OUTPUT_TOKENS",
        "QWEN_TEMPERATURE",
        "CODE_AGENT_MAX_STEPS",
        "VERIFIER_AGENT_MAX_STEPS",
        "VERIFIER_ACTION_BUDGET",
        "VIEW_ACTION_SPACE",
        "VIEW_ACTION_SPACE_CONFIG",
        "MAX_RENDER_RETRIES",
        "MAX_VISUAL_REVISIONS",
        "RUNS_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "QWEN_BASE_URL=https://example.test/v1\n"
        "QWEN_API_KEY=secret\n"
        "QWEN_MODEL=openai/qwen-test\n"
        "QWEN_MAX_OUTPUT_TOKENS=4096\n"
        "QWEN_TEMPERATURE=0.2\n"
        "CODE_AGENT_MAX_STEPS=5\n"
        "VERIFIER_AGENT_MAX_STEPS=7\n"
        "VERIFIER_ACTION_BUDGET=4\n"
        "VIEW_ACTION_SPACE=pose_grid_v2\n"
        "VIEW_ACTION_SPACE_CONFIG=configs/view_pose_grid_v2.json\n"
        "MAX_RENDER_RETRIES=2\n"
        "MAX_VISUAL_REVISIONS=3\n"
        "RUNS_DIR=my-runs\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(env_file)
    settings.validate_model()
    assert settings.qwen_base_url == "https://example.test/v1"
    assert settings.qwen_api_key == "secret"
    assert settings.qwen_model == "openai/qwen-test"
    assert settings.qwen_max_output_tokens == 4096
    assert settings.qwen_temperature == 0.2
    assert settings.code_agent_max_steps == 5
    assert settings.verifier_agent_max_steps == 7
    assert settings.verifier_action_budget == 4
    assert settings.view_action_space == "pose_grid_v2"
    assert settings.view_action_space_config == Path("configs/view_pose_grid_v2.json")
    assert settings.max_render_retries == 2
    assert settings.max_visual_revisions == 3
    assert settings.runs_dir == Path("my-runs")

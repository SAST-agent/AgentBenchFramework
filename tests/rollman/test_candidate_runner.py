import importlib.util
from pathlib import Path


TEMPLATE = (
    Path(__file__).parents[2]
    / "src/agentbench_frame/games/rollman/assets/candidate-template/ai.py"
)


def test_candidate_template_is_from_scratch_and_protocol_legal():
    spec = importlib.util.spec_from_file_location("candidate_template", TEMPLATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.ai_func(object())

    assert result == {"action": 0, "memory_id": "from-scratch-v0"}
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "opponent" not in source.lower()
    assert "grid" not in source.lower()


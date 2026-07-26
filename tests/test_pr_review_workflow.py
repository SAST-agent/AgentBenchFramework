from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "pr-review.yml"


def test_pr_review_workflow_listens_to_all_pull_request_base_branches():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "    branches:" not in text
    assert "types: [opened, synchronize, reopened, ready_for_review]" in text


def test_pr_review_workflow_keeps_required_job_names():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "  framework-tests:" in text
    assert "  ai-pr-review:" in text
    assert "name: framework-tests" in text
    assert "name: ai-pr-review" in text

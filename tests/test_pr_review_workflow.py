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


def test_pr_review_workflow_routes_forks_without_secrets():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_target" not in text
    assert "name: Run blocking AI review" in text
    assert "if: github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "name: Reject fork PR without credentialed review" in text
    assert "if: github.event.pull_request.head.repo.full_name != github.repository" in text
    assert "Fork PRs cannot receive the credentialed blocking review" in text


def test_pr_review_workflow_rejects_oversized_diffs_before_review():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Reject oversized diff" in text
    assert "diff_bytes" in text
    assert "exit 1" in text
    assert "raw_diff[:350_000]" not in text


def test_pr_review_workflow_rejects_unsupported_changed_file_counts():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Reject unsupported file count" in text
    assert "changed_files" in text
    assert "max_changed_files=300" in text

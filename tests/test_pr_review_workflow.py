from pathlib import Path


WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
PR_TEST_WORKFLOW = WORKFLOWS / "pr-review.yml"
TRUSTED_REVIEW_WORKFLOW = WORKFLOWS / "pr-review-trusted.yml"


def test_pr_review_workflow_listens_to_all_pull_request_base_branches():
    text = PR_TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "    branches:" not in text
    assert "types: [opened, synchronize, reopened, ready_for_review]" in text
    assert "ai-pr-review" not in text
    assert "PR_REVIEW_API_KEY" not in text


def test_pr_review_workflow_keeps_required_job_names():
    tests = PR_TEST_WORKFLOW.read_text(encoding="utf-8")
    review = TRUSTED_REVIEW_WORKFLOW.read_text(encoding="utf-8")

    assert "  framework-tests:" in tests
    assert "name: framework-tests" in tests
    assert "  ai-pr-review:" in review
    assert "name: ai-pr-review" in review


def test_trusted_review_workflow_uses_trusted_pull_request_target_context():
    text = TRUSTED_REVIEW_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_target:" in text
    assert "pull_request:" not in text
    assert "name: Run blocking AI review" in text
    assert "PR_REVIEW_API_KEY: ${{ secrets.PR_REVIEW_API_KEY }}" in text
    assert "PR_REVIEW_ENDPOINT: ${{ secrets.PR_REVIEW_ENDPOINT }}" in text
    assert "ref: ${{ github.event.repository.default_branch }}" in text
    assert "github.event.pull_request.head.sha" not in text
    assert "Fork PRs cannot receive the credentialed blocking review" not in text


def test_pr_review_workflow_rejects_oversized_diffs_before_review():
    text = TRUSTED_REVIEW_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Reject oversized diff" in text
    assert "diff_bytes" in text
    assert "exit 1" in text
    assert "raw_diff[:350_000]" not in text


def test_pr_review_workflow_rejects_unsupported_changed_file_counts():
    text = TRUSTED_REVIEW_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Reject unsupported file count" in text
    assert "changed_files" in text
    assert "max_changed_files=300" in text

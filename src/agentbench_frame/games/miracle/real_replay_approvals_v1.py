"""Independent production approval registry for real Judge replay manifests.

This data boundary is deliberately outside ``real_replay_adapter_v1`` so
approving a canonical manifest cannot change the source identity that the
same manifest binds. Production remains fail closed until a project-owner
change explicitly adds an independently reviewed digest.
"""

APPROVED_REAL_JUDGE_REPLAY_MANIFESTS: frozenset[str] = frozenset(
    {
        "7fe0f543abaee3e2e63b40fe2933dd541df6fc078fc635116262e55333e5ff8c",
    }
)

__all__ = ["APPROVED_REAL_JUDGE_REPLAY_MANIFESTS"]

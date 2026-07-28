"""24_miracle from-scratch strategy bootstrap template.

This file intentionally contains only the protocol adapter and a
minimal legal fallback.  It contains no copied bot tactics, opponent knowledge, or learned
rules.  A research run records this file's version and SHA-256 before creating
any mutable run resources.
"""

TEMPLATE_VERSION = "24m-minimal-bootstrap-v1"


def choose_action(_observation, action_support):
    """Choose a legal atomic command without encoding tactical knowledge."""
    if not action_support.actions:
        raise ValueError("complete legal action support must not be empty")
    for candidate in action_support.actions:
        if candidate.action.get("operation_type") == "endround":
            return candidate.action
    return action_support.actions[0].action


def distribution_for_measurement(observation, action_support):
    """Expose the deterministic fallback as a read-only full distribution."""
    selected = choose_action(observation, action_support)
    return {
        candidate.action_id: float(candidate.action == selected)
        for candidate in action_support.actions
    }

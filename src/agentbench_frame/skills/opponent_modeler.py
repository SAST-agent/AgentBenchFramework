"""
Opponent Modeler Skill

Builds and maintains opponent behavior models during gameplay.
Updates a probabilistic model of what the opponent is likely to do next
based on observed behavior and historical replay data.

Integrates with ReplayReaderSkill for historical data and updates
the model in real-time as the game progresses.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from agentbench_frame.skills.base import Skill, SkillMeta


class OpponentModelerSkill(Skill):
    """
    Models opponent behavior during gameplay.

    Tracks:
    - Action probabilities given game state
    - Response patterns (how they react to our moves)
    - Resource allocation preferences
    - Risk tolerance

    The model is updated every turn and made available to other skills
    via the shared context.
    """

    def __init__(self):
        meta = SkillMeta(
            name="opponent_modeler",
            version="1.0.0",
            description="Builds and maintains a probabilistic model of opponent behavior",
            tags=["analysis", "opponent", "prediction"],
            game="generals",
            dependencies=["replay_reader"],
        )
        super().__init__(meta)
        self._model: Dict[str, Any] = self._init_model()
        self._observation_history: List[Dict] = []

    def can_activate(self, observation: Dict[str, Any],
                     context: Dict[str, Any]) -> bool:
        """Always active — runs every turn to update the opponent model."""
        return True

    def execute(self, observation: Dict[str, Any],
                context: Dict[str, Any]) -> Optional[Any]:
        """
        Update the opponent model based on latest observation.
        Stores the model in context for other skills.
        """
        self._update_model(observation, context)

        # Store model in context
        context["opponent_model"] = self._model
        context["opponent_prediction"] = self._predict_next_action(observation)

        # Also store readable summary
        context["opponent_summary"] = self._summarize()

        return None  # Pure analysis skill

    # ---- Model ----

    def _init_model(self) -> Dict[str, Any]:
        return {
            "observed_rounds": 0,
            "action_counts": Counter(),
            "state_action_pairs": defaultdict(Counter),
            "response_patterns": defaultdict(Counter),
            "territory_changes": [],
            "resource_spending_rate": 0.0,
            "expansion_rate": 0.0,
            "aggression_events": 0,
            "defensive_events": 0,
            "tech_priority": Counter(),
        }

    def _update_model(self, observation: Dict[str, Any],
                      context: Dict[str, Any]):
        """Update the opponent model with new observation."""
        self._model["observed_rounds"] += 1
        self._observation_history.append(observation)

        state = observation.get("state", {})
        board = state.get("board", [])
        player = observation.get("player_id", 0)
        opponent = 1 - player

        # Count opponent territory
        opp_cells = 0
        opp_army = 0
        for row in board:
            for cell in row:
                if cell.get("player") == opponent:
                    opp_cells += 1
                    opp_army += cell.get("army", 0)

        # Track territory changes
        prev_opp_cells = context.get("_prev_opp_cells", 0)
        self._model["territory_changes"].append(opp_cells - prev_opp_cells)
        context["_prev_opp_cells"] = opp_cells

        # Track expansion rate
        if len(self._model["territory_changes"]) > 0:
            recent = self._model["territory_changes"][-5:]
            self._model["expansion_rate"] = sum(recent) / max(1, len(recent))

        # Track aggression (territory gain = aggression)
        delta = opp_cells - prev_opp_cells
        if delta > 2:
            self._model["aggression_events"] += 1
        elif delta < 0:
            self._model["defensive_events"] += 1

        # Clean up old history (keep last 100)
        if len(self._model["territory_changes"]) > 100:
            self._model["territory_changes"] = self._model["territory_changes"][-100:]

        # Track opponent actions if available in observation
        actions = self._extract_opponent_actions(observation, opponent)
        for action in actions:
            self._model["action_counts"][action] += 1

            # State-action pair
            state_key = f"cells_{opp_cells}_army_{opp_army // 10}"
            self._model["state_action_pairs"][state_key][action] += 1

    def _extract_opponent_actions(self, observation: Dict[str, Any],
                                  opponent: int) -> List[str]:
        """Extract opponent's recent actions from observation."""
        actions = []
        state = observation.get("state", {})

        # Look for opponent action info in various formats
        enemy_actions = state.get("enemy_actions", [])
        if enemy_actions:
            for action in enemy_actions:
                if isinstance(action, list) and action:
                    cmd_names = {
                        1: "move_army", 2: "move_general", 3: "upgrade",
                        4: "skill", 5: "tech", 6: "superweapon",
                        7: "call_sub", 8: "end_turn", 9: "surrender",
                    }
                    cmd = action[0]
                    actions.append(cmd_names.get(cmd, f"cmd_{cmd}"))
                elif isinstance(action, dict):
                    actions.append(action.get("type", "unknown"))

        return actions

    def _predict_next_action(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Predict the opponent's most likely next action."""
        state = observation.get("state", {})
        board = state.get("board", [])

        # Build state key
        opp_cells = sum(
            1 for row in board for cell in row
            if cell.get("player") == 1 - observation.get("player_id", 0)
        )

        state_key = f"cells_{opp_cells}"
        action_probs = self._model["state_action_pairs"].get(state_key, {})

        if not action_probs:
            # Fall back to overall action distribution
            total = sum(self._model["action_counts"].values()) or 1
            return {
                action: count / total
                for action, count in self._model["action_counts"].most_common(5)
            }

        total = sum(action_probs.values()) or 1
        return {
            action: count / total
            for action, count in action_probs.most_common(5)
        }

    def _summarize(self) -> Dict[str, Any]:
        """Generate a human-readable summary of the opponent model."""
        m = self._model
        rounds = max(1, m["observed_rounds"])

        total_events = m["aggression_events"] + m["defensive_events"]
        aggression_ratio = (
            m["aggression_events"] / max(1, total_events)
            if total_events > 0 else 0.5
        )

        return {
            "observed_rounds": rounds,
            "aggression_ratio": round(aggression_ratio, 2),
            "style": "aggressive" if aggression_ratio > 0.6
                     else "defensive" if aggression_ratio < 0.4
                     else "balanced",
            "expansion_trend": "growing" if m["expansion_rate"] > 0.5
                               else "shrinking" if m["expansion_rate"] < -0.5
                               else "stable",
            "top_actions": [
                (action, count)
                for action, count in m["action_counts"].most_common(3)
            ],
        }

    def reset(self):
        super().reset()
        self._model = self._init_model()
        self._observation_history = []

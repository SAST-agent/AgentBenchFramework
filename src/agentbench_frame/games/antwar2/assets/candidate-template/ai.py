"""Interpretable bootstrap policy scaffold for AntWar2."""

from __future__ import annotations

from common import BaseAgent
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle


class AI(BaseAgent):
    """Replace HOLD with a rules-derived visible-state policy during bootstrap."""

    def choose_operations(self, state: BackendState, player: int, bundles=None):
        del state, player, bundles
        return []

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        candidates = bundles or self.list_bundles(state, player)
        return next(
            (bundle for bundle in candidates if not bundle.operations),
            candidates[0],
        )

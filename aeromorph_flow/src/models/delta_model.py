from __future__ import annotations

from aeromorph_flow.src.models.baseline_mlp import MLP


class DeltaMLP(MLP):
    """Predict delta Cp plus scalar delta Cl/Cd from morph transition features."""


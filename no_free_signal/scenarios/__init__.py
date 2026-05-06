"""Scenario library — declarative 'people-in-a-box' setups."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCENARIOS_PATH = Path(__file__).parent / "SCENARIOS.json"


def load_scenarios() -> list[dict[str, Any]]:
    with _SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def get_scenario(scenario_id: str) -> dict[str, Any] | None:
    for s in load_scenarios():
        if s.get("id") == scenario_id:
            return s
    return None

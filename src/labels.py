"""Canonical behavior labels and alias resolution."""

from __future__ import annotations

import json
from pathlib import Path

BEHAVIORS = (
    "traveling",
    "milling",
    "shoaling",
    "expansion",
    "compaction",
)

TRANSITIONS = tuple(f"{a}_to_{b}" for a in BEHAVIORS for b in BEHAVIORS if a != b)

ALL_LABELS = BEHAVIORS + TRANSITIONS

_LEGACY_BEHAVIOR = {
    "traveling_polarized": "traveling",
    "tpol": "traveling",
    "polarized": "traveling",
    "travelling": "traveling",
    "t": "traveling",
    "expansion_burst": "expansion",
    "burst": "expansion",
    "spread": "expansion",
    "e+": "expansion",
    "e-": "compaction",
    "contraction": "compaction",
    "swarming": "shoaling",
    "m": "milling",
    "s": "shoaling",
}

_ROOT = Path(__file__).resolve().parents[1]
_ALIAS_PATH = _ROOT / "annotations" / "_label_aliases.json"


def is_transition(label: str) -> bool:
    return "_to_" in label


def is_behavior(label: str) -> bool:
    return label in BEHAVIORS


def ordered_labels(present) -> list[str]:
    """Canonical class order, then leftover labels."""
    present_set = set(present)
    ordered = [lab for lab in ALL_LABELS if lab in present_set]
    leftover = sorted(present_set - set(ordered))
    return ordered + leftover


def load_aliases(path: Path | None = None) -> dict[str, str]:
    p = path or _ALIAS_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k.lower(): v for k, v in raw.items()}


def _modernize(label: str) -> str:
    key = label.strip().lower()
    if key in BEHAVIORS or key in TRANSITIONS:
        return key
    if key in _LEGACY_BEHAVIOR:
        return _LEGACY_BEHAVIOR[key]
    if "_to_" in key:
        a, b = key.split("_to_", 1)
        return f"{_modernize(a)}_to_{_modernize(b)}"
    return key


def _resolve_transition(label: str, aliases: dict[str, str]) -> str | None:
    for sep in (" to ", "_to_"):
        if sep in label.lower():
            parts = label.lower().split(sep, 1)
            if len(parts) == 2:
                a_raw, b_raw = parts[0].strip(), parts[1].strip()
                a = _modernize(aliases.get(a_raw, _LEGACY_BEHAVIOR.get(a_raw, a_raw)))
                b = _modernize(aliases.get(b_raw, _LEGACY_BEHAVIOR.get(b_raw, b_raw)))
                trans = f"{a}_to_{b}"
                if trans in TRANSITIONS:
                    return trans
    return None


def canonicalize(label: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or load_aliases()
    key = label.strip().lower()
    if key in aliases:
        resolved = _modernize(aliases[key])
    elif key in ALL_LABELS:
        return key
    else:
        trans = _resolve_transition(key, aliases)
        if trans:
            return trans
        resolved = _modernize(key)
    if resolved not in ALL_LABELS:
        raise ValueError(f"Unknown label: {label!r}")
    return resolved

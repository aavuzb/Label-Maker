"""Deterministic, visually distinct colors auto-assigned to new classes."""

from __future__ import annotations

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#e6beff", "#ffe119",
]


def palette_color(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


def default_shortcut(index: int) -> str:
    return str(index + 1) if index < 9 else ""

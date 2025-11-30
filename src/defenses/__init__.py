"""Defenses module."""

from .defense_methods import (
    BaseDefense,
    KrumDefense,
    FlameDefense,
    MultiMetricsDefense,
    FedDMCDefense,
    FLPruningDefense,
    get_defense
)

__all__ = [
    'BaseDefense',
    'KrumDefense',
    'FlameDefense',
    'MultiMetricsDefense',
    'FedDMCDefense',
    'FLPruningDefense',
    'get_defense'
]

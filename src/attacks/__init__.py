"""Attacks module."""

from .backdoor_attacks import (
    AttackTaxonomy,
    DataPoisoningAttack,
    WeightPoisoningAttack,
    assign_attacks_to_clients
)

__all__ = [
    'AttackTaxonomy',
    'DataPoisoningAttack',
    'WeightPoisoningAttack',
    'assign_attacks_to_clients'
]

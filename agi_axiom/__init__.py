"""
Axiomatic Intelligence: Artificial General Intelligence via Axiomatic Reasoning
"""

from .core import Axiom, AxiomType, AxiomaticBase
from .reasoning import RecursiveReasoningEngine
from .safety import SafetyVerifier, ComptonClassSafety
from .interface import NaturalLanguageInterface, TruthLedger
from .system import AGI_Axiom

__version__ = "1.0.0"
__author__ = "Universal Standard Axiom Corp"
__license__ = "Russell Standard Open Protocol v1.0"
__copyright__ = "Copyright 2024, Universal Standard Axiom Corp"
__patent__ = "US-2026-CONSCIOUS-INTERNET"

__all__ = [
    'AGI_Axiom',
    'Axiom',
    'AxiomType', 
    'AxiomaticBase',
    'RecursiveReasoningEngine',
    'SafetyVerifier',
    'ComptonClassSafety',
    'NaturalLanguageInterface',
    'TruthLedger',
]

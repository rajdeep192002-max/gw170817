"""
Validation sub-package for GW170817 observational comparison and reporting.
"""
from gw170817.validation.observational_data import GW170817ReferenceData, ValueReference, ReferenceCategory
from gw170817.validation.comparison import ComparisonResult, compare_value
from gw170817.validation.report import ValidationReport

__all__ = [
    "GW170817ReferenceData",
    "ValueReference",
    "ReferenceCategory",
    "ComparisonResult",
    "compare_value",
    "ValidationReport"
]

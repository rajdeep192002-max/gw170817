"""
Validation Comparison Utilities for GW170817 Simulation.

Provides transparent scalar and range comparison functions with explicit tolerance handling.
"""
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import numpy as np
from gw170817.validation.observational_data import ValueReference, ReferenceCategory


@dataclass
class ComparisonResult:
    """Dataclass holding detailed comparison metrics between model value and reference."""
    name: str
    reference_nominal: float
    reference_str: str
    model_value: float
    model_str: str
    difference: float
    relative_error: float
    passed: bool
    status: str          # "PASS" or "FAIL"
    category: str        # "OBSERVATIONAL_CONSTRAINT" or "PHENOMENOLOGICAL_REFERENCE"
    notes: str


def compare_value(
    ref: ValueReference,
    model_val: float,
    tolerance_rel: float = 0.10,
    unit_scale: float = 1.0,
    unit_label: str = None
) -> ComparisonResult:
    """
    Compare a model scalar value against a ValueReference.
    Checks whether model_val falls within [ref.min_val, ref.max_val] OR within relative tolerance around nominal.
    """
    val = float(model_val)
    if not np.isfinite(val):
        return ComparisonResult(
            name=ref.name,
            reference_nominal=ref.nominal / unit_scale,
            reference_str=f"{ref.nominal / unit_scale:.3f} {unit_label or ref.unit}",
            model_value=val,
            model_str="non-finite (NaN/Inf)",
            difference=float('nan'),
            relative_error=float('nan'),
            passed=False,
            status="FAIL",
            category=ref.category.value,
            notes="Model value is non-finite (NaN or Inf)"
        )

    scale = float(unit_scale)
    label = unit_label or ref.unit

    val_scaled = val / scale
    nom_scaled = ref.nominal / scale
    min_scaled = ref.min_val / scale
    max_scaled = ref.max_val / scale

    diff = val - ref.nominal
    diff_scaled = diff / scale
    rel_err = abs(diff) / abs(ref.nominal) if ref.nominal != 0 else 0.0

    # Pass if value is within reference range [min_val, max_val] OR relative error <= tolerance_rel
    min_bound = min(ref.min_val, ref.max_val)
    max_bound = max(ref.min_val, ref.max_val)
    in_range = (min_bound <= val <= max_bound)
    within_tol = rel_err <= tolerance_rel
    passed = bool(in_range or within_tol)

    status_str = "PASS" if passed else "FAIL"
    ref_str = f"[{min_scaled:.3f}, {max_scaled:.3f}] {label} (nom: {nom_scaled:.3f})"
    model_str = f"{val_scaled:.3f} {label}"

    notes = f"Diff: {diff_scaled:+.3f} {label}, Rel Error: {rel_err * 100:.2f}%"

    return ComparisonResult(
        name=ref.name,
        reference_nominal=nom_scaled,
        reference_str=ref_str,
        model_value=val_scaled,
        model_str=model_str,
        difference=diff_scaled,
        relative_error=rel_err,
        passed=passed,
        status=status_str,
        category=ref.category.value,
        notes=notes
    )

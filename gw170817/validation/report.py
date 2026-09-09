"""
Observational Validation Report for GW170817 Simulation.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
This module evaluates the integrated simulation engine against GW170817 observational constraints
and phenomenological reference values.

DISCLAIMER:
Matching a phenomenological parameter does not mean the simulation has reproduced
the full astrophysical event or solved full general relativity / GRHD.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from gw170817.constants import M_sun, Mpc, day
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.validation.observational_data import GW170817ReferenceData
from gw170817.validation.comparison import ComparisonResult, compare_value


class ValidationReport:
    """
    Comprehensive Observational Validation Report for GW170817.
    """

    DISCLAIMER = (
        "IMPORTANT: This is a physics-informed reduced-order simulation. "
        "Matching observational constraints or phenomenological parameters demonstrates "
        "model consistency, NOT full numerical-relativity or first-principles GRHD resolution."
    )

    def __init__(self, engine: Optional[GW170817Simulation] = None, config: Optional[SimConfig] = None):
        if config is None:
            if engine is not None:
                config = engine.config
            else:
                config = SimConfig()
        self.config = config

        if engine is None:
            engine = GW170817Simulation(config=self.config)
        self.engine = engine

        self.results: List[ComparisonResult] = []
        self._evaluate()

    def _evaluate(self):
        """Perform all validation comparisons against GW170817 Reference Data."""
        self.results.clear()
        ref = GW170817ReferenceData

        # 1. Total Binary Mass M_tot (Model: config.M_total, Ref: 2.74 M_sun)
        self.results.append(
            compare_value(
                ref.TOTAL_MASS,
                self.config.M_total,
                tolerance_rel=0.05,
                unit_scale=M_sun,
                unit_label="M_sun"
            )
        )

        # 2. Chirp Mass M_c (Model: config.chirp_mass, Ref: 1.188 M_sun)
        self.results.append(
            compare_value(
                ref.CHIRP_MASS,
                self.config.chirp_mass,
                tolerance_rel=0.05,
                unit_scale=M_sun,
                unit_label="M_sun"
            )
        )

        # 3. Luminosity Distance D_L (Model: config.distance, Ref: 40.0 Mpc)
        self.results.append(
            compare_value(
                ref.DISTANCE,
                self.config.distance,
                tolerance_rel=0.15,
                unit_scale=Mpc,
                unit_label="Mpc"
            )
        )

        # 4. GW-GRB Prompt Delay (Model: engine.jet.jet_delay, Ref: 1.74 s)
        self.results.append(
            compare_value(
                ref.GRB_DELAY,
                self.engine.jet.jet_delay,
                tolerance_rel=0.15,
                unit_scale=1.0,
                unit_label="s"
            )
        )

        # 5. Jet Viewing Angle theta_obs (Model: rad2deg(engine.jet.viewing_angle), Ref: 22 deg)
        viewing_angle_deg = float(np.rad2deg(self.engine.jet.viewing_angle))
        self.results.append(
            compare_value(
                ref.VIEWING_ANGLE,
                viewing_angle_deg,
                tolerance_rel=0.25,
                unit_scale=1.0,
                unit_label="deg"
            )
        )

        # 6. Afterglow Peak Time (Model: engine.afterglow.t_peak_days, Ref: 155 days)
        self.results.append(
            compare_value(
                ref.AFTERGLOW_PEAK_TIME,
                self.engine.afterglow.t_peak_days,
                tolerance_rel=0.10,
                unit_scale=1.0,
                unit_label="days"
            )
        )

        # 7. Afterglow Rise Slope alpha_rise (Model: engine.afterglow.rise_index, Ref: 0.85)
        self.results.append(
            compare_value(
                ref.AFTERGLOW_RISE_SLOPE,
                self.engine.afterglow.rise_index,
                tolerance_rel=0.20,
                unit_scale=1.0,
                unit_label="dimensionless"
            )
        )

        # 8. Afterglow Decay Slope alpha_decay (Model: engine.afterglow.decay_index, Ref: -2.1)
        self.results.append(
            compare_value(
                ref.AFTERGLOW_DECAY_SLOPE,
                self.engine.afterglow.decay_index,
                tolerance_rel=0.20,
                unit_scale=1.0,
                unit_label="dimensionless"
            )
        )

        # 9. Total Kilonova Ejecta Mass (Model: M_EJ_BLUE + M_EJ_RED or cfg default, Ref: 0.05 M_sun)
        m_ej_total = getattr(self.config, "M_EJ_BLUE", 0.02 * M_sun) + getattr(self.config, "M_EJ_RED", 0.04 * M_sun)
        self.results.append(
            compare_value(
                ref.EJECTA_MASS_TOTAL,
                m_ej_total,
                tolerance_rel=0.50,
                unit_scale=M_sun,
                unit_label="M_sun"
            )
        )

    @property
    def all_passed(self) -> bool:
        """Return True if all validation comparisons passed."""
        return all(res.passed for res in self.results)

    def summary_dict(self) -> Dict[str, Any]:
        """Expose report metrics as a dictionary."""
        return {
            "event_name": GW170817ReferenceData.EVENT_NAME,
            "all_passed": self.all_passed,
            "passed_count": sum(1 for r in self.results if r.passed),
            "total_count": len(self.results),
            "results": [
                {
                    "name": r.name,
                    "reference": r.reference_str,
                    "model": r.model_str,
                    "difference": r.difference,
                    "relative_error": r.relative_error,
                    "status": r.status,
                    "category": r.category,
                    "notes": r.notes
                }
                for r in self.results
            ]
        }

    def print_report(self):
        """Print formatted validation report to stdout."""
        print("=== GW170817 OBSERVATIONAL VALIDATION REPORT ===")
        print(f"Event: {GW170817ReferenceData.EVENT_NAME} ({GW170817ReferenceData.EVENT_TYPE})")
        print(f"Reference UTC: {GW170817ReferenceData.MERGER_UTC}\n")
        print(f"{'Parameter':<24} | {'Reference':<26} | {'Model':<16} | {'Status':<6} | Notes")
        print("-" * 100)

        for r in self.results:
            print(f"{r.name:<24} | {r.reference_str:<26} | {r.model_str:<16} | {r.status:<6} | {r.notes}")

        print("-" * 100)
        overall = "ALL VALIDATION CHECKS PASSED" if self.all_passed else "SOME CHECKS FAILED"
        print(f"\nOverall Status: {overall} ({sum(1 for r in self.results if r.passed)}/{len(self.results)} passed)")
        print(f"\nDisclaimer: {self.DISCLAIMER}")

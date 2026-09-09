"""
Observational and Reference Data for GW170817 Multi-Messenger Analysis.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
This module encodes observational constraints derived from LIGO/Virgo and
EM follow-up (Fermi, INTEGRAL, Hubble, Chandra, VLA) alongside phenomenological
reference parameters.

OBSERVATIONAL CONSTRAINTS are empirical measurements from multi-messenger data.
PHENOMENOLOGICAL REFERENCES are parametrized model benchmarks from literature fits.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
from gw170817.constants import M_sun, Mpc, day


class ReferenceCategory(Enum):
    OBSERVATIONAL_CONSTRAINT = "OBSERVATIONAL_CONSTRAINT"
    PHENOMENOLOGICAL_REFERENCE = "PHENOMENOLOGICAL_REFERENCE"


@dataclass(frozen=True)
class ValueReference:
    """Structure for a physical parameter with reference value, range, units, and category."""
    name: str
    nominal: float
    min_val: float
    max_val: float
    unit: str
    category: ReferenceCategory
    citation: str
    description: str


class GW170817ReferenceData:
    """
    Authoritative reference dataset for GW170817 multi-messenger event.
    """
    EVENT_NAME = "GW170817"
    EVENT_TYPE = "Binary Neutron Star Merger"
    MERGER_UTC = "2017-08-17 12:41:04 UTC"

    # 1. Total Mass M_tot [kg] (Nominal 2.74 M_sun, Range 2.70 - 2.80 M_sun)
    TOTAL_MASS = ValueReference(
        name="Total Binary Mass",
        nominal=2.74 * M_sun,
        min_val=2.70 * M_sun,
        max_val=2.80 * M_sun,
        unit="kg",
        category=ReferenceCategory.OBSERVATIONAL_CONSTRAINT,
        citation="Abbott et al. (2017) PRL 119, 161101",
        description="Low-spin total mass constraint from GW170817 strain waveform."
    )

    # 2. Chirp Mass M_c [kg] (Nominal 1.188 M_sun, Range 1.184 - 1.192 M_sun)
    CHIRP_MASS = ValueReference(
        name="Chirp Mass",
        nominal=1.188 * M_sun,
        min_val=1.184 * M_sun,
        max_val=1.192 * M_sun,
        unit="kg",
        category=ReferenceCategory.OBSERVATIONAL_CONSTRAINT,
        citation="Abbott et al. (2017) PRL 119, 161101",
        description="Detector-frame / source-frame chirp mass constraint."
    )

    # 3. Luminosity Distance D_L [m] (Nominal 40.0 Mpc, Range 35.0 - 45.0 Mpc)
    DISTANCE = ValueReference(
        name="Luminosity Distance",
        nominal=40.0 * Mpc,
        min_val=35.0 * Mpc,
        max_val=45.0 * Mpc,
        unit="m",
        category=ReferenceCategory.OBSERVATIONAL_CONSTRAINT,
        citation="Abbott et al. (2017) Nature 551, 85",
        description="Standard-siren distance measurement to NGC 4993."
    )

    # 4. GW-GRB Prompt Delay [s] (Nominal 1.74 s, Range 1.5 - 2.0 s)
    GRB_DELAY = ValueReference(
        name="GW-GRB Prompt Delay",
        nominal=1.74,
        min_val=1.50,
        max_val=2.00,
        unit="s",
        category=ReferenceCategory.OBSERVATIONAL_CONSTRAINT,
        citation="Abbott et al. (2017) ApJ 848, L13",
        description="Observed delay between GW170817 merger and GRB 170817A prompt trigger."
    )

    # 5. Jet Viewing Angle [deg] (Nominal 22.0 deg, Range 15.0 - 28.0 deg)
    VIEWING_ANGLE = ValueReference(
        name="Viewing Angle",
        nominal=22.0,
        min_val=15.0,
        max_val=28.0,
        unit="deg",
        category=ReferenceCategory.PHENOMENOLOGICAL_REFERENCE,
        citation="Mooley et al. (2018) Nature 561, 355",
        description="Observer angle off the relativistic jet axis."
    )

    # 6. Afterglow Peak Time [days] (Nominal 155.0 d, Range 150.0 - 160.0 d)
    AFTERGLOW_PEAK_TIME = ValueReference(
        name="Afterglow Peak Time",
        nominal=155.0,
        min_val=150.0,
        max_val=160.0,
        unit="days",
        category=ReferenceCategory.OBSERVATIONAL_CONSTRAINT,
        citation="Mooley et al. (2018) Nature 561, 355",
        description="Time of peak broadband radio/X-ray afterglow emission."
    )

    # 7. Afterglow Temporal Rise Slope alpha_rise (Nominal 0.85, Range 0.70 - 1.00)
    AFTERGLOW_RISE_SLOPE = ValueReference(
        name="Afterglow Rise Slope",
        nominal=0.85,
        min_val=0.70,
        max_val=1.00,
        unit="dimensionless",
        category=ReferenceCategory.PHENOMENOLOGICAL_REFERENCE,
        citation="Mooley et al. (2018); Margutti et al. (2018)",
        description="Power-law rise index F_nu ~ t^alpha before peak."
    )

    # 8. Afterglow Temporal Decay Slope alpha_decay (Nominal -2.1, Range -2.4 - -1.9)
    AFTERGLOW_DECAY_SLOPE = ValueReference(
        name="Afterglow Decay Slope",
        nominal=-2.1,
        min_val=-2.4,
        max_val=-1.9,
        unit="dimensionless",
        category=ReferenceCategory.PHENOMENOLOGICAL_REFERENCE,
        citation="Mooley et al. (2018); Lamb et al. (2019)",
        description="Power-law decay index F_nu ~ t^alpha post peak."
    )

    # 9. Total Kilonova Ejecta Mass [kg] (Nominal 0.05 M_sun, Range 0.02 - 0.08 M_sun)
    EJECTA_MASS_TOTAL = ValueReference(
        name="Total Ejecta Mass",
        nominal=0.05 * M_sun,
        min_val=0.02 * M_sun,
        max_val=0.08 * M_sun,
        unit="kg",
        category=ReferenceCategory.PHENOMENOLOGICAL_REFERENCE,
        citation="Metzger (2019) LRR 23, 1; Arcavi (2018)",
        description="Combined blue and red kilonova ejecta mass estimate."
    )

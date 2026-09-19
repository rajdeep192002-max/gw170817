
# GW170817 Multimessenger Simulation

A real-time, reduced-order visualization of the GW170817 binary neutron-star merger and its multimessenger aftermath.

> **Important:** This is a scientifically motivated visualization and reduced-order simulation, not a full general-relativistic radiation-hydrodynamic or GRMHD simulation.

## Overview

This project presents a binary neutron-star merger from inspiral through post-merger evolution, with dedicated visualization modes for gravitational waves, magnetic fields, neutrino transport, and multimessenger physics.

### Main stages

```text
INSPIRAL -> LATE INSPIRAL -> MERGER -> POST-MERGER
                                      |
                                      v
                         COMPACT REMNANT / BH MODEL
````

The presentation timeline is compressed for visualization. One authoritative event time is maintained across visualization modes and continues beyond the cinematic demonstration interval.

## Visualization Modes

### 1. CORE

The main merger visualization, including:

* Binary neutron-star inspiral
* Dynamical neutron-star surfaces
* Reduced-order relativistic Doppler/beaming
* Merger flash
* Gravitational-wave wavefronts
* Dynamical ejecta
* Compact post-merger remnant
* Model black-hole formation
* Accretion disk
* Jet/outflow
* Reduced-order gravitational lensing
* Background star field

This is the primary cinematic view.

### 2. GW

Dedicated gravitational-wave visualization using a reduced-order quadrupolar model.

Includes:

* Time-dependent GW strain
* Quadrupolar waveform
* Outgoing wavefront shells
* Merger-associated strain maximum
* Frequency evolution during inspiral and merger

GW emission is represented throughout the late inspiral and reaches its strongest amplitude around merger.

The visible wavefronts are a visualization of the evolving gravitational-wave signal, not a literal reconstruction of the complete spacetime geometry.

### 3. MAGNETIC FIELD

Reduced-order visualization of magnetic-field winding during merger.

Includes:

* Differential rotation
* Magnetic-field winding
* Poloidal field structure
* Toroidal field amplification
* Compact field-line structures
* Coupling to the rotating remnant

This is **not** a full general-relativistic magnetohydrodynamic (GRMHD) calculation.

### 4. NEUTRINO

Reduced-order neutrino emission and transport visualization.

The model uses three neutrino groups:

```text
nu_e
anti-nu_e
nu_x
```

Displayed quantities include:

* Neutrino luminosity
* Mean neutrino energy
* Species-dependent luminosity fractions
* Outward transport/flux tracers
* Neutrino-driven wind mass-loss
* Compact-remnant emission

The tracers represent reduced-order transport/flux visualization, not individual neutrino particle trajectories.

Real neutrino transport involves emission, absorption, scattering, energy exchange, angular dependence, and matter coupling. These processes are not solved in full here.

#### GW170817 observational caveat

GW170817 did **not** produce a confirmed neutrino detection.

This mode therefore represents a physically motivated model of merger-associated neutrino emission rather than a reconstruction of an observed neutrino signal from GW170817.

### 5. MULTI

Multimessenger overview combining:

* Gravitational waves
* Compact-remnant evolution
* Magnetic fields
* Ejecta
* Neutrino emission
* Jet/outflow structures
* Electromagnetic consequences

It is intended as the final overview of the simulated event.

## Compact Remnant and Black Hole Model

The simulation includes a post-merger black-hole scenario with:

* Black-hole horizon/shadow representation
* Redshifted inner rim
* Hot equatorial accretion disk
* Relativistic Doppler asymmetry
* Keplerian disk rotation
* Radial inflow
* Material capture/recycling
* Jet/outflow structure

The delayed-collapse black-hole scenario is a **model continuation**, not a claim that GW170817 was directly observed to form a black hole at the displayed time.

The post-merger simulation continues beyond the short cinematic merger interval so the modelled remnant and surrounding structures can persist during longer runtime tests.

## Gravitational Lensing

The compact-object lens uses a reduced-order Schwarzschild model.

Characteristic scales are:

```text
Event horizon   : 2M
Photon sphere   : 3M
ISCO            : 6M
Critical impact : 3 sqrt(3) M
```

The null-geodesic equation is represented as:

```text
d²u/dphi² = -u + 3 M u²
```

and integrated numerically.

This implementation does **not** claim to perform:

* Kerr ray tracing
* Full GR ray tracing
* GRMHD radiative transfer
* Full spacetime evolution

The pre-merger binary lensing visualization is camera-dependent and is primarily intended to communicate the qualitative relativistic optical effect.

## Jets and Outflows

A bounded structured bipolar outflow represents:

* Central spine
* Sheath
* Outer structured component

It is a reduced-order representation of a possible relativistic outflow, not a detailed GRB jet simulation.

## Ejecta and Kilonova Connection

The simulation includes dynamical ejecta and thermal evolution to communicate the connection:

```text
Merger -> Dynamical ejecta -> Heavy-element nucleosynthesis
       -> Kilonova -> Later electromagnetic emission
```

Physical kilonova emission evolves over hours to days rather than milliseconds.

For presentation purposes, the visualization compresses long physical evolution into a short cinematic timeline. The visual time compression should therefore not be interpreted as the physical duration of a kilonova light curve.

## Scientific Approximations

This project deliberately uses reduced-order models so that the simulation can operate interactively on consumer hardware.

| Component           | Representation                            |
| ------------------- | ----------------------------------------- |
| Neutron stars       | Reduced-order compact-object surfaces     |
| Inspiral            | Reduced-order orbital dynamics            |
| Gravitational waves | Quadrupolar strain/wavefront model        |
| Lensing             | Schwarzschild reduced-order ray tracing   |
| Magnetic field      | Reduced-order differential winding        |
| Neutrinos           | Reduced-order emission/transport          |
| Ejecta              | Reduced-order dynamical/thermal model     |
| Jet                 | Structured reduced-order outflow          |
| Black hole          | Reduced-order post-merger remnant model   |
| Accretion disk      | Reduced-order rotating/inflowing disk     |
| Kilonova            | Presentation-scale thermal/emission model |

These approximations are intentional design choices for an interactive scientific visualization.

## Time Handling

The simulation uses one authoritative event time across all visualization modes.

Mode switching does not reset the physical event time.

The presentation/cinematic timeline is separate from the underlying event-time interpretation.

After the main cinematic interval, the simulation continues linearly rather than jumping to an unrelated physical timescale.

This allows the post-merger black hole, accretion disk, ejecta, and other structures to persist during longer runtime tests.

## Observational Context

The project is inspired by GW170817.

GW170817 provided observations across multiple messenger channels, including:

* Gravitational waves
* Gamma rays
* Optical/infrared emission
* Radio emission
* Other electromagnetic observations

The simulation should therefore be interpreted as a **physics-inspired visualization**, not a numerical reproduction of the observed event.

Where the simulation goes beyond directly observed information, the display identifies those components as modelled or predicted behavior.

In particular:

* Post-merger evolution is modelled.
* Black-hole formation is a model scenario.
* Neutrino emission is modelled.
* Neutrino transport is reduced-order.
* Post-merger GW behavior is modelled.
* Jet structure is reduced-order.
* Lensing is reduced-order.

## Audio

The GW audio is a sonification of the model gravitational-wave strain.

It is **not sound propagating through space**.

It is an audible representation of the strain signal, allowing the frequency and amplitude evolution to be experienced acoustically.

## Controls

The application supports interactive camera navigation and visualization-mode switching.

Typical controls include:

```text
Mouse       : Orbit camera
Arrow keys  : Camera movement/orbit
Zoom        : Camera distance
Mode keys   : Switch visualization modes
```

The on-screen dashboard shows the active mode and simulation telemetry.

Use the controls displayed by the application for the current build.

## Performance

The simulation was designed with consumer hardware in mind.

Performance-sensitive components use:

* Taichi GPU/CPU kernels
* Cached waveform data
* Cached observational data windows
* Reduced visual update cadence for selected secondary effects
* Mode-dependent activation of expensive subsystems
* Bounded particle/tracer populations
* Deterministic star-field generation

The simulation does not require a discrete high-end GPU.

Performance naturally depends on hardware, graphics backend, window size, and active visualization mode.

## Reproducibility

The star field and several visualization components use deterministic initialization.

The main star-field catalog uses a fixed seed so that the visual environment is reproducible between runs.

Physical state is separated from presentation/camera state so event timing remains consistent when switching modes.

## Installation

Clone the repository:

```bash
git clone https://github.com/rajdeep192002-max/gw170817.git
cd gw170817
```

Create an environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch the application using the project's main entry point.

## Testing

The repository contains targeted tests for the major physical and visualization components.

Examples include tests covering:

* Post-merger evolution
* Black-hole persistence
* Neutrino transport
* Magnetic-field evolution
* Magnetic rotation coupling
* Accretion
* Astronomical star field
* Visualization modes

Run:

```bash
pytest
```

Some full-system tests may depend on the available graphics backend and hardware.

## Project Structure

```text
gw170817/
|
+-- visualization/
|   +-- dashboard.py
|   +-- renderer.py
|   +-- wave_propagation.py
|   +-- field_lines.py
|   +-- ...
|
+-- tests/
|   +-- ...
|
+-- assets/
|   +-- ...
|
+-- requirements.txt
+-- README.md
```

## Limitations

This project is intended as a scientific visualization and educational simulation.

It does not attempt to replace numerical relativity or radiation-hydrodynamic simulation codes.

In particular, it does not solve the full coupled Einstein-hydrodynamics-neutrino-radiation-MHD problem.

Important limitations include:

1. Reduced-order orbital dynamics
2. Reduced-order neutron-star structure
3. Reduced-order gravitational-wave generation
4. Reduced-order neutrino transport
5. Reduced-order magnetic-field evolution
6. Simplified ejecta dynamics
7. Simplified accretion physics
8. Simplified jet structure
9. Schwarzschild rather than Kerr lensing
10. Presentation-scale time compression
11. Modelled rather than observed post-merger behavior

These limitations are necessary to maintain real-time interactivity on consumer hardware.

## Purpose

The goal of this project is to make the physics of a binary neutron-star merger visually understandable while retaining explicit connections to the underlying astrophysical processes.

Rather than presenting the merger as a purely cinematic effect, the simulation separates the event into observable and modelled physical channels:

```text
        BINARY NEUTRON-STAR MERGER
                    |
       +------------+------------+
       |            |            |
       v            v            v
 Gravitational   Matter       Compact
     Waves       Ejecta       Remnant
       |            |            |
       v            v            v
    GW signal    Kilonova     BH / Disk
                              /     \
                             v       v
                         Magnetic   Jet
                           field
                             |
                             v
                         Neutrinos
```

The result is an interactive multimessenger view of a neutron-star merger inspired by GW170817.

## References and Further Reading

The scientific interpretation of this project is based on the broader literature surrounding:

* GW170817
* Binary neutron-star mergers
* Gravitational-wave emission
* Kilonovae
* Neutron-star merger nucleosynthesis
* Neutrino emission and transport
* Relativistic jets
* Compact-object accretion
* Schwarzschild gravitational lensing

The implementation should be interpreted alongside the original observational and numerical-relativity literature rather than as a replacement for those calculations.

## License

See the repository license for the terms governing use and redistribution.

---

**This project is a reduced-order scientific visualization of GW170817-inspired multimessenger physics.**

The distinction between observation, physical interpretation, and model continuation is intentional throughout the simulation.



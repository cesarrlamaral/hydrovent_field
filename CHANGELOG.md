# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- "Experiment run" mode in the GUI now locks the *entire* real protocol
  used for the ensembles behind the Astrobiology submission (terrain
  generation, plume-physics parameters, prebiotic modules, basin,
  sensitivity sweep), not just molecule class and acoustic mode as
  before — so anyone selecting "Experiment run" reproduces the same
  scientific configuration, regardless of future changes to each
  widget's default value. The locked values are defined once in
  `_EXPERIMENT_RUN_PROTOCOL` (`gui.py`) and enforced twice: the
  corresponding widgets are disabled/repopulated, and `_build_args()`
  independently overrides them regardless of widget state (defense in
  depth). Seed, run count, image generation, parallelism, output
  location, and 3D-visualization options remain free, since they don't
  affect the scientific result — run count specifically switches the
  execution mode to "ensemble" (single/vardecomp don't use it the same
  way) and pre-fills it with 1000 (the larger of the two paper
  ensembles) as a starting suggestion, editable like before.
- GUI now exposes every remaining CLI-only option: pore aspect ratio
  (`--pore-aspect-ratio`), custom acoustic particle radius/density
  (`--acoustic-particle-radius-um`/`--acoustic-particle-density`, enabled
  only when the acoustic mode uses particle trapping), and the parallel
  worker-process count (`--workers`, enabled alongside the parallel
  checkbox). All three are blank-by-default text fields that map to the
  same "omit the flag" default as the CLI. The GUI previously hardcoded
  the two acoustic particle fields to `None` and had no field at all for
  `--pore-aspect-ratio`/`--workers`, so single/ensemble/variance-decomposition/
  resume runs from the GUI could not reach these options.
- Nested-design variance decomposition (`--variance-decomposition`):
  separates stochastic (vent-field seed) from parametric (entrainment
  coefficient, acoustic aggregate radius/density) variance via one-way
  random-effects ANOVA, with bootstrap confidence intervals.
- Global sensitivity analysis: Sobol' first-order/total-effect indices
  per swept parameter, computed on a from-scratch Gaussian Process
  surrogate fit to the nested-design data (no extra simulations).
- Multivariate driver regression (`driver_regression.py`): rank-transform
  multiple regression controlling all swept parameters simultaneously,
  replacing one-at-a-time correlation for identifying which parameter
  actually drives a given outcome (with VIF and Holm-corrected p-values).
- `--sensitivity-sweep` now uses a single discrepancy-optimized joint
  Latin Hypercube design instead of independent per-parameter draws.
- `ensemble_stats.describe()`: robust statistics (IQR, scaled MAD,
  skewness, kurtosis) alongside mean/std/median, and an opt-in,
  vectorized bootstrap confidence interval for every continuous
  statistic (previously only the rare-event fraction had a CI).
- Open, no-login ensemble statistical report in the GUI
  (`ensemble_report.py`, "Generate statistical report (HTML)" button in
  the stats tab): descriptive-statistics table with 95% CIs, the same
  ensemble-level charts already shown live in the GUI (with full figure
  captions), the per-run results table, and — when the ensemble used
  `--sensitivity-sweep` — a driver-regression table. Deliberately no
  interpretation/discussion, article framing, or per-run representative
  images (topview/3D/artistic renders) — that remains the separate,
  gitignored, Administrator-only `report.py`/`relatorios_admin.py`.
- `--variance-decomposition` is now also available from the GUI as a
  third execution mode ("Variance decomposition (nested)") alongside
  single run/ensemble, with a live results panel and full sections
  (with 95% CIs) in the statistical report above when applicable.
- First automated test coverage for the GUI (`tests/test_gui.py`).
- Systematized multiple-comparisons correction: the Administrator report's
  driver-analysis text (`report._relevance_drivers`) now uses the
  Holm-corrected multivariate regression instead of the old uncorrected
  one-at-a-time Spearman correlation.
- Monte Carlo convergence analysis (`convergence_analysis.py`): running
  fraction/mean traces with confidence intervals vs. ensemble size, and
  analytic forecasting of CI width at a larger N — answers "would a
  bigger ensemble meaningfully change the conclusion?" from existing
  ensemble data, without running new simulations.
- Numerical solution verification (`numerical_convergence.py`): tolerance
  convergence study for the plume ODE integrator and grid-refinement
  convergence study (with observed-order estimation via Richardson
  extrapolation) for the acoustic PDE solver — confirms the solvers'
  numerical accuracy independently of the physical model's validity.
  `plume_physics.integrate_plume` gained optional `rtol`/`atol`
  parameters (defaults unchanged) to enable this.
- Automated per-run integrity QA (`run_qa.py`): systematic checks for
  NaN/Inf, physically-impossible negative values, duplicate seeds, and
  internal-field inconsistencies (hard errors), kept strictly separate
  from robust (median/MAD) statistical outlier flags — this project's
  distributions are known to be heavy-tailed by construction, so an
  outlier is plausibly a genuine rare event, not a bug. Applied to both
  existing real ensembles (100 and 1000 runs): zero real errors found.
- Investigated (and explicitly declined to implement) a radius-density
  correlation structure for the acoustic aggregate particle class:
  reading the cited primary source (González-Santana et al. 2020)
  directly showed it does not report any such relationship - the
  density range is a fixed, radius-independent assumption borrowed from
  German & Sparks (1993), not measured data. Independent sampling
  remains the correct, defensible choice; documented in
  docs/PHYSICS_MODEL.md §7.5b to avoid repeating the incorrect
  assumption in a future session.

### Fixed

- **Core plume physics**: the MTT entrainment ODE system in
  `plume_physics.py` was missing a factor of 2 in both the momentum
  (`dM/dz`) and buoyancy (`dB/dz`) equations, present since the original
  implementation. Found by reading Morton, Taylor & Turner (1956) in
  full (previously inaccessible, paywalled) and re-deriving the
  flux-variable form directly from the paper's own equations (7)/(8),
  confirmed by three independent checks including exact reproduction of
  the paper's own tabulated numerical solution (Table 1). The
  entrainment equation (`dQ/dz`) was already correct. Reduces computed
  plume rise height by ~23% at project defaults (328m → 254m for a
  350°C black smoker); all 167 existing tests still pass (the field
  benchmarks' order-of-magnitude/factor-of-3+ tolerances were too loose
  to have caught this). The rise-height validation test now uses a
  closed form derived directly from MTT (1956) instead of a previously
  unverified, and as it turns out fabricated, "5·π^-0.25"/"2.98"
  coefficient. See docs/PHYSICS_MODEL.md §2/§2.2 and
  docs/CITATIONS_TO_VERIFY.md for the full derivation and correction.
- **Citation-accuracy fixes** found while verifying ~20 newly-obtained
  primary-source PDFs against every number/claim currently in the
  codebase (full detail in docs/CITATIONS_TO_VERIFY.md):
  - `report.py`: Lemaréchal, Roullet & Gula (2025) reference had a wrong
    first author initial and an entirely fabricated title ("On the
    entrainment coefficient near hydrothermal vent orifices" — the real
    title is "Hydrothermal Plume Near-Field Dynamics From LES and
    Observations"); Field & Sherrell (2000) reference also had a wrong
    title.
  - `acoustics.py`: the comment on `PARTICLE_CLASSES["fine_sulfide_colloid"]`
    conflated Klevenz et al. (2011)'s sample-filter pore size (0.2 μm,
    methodology) with a claimed particle-size measurement the paper
    never reports; corrected to reflect what the source actually
    supports. Also documented (no code change) that the boundary
    streaming coefficient `-0.75/omega` is the RMS-amplitude form of
    Nyborg (1958) eq. 28c's peak-amplitude coefficient `-3/(8ω)` — an
    apparent factor-of-2 that isn't a real discrepancy.
  - `reaction_kinetics.py`: `FE_PROMPT_SULFIDE_FRACTION_DEFAULT`
    corrected from 0.65 to 0.68 (Field & Sherrell 2000's own calculated
    value, "~68% of the total Fe vented" — 0.65 didn't appear anywhere
    in the source).
  - `acoustics.py`: `sound_speed_seawater` docstring's validity range
    corrected from "T=2-30°C" to the paper's actual "-2 to 30°C"
    (Mackenzie 1981 — code itself was already correct, only the
    documented range understated the margin).
  - Verified, with no changes needed: Ainslie & McColm (1998), Nyborg
    (1958), Eckart (1948, uncited/unused), Lavelle (1997), Mittelstaedt
    et al. (2012), Millero/Sotolongo/Izaguirre (1987), Cowen/Massoth/
    Feely (1990), Rudnicki & Elderfield (1992, 1993), Mottl & McConachy
    (1990), McKay/Beckman/Conover (1979), Baross & Hoffman (1985), and
    — found already present in `biblios/` despite being marked
    unlocatable — Gor'kov's original 1961 Russian paper, whose eq. (12)
    matches the `acoustics.py` Gor'kov-potential implementation exactly,
    term for term.

## [1.0.0] — 2026-08-07

Initial public release.

### Added

- Procedural mid-ocean-ridge terrain and hydrothermal vent field generator
  (diamond-square heightmap, axial rift valley, clustered vent classification).
- Validated turbulent plume physics (Morton–Taylor–Turner 1956) and cited
  reactive-species decay kinetics, benchmarked against real field measurements.
- Four classical prebiotic concentration modules (dilution, pore
  thermophoresis, mineral adsorption, proton-gradient compartmentalization).
- Original acoustic-concentration hypothesis module (boundary streaming +
  Gor'kov radiation-force trapping), evaluated against real measured vent
  sound pressures, cross-checked against an independent bench-scale dataset.
- Latin Hypercube sensitivity sweep over parameters with a documented
  literature uncertainty range.
- Ensemble simulation of up to 10,000 runs, sequential or parallelized across
  CPU cores, with reproducible per-run seeding, crash detection, and resume
  support.
- Desktop GUI (Tkinter) and CLI, both driving the same simulation code.
- Full physics documentation with citations and an explicit limitations
  section (`docs/PHYSICS_MODEL.md`).

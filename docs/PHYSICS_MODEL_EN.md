# Hydrothermal plume physicochemical model — reference documentation

This document describes the near-field physicochemical gradient model
implemented in `plume_physics.py` and `reaction_kinetics.py` (Phase 1
of the project — see `simulate_plume()` in `fumarola_field.py` for the
integration point). Every default and every test tolerance must be
traceable to a citation or an explicit justification in this file. If
you change a constant in `plume_physics.py` or `reaction_kinetics.py`,
update the corresponding entry here.

## 1. Scope

Modeled: the near-field buoyant turbulent plume (from the vent orifice
to the neutral-buoyancy height, typically tens to hundreds of meters),
including conservative dilution and reactive transport of H2S, Fe(II),
and Mn(II).

**Out of scope at this phase** (see section 5): background/diffuse-flow
dispersion by long-range advection-diffusion, background currents,
plume coalescence between vents of the same cluster, CH4 oxidation
kinetics, diffusive mixing in chimney-wall pores (a regime distinct
from the free-water-column plume), real-site mode (geometrically
replicated Endeavour/TAG).

## 2. Equation system — buoyant turbulent plume (MTT)

Primary reference: Morton, B.R., Taylor, G.I., & Turner, J.S. (1956).
"Turbulent gravitational convection from maintained and instantaneous
sources." *Proc. R. Soc. Lond. A* 234(1196), 1-23.

Numerical form implemented (derived from the standard flux variables
Q=πb²w, M=πb²w², B=πb²wg′; see Jones, Hogg, Kerr et al. (2020), *Phil.
Trans. R. Soc. A*, PMC7422873, for the equivalent per-angle form):

```
dQ/dz = 2*sqrt(pi)*alpha*sqrt(M)      (entrainment)
dM/dz = 2*Q*B/M                       (momentum)
dB/dz = -2*N^2 * Q                    (buoyancy consumed by stratification)
```

**Real error found and fixed (2026-08-08)**: until this date, `dM/dz`
and `dB/dz` were implemented without the factor of 2 (`dM/dz = Q*B/M`,
`dB/dz = -N²*Q`) — a real error in the core physics of the entire
model, present since the original implementation. Found by reading
`biblios/rspa.1956.0011.pdf` (Morton, Taylor & Turner 1956) in full and
deriving `Q=πb²w, M=πb²w², B=πb²wg'` directly from the paper's own
equations, instead of trusting an already-reduced form cited
secondhand. The derivation, starting from the paper's eqs. (7,ii-iii)
(`d/dx(b²u²) = 2b²g(ρ0-ρ)/ρ1` and `d/dx(b²ug(ρ0-ρ)/ρ1) =
2b²u(g/ρ1)(dρ0/dx)`, both with the factor of 2 explicit on the right
side, also confirmed by the equivalent reduced form of eq. (8),
`dV⁴/dx=4F*W`, `dF*/dx=-2WG`) shows that the factor of 2 survives the
conversion to the standard flux variables: with `Q=πW`, `M=πV²`,
`B=πF*` (W=b²u, V=bu, F*=b²ug(ρ0-ρ)/ρ1, the paper's own variables),
one obtains `dM/dz = 2QB/M` and `dB/dz = -2N²Q`, not `QB/M` and
`-N²Q`. The `dQ/dz = 2√π·α·√M` entry was already correct (needed no
correction). The check was cross-verified numerically: integrating the
paper's Table 1 (reduced system eq. 11, `dw/dx1=v, dv⁴/dx1=fw,
df/dx1=-w`) exactly reproduces the tabulated values (e.g. at x1=1:
v=0.5971, w=0.3624, f=0.8636) and the two notable points cited in the
text (x1=2.8 where velocity vanishes, x1=2.125 where buoyancy
vanishes) — confirming that the reading of equations (7)/(8)/(10)/(14)
is correct before applying it to the code fix. Impact: the computed
rise height (`rise_height_m`) decreases (the plume consumes buoyancy
and loses momentum faster than before) — e.g. for a 350°C black smoker
with the project defaults, from 328m to 254m. All field benchmark
tests in `tests/test_plume_physics.py` (Mottl & McConachy 1990, Lupton
et al. 1985, Rudnicki & Elderfield 1993 — all with order-of-magnitude
or factor-≥3 tolerance) still pass after the fix; see §2.2 for the
corresponding adjustment to the closed-form reconciliation formula.

**Additional independent confirmation**: Lemaréchal, Roullet & Gula
(2025), *JGR Oceans* 130(10) (verified 2026-08-08), rewrites the MTT
system in radius/velocity variables in their eq. (1a-1c):
`d/dz(r²w)=2αrw`, `d/dz(r²w²)=2r²b`, `d/dz(r²wb)=2r²w(g/ρr)(∂ρa/∂z)` —
the second and third equations have the factor of 2 exactly as
corrected here, independently confirming (a third route, in addition
to the direct reading of MTT 1956 and the numerical reproduction of
Table 1) that the correction is right.

`Q`=volume flux [m³/s], `M`=momentum flux [m⁴/s²], `B`=buoyancy flux
[m⁴/s³], `alpha`=entrainment coefficient, `N`=Brunt-Väisälä frequency
[s⁻¹]. Implemented in `plume_physics.integrate_plume()` via
`scipy.integrate.solve_ivp` (RK45, adaptive step, `rtol=1e-8`) —
chosen for error control and deterministic reproducibility (same seed
+ same parameters → same result, within the integrator's tolerance),
not a homemade fixed-step integrator.

**Dilution** is defined as `D(z) = Q(z)/Q0` — a direct result of the
integration, not a separate formula.

**Stopping condition**: integration ends at the first of two events:
`B(z)=0` (neutral-buoyancy height — the layer effectively observed in
the field) or `M(z)≈0` (momentum exhausted). For `N=0` (unstratified)
neither event fires and integration proceeds to `z_max` (default
500m).

**Direct precedent for adding a chemical tracer to this system**:
Rudnicki, M.D., & Elderfield, H. (1992). "Theory applied to the
Mid-Atlantic Ridge hydrothermal plumes: the finite-difference
approach." *J. Volcanol. Geotherm. Res.* 50(1-2), 161-172.

**Local plume temperature** `T(z)` is derived self-consistently from
`g'(z)=B(z)/Q(z)` (it is not an independent field): `T(z) =
T_ambient + g'(z)/(g*alpha_thermal)`. This feeds the
temperature-dependent rate constants (section 3).

### 2.1 Boundary conditions (source)

`plume_physics.build_source()`: `Q0 = orifice_area * exit_velocity`,
`M0 = Q0*exit_velocity`, `B0 = g*alpha_thermal*ΔT*Q0` (Boussinesq
approximation). Exit velocities by vent type (1.5/0.6/0.05 m/s for
black smoker/white smoker/diffuse flow) are plausible but **not
cited** — kept from the previous model, not a direct measurement.

### 2.2 Reconciliation with the previous closed-form formula

**Citation error found in 2026-08-08** (direct reading of
`biblios/jc094ic05p06213.pdf`, now available — previously accessible
only behind a paywall): this section's previous claim, that the
consulted literature gave `5*pi^-0.25 = 3.76` (overshoot height, M=0)
and `4*pi^-0.25 = 3.01` (intrusion height), **does not correspond to
any equation in Speer & Rona (1989)** nor to any verifiable source —
the `pi^-0.25` factor multiplying the "5" does not appear anywhere in
the primary text; it appears to have been an invented combination in
an earlier session. The paper's real equation (eq. 5, p. 6214,
attributed by them to Turner, J.S. (1973), *Buoyancy Effects in
Fluids*, Cambridge Univ. Press — not verified here, text not obtained)
is

```
z* = 5 * Bo^0.25 * N^-0.75    (penetration height, where velocity vanishes)
```

with no `pi` factor, and with `Bo` defined (paper's eq. 6) exactly as
this project's `B0` (`Bo = g'*A*W` in the source, evaluated at `z=0` —
identical to `B0 = g'0*Q0` used here). Speer & Rona do not use the MTT
(7)/(8) system directly; their model (paper's eqs. 1-4) solves
temperature and salinity separately with its own entrainment
coefficient (`E=0.255`, equivalent to `α=E/(2√π)=0.072` in MTT
notation) and cites Turner's (1973) closed form only for the
penetration height, not as an equation they themselves derived. The
paper's own validation: Atlantic case (TAG) predicts 330m vs. ~360m
observed; Pacific case predicts 180m vs. ~200m observed (Lupton et
al. 1985) — not replicated here (would require their full T+S model,
out of scope for this fix).

The closed-form formula used as a **validation test**
(`tests/test_plume_physics.py::test_rise_height_reconciles_with_mtt_closed_form`)
was replaced by the closed form derived directly from MTT (1956) eqs.
(10)/(14), verified through three independent routes in this session
(exact reproduction of the paper's Table 1 by numerical integration of
the reduced system eq. 11; algebraic verification of eqs. (7)/(8)/(10);
and cross-checking of the dimensional coefficient `0.410` from eq.
(14)). For the neutral-buoyancy height (`x1=2.125`, where `f=0` — the
same locus as the code's `B(z)=0` event):

```
z_neutral ≈ 0.7326 * alpha^-0.5 * B0^0.25 * N^-0.75
```

(constant `0.7326 = 0.410 * 2^-0.25 * 2.125`, with `F0(paper) = B0/2`
via the paper's relation `F0* = (2/π)F0` combined with `B0 = π*F0*`).
Unlike the previous formula, this one has explicit dependence on
`alpha` — correct, since the paper's own eq. (10) shows the height
scale to be proportional to `alpha^-0.5`; a closed-form formula with
no dependence on `alpha` (like the previously used "2.98") cannot
literally be MTT (1956), even though it could be a legitimate
approximation from another source with `alpha` implicitly fixed. The
earlier citation to Jones, Hogg, Kerr et al. (2020) for `z ≈
2.98*(B0/N³)^0.25` remains **unverified** in this session (PDF not
obtained) — kept only in the section 2 note about the equivalent
per-angle form, no longer used as the basis of the validation test.
Since the new closed form's locus `x1=2.125` is exactly the same
`B(z)=0` event that the numerical integration uses as its stopping
criterion, no structural difference is expected between the two
formulations — the observed agreement with the project defaults is
<1% (254.0m numerical vs. 255.0m closed-form), and the test's
tolerance was tightened from 25% to 10% accordingly.

## 3. Per-species reaction kinetics (`reaction_kinetics.py`)

General approach: the *magnitude* of each rate is anchored to a
field-measured half-life, in a real hydrothermal plume, at the
reference temperature `T_REF_C=2°C` (ambient seawater); the
*temperature dependence* uses a laboratory activation energy
(Arrhenius), when available. **This is an explicit approximation**:
the cited field half-lives are effective values, integrated over the
plume's entire thermal history (from the hot orifice to near-ambient),
not measured at a single controlled temperature. Anchoring at T=2°C
assumes that most of the residence time relevant to the observed
effective rate occurs near ambient temperature (since the plume cools
rapidly within the first few meters) — a reasonable modeling choice,
not a direct measurement.

| Species | Law | Magnitude (anchor) | Ea | Source |
|---|---|---|---|---|
| H2S | Arrhenius from t½ | 26±9 h, seawater, pH 8, 25°C, × plume enhancement factor 100× | 39±2 kJ/mol | Millero et al. (1987) *ES&T* 21:439-443 (kinetics); Radford-Knoery et al. (2001) *L&O* 46:461-464 (enhancement) |
| Fe(II) | Arrhenius from t½, by basin | Atlantic (TAG): 2.1 min; Pacific (EPR 9°45'N): 3.3 h | 29±2 kJ/mol | Rudnicki & Elderfield (1993) *GCA* 57:2939-2957; Field & Sherrell (2000) *GCA* 64:619-628; Ea from Millero et al. (1987) *GCA* 51:793-801 |
| Fe (sulfide) | instantaneous removal, fixed fraction | 65% (range 40-90%) removed near the orifice, before continued oxidation | — | Field & Sherrell (2000); Mottl & McConachy (1990) *GCA* 54:1911-1927 |
| Mn(II) | first order, no thermal scaling | k₁ < 0.2/yr (buoyant plume) | not found | Cowen, Massoth & Feely (1990) *Deep-Sea Res.* 37:1619-1637 |
| CH4 | conservative tracer (k=0) | — | — | no kinetics found in the literature search |

**H2S enhancement factor (100×)**: Radford-Knoery et al. (2001) report
sulfide removal in real plumes ~2 orders of magnitude faster than
laboratory seawater kinetics (likely mechanism: metal/particle-
catalyzed oxidation — see also PNAS 2021 on "Fe-catalyzed sulfide
oxidation" for the mechanism, whose numerical values were not verified
in this research). Exposed as the `plume_enhancement` parameter in
`k_h2s()`, not hidden inside an inflated "base" constant.

**Fe(II) basin asymmetry**: the Atlantic/Pacific rate ratio implemented
here is ~94× at T=2°C (see
`test_atlantic_fe_oxidation_faster_than_pacific_by_at_least_one_order_of_magnitude`
in `tests/test_plume_physics.py`) — stronger than the "~1 order of
magnitude" qualitatively cited by Field & Sherrell (2000), because we
are comparing the cited extremes (2.1 min vs. 3.3h) rather than a
basin average. Treat as a plausible upper bound, not a universal
ratio.

**Not verified in the primary text**: the ionic-strength/salinity
correction terms of Millero et al.'s (1987, *GCA*) full Fe(II)
equation, and the coefficients of the more recent multiparameter
parametrization of González-Santana et al. (2021), *GCA* 297:143-157,
could not be accessed (paywalled papers). Not implemented; if added in
the future, they should come with those coefficients verified in the
primary text, not reconstructed by inference.

## 4. Reactive tracer transport

Extension of Rudnicki & Elderfield (1992) to the MTT system: `d(QC)/dz
= -k(T(z))*Q*C/w(z)`. Using `dt/dz=1/w` and approximating the ambient
background concentration as zero (C represents the excess above
background), the closed-form solution implemented in
`plume_physics.integrate_species_transport()` is:

```
C(z) = (C0_effective / D(z)) * exp(-∫[0,t(z)] k dt')
```

`C0_effective` already incorporates the instantaneous-removal fraction
(Fe, sulfide precipitation) when applicable. This avoids a second
coupled ODE integration — reactive decay is solved by direct
quadrature (`scipy.integrate.cumulative_trapezoid`) over the `(t(z),
k(T(z)))` profile already produced by the base plume integration.

## 5. Limitations and unverified elements (read before citing this model)

- **N constant across the whole field**: no verified N value in the
  sources consulted for Atlantic (MAR/TAG) or Pacific (EPR) ridges —
  the only verified oceanic ridge value is Juan de Fuca, 7.9×10⁻⁴ s⁻¹
  (Lavelle, 1997). Used as a citable default/fallback, not as a
  universal value. The old ad hoc depth-dependence formula
  (`1e-3*(1+depth/6000)`) was removed for having no support in any
  source found — not replaced by another depth formula, for lack of a
  citable one.
- **alpha constant**: documented to fail within the first ~2m above
  the orifice (Lemaréchal, Roullet & Gula, 2025, *JGR Oceans* 130(10))
  — not corrected at this phase.
- **No pressure/depth dependence** in the reaction rates — the kinetic
  laws used include no pressure term.
- **pH**: conservative mixing of [H+] (additive, physically correct),
  but with no carbonate/borate buffering chemistry — a real
  simplification of ocean carbonate chemistry.
- **CH4**: treated as a purely conservative tracer — no oxidation
  kinetics (microbial or abiotic) was found in the literature search
  underlying this model.
- **Orifice geometry (radius, exit velocity)**: plausible values kept
  from the previous model, not cited measurements — they affect
  Q0/M0/B0 and therefore dilution at any absolute height.
- **Mixing regime for prebiotic hypotheses**: `prebiotic.py` uses
  `dilution_near_field_1m` (D(z=1m) from the turbulent plume model) as
  a physical proxy for the dilution input of the origin-of-life
  module. This is an approximation: the real regime of interest for
  these hypotheses (diffusive mixing in chimney-wall pores, e.g. Lost
  City / Russell & Martin) is physically distinct from the free-
  water-column turbulent plume modeled here. The remaining prebiotic
  modules (thermophoresis, mineral adsorption, proton gradient) remain
  explicitly labeled as illustrative/speculative.
- **No background dispersion, currents, plume coalescence between
  vents of the same cluster, or real-site mode** — see section 1.

## 6. How to run validation

```
pytest tests/test_plume_physics.py -v
```

Each test cites its source and the reason for the chosen tolerance in
its docstring/comment — see `tests/test_plume_physics.py`.

---

# Phase 2 — Exploratory acoustic model of prebiotic concentration

Implemented in `acoustics.py` (integration point: `--acoustic-mode` in
`fumarola_field.py`, "Acoustic model" selector in the GUI). **Read this
entire section, especially 7.6, before citing any result from this
module** — it is an original hypothesis explored computationally, not
the implementation of an already-published model.

## 7.1 Hypothesis tested

This project's user's own formulation: the real acoustic field
generated by hydrothermal vents might transport/concentrate prebiotic
molecules in regions of interference of the waves propagating in that
field. No publication found in the research underlying this module
proposes this mechanism for hydrothermal vents — the literature search
on prebiotic concentration at vents (tidal-pool evaporation, clay
adsorption, ice eutectics, Russell & Martin-type mineral compartments)
returned no mention of acoustic fields.

## 7.2 Central physical problem (why the model has two mechanisms, not one)

The classical acoustic radiation force (Gor'kov potential) scales with
particle volume (∝r³) and is demonstrated experimentally effective
starting at ~1 μm; below that it vanishes faster than r³ and
Brownian/thermal motion dominates (see Estimation of acoustic forces
on submicron aerosol particles, *Aerosol Sci. Technol.* 2017). Free
prebiotic molecules (amino acids, nucleotides: sub-nm to a few nm) are
100-1000× below that floor — "molecules trapped directly at
interference nodes" is not defensible with known acoustofluidics. This
is why the module implements two physically distinct mechanisms,
selectable via `--acoustic-mode {streaming,particle_trap,both}`:

- **A. `streaming`**: boundary acoustic streaming advects/retains
  DISSOLVED SOLUTE directly — the correct mechanism for free
  molecules, independent of size.
- **B. `particle_trap`**: Gor'kov potential traps a MINERAL COLLOID
  (>1 μm, the regime where radiation force actually works); molecular
  concentration is assumed proportional to local particle
  concentration (an approximation, not a real adsorption
  measurement).

## 7.3 Real acoustic source

Crone, T.J., Wilcock, W.S.D., Barclay, A.H., & Parsons, J.D. (2006).
"The Sound Generated by Mid-Ocean Ridge Black Smoker Hydrothermal
Vents." *PLOS ONE* 1(1): e133. "Sully"/"Puffer" vents (Endeavour, Juan
de Fuca): broadband 5-500 Hz (10-30 dB above ambient noise), narrow
tones 10-250 Hz (5-15 Hz linewidth, 10-20 dB above broadband), RMS
pressure 0.4-2.6 Pa. Proposed generation mechanisms (not mutually
exclusive): pulsating flow (orifice monopole), turbulent mixing
(dipole), fluid-structure interaction in the chimney, Helmholtz/
quarter-wave-type resonance in the chimney itself.

**Unverified/assumed**: the exact hydrophone-to-orifice distance was
not confirmed in the material consulted (the full text/methods was not
accessed) — the measured RMS pressure values are treated as
representative at a nominal reference distance of 1 m
(`REFERENCE_DISTANCE_M`), an explicit approximation. Each vent's tonal
frequency is **sampled** from the empirical range (10-250 Hz), not
derived from chimney height via a resonance formula: Crone et al. cite
Helmholtz/quarter-wave resonance as one hypothesis among several,
without providing a closed, verified geometry→frequency relation —
inventing that relation would be false precision.

## 7.4 Propagation and self-interference (Lloyd's-mirror effect)

Sound speed: Mackenzie, K.V. (1981), "Nine-term equation for sound
speed in the oceans," *JASA* 70(3), 807-812 — ~1499 m/s at this
project's ambient conditions (T=2°C, D~2500m), consistent with the
deep-ocean reference value (~1500 m/s). Absorption: Ainslie, M.A., &
McColm, J.G. (1998), *JASA* 103(3), 1671-1672 — on the order of 1e-3
dB/km at vent frequencies, hence negligible over the domain scale (the
term is computed and applied, not simply omitted; see
`test_absorption_negligible_over_domain_scale`).

There is no evidence of phase synchronism between independent vents
(Crone et al. only report AMPLITUDE modulation by tide, not phase
locking) and the tonal lines have finite bandwidth (5-15 Hz, implying
a coherence time of ~tens of ms) — interference fringes BETWEEN
different vents would not be stable in time. That is why the default
mode (`cross_vent_coherence="incoherent"`) always sums different
vents' fields IN POWER (never in phase) and only models each vent's
self-interference with its OWN image reflected off the local seafloor
(image method for a rigid boundary — basalt/sulfide impedance ≫ water,
`BOTTOM_REFLECTION_COEFF=0.9`, a plausible unmeasured value). This
source+image self-interference is the classical "Lloyd's mirror"
effect (see Urick, R.J., *Principles of Underwater Sound*), always
coherent (same source, fixed geometry), independent of any assumption
about phase between different vents.

**Upper-bound test (`--acoustic-cross-vent-coherence coherent`, added
2026-08-06)**: the "incoherent" choice above is an assumption from
ABSENCE of evidence (Crone et al. 2006 do not measure phase coherence
between neighboring vents), not a measurement that rules out the
opposite scenario — there is a physically plausible path to PARTIAL
coherence (vents in the same cluster with similarly sized chimneys
could have close tonal resonance frequencies). To test whether this
assumption is load-bearing for the model's conclusion, `acoustics.py`
also accepts a "coherent" mode: ALL vents receive a single sampled
tonal frequency (instead of one per vent, see `build_acoustic_sources`)
and their pressure/velocity phasors are summed in phase — a
deliberately idealized best-case scenario (upper bound), not a
prediction. 20-run pilot (`--seed 100 --size 65 --n-clusters 5
--acoustic-mode particle_trap`, 2026-08-06): the coherent mode raised
the mean Gor'kov trap depth (near-field Fe-oxyhydroxide aggregate, the
most favorable class) from 2.35e-2 to 3.31e-2 kT (~1.4×) and the
maximum from 1.48e-1 to 2.03e-1 kT — in the physically expected
direction (constructive interference in the best case), but BOTH modes
remain 1-2 orders of magnitude below the kT=1 threshold. Preliminary
finding (small pilot, not a full factorial design — see limitation
below): even the optimistic upper bound of full coherence between
vents does not change the conclusion that mechanism B is physically
irrelevant at real measured acoustic pressures.

**FIX (2026-08-06)**: prior to this version, the VELOCITY term (used by
both streaming and Gor'kov) summed the complex phasors of DIFFERENT
vents in phase — inconsistent with the power summation already
(correctly) used for pressure, and without well-defined physical
meaning when the summed vents have different tonal frequencies. This
leaked an undocumented, unintended form of coherent interference
between vents, through a different path than the pressure term. Fixed:
the velocity term now follows exactly the same `cross_vent_coherence`
choice as the pressure term. Regression covered in
`tests/test_acoustics.py`
(`test_different_vents_still_combine_incoherently_by_default_after_fix`,
`test_single_vent_incoherent_and_coherent_modes_are_identical`).

**Non-obvious fringe-visibility limitation**: the maximum source-vs-
image path difference is `2×(receiver height)=2m` (receiver fixed at 1
m above the bottom, the same near-field convention used in
`dilution_near_field_1m`), **independent of chimney height**. At the
real measured frequencies (10-250 Hz, λ=6-150m), this difference
covers LESS than one full interference cycle — in practice,
self-interference appears as a single smooth enhancement/shadow near
the vent, not multiple periodic nodes/antinodes (see
`tests/test_acoustics.py`, where 1000 Hz — well above the measured
range — is deliberately used only to validate the multiple-fringe
math).

## 7.5 Mechanism B — Gor'kov potential + cited particle population

Gor'kov, L.P. (1962), *Sov. Phys. Dokl.* 6, 773-775; modern formulation
in Bruus, H. (2012), "Acoustofluidics 7," *Lab Chip* 12, 1014-1021.
Particle velocity derived from the full linearized momentum equation
(`v=∇p/(iωρ)`), not the far-field shortcut `v≈p/(ρc)` — that shortcut
would erase exactly the pressure-node/velocity-antinode separation
that trapping depends on. Particle distribution: equilibrium
Boltzmann, `n∝exp(-U/k_BT)` (standard statistical mechanics, not a
fitted heuristic formula).

**Phase 2b (2026-08-05) — a two-size-class population, instead of a
single arbitrary colloid** (`acoustics.PARTICLE_CLASSES`): testing the
recommendation "model real aggregate/floc populations, not fine
colloids" requires anchoring size to field data, not an assumption.
The literature search supports two anchor points (not a fitted
continuous distribution — that would be false precision with only two
points):

1. **Fine sulfide colloid** (`fine_sulfide_colloid`): 2 μm, pyrite
   (ρ=5010 kg/m³) — order-of-magnitude reference to the >0.2 μm
   threshold of Cu/Zn particles near the orifice (Klevenz, V., Bach,
   W., Schmidt, K., Hentscher, M., Koschinsky, A., & Petersen, S.,
   2011, "Geochemistry of vent fluid particles formed during initial
   hydrothermal fluid-seawater mixing along the Mid-Atlantic Ridge,"
   *Geochem. Geophys. Geosyst.* 12, Q0AE05).
2. **Near-field Fe-oxyhydroxide aggregate**
   (`near_field_fe_oxyhydroxide_aggregate`): 17 μm (midpoint of 14-20
   μm), ρ=3000 kg/m³ (midpoint of 2400-3600 kg/m³) —
   González-Santana, D., Planquette, H., Cheize, M., Whitby, H.,
   Gourain, A., Holmes, T., et al. (2020), "Processes driving iron and
   manganese dispersal from the TAG hydrothermal plume (Mid-Atlantic
   Ridge): Results from a GEOTRACES process study," *Front. Mar. Sci.*
   7:568 — radii derived from settling velocity (Stokes' law) in the
   first km of the TAG plume. This is the largest hydrothermal
   particle/aggregate radius DIRECTLY documented in the literature
   search underlying this model (the same source reports smaller
   aggregates, 2-4 μm, between 2-30 km — the larger aggregates settle
   out and are removed with distance, not modeled here).

Both classes are always computed (`gorkov_potential_field` +
`particle_boltzmann_enhancement` for each), reported in
`diagnostics["particle_classes"]`; the aggregate class (larger, more
favorable) is used as the primary field
(`PRIMARY_PARTICLE_CLASS`). A single custom size can still be passed
explicitly via `particle_radius_m`/`particle_density_kg_m3` (or
`--acoustic-particle-radius-um`/`--acoustic-particle-density` on the
CLI) for pointed exploration — this TURNS OFF the cited population.

**Central finding (recorded as permanent regression tests,
`test_gorkov_trap_not_physically_relevant_for_fine_colloid_at_measured_vent_pressures`
and `test_gorkov_trap_remains_negligible_for_largest_documented_aggregate`)**:
at the REAL measured acoustic pressures at vents (0.4-2.6 Pa RMS — many
orders of magnitude below the kPa-MPa used in laboratory
acoustophoresis devices):

- Fine colloid (2 μm): trap depth ≈ 2.05×10⁻⁵ `k_BT`.
- Near-field aggregate (17 μm, the largest documented): trap depth ≈
  1.26×10⁻² `k_BT` (~1.3% of `k_BT`) — ~79× below the threshold of
  thermal relevance, even though it is the best size/density
  combination directly documented in the consulted literature.

Thermal agitation completely dominates in BOTH classes — the trap **is
not physically relevant**, and this result now rests on the largest
real documented size, not on an arbitrary "fine colloid" assumption.
Numerically verified that trap depth scales as r³ (see
`test_gorkov_potential_scales_as_particle_radius_cubed`): trapping
would only become relevant for particles larger than any hydrothermal
size with a directly documented radius in the consulted sources — a
stronger, falsifiable refinement than this document's previous version
(which extrapolated a "~0.1-0.5 mm" threshold with no comparison to
real particle-size data).

## 7.5b Investigated and rejected: radius-density copula for the aggregate (2026-08-08)

The statistical-robustness item "realistic correlation structure
between swept parameters" (this conversation, data-science improvement
list) started from the assumption that the near-field aggregate's
radius and density are probably NOT physically independent ("larger
aggregates tend to be less dense, the cited source itself has this
relationship in the raw data, plausibly extractable") — today
`--sensitivity-sweep`/`--variance-decomposition` sample the two as
independent (§7.8/§7.8.2).

**That assumption was WRONG — verified by reading the full primary
text of González-Santana et al. (2020) directly** (not a secondhand
summary/citation): the paper does NOT report any radius-density
relationship. The density range (2400-3600 kg/m³) is a FIXED
uncertainty range, borrowed from German & Sparks (1993) for
goethite/amorphous Fe-oxyhydroxide — "we used the SAME range of
particle densities... as German and Sparks (1993)" — and applied
UNIFORMLY via Stokes' law to invert settling velocity → radius,
independent of whichever radius results from the inversion. The radius
itself varies with DISTANCE from the vent (14-20 μm in the first km,
2-4 μm between 2-30 km — already correctly captured in
`PRIMARY_PARTICLE_CLASS`, §7.5), not with density. There is no raw
radius-vs-density data in the paper, published or supplementary, that
would allow estimating a real copula.

**Decision**: no copula was implemented. Inventing a correlation
structure with no basis in the cited source would be less defensible
than the currently assumed independence — false precision disguised as
realism, exactly the kind of choice this project avoids
(`--sensitivity-sweep`/`--variance-decomposition` already document this
same standard for other "illustrative, no citable range" parameters,
see §7.8). The current independent sampling remains the correct,
already-validated choice (`joint_latin_hypercube`, §7.8/regression test
`test_sweep_produces_no_spurious_correlation`).

**Corrective note**: this session's earlier memory/conversation entries
that motivated this item contained this unverified assumption about the
paper's "raw data" — there is no raw radius-density data in that paper.
Recorded here so the assumption is not repeated in a future session.

## 7.6 Mechanism A — boundary streaming + real advection-diffusion

Boundary streaming: Rayleigh, Lord (1884), *Phil. Trans. R. Soc.* 175,
1-21; explicit form `U_slip=-(3/4ω)·d⟨U₀²⟩/dx` from Nyborg, W.L.
(1958), "Acoustic streaming near a boundary," *JASA* 30(4), 329-339 —
applied here as an engineering extension to the full 2D gradient (not
a literal transcription of the 1D formula). Bulk streaming (Eckart,
C., 1948, *Phys. Rev.* 73(1), 68-76) was deliberately **not** used as
the primary mechanism: at vent frequencies absorption is negligible
(section 7.4), making the Eckart force (∝absorption×intensity)
irrelevant at this scale — boundary streaming dominates.

Solute concentration is NOT a heuristic formula: the real steady-state
advection-diffusion-loss equation is solved, `D∇²C - u·∇C - k·C + S =
0`, by finite differences with 1st-order upwind advection (necessary —
the mesh Péclet number here is typically ≫1, making central
differences unstable) and a sparse linear system
(`scipy.sparse.linalg.spsolve`), comparing the solution WITH streaming
(`u` from the Rayleigh/Nyborg field) against a control WITHOUT
streaming (`u=0`), same source and same loss — the same
comparison-against-control pattern used throughout the rest of
`prebiotic.py`. `D` (small organic solute diffusivity in water, ~8e-10
m²/s) is a textbook order of magnitude (Cussler, E.L., *Diffusion:
Mass Transfer in Fluid Systems*), not a measurement for amino acids at
2°C specifically. `k` (loss) is a **numerical-conditioning choice**
(pure steady 2D diffusion, with no loss at all, has no finite localized
solution for a point source in an infinite domain) — not a measured
chemical kinetic.

At the real measured pressures, the resulting streaming velocity is
negligible (order 1e-18 m/s in verification runs, see the
`metadata.json` of any run with `--acoustic-mode streaming`) — the
same qualitative conclusion as mechanism B: the real acoustic energy
available at vents is orders of magnitude too small, in both of these
mechanistic formulations, to produce a detectable concentration
effect.

## 7.7 Limitations and unverified elements (read before citing this model)

- **No experimental validation exists, in any source found, for the
  acoustic-prebiotic mechanism specifically** — this is an original
  hypothesis explored computationally, not a published model being
  implemented. Treat every result from this module as a theoretical-
  plausibility test, not a prediction.
- **Source reference distance (1 m) and bottom reflection coefficient
  (0.9) are not measurements** — plausible unverified values for the
  specific mineralized material of a vent field.
- **Tonal frequency per vent is sampled, not derived from chimney
  geometry** — Crone et al. (2006) do not provide a verified closed
  geometry→frequency relation.
- **Absence of phase coherence between vents is an inference, not a
  direct measurement** — based on the finite bandwidth of the tonal
  lines (5-15 Hz) and the absence of any reported phase locking in
  Crone et al.; if future data show partial coherence between nearby
  vents, the incoherent-summation model would underestimate the
  effect. Testable via `--acoustic-cross-vent-coherence coherent` (see
  §7.4) — an idealized upper bound (all vents at the same frequency,
  summed in phase), not a measurement nor an equally arbitrary second
  assumption: in the 20-run pilot already run (see §7.4), the upper
  bound raised the trap depth by ~1.4× but did not change the
  qualitative conclusion (still orders of magnitude below kT) — a
  larger factorial design (fixed field, coherence varying) would be
  needed for a stronger quantitative claim than "does not change the
  conclusion in this pilot."
- **The vertical component of particle velocity is not resolved**
  (the field is evaluated only on a fixed horizontal plane 1 m above
  the bottom) — vertical trapping force is not modeled.
- **Mechanism B particles in the rigid limit**: the real
  compressibility modulus of sulfide/oxyhydroxide colloids/aggregates
  at micro/nanometer scale was not verified for either class — using a
  specific value would have been false precision, but the rigid limit
  is itself an approximation.
- **The aggregate class (14-20 μm) rests on a single documented plume**
  (TAG, Mid-Atlantic Ridge; González-Santana et al., 2020) — it is the
  largest hydrothermal aggregate radius directly documented in the
  literature search underlying this model, not necessarily the largest
  that exists in nature at other sites or under other flow/plume-age
  conditions.
- **Static size classes, not a dynamic aggregation/flocculation
  model** — real particle populations coarsen over the plume's
  residence time (settling first removes the larger aggregates, per
  the near-field/far-field size difference itself reported by
  González-Santana et al., 2020); this temporal evolution is not
  simulated here.
- **Mechanism A's solute diffusivity and loss rate are not calibrated
  for a specific prebiotic molecule at 2°C** — a textbook order of
  magnitude and a numerical-conditioning choice, respectively.
- **Current result (a finding, not a limitation, but worth repeating
  here)**: in both mechanisms, at the real measured acoustic pressures
  at vents, the effect is physically negligible for the default
  particle-size/solute parameters tested. This does not invalidate the
  hypothesis in principle (larger particles, sites with higher
  acoustic pressure, or mechanisms not modeled here could change this
  conclusion), but any claim that "the mechanism works" would need to
  justify why the parameters used here would be substantially
  different from those cited in this section.

## 7.8 Sensitivity sweep (Latin Hypercube)

Implemented in `fumarola_field.latin_hypercube_1d()` +
`run_experiment()` (`--sensitivity-sweep` on the CLI, checkbox in the
GUI). Motivation: a regular ensemble already produces enrichment
variability between runs, but that variability comes only from vent-
field randomness (different seeds) — the uncertain physical parameters
stay FIXED at their default value across all runs, hiding how much the
result depends on those choices (see section 7.7).

**Method**: JOINT/multi-D Latin Hypercube sampling (McKay, M.D.,
Beckman, R.J., & Conover, W.J., 1979, "A comparison of three methods
for selecting values of input variables in the analysis of output
from a computer code," *Technometrics* 21(2), 239-245), via
`fumarola_field.joint_latin_hypercube()`
(`scipy.stats.qmc.LatinHypercube` with `optimization="random-cd"`) —
each dimension comes out stratified into N equal intervals (N = number
of runs in the ensemble), exactly like the 1D LHS
(`latin_hypercube_1d()`, still used standalone in
`tests/test_fumarola_field.py` and available as a utility), but the N
combinations across dimensions are chosen with centered-discrepancy
optimization instead of a purely random per-dimension permutation —
reduces the chance of residual spurious correlation between swept
parameters in a small sample, without introducing any real physical
correlation between them (each margin keeps the same documented
range). **Updated 2026-08-07**: before this adjustment, each parameter
was sampled by 3 SEQUENTIAL, independent calls to
`latin_hypercube_1d()`, which were already mathematically equivalent
to a joint LHS by simple permutation (without discrepancy
optimization); `tests/test_fumarola_field.py::test_sensitivity_sweep_swept_parameters_
show_no_spurious_correlation` numerically confirms (Spearman |rho|
below the asymptotic threshold under independence, n=60) that the swap
neither introduced nor left uncorrected any detectable correlation
between `entrainment_alpha` and the aggregate radius in a real sweep.
The sweep's RNG derives from the SAME base `--seed` used for the vent
fields (via an additional child of `SeedSequence.spawn`), so the whole
sweep — not just each individual field — is reproducible.

**What is swept, and why only that**: only parameters with a
DOCUMENTED uncertainty range from a citable source:
- `entrainment_alpha` ∈ [0.07, 0.18] — field-measured range (Grotto
  vent, Main Endeavour Field; Rona, P.A., Bemis, K.G., Jones, C.D.,
  Jackson, D.R., Mitsuzawa, K., & Silver, D., 2006, "Entrainment and
  bending in a major hydrothermal plume, Main Endeavour Field, Juan de
  Fuca Ridge," *GRL* 33, L19313, doi:10.1029/2006GL027211 — corrected
  2026-08-06 after directly reading the full primary PDF,
  `biblios/2006gl027211.pdf`: the lead author, title, and article
  number previously cited here were wrong — "Bemis, Jones & Jackson,
  'Plume anomaly detected by acoustic Doppler current profiler,'
  L02613" — only the DOI was correct; the 0.07-0.18 value already
  matched Table 1 of the real paper exactly, so it does not change).
- If `--acoustic-mode` is `particle_trap`/`both`: aggregate radius ∈
  [14, 20] μm and density ∈ [2400, 3600] kg/m³ (the full range, not
  just the midpoint used by default — González-Santana et al., 2020;
  see section 7.5), overriding only the
  `near_field_fe_oxyhydroxide_aggregate` class, keeping the fine
  colloid fixed.

Parameters that are "illustrative, with no citable range" (Soret
coefficients, Langmuir capacity, proton-gradient gain) are
DELIBERATELY excluded from the quantitative sweep — inventing a ±
range for them would be false precision. The relative contribution of
those modules continues to be assessed only through the existing
on/off ablation design.

**Limitation (resolved by an alternative mode, see §7.8.2)**: since
each sweep run also has an independently generated vent field
(different seed), the resulting enrichment spread mixes STOCHASTIC
field variability with PARAMETER uncertainty — it does not isolate one
from the other. Isolating the two requires a nested design (same
field/fixed parameter within a group, varying only between groups),
implemented separately as `--variance-decomposition` (§7.8.2) — this
"plain" `--sensitivity-sweep` continues to exist as is, with no
behavior change beyond the sampling swap described above. See
`tests/test_fumarola_field.py` for the Latin Hypercube stratification/
reproducibility tests.

## 7.8.1 Rare event observed in the 1000-run ensemble: when Gor'kov depth crosses k_BT

Analyzing the real 1000-run ensemble generated with
`--sensitivity-sweep --acoustic-mode particle_trap` (base seed
1657119425, `outputs/experimento_260807_021219`), 7 of the 1000
realizations (0.7%; 95% Wilson CI [0.34%, 1.44%]) had the near-field
aggregate class's Gor'kov trap depth cross the thermal-relevance
threshold (>1 k_BT) — maximum observed 2.82 k_BT (seed 3074131324, 28
vents). Across the 7 realizations, the aggregate radius sampled by the
Latin Hypercube (real range 14-20 μm, González-Santana et al., 2020)
fell between 15.5 and 20.0 μm — always near the TOP of the documented
range, never in the middle or near the floor. This matches exactly the
already-established r³ scaling (section 7.5, numerically verified in
`test_gorkov_trap_not_physically_relevant_at_measured_vent_pressures`
and related tests): this is not noise — it is the expected tail of the
aggregate-size distribution colliding with a mechanism hypersensitive
to that parameter. Spearman correlation between the swept parameters
and trap depth, computed over the full ensemble: aggregate radius
ρ=+0.48 (p≈7×10⁻⁶⁰, dominant), number of vents ρ=+0.13 (p≈3×10⁻⁵, weak
in MAGNITUDE — but see §7.8.4: multivariate regression shows this
effect is REAL, not a confounding artifact, and mechanistically
explainable from the metric's own definition, not dismissible as
initially suspected here), entrainment alpha and aggregate density with
no significant correlation (|ρ|<0.03). Reproducible directly from the
ensemble's per-run `summaries`.

**Why this is discussed under an origin-of-life "rare events" framing**
(explicit user request, 2026-08-07): origin-of-life hypotheses do not require a concentration
mechanism to work reliably ON AVERAGE — they require it to work AT
LEAST ONCE, at some vent, at some point during the Hadean/early
Archean history of Earth, after which autocatalytic/template-
replicating chemistry (if it exists) would no longer need the
mechanism. This is the "many trials" argument: even a small
per-realization probability, multiplied by a plausibly enormous number
of independent hydrothermal systems over geological time, can yield an
expected number of successes >> 1. Two references anchor this framing
(citations RETRIEVED FROM TRAINING MEMORY, not read in primary text
during this session — flagged for future verification, the same
process already applied to every other citation reconstructed this way
in the project):
- Lineweaver, C.H., & Davis, T.M. (2002). "Does the rapid appearance of
  life on Earth suggest that life is common in the universe?"
  *Astrobiology* 2(3), 293-304 — argues that the rapid appearance of
  life on Earth is (Bayesian) evidence that abiogenesis may not
  require an extremely improbable step.
- Carter, B. (1983). "The anthropic principle and its implications for
  biological evolution." *Philosophical Transactions of the Royal
  Society A* 310(1512), 347-363 — the anthropic "hard steps" argument:
  observers can only find themselves in a history where a rare
  transition has already happened, so the mere existence of life is
  not strong evidence that abiogenesis is common — the two arguments
  are in real debate in the literature (Lineweaver & Davis directly
  engage Carter), and the report treats both with the SAME editorial
  weight already used for other contested mechanisms in the project
  (e.g. Sojo et al. 2016 vs. Jackson 2016 on the proton gradient,
  section 8.9).

**Honest, critical caveat**: the 0.7% is NOT an estimate of a real-
world frequency. It is a direct consequence of the sweep's Latin
Hypercube sampling design, which covers the real documented range
(14-20 μm) approximately uniformly by construction — not an
independent measurement of how often 18-20 μm aggregates actually
occur in real hydrothermal systems (which would require a real
particle-size-distribution dataset, not just the documented range).
What IS defensible: a physically real, reachable regime exists, within
the range directly documented in the literature, in which the
mechanism stops being negligible relative to thermal motion —
converting what would be dismissed outright into a concrete, falsifiable
question about the real frequency of large aggregates at active vents.

## 7.8.2 Stochastic vs. parametric variance decomposition (nested design)

Implemented in `variance_decomposition.py` (pure statistics) +
`fumarola_field.run_nested_variance_experiment()` (run orchestration)
— `--variance-decomposition` on the CLI (`--outer-samples`/
`--inner-replicates`), an alternative mode to `--runs`/
`--sensitivity-sweep` (derives its own number of runs = outer ×
inner). **Also exposed in the GUI** (`gui.py`): a third execution mode,
"Variance decomposition (nested)," alongside single/ensemble, with its
own fields for number of outer points/inner replicates; on completion,
a live panel in the statistics tab shows the summary
(`HydroventGUI._render_vardecomp_summary`) and the HTML statistical
report (`ensemble_report.py`, §10.4) includes the full sections with
95% CIs.

**Motivation**: quantitatively answer "how much of the spread observed
in an ensemble is vent-field randomness, and how much is uncertainty
about `entrainment_alpha`/aggregate radius-density?" — a question
`--sensitivity-sweep` (§7.8) cannot answer by design, since it varies
both sources simultaneously run by run.

**Method — balanced nested design**: N_outer parameter points sampled
by `joint_latin_hypercube()` (same method as §7.8); for EACH outer
point, N_inner replicates with distinct field seeds and the physical
parameter FIXED within the group. Seeds are derived deterministically
from `--seed` (one `SeedSequence` child per outer point, `N_inner`
grandchildren per point for that group's field seeds) — the whole
design is reproducible.

**Statistics — one-way random-effects ANOVA** (method of moments,
Searle, S.R., Casella, G., & McCulloch, C.E., 1992, "Variance
Components," Wiley, ch. 3): by the law of total variance, `Var(Y) =
E[Var(Y|θ)] + Var(E[Y|θ])`, where θ indexes the outer parameter point.
`MSW` (mean of the within-group sample variances) directly estimates
the STOCHASTIC component (field σ², parameter fixed). `MSB` (N_inner ×
sample variance of group means) has `E[MSB] = σ²_stochastic +
N_inner·σ²_parametric`, so `σ²_parametric = (MSB − MSW) / N_inner` —
clamped to 0 when negative (parametric signal at or below finite-
sample noise; `between_group_variance_was_clipped` records when this
happens, treated as informational, not an error). 95% CIs for the
fractions via 2-stage NESTED bootstrap (resamples which outer groups
enter AND, within each, which inner replicates — preserves the
hierarchical structure; Davison, A.C., & Hinkley, D.V., 1997,
"Bootstrap Methods and Their Application," Cambridge University Press,
ch. 3.8).

**Response variable**: by default, the near-field aggregate class's
Gor'kov trap depth (`trap_depth_over_kT`, the same metric as
§7.5/§7.8.1) when `--acoustic-mode` is `particle_trap`/`both`; falls
back to the leading hotspot's enrichment vs. control
(`top_hotspot_enrichment_vs_control`) otherwise — both already used
elsewhere in this project, not a new metric invented just for this.
Customizable via the function's `response_extractor` parameter (not
yet exposed on the CLI).

**Validated with synthetic data** (`tests/test_variance_decomposition.py`):
generating groups with KNOWN true variance components (group effect ~
N(0,σ²_between), internal noise ~ N(0,σ²_within)), the TRUE parametric
fraction falls inside the reported 95% CI, and degenerate cases (only
stochastic noise; only group effect, no noise) are correctly recognized
(parametric/stochastic fraction ≈ 0, respectively) — not just a
code-shape test, it tests whether the math recovers a known result.

**Cost**: each replicate is a full physical simulation (same cost as a
normal run, ~10-11s/run with acoustics+plume ODE, measured in a
previous session) — the default design (`--outer-samples 20
--inner-replicates 10` = 200 runs) is deliberately more modest than a
typical ensemble of thousands of runs; more N_outer improves the
parametric component's resolution, more N_inner improves the
stochastic component's — they are not interchangeable.

**Limitation (resolved by §7.8.3)**: assumes a BALANCED design (same
N_inner in every outer group — imposed by the orchestration itself,
not a user choice) and does not, by itself, decompose the INDIVIDUAL
contribution of `entrainment_alpha` vs. aggregate radius vs. density
within the parametric component when all three are active — it only
tells "how much of the variance is stochastic vs. parametric in
total." Isolating each parameter's individual contribution (and their
interactions) is exactly what §7.8.3 (Sobol' indices) does, reusing
the same data collected here.

## 7.8.3 Global per-parameter sensitivity (Sobol' indices via surrogate)

Implemented in `global_sensitivity.py` — called automatically by
`run_nested_variance_experiment()` (§7.8.2) over the SAME data already
collected (outer points + inner replicates), with no additional
physical simulation run. **Exposed in the GUI and in the statistical
report** alongside §7.8.2 (same "Variance decomposition (nested)"
execution mode, same live panel, same HTML report section — see
§10.4).

**Motivation**: §7.8.2 separates stochastic from parametric variance,
but when MORE than one parameter is active (α + aggregate radius +
density) it does not say which ONE dominates the parametric component
— that is the question Sobol' indices (Sobol, I.M., 1993, "Sensitivity
estimates for nonlinear mathematical models," *Mathematical Modeling
and Computational Experiment* 1(4), 407-414) answer: how much of the
output variance is attributable to EACH parameter individually
(first-order index, S_i) and to each parameter including its
interactions with the others (total-effect index, S_Ti).

**Why a surrogate**: a stable Monte Carlo estimate of Sobol' indices
via the Saltelli scheme needs thousands of function evaluations —
infeasible with the real physical simulation (~10s/run). A Gaussian
Process (RBF kernel, per-dimension length scales; Rasmussen, C.E., &
Williams, C.K.I., 2006, "Gaussian Processes for Machine Learning," MIT
Press, ch. 2/5) is fit to the §7.8.2 nested design's GROUP MEANS
(approximating E[Y|θ], already isolated from stochastic noise by the
design itself), with each point's measurement noise known in advance
(`within_group_variance / N_inner`, reused from §7.8.2 — not
re-estimated). Implemented from scratch (numpy/scipy, no
scikit-learn) to keep the same lean-dependency philosophy as the rest
of the project. Kernel hyperparameters fit by maximum marginal
likelihood (multiple restarts, since the marginal log-likelihood is
not convex).

**Sobol' indices via the Saltelli scheme**: S_i via the Saltelli
estimator (Saltelli, A., et al., 2010, "Variance based sensitivity
analysis of model output. Design and estimator for the total
sensitivity index," *Computer Physics Communications* 181(2),
259-270); S_Ti via the Jansen estimator (Jansen, M.J.W., 1999,
"Analysis of variance designs for model output," *Computer Physics
Communications* 117(1-2), 35-43), numerically more stable for the
total index. The Saltelli scheme's A/B matrices come from a SINGLE
Sobol' sequence (low-discrepancy — Sobol, I.M., 1967/1976; a DIFFERENT
tool from the same-named sensitivity indices) of dimension 2d split
into columns, not two independently constructed sequences.

**Real bug found and fixed in this session**: the first implementation
generated A and B as two INDEPENDENT instances of `qmc.Sobol` — each
well distributed in its own dimension, but without the correlation
structure the Saltelli/Jansen estimator requires between A and B.
Numerically verified against an additive function with a known
analytic response (`f = x1 + 2·x2`, Var(Uniform(0,1))=1/12 —
elementary, not a memorized number): the independent-sequences version
measured S1≈0.09 when the correct analytic value is S1=0.2 (error
>2x). Fixed by generating a single joint sequence of dimension 2d and
splitting it into A (first d columns) / B (last d columns) — the same
approach used by the SALib reference library — and reconfirmed to
match S1=0.2/S2=0.8 within Monte Carlo error. Fixed as a permanent
regression in
`tests/test_global_sensitivity.py::test_sobol_matrices_use_joint_sequence_not_independent_ones`.

**Central honesty point — indices are about the SURROGATE, not the
simulation directly**: they are only as trustworthy as the Gaussian
Process's approximation of the real response, which with typically
modest N_outer (tens, not thousands — each point is a full physical
simulation) can be poor. Every result comes with `loo_cv_r2`
(leave-one-out cross-validation R², via the closed-form GP formula —
Rasmussen & Williams 2006, §5.4.2 — validated against a genuinely
held-out test set in `tests/test_global_sensitivity.py`) and an
explicit warning (`loo_cv_r2_warning`, threshold R²<0.5) when the fit
is too weak to trust the indices — tested on a real case from this
session (`--outer-samples 8` with 3 parameters: R²=0.000, indices
discardable, warning correctly fired; `--outer-samples 20`, the
production default: R²=0.313, still an honest warning, but indices
consistent with the driver already identified in §7.8.1 — aggregate
radius dominant). Individual indices are clamped to [0,1] (the
mathematical definition never leaves that interval; a finite-sample MC
estimator over a nearly-constant surrogate can escape it through pure
noise — seen in practice with `--outer-samples 8`, one index came out
1.24 before clamping).

**Validated with analytic functions** (`tests/test_global_sensitivity.py`,
without involving the GP — isolates Monte Carlo estimator bugs from
surrogate-fit ones): an additive function with no interaction (S_i=S_Ti,
sum of S_i=1, an unused parameter has S≈0) and a product function with
genuine interaction (S_Ti > S_i strictly, sum of S_i < 1) — both match
theory. The full pipeline (surrogate + Sobol') validated with synthetic
data with known-by-construction sensitivity (one dominant parameter vs.
a mildly influential one) and with a real single-swept-dimension case
(`--acoustic-mode off`/`streaming`), where all explained variance
necessarily belongs to the single parameter (S1>0.85 in the tests).

## 7.8.4 Vent-count driver: confounding or real effect? (rank-based multivariate regression)

Implemented in `driver_regression.py` — generalizes the one-parameter-
at-a-time Spearman correlation of `report._relevance_drivers` (§7.8.1)
to a regression that controls ALL predictors simultaneously, via rank
transformation (Iman, R.L., & Conover, W.J., 1979, "The use of the rank
transform in regression," *Technometrics* 21(4), 499-509) — each
predictor and the response are turned into their own ranks (average
ranks for ties), fit by ordinary least squares, preserving Spearman
correlation's robustness to monotonic nonlinearity/non-normality (this
project's response has a known long tail — see §7.8.1), but now with
PARTIAL coefficients, standard error, t-test, 95% CI by case
bootstrap, and VIF (variance inflation factor) per predictor.
Holm-Bonferroni correction (Holm, 1979, *Scand. J. Statist.* 6(2),
65-70) applied to this regression's p-values specifically.

**Systematized in the Administrator report (2026-08-08)**:
`report._relevance_drivers()`/`_relevance_driver_sentence()` (the real
text that becomes Discussion in the 3 admin-gated generators —
`generate_scientific_report`/`generate_admin_report`/
`generate_admin_paper_plosone`) used the OLD one-at-a-time Spearman
correlation (no correction), even after this section had already shown
the better method — switched to call
`driver_regression.rank_transform_regression` directly. **Real bug
found and fixed in this switch**: the sentence generator's first
version grouped EVERY predictor that was not the dominant one into a
single "no significant effect" block — but with real data, `n_vents`
has p_Holm=8.3×10⁻⁷ (survives Holm) while
`entrainment_alpha`/`agg_density_kg_m3` do not; the old sentence would
have literally contradicted the very numbers it printed alongside it.
Fixed by explicitly separating "also significant" predictors (survive
Holm, but with a smaller coefficient than the dominant one) from
"not significant" ones. Two citations for the method (Iman & Conover
1979; Holm 1979) were checked for correct formatting.

**Why the bench analysis (Phase 5, `data/chladni_bench_2021/
analysis.py`) did NOT get the same correction**: it is the only other
place in the project with hypothesis tests (Student t/Welch t/
Mann-Whitney/paired/Wilcoxon + Shapiro-Wilk/Levene diagnostics). But
these are MULTIPLE METHODS testing the SAME single question (does
wave+ differ from wave-?), a deliberate robustness-across-methods
design (documented since Phase 5), not a family of independent
comparisons — applying Holm-Bonferroni there would be an incorrect use
of the concept, not an improvement. Systematizing correctly includes
recognizing where the correction does NOT apply, not just where to
apply it.

**Motivation — resolving a documented suspicion, not just building the
capability**: §7.8.1 had left open whether vent count (ρ=+0.13, p≈3×10⁻⁵
in the marginal correlation) was a real driver of Gor'kov trap depth or
an artifact of multiple comparisons — marginal correlation does not
distinguish "X affects Y on its own" from "X and Y are both affected
by a third variable correlated with X" (here, hypothetically, the
aggregate radius).

**Real finding, run over the already-existing real 1000-run ensemble**
(`outputs/experimento_260807_021219`, the same dataset as §7.8.1 — no
new simulation, only reanalysis): multivariate regression of
`entrainment_alpha`, `agg_radius_um`, `agg_density_kg_m3`, and
`n_vents` against `gorkov_trap_depth_over_kT` (n=1000, R²=0.255):

| predictor | standardized coef. | p | p (Holm) | VIF |
|---|---|---|---|---|
| agg_radius_um | +0.489 | 1.0×10⁻⁶¹ | 4.1×10⁻⁶¹ | 1.01 |
| **n_vents** | **+0.142** | **2.8×10⁻⁷** | **8.3×10⁻⁷** | **1.00** |
| entrainment_alpha | −0.028 | 0.31 | 0.61 | 1.00 |
| agg_density_kg_m3 | +0.016 | 0.57 | 0.61 | 1.00 |

**Vent count SURVIVES multivariate control and the Holm correction** —
it is not a confounding artifact with the aggregate radius (VIF=1.00,
confirming that vent count is statistically independent of the LHS-
swept parameters, as the sampling design itself already guaranteed by
construction). The suspicion recorded in §7.8.1 ("plausibly a
multiple-comparisons effect") was right to be suspicious, but the
tested answer is that the effect is REAL — it is just not a
multiple-comparisons effect in the sense of a statistical false
positive.

**Why this makes physical sense — not a coincidence, it is
extreme-value statistics**: `particle_boltzmann_enhancement()`
(`acoustics.py`, line ~586) defines trap depth as `(U_max − U_min) /
k_BT` over the ENTIRE spatial potential field `U(x,y)` (the sum of the
acoustic contributions of ALL vents in the field, not an isolated
per-vent value). More vents in the same field → more local potential
wells overlapping in the same spatial domain → higher chance that the
MAXIMUM of the range (U_max−U_min) lands on a large value somewhere in
the domain — a classic result of extreme-value statistics (the maximum
of more variables, even iid, tends to be larger), not an artifact of
multiple hypothesis tests. This is consistent with (and mechanistically
explains) the pattern already documented in §7.8.1: the 7 rare
realizations that crossed the threshold tended to have more vents.

**Honest limitation**: the model's R² is modest (0.255) — the 4
predictors tested (only the already-swept parameters + vent count) do
not explain most of the variance in `gorkov_trap_depth_over_kT`; a
substantial part comes from other stochastic degrees of freedom of the
field (relative vent position, per-vent Lloyd's-mirror interference
geometry — §7.4) not included here as explicit predictors. 95% CI by
case bootstrap (not nested — each row of the flat ensemble is an
independent resampling unit, unlike §7.8.2's nested bootstrap, which
preserves a hierarchical structure that does not exist in this
dataset).

**Validated with synthetic data** (`tests/test_driver_regression.py`,
the module's central case): a predictor confounded by construction
(correlated with the real driver, with no effect of its own on the
response) shows a strongly "significant" marginal Spearman correlation
(inherited from the real driver) but a partial coefficient ≈0/
non-significant in the multivariate regression — reproduces exactly the
kind of question tested above with real data, before applying it to
the real ensemble.

## 7.9 How to run validation

```
pytest tests/test_acoustics.py -v
pytest tests/test_fumarola_field.py -v
pytest tests/test_variance_decomposition.py -v
pytest tests/test_global_sensitivity.py -v
pytest tests/test_driver_regression.py -v
```

Each test cites its source/justification in its docstring — including
the tests that record the model's NEGATIVE findings (trap not
thermally relevant) as a permanent regression, not just the ones that
verify the code runs without error.

---

# Phase 3 — Calibration of classical prebiotic modules (`prebiotic.py`)

Motivation: of the four classical prebiotic-concentration modules
(dilution, thermophoresis, mineral adsorption, proton gradient), only
dilution had a real physical basis (a validated plume model); the
other three used "illustrative, order-of-magnitude" parameters with no
measurement behind them — the same gap that motivated Phase 2b for the
acoustic model. This phase tackles thermophoresis, which has real
measurement literature available (Baaske et al., 2007).

## 8.1 The real mechanism is not a static Soret equilibrium

The original implementation used `enhancement = exp(S_T · ΔT)` — a
simple static Soret equilibrium. Baaske, P., Weinert, F.M., Duhr, S.,
Lemke, K.H., Russell, M.J., & Braun, D. (2007), "Extreme accumulation
of nucleotides in simulated hydrothermal pore systems," *PNAS*
104(22), 9346-9351, measured a much richer mechanism: **thermal
convection along an elongated pore coupled to thermodiffusion across
it** — the fluid convects lengthwise along the pore while molecules
migrate toward the edges by thermodiffusion, creating an accumulation
that scales with the pore's ASPECT RATIO (length/width), not just with
S_T·ΔT. This produces accumulation factors of 10⁸ to 10¹⁵×, many
orders of magnitude larger than `exp(S_T·ΔT)` alone — and explains why
the authors call the effect "extreme."

## 8.2 Formula implemented — verified against the full primary text

`module_thermophoresis()` implements, for classes with
`thermophoresis_convection_coupled=True`:

```
enhancement = exp(k · S_T · ΔT · aspect_ratio)
```

**2026-08-06 update**: the user obtained the paper's full PDF
(`biblios/baaske2007.pdf`) and the formula was verified by direct
reading of the primary text, no longer reconstructed by regression. It
is exactly the paper's Eq. 1 (p. 9348):

> c_BOTTOM / c_TOP = exp(0.42 × S_T × ΔT × r)

— the analytic solution of Furry, Jones & Onsager (1939) / Debye
(1939) for a thermogravitational column (Clusius-Dickel type), which
the authors themselves confirm against their finite-element numerical
simulation. So **k=0.42, read directly from the equation, not a
regressed value**. The previously used value (k=0.4, reconstructed by
regression to 3 numeric examples before the full text was available)
was already within ~5% of the correct value — updated to 0.42 in this
revision. `tests/test_prebiotic.py` reproduces the paper's numeric
examples (Fig. 2a/Table 2) within a wide tolerance.

## 8.3 Calibration scope: only the "nucleotideos" class

Baaske et al. (2007) measured single nucleotides and single-/double-
stranded DNA/RNA — NOT amino acids, lipids, or sugars. Generalizing
their formula/coefficients to those other classes by analogy would
have no citable basis (an option explicitly rejected when deciding
this phase's scope). Hence:

- **"nucleotideos"**: `thermophoresis_convection_coupled=True`,
  `soret_coefficient_per_k=0.006` (measured, see 8.4), `pore_aspect_ratio`
  in `SHARED_PARAMS` (default 10.0, adjustable via
  `--pore-aspect-ratio`).
- **"aminoacidos", "lipideos", "acucares"**:
  `thermophoresis_convection_coupled=False` → the formula reduces
  exactly to `exp(S_T·ΔT)` (aspect ratio and k become 1, no effect) —
  IDENTICAL behavior to before this phase, with the same illustrative
  coefficients as always. They remain explicitly labeled as
  uncalibrated.

## 8.4 Choice of the measured S_T and the salinity caveat

**Verified 2026-08-06 by direct reading of the full primary text**
(the paper's Table 1): Baaske et al. (2007) report the single-
nucleotide S_T at two monovalent-salt concentrations: 0.015/K (1.7 mM)
and 0.006/K (170 mM) — values confirmed exactly, with no divergence
from what was already documented here. S_T DROPS with salinity. We use
0.006/K (the saltiest tested condition) as the closest analog to real
seawater ionic strength (~500-600 mM) compared to the dilute
condition — but 170 mM is still ~3× more dilute than real seawater,
and the observed trend (S_T drops with salt) suggests the real value
in seawater could be EVEN LOWER. This is an explicit extrapolation,
not a measurement under this model's exact conditions — the value used
probably still overestimates the thermophoretic effect at real vents.

## 8.5 Aspect ratio: laboratory geometry, not field geometry

`pore_aspect_ratio=10.0` (default) is the most conservative single-
segment value experimentally tested by Baaske et al. (tested range
10:1-125:1, or segment cascades reaching larger effective ratios via
concatenation). The REAL microporous geometry of active hydrothermal
chimney walls was not found in the literature search underlying this
model — using the laboratory apparatus's geometry as a plausible
analog is an explicit choice, adjustable via `--pore-aspect-ratio`,
not a field measurement.

## 8.6 Additional limitations at this phase

- Real pore aspect ratio in hydrothermal chimneys not measured in the
  consulted literature (see 8.5).
- S_T extrapolated from 170 mM to real seawater salinity (see 8.4),
  probably an overestimate.
- "Nucleotide" (measured) and "free nitrogenous base" (not measured
  separately by Baaske et al., but included under the same
  `MOLECULE_CLASS_LABELS["nucleotideos"]` class label) may have quite
  different thermophoresis — free bases are smaller and simpler than
  full nucleotides (base+sugar+phosphate).
- Amino acids, lipids, and sugars remain with no equivalent
  calibration — no Baaske-type thermophoresis measurement was found
  for these classes in the consulted literature search.

## 8.7 How to run validation (thermophoresis)

```
pytest tests/test_prebiotic.py -v
```

## 8.8 Proton gradient: from self-normalizer to a real biological reference

The proton-gradient module (`module_proton_gradient`) used
`gradient_frac = ΔpH / MAX_DELTA_PH`, where `MAX_DELTA_PH` was the
MODEL'S OWN maximum ΔpH (the difference between the hydrothermal
end-member pH and seawater, both already used elsewhere) — a self-
normalizer with no external reference: `gradient_frac=1` meant nothing
beyond "this is the most extreme vent this model can generate."

**Reformulation**: `ΔpH` is converted into a real transmembrane
potential via the Nernst equation (`ΔV = 59.2 mV × ΔpH`, the ideal
slope at 25°C — standard electrochemistry), and compared against a
REAL BIOLOGICAL REFERENCE: Sojo, V., Herschy, B., Whicher, A.,
Camprubí, E., & Lane, N. (2016), "The Origin of Life in Alkaline
Hydrothermal Vents," *Astrobiology* 16(2), 181-197, state in the main
text and in Fig. 1's caption that alkaline vent pores have a pH
gradient of 3 units across the inorganic barrier (proton-motive force
of ~200 mV), and that this value is "exactly equivalent in both
magnitude (about 3 pH units) and polarity" to the proton-motive force
used by extant autotrophic cells today. We use this reference (≈177.6
mV) as the comparison scale: `gradient_frac=1` now means "this vent
produces, in magnitude, the same potential modern life uses to fix
carbon" — a quantitative comparison with external meaning, not an
internal-normalization artifact. No artificial ceiling: vents with a
ΔpH larger than the reference deliberately give `gradient_frac>1`.

**Correction (2026-08-06, verified by direct reading of the full
primary PDF, `biblios/ast.2015.1406.pdf`)**: this section previously
cited "5-6 pH units" as the gradient to which the "3 units" would be
equivalent — this conflated two distinct numbers from the paper. The
"3 units / 200 mV" is the gradient across alkaline vent pores UNDER
MODERN-ANALOG CONDITIONS, explicitly compared by the authors to the
extant biological PMF — this is the number used in the calibration.
Separately, the paper also discusses a more extreme Hadean scenario
(more acidic, CO2-rich ocean) in which the gradient could reach up to
6 units (~400 mV) — but that number is presented as an additional
potential MAXIMUM, not as "equivalent" to anything biological; it
should not have been cited here as if it were the same comparison as
the "3 units." The Arndt & Nisbet (2012) citation remains correct as
the source, within Sojo et al., of the "5-6 units" range used for the
Hadean scenario — only the attribution of the biological equivalence
was wrong.

## 8.9 The mechanism itself remains contested — equal-weight criticism

**Verified 2026-08-06 by direct reading of the full primary text**
(`biblios/s00239-016-9756-6.pdf`): the four numbers below (1 μm, >200×,
0.004 pH units, 24 J/mol vs. 24 kJ/mol) were confirmed exactly, with no
divergence from what was already documented here.

Unlike thermophoresis (where the mechanism's physics is well
established and only measured numbers were missing), here the
CALIBRATION does not resolve a real dispute about whether the
mechanism works:

Jackson, J.B. (2016), "Natural pH gradients in hydrothermal alkali
vents were unlikely to have played a role in the origin of life,"
*J. Mol. Evol.* 83(1), 1-11, argues quantitatively that:

1. Thin inorganic membranes in real alkaline vents (~1 μm) are >200×
   thicker than lipid bilayers that run real chemiosmotic circuits —
   molecular machinery (~1 nm) could not use a gradient across such a
   thick barrier.
2. In any H+-permeable channel through that membrane (necessary for
   molecular machinery to actually ACCESS the gradient), diffusion
   would make the ΔpH collapse to ~0.004 units — an available work of
   ~24 J/mol of protons, far below the ~24 kJ/mol needed for useful
   work (3 orders of magnitude difference).
3. If fluid flows faster than H+/OH- diffusion, pH along the channel
   stays close to that of the source fluid, preventing the gradient
   from being used exactly where the machinery would be.
4. There is no evidence of thin inorganic membranes sustaining sharp
   pH gradients in modern alkaline vents (Lost City).

In other words: the mechanism modeled by `module_proton_gradient`
itself — a bulk pH gradient (fluid↔ocean) translated into a
transmembrane potential USABLE by molecular machinery — is a
seriously contested hypothesis in the primary literature, with
concrete numbers on the critical side. This calibration improves the
comparison REFERENCE (section 8.8), it does not resolve this dispute —
treat any result from this module as illustrating a hypothesis under
active debate, not an established mechanism.

## 8.10 Additional limitations at this phase (proton gradient)

- The mechanism itself is contested in the primary literature
  (Jackson, 2016) — see 8.9.
- The model uses only the MAGNITUDE of ΔpH as a proxy (it does not
  distinguish acid/alkaline direction) — a pre-existing approximation,
  not changed by this calibration; the per-vent-type weight
  (`proton_vent_type_weight`) partially compensates, not fully
  corrects.
- The Nernst slope used (59.2 mV/unit at 25°C) is not corrected for
  the real hydrothermal fluid temperature (up to ~400°C) — the Nernst
  slope scales with T, so this underestimates the potential in hotter
  fluids; the correction is not implemented at this phase.
- `proton_max_factor` (the final gain applied to `gradient_frac`)
  remains an illustrative, uncalibrated parameter — only the
  comparison REFERENCE (denominator) was calibrated at this phase, not
  the gain itself.

## 8.11 How to run validation (proton gradient)

```
pytest tests/test_prebiotic.py -v -k proton
```

---

# Phase 4 — Exit-velocity validation (`plume_physics.py`)

`EXIT_VELOCITY_BY_TYPE` was documented as "plausible order of
magnitude, not cited." A literature search found real field
measurements for two of the three vent types:

- **black_smoker**: 0.7-2.4 m/s (direct in-situ turbine-flowmeter
  measurement, "Alvin") — Converse, D.R., Holland, H.D., & Edmond,
  J.M. (1984), "Flow rates in the axial hot springs of the East
  Pacific Rise (21°N)," *Earth Planet. Sci. Lett.* 69, 159-175. The
  model's value (1.5 m/s) already fell within this range — promoted
  from "plausible" to "validated within the measured range," with no
  change to the number. **Correction (2026-08-06, verified by direct
  reading of the full primary PDF, `biblios/0012-821x2990080-3.pdf`)**:
  this section previously cited "1-5 m/s" as the Converse et al.
  range — that was wrong. 1-5 m/s is the estimate of Macdonald, K.C.,
  Becker, K., Spiess, F.N., & Ballard, R.D. (1980), *Earth Planet.
  Sci. Lett.* 48, 1-7, for the "National Geographic" vent, cited
  secondhand WITHIN the Converse et al. paper — it is not their own
  measurement. Converse et al.'s direct measurement is 0.7-2.4 m/s;
  the model's value (1.5 m/s) remains validated within that range, so
  the conclusion does not change, only the citation attribution.
- **diffuse_flow**: ~0.001-0.111 m/s, combining Mittelstaedt, E., et
  al. (2012), "Quantifying diffuse and discrete venting at the Tour
  Eiffel vent site, Lucky Strike hydrothermal field," *Geochem.
  Geophys. Geosyst.* 13, Q0AF04 (0.009-0.111 m/s, optical
  velocimetry) and Sarrazin, J., Rodier, P., Tivey, M.K., Singh, H.,
  Schultz, A., & Sarradin, P.-M. (2009), "A dual sensor device to
  estimate fluid flow velocity at diffuse hydrothermal vents,"
  *Deep-Sea Res. I* 56(11), 2065-2074, at the same edifice
  (0.0011-0.0049 m/s, low-temperature fractures). The ~1-order-of-
  magnitude discrepancy between the two methods/conditions is NOT
  resolved — reported as the range's real uncertainty, not hidden.
  The model's value (0.05 m/s) falls within the combined range.
- **white_smoker**: no specific measurement found — remains a
  plausible, uncited value (0.6 m/s, assumed to be intermediate).

No numeric value was changed at this phase — only two of the three
were promoted from "assumption" to "validated against real field
measurement," with the range and source now explicit.

```
pytest tests/test_plume_physics.py -v -k exit_velocity
```

---

# Phase 5 — Bench-scale test (2021) and the conduit-resonance hypothesis

On 2026-08-06, the user brought their own 2021 bench-scale test: a
5×5cm container with 5mL of nuclease-free water and purified K-562
standard DNA, resting directly on a loudspeaker generating 33-34Hz
tones, with DNA samples (quantified by Qubit) collected at regions
labeled by the user as "wave+" (node) and "wave-" (antinode), plus
control and blank (n=32 independent experiments, container cleaned
with hypochlorite+alcohol and reassembled each run). Raw data and
video of the experiment are in `data/chladni_bench_2021/`
(`medias.xlsx`, `experiment_video.mp4`, `analysis.py` reproduces all
the statistics below).

This section documents that reanalysis and connects the finding to
Phase 2 (§7): why bench/laboratory systems succeed at organizing
particles and cells into Chladni patterns, while the free-field vent
model (§7.5-7.7) finds no relevant effect in kT.

## 9.1 Statistical reanalysis (do not rely on the original spreadsheet's p-values alone)

| Group | Mean | SD | n |
|---|---|---|---|
| wave+ (node) | 34.10 | 14.98 | 32 |
| wave− (antinode) | 6.54 | 2.18 | 32 |
| control | 6.31 | 2.47 | 32 |
| blank | 0.57 | 0.15 | 6 |

wave+ differs from wave− and from control with a large effect, robust
across multiple test choices (Student p=4.8e-15; Welch p=9.9e-12,
since Levene rejects equal variances p<0.0001; non-parametric
Mann-Whitney p=6.5e-12, since wave+ fails Shapiro normality p=0.003;
**paired test** — the correct design here, since each run produces one
wave+ AND one wave- sample from the same experiment — t=1.3e-11,
Wilcoxon signed-rank p=4.7e-10). Cohen's d=2.57. wave− does not differ
from control (p≈0.69 across all tests, d=0.10). Control differs
strongly from blank (p~1e-6 to 1e-14), validating that the Qubit assay
is detecting real DNA above background. Conclusion: the effect is real
and statistically robust, regardless of the choice between
parametric/non-parametric/paired tests.

```
cd data/chladni_bench_2021 && python analysis.py
```

## 9.2 The video shows a physical regime different from the one modeled in Phase 2

Inspection of video frames (`data/chladni_bench_2021/frames/`) shows
the container in **direct rigid contact** with the speaker driver, and
a classical Chladni pattern (sharp checkerboard grid of nodal lines)
formed by a light tracer (fine powder/foam) on the liquid surface.

This is not a free-field standing acoustic wave in the fluid: at
33-34Hz, the sound wavelength in water is λ=c/f≈1500/33.5≈45m, much
larger than the 5cm container — a pressure node/antinode pattern does
not fit at that scale. The visible pattern is therefore almost
certainly a **bending resonance of the container's solid plate/wall**
(thin-plate bending wave — Kirchhoff-Love theory, dispersion relation
k⁴=ρhω²/D — has a much lower propagation speed than sound in the
fluid, so a pattern of a few centimeters fits at a frequency of tens of
Hz), coupled into the liquid as a "wet Chladni plate."

This is physically distinct from the free-acoustic-field Gor'kov
trapping mechanism modeled in `acoustics.py` (§7.5) — the mechanism
here is transport by **confined acoustic streaming** (liquid
recirculation vortices induced by the plate's vibration), not direct
radiation force on the molecule.

## 9.3 Node vs. antinode: what the literature has documented since Faraday (1831)

The question "why does Chladni organize particles while our vent model
does not" has a quantitative answer, not a qualitative one — the
mechanism is the same (radiation force/streaming in a standing-wave
field), it is the field intensity that differs by many orders of
magnitude (see 9.4).

But the direction of the effect (node vs. antinode) is not universal —
it has been documented as depending on particle size/density since the
primary source:

- **Faraday, M. (1831), "On a peculiar class of Acoustical Figures; and
  on certain Forms assumed by groups of particles upon vibrating
  elastic Surfaces," *Philosophical Transactions of the Royal Society
  of London* 121, 299-340** (`biblios/faraday1831.pdf`, full primary
  text read directly). Chladni had already shown that coarse sand/
  grains stick to the nodal lines (§1 of the paper). Faraday shows that
  fine powder (lycopodium) does the opposite — it accumulates at
  **antinodes** ("centres of oscillation")# — and demonstrates
  experimentally the cause: air currents (streaming) flowing toward the
  antinode. In the gold-leaf experiment (§16), Faraday shows air
  entering under the leaf and lifting it "into the form of a blister"
  exactly at the center of vibration — direct evidence of streaming,
  150 years before the term existed.
- **Vuillermet, G., Gires, P.-Y., Casset, F., & Poulain, C. (2016),
  "Chladni Patterns in a Liquid at Microscale," *Physical Review
  Letters* 116(18), 184501** (citation verified via Crossref; full PDF
  not obtained through a legitimate free channel, flagged for future
  verification) and **Lei, J. (2017), "Formation of
  inverse Chladni patterns in liquids at microscale: roles of acoustic
  radiation and streaming-induced drag forces," *Microfluidics and
  Nanofluidics* 21(3), 50** (idem) confirm in microscale liquid: there
  is a competition between acoustic radiation force (favors nodes,
  dominates for larger/denser particles) and streaming drag (favors
  antinodes, dominates for small/light particles), with a size
  threshold separating the two regimes.

**Consequence for interpreting this experiment**: free-solution DNA is
too small a molecule for Gor'kov radiation force to play any role (§7.5
already establishes that even 17μm aggregates — a thousand times
larger than a DNA fragment — remain ~100x below the thermal kT
threshold at real measured vent pressures). By this literature's logic,
dissolved DNA should behave as a passive streaming tracer, not as a
particle subject to radiation — that is, it should go wherever the
streaming concentrates fluid mass, not necessarily to the "node" in the
classical dry-Chladni sense.

**This is a real tension, not resolved by this analysis**: the visible
tracer in the video behaves like the classical case (concentrates at
nodal lines, like coarse sand, not like Faraday's fine powder). If the
dissolved DNA is simply co-transported by the same streaming flow that
concentrates the visible tracer — plausible in a thin liquid film,
where the plate's geometric node can be a natural convergence point for
the surface flow, a regime distinct from Vuillermet/Lei's deep-liquid
experiments — that would explain the observed result without
contradicting the literature. But this was not directly verified (the
visible tracer is not the DNA; there is no independent measurement
confirming the two co-localize by the same mechanism). It remains an
open question, not a conclusion.

## 9.4 Connection to hydrothermal vents: conduit resonance is already a documented mechanism — but quantitatively insufficient

The "missing ingredient" in Phase 2's free-field model may not be more
raw pressure, but rather confinement/mechanical resonance — exactly
what the bench experiment demonstrates by analogy (a large effect via
confined streaming, with no relevant Gor'kov force at all).

**This is not new speculation**: **Crone, T.J., Wilcock, W.S.D.,
Barclay, A.H., & Parsons, J.D. (2006), "The Sound Generated by
Mid-Ocean Ridge Black Smoker Hydrothermal Vents," *PLoS ONE* 1(1),
e133** — already in `biblios/journal.pone.0000133.pdf`, already cited
in §7.3 for the measured pressure amplitude — also explicitly discusses
conduit resonance as a source of the observed narrow tones (10-250Hz),
citing as candidate resonators "Helmholtz resonators, half-wave or
quarter-wave resonators, and solid structures such as **tubes, plates,
or cavities** within the chimneys" (primary text, directly verified in
the PDF). The authors give a concrete numeric example: a 2L cavity
connected to the conduit by a 0.02m-diameter, 0.04m-long opening,
filled with hot hydrothermal fluid (c=450m/s), gives a Helmholtz
fundamental frequency f≈120Hz; a 1m tube closed at one end (quarter-
wave resonator) gives f≈113Hz — both matching the real observed range
(10-250Hz). The paper itself notes that "plates" are one of the
candidate resonating-structure types — the same mechanism class
(plate-bending resonance) that the bench experiment's video visually
suggests.

**Quantitative evidence of resonant amplification, already measured in
the field**: the narrow tones have ~10-20dB more power than the
broadband level at the same hydrophone (primary text, verified). This
corresponds to ~3-10x in pressure amplitude (~10-100x in power) of
resonant amplification — real, but **many orders of magnitude below**
what would be needed to close the ~1e9-1e11x gap (in Gor'kov trap
depth/kT) between the pressure measured in the free field (~1-3 Pa) and
the pressure used in bench-scale cell-patterning systems (~0.1-0.2
MPa — Engineering Anisotropic Muscle Tissue using Acoustic Cell
Patterning, PubMed 30277617).

**Open question, not resolved here**: the measured 10-20dB is the
amplification left over in the RADIATED sound, measured at a distance
by the hydrophone — not the pressure amplitude INSIDE the resonant
conduit/cavity itself, which could be substantially larger (analogous
to how the sound inside a resonant chamber is louder than what escapes
and is measured from outside). As far as this research was able to
verify, there is no published in-conduit pressure measurement for real
hydrothermal vents. This is the most promising line of investigation
opened by this section — not closed, only delimited.

## 9.5 Limitations of this phase

- The bench experiment is from 2021, conducted before this project
  existed — reanalyzed here, not designed for this model's purposes.
  Protocol details that would affect interpretation are missing (the
  exact definition of "control" — speaker off in the same container,
  or a separate setup; the node/antinode localization method — visual
  tracer, calculation, or fixed point; container material/thickness).
- Vuillermet (2016) and Lei (2017) are cited only via verified metadata
  (Crossref) and search abstracts — the primary text was not obtained
  (PDF attempts via Springer/APS/ResearchGate were blocked). Faraday
  (1831) and Crone et al. (2006) were verified by direct reading of the
  full primary text.
- The DNA↔visible-tracer co-localization hypothesis (9.3) and the
  real-chimney conduit-resonance hypothesis (9.4) are proposals
  motivated by indirect evidence, not confirmed — neither was tested
  computationally at this phase.
- No code change was made to `acoustics.py`/`fumarola_field.py` at
  this phase — this section is entirely documentary/interpretive.

---

# Phase 6 — General ensemble statistical robustness (infrastructure, not module-specific)

Unlike Phases 1-5 (specific physical model/modules) and sections
7.8.2-7.8.4 (tools specific to the `--variance-decomposition`
pipeline), this phase documents an improvement to GENERAL-PURPOSE
statistical infrastructure already used across the whole project —
`ensemble_stats.describe()`, consumed by the GUI's statistics tab for
ANY aggregated ensemble quantity (prebiotic concentration/enrichment,
acoustic diagnostics, vent count — not just the acoustic module).

## 10.1 Why mean/std alone are misleading in this project

`describe()` used to report only n/mean/std/min/median/max. This
project's real distributions are repeatedly heavy-tailed for
documented physical reasons (long-tailed chimney height —
`sample_chimney_height` in `fumarola_field.py`; the Gor'kov rare event,
§7.8.1; extreme-value statistics by vent count, §7.8.4) — in that
regime, mean/std alone can suggest a symmetric spread that does not
exist.

**Confirmed with the real 1000-run ensemble** (same dataset as
§7.8.1/§7.8.4, `outputs/experimento_260807_021219`,
`gorkov_trap_depth_over_kT` of the aggregate class, a reanalysis — no
new simulation):

| statistic | value |
|---|---|
| mean | 0.153 |
| std | 0.189 |
| median | 0.108 |
| IQR (Q1–Q3) | 0.067–0.177 |
| scaled MAD | 0.072 |
| skewness | **7.22** |
| kurtosis (excess) | **78.8** |
| (mean−median)/IQR | 0.41 |

`mean±std` (0.153±0.189) suggests an approximately symmetric range that
would come close to 0 on the lower side — but the real distribution is
a narrow median (0.108) with an extremely long, thin upper tail (max
observed 2.82, section 7.8.1): skewness=7.22 and kurtosis=78.8 are
orders of magnitude above what an approximately normal distribution
would have (~0 for both) — a direct quantitative signal that
median/IQR describe this distribution far better than mean/std.

## 10.2 Statistics added

Implemented in `ensemble_stats.describe()` (additive — all pre-
existing keys kept with identical semantics, no existing consumer
breaks):
- `q1`/`q3`/`iqr`: quartiles and interquartile range (elementary
  percentile definition) — spread robust to outliers.
- `mad`/`mad_scaled`: raw median absolute deviation and the version
  scaled by 1/Φ⁻¹(3/4) ≈ 1.4826 (Φ⁻¹ = standard normal quantile) —
  this specific constant makes the MAD a CONSISTENT estimator of the
  standard deviation under normality (derivation: under
  X~N(μ,σ²), E[|X−median|]=σ·Φ⁻¹(3/4)), directly comparable to `std`
  but without the disproportionate weight outliers have on `std`.
- `skewness`/`kurtosis`: skewness coefficient and excess kurtosis
  (normal=0), Fisher-Pearson adjusted estimator via
  `scipy.stats.skew`/`kurtosis` (`bias=False`, finite-sample bias
  correction). Return NaN for n<3/n<4 respectively (below that the
  value is mathematically degenerate — e.g. the skewness of 2 points
  is ALWAYS 0 by symmetry, not because the real distribution is
  symmetric; NaN is more honest than a misleading zero). Explicit
  guard for zero variance (all identical values) avoids a noisy scipy
  `RuntimeWarning` without changing the result (same correct NaN).
- `mean_median_gap_over_iqr`: `(mean−median)/iqr` — how much the mean
  is "pulled" from the median by skewness/outliers, on the scale of
  the data's typical spread. A SELF-EXPLANATORY diagnostic (a ratio of
  two already-reported quantities), deliberately with no external
  threshold like "|skewness|>1 is 'very skewed'" — that kind of rule
  of thumb appears in several textbooks with inconsistent attributions
  to each other; rather than cite a not-fully-verified source for an
  arbitrary cutoff number, the module reports the raw metric and
  leaves interpretation to the reader.

## 10.2b 95% bootstrap CI for EVERY continuous statistic

Before this section, only the binary rare-event fraction had a CI
(Wilson score interval, §7.8.1) — every CONTINUOUS statistic
(mean, median, iqr, skewness...) was reported as a bare number, with
no uncertainty. `describe()` gained the `n_bootstrap` parameter
(default 0 — behavior IDENTICAL to before, no new key unless
explicitly requested): when >0, it adds `<name>_ci95` for EVERY
continuous statistic via case bootstrap (Efron & Tibshirani, 1993, "An
Introduction to the Bootstrap," Chapman & Hall), fully vectorized (a
single `(n_bootstrap, n)` resampling matrix, no Python loop — see
`_bootstrap_point_estimates`). `compute_ensemble_stats()` passes the
same parameter down to every internal `describe()`.

**Why the default is off**: a real measured cost, not estimated — ~8s
for `n_bootstrap=2000` on a 30-thousand-point pooled array (the
resampling itself is fast; SORTING 2000×30000 elements for
median/percentiles is what dominates — inherent to bootstrapping
order-based statistics, not an implementation inefficiency). Turning
it on by default in EVERY `compute_ensemble_stats()` would make the
GUI's statistics tab/report generation noticeably slower on large
ensembles without the user asking — the same logic already applied to
`--ensemble-images`/`--sensitivity-sweep` elsewhere in the project.
Per-run arrays (typically hundreds to thousands of points, not tens of
thousands) are much faster: 0.25s for `n_bootstrap=2000` on a
1000-point array (measured below).

**Real finding, applied to the same 1000-run ensemble from §10.1** (no
new simulation): the CI of skewness/kurtosis itself is WIDE —
skewness=7.22 (95% CI [3.72, 8.38]), kurtosis=78.8 (95% CI [22.4,
105.3]). Even with n=1000, the HIGHER-order moments (skewness/kurtosis)
are estimated with much less precision than mean/median (95% CI
[0.142, 0.167] and [0.102, 0.114] respectively, much narrower relative
ranges) — statistically expected (the standard error of order-k moment
estimators grows with the order), but before this section there was no
way to know THIS without a separate calculation; now it comes for free
in any `describe()` call with `n_bootstrap>0`.

## 10.3 Tested

`tests/test_ensemble_stats.py` (this module's first dedicated test
coverage — extracted from gui.py in a previous session with no tests
of its own at the time): exact backward compatibility of pre-existing
keys; IQR against direct `np.percentile`; recovery of skewness≈0/
kurtosis≈0 on a large synthetic normal; skewness>1 on a synthetic
exponential (known theoretical skewness=2); scaled MAD recovers the
real standard deviation under normality within 5%; central property —
a single extreme outlier inflates `std` >5x but moves `iqr`/
`mad_scaled` less than 1.5x; degenerate cases (small n, zero variance)
return NaN with no warning. **Bootstrap CI**: empirical coverage
measured over 150 independent replicates comes out near 95% (not a
single seed — a 95% CI by definition misses ~5% of the time; testing
just 1 case would carry real risk of failing by luck); CI width
decreases with larger sample; the vectorized version matches manual
row-by-row resampling exactly; `compute_ensemble_stats(n_bootstrap=...)`
wiring tested end to end. 19 tests in this file, full project suite
110/110 passing.

```
pytest tests/test_ensemble_stats.py -v
```

## 10.4 Open statistical report in the GUI (no interpretation, no login)

Implemented in `ensemble_report.py` — explicit user request
(2026-08-07, the session following this one): "I want us to have an
open report section in the GUI again, but only the statistical
report... no discussion or article format... no representative
images." Deliberately no interpretation/discussion or article framing,
by the SAME standard already used in the project's
`.gitignore`/`CONTRIBUTING.md` to decide what is generic software vs.
author-specific content.

**Report contents** (the "Generate statistical report (HTML)" button in
the statistics tab, available with no login as soon as an ensemble
finishes): a descriptive-statistics table with 95% bootstrap CIs
(§10.2b), the SAME 4 charts already displayed live in the tab
(`build_ensemble_charts_figure`, reused — the GUI and the report never
diverge, a single source of truth) with a full caption (not just a
short axis title), a per-run table, and — when `experiment_dir`
contains `vardecomp_summary.json` (see below) — the variance-
decomposition (§7.8.2) and Sobol' (§7.8.3) sections; when the ensemble
used `--sensitivity-sweep` in flat (non-nested) mode, the multivariate
driver table (§7.8.4). Deliberately WITHOUT: title/abstract/
discussion, manuscript framing, or any image of ONE specific run
(topview/3D/artistic) — restrictions explicitly tested
(`test_report_does_not_contain_discussion_or_article_framing`,
`test_report_does_not_embed_representative_run_images`).

**`--variance-decomposition` brought into the GUI** (same session, the
user's next request after noticing that items marked "done" did not
imply they showed up in the GUI): a third execution mode, "Variance
decomposition (nested)" (`HydroventGUI._on_run_clicked`/
`_vardecomp_worker`), with its own fields for number of outer points/
inner replicates (same CLI defaults, 20/10). On completion,
`_on_vardecomp_finished` loads the individual runs from disk (the same
"open existing experiment" pattern — `find_run_dirs`/
`load_run_summary`) to feed the image viewer/statistics tab normally,
and a live panel (`_render_vardecomp_summary`) summarizes decomposition/
Sobol' without needing to generate the report. The nested design's
specific results are read from the `vardecomp_summary.json` already
written to disk by `run_nested_variance_experiment`
(`_read_vardecomp_summary`) — not passed in memory — so they work both
for a freshly finished run and for reopening an old `vardecomp_*`
folder later.

**Real bugs found and fixed while building this**:
1. The per-run table assembly had a Python ternary expression that, by
   operator precedence, applied the condition to the ENTIRE concatenated
   HTML row instead of just to the cell — if
   `top_hotspot_enrichment_vs_control` was `None`, the whole row
   silently became just "n/a," discarding run/seed/vent-count. Fixed
   by building each cell as its own variable before assembling the
   row.
2. The generator function accepted a `stats` parameter that was never
   actually used (it always recomputed internally to guarantee the
   bootstrap CIs) — a misleading API, removed.
3. **Test-infrastructure bug, not a production-code one**: running all
   of `tests/test_gui.py` caused, depending on order, anything from a
   fatal Windows exception to a silent failure creating the next
   `tk.Tk()` — creating/destroying multiple Tk roots in the same
   process proved unstable on this platform/Tcl combination (each
   isolated test passed normally). Fixed with a `scope="module"`
   fixture that reuses ONE Tk root for the entire file.

**Tested**: `tests/test_ensemble_report.py` (15 tests — balanced HTML,
expected content always present, absence of discussion/representative
images, bilingual PT/EN, conditional driver table, conditional
decomposition/Sobol' sections read from disk) and `tests/test_gui.py`
(7 tests — this project's first automated GUI test coverage: widget
switching across the 3 execution modes, input validation, and one full
real flow — click → background thread → result populating
`HydroventGUI` → live panel → report button — through the real
`HydroventGUI` object, not just isolated
`fumarola_field.py`/`ensemble_report.py` functions). Full project
suite 132/132 passing.

```
pytest tests/test_ensemble_report.py -v
pytest tests/test_gui.py -v
```

## 10.5 Monte Carlo convergence: is 1000 runs enough, or would 10000 change the conclusion?

Implemented in `convergence_analysis.py` — formally answers a question
that was previously only answered qualitatively ("0 out of 100 is
plausible when the real rate is ~0.7%, via Poisson probability"): as
more runs accumulate, the rare-event fraction estimate STABILIZES as
expected, and how much running 10000 instead of 1000 would actually buy
in precision — with no new simulation run, just reanalyzing the two
real ensembles that already exist.

**Wilson CI reimplemented here** (this module is tracked and needs to
work standalone in a public clone of the repository) — validated
against the value already
documented in §7.8.1 (k=7,n=1000 → 95% CI [0.34%, 1.44%], exactly
reproduced) and against average empirical coverage ~94.8% across
several (p,n) combinations (Wilson has known OSCILLATING coverage
around the nominal value — Brown, Cai & DasGupta, 2001, *Statistical
Science* 16(2), 101-133 — testing a single isolated case would be
misleading).

**Real convergence trace, 1000-run ensemble**
(`outputs/experimento_260807_021219`, in the runs' REAL order — order
matters for a genuine trace, not a shuffled resampling):

| N | fraction | 95% CI | width |
|---|---|---|---|
| 100 | 1.00% | [0.18%, 5.45%] | 5.27pp |
| 316 | 0.95% | [0.32%, 2.75%] | 2.43pp |
| 1000 | 0.70% | [0.34%, 1.44%] | 1.10pp |

The cumulative fraction converges smoothly to ~0.7% as N grows (it does
not get "stuck" on a misleading initial value), and the CI width
decreases consistently with what is expected — no sign that the
1000-run ensemble is still "unstable" or needs more data for this
specific qualitative conclusion.

**Analytic projection for N=10000** (assuming the SAME observed rate of
0.7% — not a guarantee, an answer to "if the real rate is what we
already measured, what would 10000 runs change"): the CI would narrow
from [0.34%, 1.44%] (width 1.10pp) to [0.55%, 0.88%] (width 0.33pp) —
a ~3.3x reduction, close to the 1/√10≈0.316 theoretical ratio expected
asymptotically for a binomial proportion. **Actionable conclusion**:
running 10000 runs (~2.7h in parallel, see §7.8) would not change the
qualitative conclusion (a real rare event, order of magnitude ~0.7%,
already established with margin), but it would triple the PRECISION of
the estimate — a legitimate trade-off between compute time and
precision, now quantified, not just intuited.

**Validated with synthetic data where the answer is known**
(`tests/test_convergence_analysis.py`): a binomial trace recovers the
real rate within the final CI; a continuous-mean trace (reuses
`ensemble_stats.describe`, the same bootstrap from §10.2b) recovers
the real mean; and — the strongest check — the N=10000 projection made
with only the first 1000 points of a synthetic 20000-point sample is
compared against the CI actually computed over the real 20000 points
(widths match within 50% relative tolerance), confirming the
extrapolation is usable, not just mathematically plausible.

```
pytest tests/test_convergence_analysis.py -v
```

## 10.6 Numerical convergence of the solvers: ODE tolerance and PDE mesh

Implemented in `numerical_convergence.py` — verification of NUMERICAL
SOLUTION (does the method correctly solve the equation as written,
does refining tolerance/mesh leave the result unchanged) for the
project's two solvers, never checked before: the plume ODE integrator
(`plume_physics.integrate_plume`, adaptive RK45) and the acoustic PDE
solver (`acoustics.solve_steady_advection_diffusion`, 1st-order
upwind finite differences). Orthogonal to PHYSICAL validation (the
equation is calibrated against real data — already done in other
sections) — here the question is purely numerical. `integrate_plume`
gained `rtol`/`atol` as optional parameters (defaults unchanged,
`1e-8`/`1e-12`) only to enable this study.

**Plume ODE — already converged at the default**: comparing
`rise_height_m` (neutral-buoyancy height) and dilution at z=1m between
the project default (rtol=1e-8, atol=1e-12) and a tolerance 100x
tighter (rtol=1e-10, atol=1e-14), across the 3 vent types: relative
change <10⁻⁵ in every case (typically ~10⁻⁹-10⁻¹²) — the default is
already effectively at machine precision for the physical quantities
that matter. Tightening the tolerance would not change any significant
digit of any result already published in this project.

**Acoustic PDE — two complementary, non-contradictory findings**:

1. **At the REAL production parameters** (`DEFAULT_SOLUTE_DIFFUSIVITY_
   M2_S`=8×10⁻¹⁰ m²/s, tiny — real molecular diffusion of a small
   solute), the relative change in concentration at a fixed probe
   point, between meshes bracketing the real default `--size`
   (129→257→513 cells, 1200m domain), is <10⁻³ (typically
   ~10⁻⁵-10⁻⁶) — the production mesh is not this simulation's
   precision bottleneck. The OBSERVED convergence order in that regime
   comes out noisy/without physical meaning (the change is already so
   small that it becomes dominated by the sparse linear solver's
   precision, not the scheme's truncation error) — an expected,
   coherent result, not a bug: when the change is already negligible,
   measuring its "order" stops being a well-posed question.
2. **In a deliberately more demanding synthetic scenario** (diffusivity
   and velocity larger than the real defaults, specifically to isolate
   truncation error from the linear solver's noise and actually
   measure a real convergence order): the observed order comes out
   around 0.78-0.88 as the mesh refines, approaching the theoretical 1
   expected for a 1st-order upwind scheme — confirms the METHOD itself
   behaves as the math predicts, a real check of the solver,
   independent of which parameter regime is in use.

**Real methodological bug found and fixed while building the mesh
study**: the first version placed the Gaussian source and sampled the
probe point by the CELL INDEX nearest the domain's center/fraction —
at different mesh resolutions, the same index corresponds to a
slightly different PHYSICAL location (up to half a cell of shift), so
each refinement was, unintentionally, solving a slightly different
problem at each mesh. This produced observed convergence orders with
no physical meaning at all (-1.75, then +2.56, oscillating) — a clear
symptom of positioning noise, not real discretization error. Fixed by
using PHYSICAL coordinates (meters) everywhere — the source always at
the domain's exact physical center, the probe sampled by bilinear
interpolation (`scipy.ndimage.map_coordinates`) at the exact physical
position, not the nearest cell index — after which the observed order
converged smoothly to near 1, as expected.

**Tested** (`tests/test_numerical_convergence.py`): Richardson
extrapolation recovers exact order 1 and order 2 on synthetic
functions with analytically known error (validates the formula itself,
independent of any real solver); tolerance convergence tested on the 3
real vent types (not a single isolated case); mesh convergence tested
both in the "stressed" synthetic regime (observed order checked
against the [0.5, 1.5] range) and in the real production-parameter
regime (relative change checked against <10⁻³); regression of the
cell-index-positioning bug. 8 tests, full project suite 154/154
passing.

```
pytest tests/test_numerical_convergence.py -v
```

## 10.7 Automated per-run integrity QA

Implemented in `run_qa.py` — systematic verification of an ensemble
(NaN/Inf, physically impossible negative values, duplicate seeds,
internal inconsistency between already-computed fields, runs with
missing/corrupted `metadata.json`) instead of relying only on
occasional manual inspection. Two DELIBERATELY separate levels:

- `hard_errors`: unambiguous bugs — always worth investigating.
- `soft_flags`: statistical outliers via a ROBUST z-score
  (median/MAD, the same 1.4826 scale factor from
  `ensemble_stats.describe`, §10.2) — candidates for manual review,
  EXPLICITLY not treated as bugs. This project's distributions are
  heavy-tailed by construction (chimney height, the Gor'kov rare event
  — §7.8.1); a statistical outlier here has a good chance of being the
  same kind of real rare event the rest of the project was built to
  study. Mixing the two levels would repeat, in reverse, the error
  §7.8.4 already showed to be real (treating genuine signal as if it
  were noise) — here the risk would be treating noise/a bug as if it
  were signal, or worse, drowning a real bug among legitimate
  rare-event alerts.

**Real bug in this very tool's FIRST version, found by applying it to
the two existing real ensembles**: the initial check required
`increased + decreased + unchanged == n_vents` — and it fired "error"
on 100% of the 1100 combined real runs from the two ensembles (100 and
1000 runs). An obvious signal (100% failure is never "many simulation
bugs," it is the check itself being wrong) led to rereading
`prebiotic.compute_field_hotspots`: that count only sums vents with
`enrichment_vs_control != None`, a real SUBSET of `n_vents` (not every
vent has a valid comparison against control) — it should never have
been equality, only "never exceeds." Fixed to `<=`; the same two real
ensembles pass clean (`hard_errors: 0`, `ok: True`) after the fix.
Fixed as a permanent regression test.

**Applied to the project's two real ensembles** (100 and 1000 runs,
1100 runs total): 0 real errors found (expected — data already
extensively analyzed in this session); 66 `soft_flags` total, all in
`top_hotspot_enrichment_vs_control`/
`gorkov_trap_depth_over_kT` — matching exactly the already-known long
tail of these distributions (§7.8.1/§10.1), not new problems.

**Tested** (`tests/test_run_qa.py`): each `hard_error` type isolated
with minimal synthetic data (NaN, negative, duplicate seed, n_vents=0,
count exceeding n_vents); regression of the real bug above (a count
BELOW n_vents is not an error); a planted outlier detected as a
`soft_flag`, uniform data produces none; the robust z-score formula
verified against direct calculation; runs with missing `metadata.json`
explicitly reported by `check_experiment_dir_integrity` (not silently
ignored); and a real small ensemble (15 runs, via
`fumarola_field.run_experiment`, not synthetic) passes clean. 13
tests, full project suite 167/167 passing.

```
pytest tests/test_run_qa.py -v
```

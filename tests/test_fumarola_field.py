"""
Testes da infraestrutura de varredura de sensibilidade (Hipercubo
Latino) em fumarola_field.py. Ver docs/PHYSICS_MODEL.md, seção da
varredura de sensibilidade, para a justificativa de por que só
parâmetros com faixa de incerteza documentada (não escolhas
ilustrativas) são varridos.

Rodar com: pytest tests/test_fumarola_field.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fumarola_field as ff


# --------------------------------------------------------------------------
# 1. Amostragem por Hipercubo Latino — McKay, Beckman & Conover (1979)
# --------------------------------------------------------------------------

def test_latin_hypercube_respects_count_and_range():
    rng = np.random.default_rng(1)
    samples = ff.latin_hypercube_1d(20, 0.07, 0.18, rng)
    assert samples.shape == (20,)
    assert np.all(samples >= 0.07)
    assert np.all(samples <= 0.18)


def test_latin_hypercube_stratifies_evenly():
    """
    Propriedade definidora do Hipercubo Latino: dividindo [low,high] em
    n estratos iguais, cada estrato deve conter exatamente UM ponto
    amostrado — diferente de amostragem aleatória simples, que pode
    deixar estratos vazios ou super-representados.
    """
    rng = np.random.default_rng(2)
    n = 15
    low, high = 0.0, 1.0
    samples = ff.latin_hypercube_1d(n, low, high, rng)
    edges = np.linspace(low, high, n + 1)
    counts, _ = np.histogram(samples, bins=edges)
    assert np.all(counts == 1), f"contagem por estrato = {counts}, esperado exatamente 1 em cada"


def test_latin_hypercube_is_reproducible_with_same_seed():
    samples_a = ff.latin_hypercube_1d(10, 0.07, 0.18, np.random.default_rng(42))
    samples_b = ff.latin_hypercube_1d(10, 0.07, 0.18, np.random.default_rng(42))
    np.testing.assert_array_equal(samples_a, samples_b)


# --------------------------------------------------------------------------
# 1b. Hipercubo Latino CONJUNTO (multi-D) — usado pela varredura de
#     sensibilidade e pelo desenho aninhado de decomposição de variância
# --------------------------------------------------------------------------

def test_joint_latin_hypercube_respects_bounds_and_shape():
    rng = np.random.default_rng(11)
    bounds = [(0.07, 0.18), (14e-6, 20e-6), (2400.0, 3600.0)]
    sample = ff.joint_latin_hypercube(25, bounds, rng)
    assert sample.shape == (25, 3)
    for j, (low, high) in enumerate(bounds):
        assert np.all(sample[:, j] >= low)
        assert np.all(sample[:, j] <= high)


def test_joint_latin_hypercube_each_dimension_is_marginally_stratified():
    """Mesma propriedade definidora do LHS 1D, agora checada em CADA
    dimensão do desenho conjunto: n estratos, um ponto por estrato."""
    rng = np.random.default_rng(12)
    n = 20
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    sample = ff.joint_latin_hypercube(n, bounds, rng)
    edges = np.linspace(0.0, 1.0, n + 1)
    for j in range(2):
        counts, _ = np.histogram(sample[:, j], bins=edges)
        assert np.all(counts == 1), f"dimensão {j}: contagem por estrato = {counts}"


def test_joint_latin_hypercube_is_reproducible_with_same_seed():
    bounds = [(0.07, 0.18), (14e-6, 20e-6)]
    sample_a = ff.joint_latin_hypercube(15, bounds, np.random.default_rng(77))
    sample_b = ff.joint_latin_hypercube(15, bounds, np.random.default_rng(77))
    np.testing.assert_array_equal(sample_a, sample_b)


def test_joint_latin_hypercube_dimensions_have_no_spurious_correlation():
    """Propriedade central do desenho conjunto (McKay, Beckman & Conover
    1979, generalização multi-D): dimensões marginalmente estratificadas
    mas SEM correlação artificial entre si — cada uma reflete só a
    incerteza real e independente daquele parâmetro. Limiar de |rho| usa
    o erro-padrão assintótico de Spearman sob independência,
    aproximadamente 1/sqrt(n-1) (ver Fieller, Hartley & Pearson, 1957) —
    3 desvios-padrão dá uma margem folgada, não-flaky, para n=60."""
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(2024)
    n = 60
    bounds = [(0.07, 0.18), (14e-6, 20e-6), (2400.0, 3600.0)]
    sample = ff.joint_latin_hypercube(n, bounds, rng)

    se = 1.0 / np.sqrt(n - 1)
    threshold = 3.0 * se
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        rho, _ = scipy_stats.spearmanr(sample[:, a], sample[:, b])
        assert abs(rho) < threshold, f"correlação espúria entre dims {a}/{b}: rho={rho:.3f} (limiar {threshold:.3f})"


def test_sensitivity_sweep_swept_parameters_show_no_spurious_correlation(tmp_path):
    """Mesmo diagnóstico acima, mas de ponta a ponta via --sensitivity-sweep
    real (não só a função de amostragem isolada) — confirma que
    `_derive_run_seeds_and_sweep` de fato usa o desenho conjunto."""
    import argparse
    import plume_physics as pp
    from scipy import stats as scipy_stats

    n_runs = 60
    args = argparse.Namespace(
        seed=321, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
        spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
        entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
        basin="atlantic", export_plume_profiles=False, outputs_dir=str(tmp_path),
        basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
        chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
        no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
        molecule_class="aminoacidos", acoustic_mode="particle_trap",
        acoustic_particle_radius_um=None, acoustic_particle_density=None,
        sensitivity_sweep=True,
    )
    result = ff.run_experiment(args, n_runs=n_runs, make_images=False)
    alphas = np.array([s["entrainment_alpha"] for s in result["summaries"]])
    radii = np.array([s["acoustic_diagnostics"]["particle_classes"]["near_field_fe_oxyhydroxide_aggregate"]["radius_um"]
                       for s in result["summaries"]])

    se = 1.0 / np.sqrt(n_runs - 1)
    rho, _ = scipy_stats.spearmanr(alphas, radii)
    assert abs(rho) < 3.0 * se


# --------------------------------------------------------------------------
# 2. Integração: --sensitivity-sweep produz alpha (e, se acústico ativo,
#    tamanho do agregado) distintos por run, dentro das faixas citadas
# --------------------------------------------------------------------------

def test_sensitivity_sweep_samples_distinct_alpha_per_run(tmp_path):
    import argparse
    import plume_physics as pp

    args = argparse.Namespace(
        seed=7, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
        spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
        entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
        basin="atlantic", export_plume_profiles=False, outputs_dir=str(tmp_path),
        basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
        chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
        no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
        molecule_class="aminoacidos", acoustic_mode="off",
        acoustic_particle_radius_um=None, acoustic_particle_density=None,
        sensitivity_sweep=True,
    )

    result = ff.run_experiment(args, n_runs=5, make_images=False)
    alphas = [s["entrainment_alpha"] for s in result["summaries"]]

    assert len(set(alphas)) == 5, f"esperado 5 valores distintos de alpha, obtido {alphas}"
    assert all(pp.ALPHA_ENTRAINMENT_RANGE[0] <= a <= pp.ALPHA_ENTRAINMENT_RANGE[1] for a in alphas)


def test_sensitivity_sweep_is_reproducible_with_same_base_seed(tmp_path):
    """A varredura inteira (não só o campo de fumarolas) deve ser reprodutível com --seed."""
    import argparse
    import plume_physics as pp

    def _make_args(outdir):
        return argparse.Namespace(
            seed=99, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
            spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
            entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
            basin="atlantic", export_plume_profiles=False, outputs_dir=str(outdir),
            basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
            chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
            no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
            molecule_class="aminoacidos", acoustic_mode="off",
            acoustic_particle_radius_um=None, acoustic_particle_density=None,
            sensitivity_sweep=True,
        )

    r1 = ff.run_experiment(_make_args(tmp_path / "a"), n_runs=4, make_images=False)
    r2 = ff.run_experiment(_make_args(tmp_path / "b"), n_runs=4, make_images=False)

    alphas1 = [s["entrainment_alpha"] for s in r1["summaries"]]
    alphas2 = [s["entrainment_alpha"] for s in r2["summaries"]]
    assert alphas1 == alphas2


def test_sensitivity_sweep_varies_acoustic_aggregate_when_active(tmp_path):
    import argparse
    import plume_physics as pp

    args = argparse.Namespace(
        seed=5, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
        spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
        entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
        basin="atlantic", export_plume_profiles=False, outputs_dir=str(tmp_path),
        basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
        chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
        no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
        molecule_class="aminoacidos", acoustic_mode="particle_trap",
        acoustic_particle_radius_um=None, acoustic_particle_density=None,
        sensitivity_sweep=True,
    )

    result = ff.run_experiment(args, n_runs=4, make_images=False)
    radii = [s["acoustic_diagnostics"]["particle_classes"]["near_field_fe_oxyhydroxide_aggregate"]["radius_um"]
             for s in result["summaries"]]

    assert len(set(radii)) == 4
    assert all(14.0 <= r <= 20.0 for r in radii)


# --------------------------------------------------------------------------
# 3. Desenho aninhado (--variance-decomposition) — parâmetro fixo dentro
#    de cada grupo externo, distinto entre grupos; decomposição de fato
#    calculada e gravada. Ver variance_decomposition.py para a estatística.
# --------------------------------------------------------------------------

def _vardecomp_args(tmp_path, acoustic_mode="particle_trap"):
    import argparse
    import plume_physics as pp

    return argparse.Namespace(
        seed=555, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
        spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
        entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
        basin="atlantic", export_plume_profiles=False, outputs_dir=str(tmp_path),
        basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
        chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
        no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
        molecule_class="aminoacidos", acoustic_mode=acoustic_mode,
        acoustic_particle_radius_um=None, acoustic_particle_density=None,
        sensitivity_sweep=False,
    )


def test_nested_variance_experiment_fixes_params_within_outer_group(tmp_path):
    args = _vardecomp_args(tmp_path, acoustic_mode="particle_trap")
    result = ff.run_nested_variance_experiment(args, outer_n=3, inner_n=3, make_images=False,
                                                gsa_n_mc=64, gsa_n_bootstrap=5)

    import csv as csv_mod
    with open(result["csv_path"], newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 9

    by_outer: dict = {}
    for row in rows:
        by_outer.setdefault(row["outer_idx"], []).append(row)

    assert len(by_outer) == 3
    for outer_idx, group_rows in by_outer.items():
        alphas = {r["entrainment_alpha"] for r in group_rows}
        radii = {r["aggregate_radius_m"] for r in group_rows}
        assert len(alphas) == 1, f"grupo {outer_idx}: alpha deveria ser fixo, obtido {alphas}"
        assert len(radii) == 1, f"grupo {outer_idx}: raio deveria ser fixo, obtido {radii}"
        seeds = {r["seed"] for r in group_rows}
        assert len(seeds) == 3, f"grupo {outer_idx}: seeds deveriam ser distintas dentro do grupo"

    all_alphas_across_groups = {rows_[0]["entrainment_alpha"] for rows_ in by_outer.values()}
    assert len(all_alphas_across_groups) == 3, "alpha deveria variar ENTRE grupos externos"


def test_nested_variance_experiment_is_reproducible_with_same_seed(tmp_path):
    r1 = ff.run_nested_variance_experiment(_vardecomp_args(tmp_path / "a"), outer_n=3, inner_n=3, make_images=False,
                                            gsa_n_mc=64, gsa_n_bootstrap=5)
    r2 = ff.run_nested_variance_experiment(_vardecomp_args(tmp_path / "b"), outer_n=3, inner_n=3, make_images=False,
                                            gsa_n_mc=64, gsa_n_bootstrap=5)
    assert r1["decomposition"]["group_means"] == r2["decomposition"]["group_means"]
    assert r1["global_sensitivity"]["first_order"] == r2["global_sensitivity"]["first_order"]


def test_nested_variance_experiment_produces_decomposition_with_expected_keys(tmp_path):
    args = _vardecomp_args(tmp_path, acoustic_mode="off")
    result = ff.run_nested_variance_experiment(args, outer_n=3, inner_n=3, make_images=False,
                                                gsa_n_mc=64, gsa_n_bootstrap=5)

    decomp = result["decomposition"]
    for key in ("within_group_variance", "between_group_variance", "stochastic_fraction",
                "parametric_fraction", "stochastic_fraction_ci95", "parametric_fraction_ci95"):
        assert key in decomp
    assert result["swept_parameters"] == ["entrainment_alpha"]
    assert os.path.exists(os.path.join(result["experiment_dir"], "vardecomp_summary.json"))

    gsa = result["global_sensitivity"]
    assert gsa["param_names"] == ["entrainment_alpha"]
    assert set(gsa["first_order"]) == {"entrainment_alpha"}
    assert set(gsa["total_order"]) == {"entrainment_alpha"}
    # R² de LOO-CV pode sair negativo para um ajuste ruim (não é um bug,
    # é a própria definição de R²) — só verificamos que é um float finito.
    assert isinstance(gsa["loo_cv_r2"], float) and np.isfinite(gsa["loo_cv_r2"])

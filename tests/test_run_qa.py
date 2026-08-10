"""
Testes de `run_qa.py` — QA automatizada de integridade de ensemble.
Dois níveis testados separadamente: `hard_errors` (bugs inequívocos) e
`soft_flags` (outliers estatísticos, explicitamente NÃO tratados como
bug — ver docs/PHYSICS_MODEL.md §10.7 e a razão: distribuições deste
projeto são caudal-pesadas por construção, um outlier é plausivelmente
um evento raro real, não um erro).

Rodar com: pytest tests/test_run_qa.py -v
"""

import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_qa
import fumarola_field as ff
import plume_physics as pp


def _base_summary(run_dir="run_x", seed=1, n_vents=10, n_up=3, n_down=2, n_eq=1,
                   max_conc=5.0, mean_conc=2.0, acoustic_diag=None):
    return {
        "run_dir": run_dir, "seed": seed, "n_vents": n_vents,
        "prebiotic_summary": {
            "max_concentration_uM": max_conc, "mean_concentration_uM": mean_conc,
            "n_vents_increased_vs_control": n_up, "n_vents_decreased_vs_control": n_down,
            "n_vents_unchanged_vs_control": n_eq,
            "top_hotspot_enrichment_vs_control": max_conc,
        },
        "acoustic_diagnostics": acoustic_diag,
    }


# --------------------------------------------------------------------------
# 1. hard_errors — cada tipo de bug testado isoladamente com dado sintético
# --------------------------------------------------------------------------

def test_clean_summaries_have_no_hard_errors():
    rng = np.random.default_rng(0)
    summaries = [_base_summary(run_dir=f"run_{i}", seed=i, n_vents=10,
                                n_up=int(rng.integers(0, 4)), n_down=int(rng.integers(0, 4)),
                                n_eq=0, max_conc=float(rng.uniform(1, 5)))
                 for i in range(30)]
    result = run_qa.check_run_integrity(summaries)
    assert result["ok"]
    assert result["hard_errors"] == []


def test_detects_nan_concentration():
    summaries = [_base_summary(max_conc=float("nan"))]
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]
    assert any("max_concentration_uM" in e and "NaN" in e for e in result["hard_errors"])


def test_detects_negative_concentration():
    summaries = [_base_summary(max_conc=-1.5)]
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]
    assert any("negativo" in e for e in result["hard_errors"])


def test_detects_zero_vents():
    summaries = [_base_summary(n_vents=0, n_up=0, n_down=0, n_eq=0)]
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]
    assert any("n_vents=0" in e for e in result["hard_errors"])


def test_detects_duplicate_seed():
    summaries = [_base_summary(run_dir="run_a", seed=42), _base_summary(run_dir="run_b", seed=42)]
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]
    assert any("duplicada" in e for e in result["hard_errors"])


def test_detects_count_exceeding_n_vents():
    summaries = [_base_summary(n_vents=5, n_up=4, n_down=3, n_eq=0)]  # 7 > 5
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]
    assert any("EXCEDE" in e for e in result["hard_errors"])


def test_count_below_n_vents_is_not_an_error():
    """Regressão de um bug real encontrado nesta sessão: a primeira
    versão exigia IGUALDADE entre aumentaram+diminuíram+inalterados e
    n_vents — mas essa contagem só inclui vents com enrichment_vs_control
    != None (ver prebiotic.compute_field_hotspots), um SUBCONJUNTO real
    de n_vents. A versão errada disparava hard_error em 100% das runs
    reais dos dois ensembles já existentes do projeto — um sinal claro
    de que a checagem, não o dado, estava errada."""
    summaries = [_base_summary(n_vents=10, n_up=3, n_down=2, n_eq=0)]  # 5 < 10, válido
    result = run_qa.check_run_integrity(summaries)
    assert result["ok"]


def test_detects_negative_acoustic_diagnostic():
    summaries = [_base_summary(acoustic_diag={"gorkov_trap_depth_over_kT": -0.1})]
    result = run_qa.check_run_integrity(summaries)
    assert not result["ok"]


def test_missing_metadata_reported_by_check_experiment_dir_integrity(tmp_path):
    """`check_run_integrity` sozinha não vê runs com metadata.json
    ausente (já filtradas por load_run_summary) — `check_experiment_dir_
    integrity` precisa reportar isso explicitamente, não deixar passar
    em silêncio."""
    exp_dir = tmp_path / "experimento_test"
    exp_dir.mkdir()
    (exp_dir / "run_ok").mkdir()
    (exp_dir / "run_ok" / "metadata.json").write_text(
        '{"run_dir": "run_ok", "seed": 1, "n_vents": 5, '
        '"prebiotic_summary": {"max_concentration_uM": 1.0, "mean_concentration_uM": 1.0, '
        '"n_vents_increased_vs_control": 1, "n_vents_decreased_vs_control": 0, '
        '"n_vents_unchanged_vs_control": 0}, "acoustic_diagnostics": null}', encoding="utf-8")
    (exp_dir / "run_crashed").mkdir()  # sem metadata.json — run interrompida

    result = run_qa.check_experiment_dir_integrity(str(exp_dir))
    assert not result["ok"]
    assert result["n_runs_expected"] == 2
    assert result["n_runs_checked"] == 1
    assert any("ausente" in e for e in result["hard_errors"])


# --------------------------------------------------------------------------
# 2. soft_flags — outliers estatísticos (z-score robusto), separados dos
#    hard_errors, com o dado plantado corretamente identificado
# --------------------------------------------------------------------------

def test_detects_planted_outlier_as_soft_flag_not_hard_error():
    rng = np.random.default_rng(1)
    summaries = [_base_summary(run_dir=f"run_{i}", seed=i,
                                max_conc=float(rng.normal(5.0, 0.3)))
                 for i in range(30)]
    summaries.append(_base_summary(run_dir="run_outlier", seed=999, max_conc=50.0))
    # o campo testado por soft_flags é top_hotspot_enrichment_vs_control,
    # não max_concentration_uM — usa o mesmo valor de max_conc aqui.
    for s in summaries:
        s["prebiotic_summary"]["top_hotspot_enrichment_vs_control"] = s["prebiotic_summary"]["max_concentration_uM"]

    result = run_qa.check_run_integrity(summaries, z_threshold=5.0)
    assert result["ok"]  # outlier não é hard_error
    assert any("run_outlier" in f for f in result["soft_flags"])


def test_no_soft_flags_for_uniform_data():
    summaries = [_base_summary(run_dir=f"run_{i}", seed=i, max_conc=5.0) for i in range(20)]
    result = run_qa.check_run_integrity(summaries)
    assert result["soft_flags"] == []


def test_robust_z_scores_matches_known_formula():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    z = run_qa._robust_z_scores(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    expected = (values - median) / (mad * 1.4826)
    np.testing.assert_allclose(z, expected)


# --------------------------------------------------------------------------
# 3. Integração: ensemble real pequeno (não sintético) roda limpo
# --------------------------------------------------------------------------

def test_real_small_ensemble_passes_qa_clean(tmp_path):
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
        sensitivity_sweep=False,
    )
    result_run = ff.run_experiment(args, n_runs=15, make_images=False)
    result = run_qa.check_experiment_dir_integrity(result_run["experiment_dir"])
    assert result["ok"], result["hard_errors"]
    assert result["n_runs_checked"] == 15

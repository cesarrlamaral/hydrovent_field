"""
Testes de `ensemble_report.py` — relatório estatístico do ensemble aberto
a todo usuário da GUI (sem login, sem interpretação/discussão, sem
imagem representativa de uma run específica). Ver docs/PHYSICS_MODEL.md
§10.4 e a conversa que motivou este módulo: "quero que a gente volte a
ter uma seção de relatório aberto na GUI... apenas o relatório
estatístico... nada de discussão ou formato de artigo... nada de
imagens representativas."

Rodar com: pytest tests/test_ensemble_report.py -v
"""

import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fumarola_field as ff
import ensemble_stats as es
import ensemble_report as er
import i18n
import plume_physics as pp


def _base_args(tmp_path, acoustic_mode="off", sensitivity_sweep=False):
    return argparse.Namespace(
        seed=42, size=33, roughness=0.55, n_clusters=2, vents_min=2, vents_max=3,
        spreading_rate=60.0, local_relief_m=150.0, ocean_depth_baseline_m=2500.0,
        entrainment_alpha=pp.DEFAULT_ALPHA_ENTRAINMENT, stratification_n=pp.DEFAULT_N_BRUNT_VAISALA,
        basin="atlantic", export_plume_profiles=False, outputs_dir=str(tmp_path),
        basename="test", no_3d=True, z_exag=25.0, view_elev=55.0, view_azim=-50.0,
        chimney_scale=1.0, true_scale=False, domain_size_m=1200.0,
        no_dilution=False, no_thermophoresis=True, no_mineral_adsorption=True, no_proton_gradient=True,
        molecule_class="aminoacidos", acoustic_mode=acoustic_mode,
        acoustic_particle_radius_um=None, acoustic_particle_density=None,
        sensitivity_sweep=sensitivity_sweep,
    )


def _run_small_ensemble(tmp_path, n_runs=8, acoustic_mode="off", sensitivity_sweep=False):
    args = _base_args(tmp_path, acoustic_mode=acoustic_mode, sensitivity_sweep=sensitivity_sweep)
    result = ff.run_experiment(args, n_runs=n_runs, make_images=False)
    summaries = result["summaries"]
    pooled = es.load_pooled_hotspots(summaries)
    stats = es.compute_ensemble_stats(summaries, pooled)
    return summaries, stats, pooled, result["experiment_dir"]


# --------------------------------------------------------------------------
# 1. Geração básica — estrutura HTML válida, sem crash
# --------------------------------------------------------------------------

def test_generates_valid_html_file(tmp_path):
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)

    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert content.startswith("<!doctype html>")
    assert content.count("<tr>") == content.count("</tr>")
    assert content.count("<td>") == content.count("</td>")
    assert content.count("<table") == content.count("</table>")


def test_default_output_path_is_inside_experiment_dir(tmp_path):
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(exp_dir)


def test_custom_output_path_is_respected(tmp_path):
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    custom = str(tmp_path / "custom_report.html")
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir,
                                                    out_path=custom, n_bootstrap=50)
    assert path == custom
    assert os.path.exists(custom)


# --------------------------------------------------------------------------
# 2. Conteúdo esperado sempre presente
# --------------------------------------------------------------------------

def test_report_contains_descriptive_stats_and_figure_and_runs_table(tmp_path):
    i18n.set_language("en")
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()

    assert "Ensemble Statistical Report" in content
    assert "Descriptive statistics" in content
    assert "data:image/png;base64" in content
    assert "Per-run results" in content
    assert "Figure 1" in content


def test_report_does_not_contain_discussion_or_article_framing(tmp_path):
    """Restrição explícita do usuário: nada de discussão/hipótese/moldura
    de artigo neste relatório. Checa seções/títulos de estilo artigo (não uma busca ingênua por
    substring — a própria frase de aviso deste relatório MENCIONA a
    palavra "discussion" para dizer que não tem uma, o que é esperado)."""
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read().lower()

    for forbidden_heading in ("<h2>abstract", "<h2>discussion", "<h2>hypothesis",
                               "<h2>recommended next steps", "<h1>abstract"):
        assert forbidden_heading not in content


def test_report_does_not_embed_representative_run_images(tmp_path):
    """Restrição explícita do usuário: "nada de imagens representativas"
    — o relatório não deve referenciar os PNGs de uma run específica
    (topview/3D/artístico), só o gráfico agregado do ensemble."""
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()

    for s in summaries:
        for key in ("png_2d_path", "png_3d_path", "png_truescale_path", "png_artistic_path",
                    "png_hotspots_path", "png_acoustic_path"):
            path_val = s.get(key)
            if path_val:
                assert os.path.basename(path_val) not in content


def test_report_language_follows_i18n_setting(tmp_path):
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)

    i18n.set_language("pt")
    path_pt = er.generate_ensemble_statistics_report(
        summaries, pooled, exp_dir, out_path=str(tmp_path / "pt.html"), n_bootstrap=50)
    content_pt = open(path_pt, encoding="utf-8").read()
    assert "Relatório Estatístico do Ensemble" in content_pt

    i18n.set_language("en")
    path_en = er.generate_ensemble_statistics_report(
        summaries, pooled, exp_dir, out_path=str(tmp_path / "en.html"), n_bootstrap=50)
    content_en = open(path_en, encoding="utf-8").read()
    assert "Ensemble Statistical Report" in content_en


# --------------------------------------------------------------------------
# 3. Tabela de drivers — só aparece quando há dados suficientes de
#    --sensitivity-sweep (mesmo gate de report._relevance_drivers)
# --------------------------------------------------------------------------

def test_driver_table_absent_without_sensitivity_sweep(tmp_path):
    i18n.set_language("en")
    summaries, stats, pooled, exp_dir = _run_small_ensemble(
        tmp_path, n_runs=8, acoustic_mode="off", sensitivity_sweep=False)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()
    assert "driver analysis" not in content.lower()


def test_driver_table_present_with_enough_sensitivity_sweep_data(tmp_path):
    i18n.set_language("en")
    summaries, stats, pooled, exp_dir = _run_small_ensemble(
        tmp_path, n_runs=25, acoustic_mode="particle_trap", sensitivity_sweep=True)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()
    assert "driver analysis" in content.lower()
    assert "entrainment_alpha" in content
    assert "agg_radius_um" in content


def test_driver_table_helper_returns_none_below_minimum_rows():
    summaries = [{"sensitivity_sweep": True, "entrainment_alpha": 0.1, "n_vents": 5,
                  "acoustic_diagnostics": None,
                  "prebiotic_summary": {"top_hotspot_enrichment_vs_control": 2.0}}] * 5
    assert er._driver_table(summaries) is None


# --------------------------------------------------------------------------
# 4. build_ensemble_charts_figure — reaproveitado pela GUI (Tkinter) e
#    pelo relatório (PNG embutido); testado aqui isoladamente da GUI.
# --------------------------------------------------------------------------

def test_build_ensemble_charts_figure_has_four_panels(tmp_path):
    _, stats, _, _ = _run_small_ensemble(tmp_path)
    fig = er.build_ensemble_charts_figure(stats)
    assert len(fig.axes) == 4


# --------------------------------------------------------------------------
# 5. Decomposição de variância / Sobol' — só aparecem quando
#    experiment_dir vem de --variance-decomposition (vardecomp_summary.json
#    presente); ausentes num ensemble normal.
# --------------------------------------------------------------------------

def _run_small_vardecomp(tmp_path, outer_n=4, inner_n=3, acoustic_mode="particle_trap"):
    args = _base_args(tmp_path, acoustic_mode=acoustic_mode, sensitivity_sweep=False)
    result = ff.run_nested_variance_experiment(args, outer_n, inner_n, make_images=False,
                                                gsa_n_mc=64, gsa_n_bootstrap=5)
    exp_dir = result["experiment_dir"]
    run_dirs = ff.find_run_dirs(exp_dir)
    summaries = [ff.load_run_summary(rd) for rd in run_dirs]
    pooled = es.load_pooled_hotspots(summaries)
    return summaries, pooled, exp_dir, result


def test_vardecomp_sections_absent_for_normal_ensemble(tmp_path):
    i18n.set_language("en")
    summaries, stats, pooled, exp_dir = _run_small_ensemble(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()
    assert "Variance decomposition" not in content
    assert "Sobol" not in content


def test_vardecomp_sections_present_for_nested_experiment(tmp_path):
    i18n.set_language("en")
    summaries, pooled, exp_dir, result = _run_small_vardecomp(tmp_path)
    path = er.generate_ensemble_statistics_report(summaries, pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()

    assert "Variance decomposition" in content
    assert "Global sensitivity" in content
    assert "Sobol" in content
    for name in result["swept_parameters"]:
        assert name in content
    assert content.count("<tr>") == content.count("</tr>")
    assert content.count("<td>") == content.count("</td>")


def test_vardecomp_report_reads_from_disk_not_memory(tmp_path):
    """A leitura é do vardecomp_summary.json gravado em disco (não de um
    objeto em memória passado pelo chamador) — precisa funcionar mesmo
    "reabrindo" o experimento numa chamada separada, simulando o caso
    real de o usuário rodar via CLI e depois abrir na GUI depois."""
    i18n.set_language("en")
    summaries, pooled, exp_dir, _ = _run_small_vardecomp(tmp_path)

    # "reabre" só com os dados que find_run_dirs/load_run_summary dariam,
    # sem nenhuma referência ao resultado retornado por run_nested_variance_experiment.
    import fumarola_field as ff2
    run_dirs = ff2.find_run_dirs(exp_dir)
    reopened_summaries = [ff2.load_run_summary(rd) for rd in run_dirs]
    reopened_pooled = es.load_pooled_hotspots(reopened_summaries)

    path = er.generate_ensemble_statistics_report(reopened_summaries, reopened_pooled, exp_dir, n_bootstrap=50)
    content = open(path, encoding="utf-8").read()
    assert "Variance decomposition" in content


def test_read_vardecomp_summary_returns_none_when_absent(tmp_path):
    assert er._read_vardecomp_summary(str(tmp_path)) is None

"""
Relatório estatístico do ensemble — HTML autocontido, aberto a QUALQUER
usuário da GUI (sem login), gerado a partir dos MESMOS dados já
computados por `ensemble_stats.py`. Deliberadamente SEM nenhuma camada
de interpretação/discussão científica, moldura de manuscrito ou imagem
representativa de uma run específica (topview/3D/artístico) — isso é
`report.py` (gitignored, texto interpretativo específico do autor,
atrás de senha de Administrador via `relatorios_admin.py`, não tocado
por este módulo). Aqui: só tabelas descritivas, os mesmos 4 gráficos já
mostrados ao vivo na aba de estatísticas da GUI (com legenda completa,
não só o título curto de eixo), a tabela por-run, e — quando aplicável —
uma tabela de análise de drivers (regressão multivariada).

Módulo tracked no git (ao contrário de report.py): não carrega texto
interpretativo específico do autor nem lógica de senha — é "software
genérico" pela mesma régua já usada no CONTRIBUTING.md/`.gitignore` do
projeto para decidir o que fica de fora do repositório público.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure

import driver_regression as dr
import variance_decomposition as vd
from i18n import t


# --------------------------------------------------------------------------
# 1. Gráficos — MESMA lógica antes embutida em gui.py._render_ensemble_charts,
#    extraída aqui para ser reaproveitada tanto pela GUI (Tkinter) quanto
#    pelo relatório (PNG embutido) sem duplicar/divergir.
# --------------------------------------------------------------------------

def build_ensemble_charts_figure(stats: dict) -> Figure:
    fig = Figure(figsize=(11, 4.6), dpi=100)
    fig.patch.set_facecolor("#fcfcfb")
    axes = fig.subplots(1, 4)

    pooled_enrich = stats["pooled_enrich_array"]
    top_enrich = stats["top_enrich_array"]
    type_counts = stats["top_hotspot_type_counts"]

    ax = axes[0]
    if pooled_enrich.size:
        ax.hist(np.log2(pooled_enrich), bins=30, color="#d95926")
        ax.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title(t("chart_enrichment_hist_title"), fontsize=9)
    ax.set_xlabel(t("chart_log2_xlabel"), fontsize=8)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    if top_enrich.size:
        ax.hist(top_enrich, bins=min(20, max(3, top_enrich.size)), color="#c98500")
        ax.axvline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title(t("chart_top_hist_title"), fontsize=9)
    ax.set_xlabel(t("chart_xcontrol_xlabel"), fontsize=8)
    ax.tick_params(labelsize=7)

    ax = axes[2]
    vtypes = list(type_counts.keys())
    counts = [type_counts[k] for k in vtypes]
    ax.bar([k.replace("_", "\n") for k in vtypes], counts, color="#2a78d6")
    ax.set_title(t("chart_type_bar_title"), fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[3]
    n_vents_for_top_enrich = stats["n_vents_for_top_enrich_array"]
    if top_enrich.size:
        ax.scatter(n_vents_for_top_enrich, top_enrich, s=14, color="#1baf7a", alpha=0.7)
        ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title(t("chart_scatter_title"), fontsize=9)
    ax.set_xlabel(t("chart_scatter_xlabel"), fontsize=8)
    ax.set_ylabel(t("chart_xcontrol_xlabel"), fontsize=8)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def _fig_to_base64_png(fig: Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# 2. Metadados do experimento (lidos do disco — experiment_metadata.json
#    já gravado por fumarola_field.run_experiment, tolerante a ausência)
# --------------------------------------------------------------------------

def _read_experiment_metadata(experiment_dir: str) -> dict:
    path = os.path.join(experiment_dir, "experiment_metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _read_vardecomp_summary(experiment_dir: str) -> Optional[dict]:
    """Lê `vardecomp_summary.json` (gravado por `fumarola_field.
    run_nested_variance_experiment`) se `experiment_dir` for uma pasta de
    decomposição de variância aninhada — None caso contrário (experimento
    de ensemble normal, sem esse arquivo). Ler do DISCO em vez de exigir
    que o chamador passe o resultado em memória funciona tanto para um
    experimento recém-rodado quanto para um reaberto depois."""
    path = os.path.join(experiment_dir, "vardecomp_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------
# 3. Tabela 1 — descritivas com IC 95% por bootstrap (reaproveita
#    ensemble_stats.describe(), pede n_bootstrap explicitamente aqui —
#    custo aceitável só na hora de gerar o relatório, não em toda
#    atualização ao vivo da aba de estatísticas)
# --------------------------------------------------------------------------

def _fmt_ci(point: float, ci: tuple) -> str:
    lo, hi = ci
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _table1_rows(stats: dict) -> List[tuple]:
    entries = [
        (t("stat_pooled_enrich"), stats["pooled_enrichment"]),
        (t("stat_top_hotspot"), stats["top_hotspot_enrichment"]),
        (t("stat_max_conc"), stats["max_concentration"]),
        (t("stat_mean_conc"), stats["mean_concentration"]),
        (t("stat_n_vents"), stats["n_vents"]),
    ]
    rows = []
    for label, d in entries:
        mean_txt = _fmt_ci(d["mean"], d["mean_ci95"]) if "mean_ci95" in d else f"{d['mean']:.3f}"
        median_txt = _fmt_ci(d["median"], d["median_ci95"]) if "median_ci95" in d else f"{d['median']:.3f}"
        skew_txt = "n/a" if not np.isfinite(d["skewness"]) else f"{d['skewness']:.2f}"
        rows.append((label, d["n"], mean_txt, median_txt, f"{d['std']:.3f}",
                     f"{d['q1']:.3f}–{d['q3']:.3f}", skew_txt))
    return rows


# --------------------------------------------------------------------------
# 4. Tabela 3 (condicional) — análise de drivers multivariada, só quando
#    o ensemble rodou com --sensitivity-sweep (mesmo gate de
#    report._relevance_drivers) — reaproveita driver_regression.py.
# --------------------------------------------------------------------------

def _driver_table(summaries: List[dict]) -> Optional[dict]:
    rows = []
    for s in summaries:
        if not s.get("sensitivity_sweep"):
            continue
        alpha = s.get("entrainment_alpha")
        n_vents = s.get("n_vents")
        diag = s.get("acoustic_diagnostics") or {}
        agg = (diag.get("particle_classes") or {}).get("near_field_fe_oxyhydroxide_aggregate")
        response = vd.default_response_value(s)
        if alpha is None or n_vents is None or response is None:
            continue
        if agg is not None:
            rows.append((alpha, agg["radius_um"], agg["density_kg_m3"], n_vents, response))
        else:
            rows.append((alpha, n_vents, response))

    if len(rows) < 20:
        return None
    arr = np.array(rows, dtype=float)
    has_acoustic = arr.shape[1] == 5
    if has_acoustic:
        X, y = arr[:, :4], arr[:, 4]
        names = ["entrainment_alpha", "agg_radius_um", "agg_density_kg_m3", "n_vents"]
        response_name = "gorkov_trap_depth_over_kT"
    else:
        X, y = arr[:, :2], arr[:, 2]
        names = ["entrainment_alpha", "n_vents"]
        response_name = "top_hotspot_enrichment_vs_control"
    if np.std(y) == 0:
        return None
    try:
        result = dr.rank_transform_regression(X, y, names, n_bootstrap=1000, rng=np.random.default_rng(0))
    except (ValueError, np.linalg.LinAlgError):
        return None
    result["response_name"] = response_name
    return result


# --------------------------------------------------------------------------
# 4b. Seções de decomposição de variância / Sobol' — só quando
#     `experiment_dir` é uma pasta de --variance-decomposition (ver
#     _read_vardecomp_summary). Reaproveita os dicts já calculados por
#     fumarola_field.run_nested_variance_experiment, gravados em
#     vardecomp_summary.json — nenhum cálculo novo acontece aqui.
# --------------------------------------------------------------------------

def _vardecomp_section_html(vardecomp: dict) -> str:
    d = vardecomp["decomposition"]
    params_txt = html.escape(" + ".join(vardecomp["swept_parameters"]))
    rows = [
        (t("ensemble_report_vardecomp_row_stochastic"), d["stochastic_fraction"], d["stochastic_fraction_ci95"]),
        (t("ensemble_report_vardecomp_row_parametric"), d["parametric_fraction"], d["parametric_fraction_ci95"]),
    ]
    parts = [
        f"<h2>{t('ensemble_report_vardecomp_title')}</h2>\n"
        f'<p class="note">{t("ensemble_report_vardecomp_note", outer=vardecomp["outer_n"], inner=vardecomp["inner_n"], params=params_txt)}</p>\n'
        "<table>\n<tr>"
        f"<th>{t('ensemble_report_vardecomp_th_component')}</th>"
        f"<th>{t('ensemble_report_vardecomp_th_fraction')}</th></tr>\n"
    ]
    for label, frac, ci in rows:
        lo, hi = ci
        parts.append(f"<tr><td>{label}</td><td>{frac:.3f} [{lo:.3f}, {hi:.3f}]</td></tr>\n")
    parts.append("</table>\n")
    if d.get("between_group_variance_was_clipped"):
        parts.append(f'<p class="note">{t("ensemble_report_vardecomp_clipped_note")}</p>\n')
    return "".join(parts)


def _sobol_section_html(vardecomp: dict) -> str:
    gs = vardecomp.get("global_sensitivity")
    if gs is None:
        return ""
    warning = f' <strong>{t("ensemble_report_sobol_warning")}</strong>' if gs["loo_cv_r2_warning"] else ""
    parts = [
        f"<h2>{t('ensemble_report_sobol_title')}</h2>\n"
        f'<p class="note">{t("ensemble_report_sobol_note", loo_r2=gs["loo_cv_r2"])}{warning}</p>\n'
        "<table>\n<tr>"
        f"<th>{t('ensemble_report_sobol_th_param')}</th>"
        f"<th>{t('ensemble_report_sobol_th_s1')}</th>"
        f"<th>{t('ensemble_report_sobol_th_st')}</th></tr>\n"
    ]
    for name in gs["param_names"]:
        s1, s1_lo, s1_hi = gs["first_order"][name], *gs["first_order_ci95"][name]
        st, st_lo, st_hi = gs["total_order"][name], *gs["total_order_ci95"][name]
        parts.append(f"<tr><td>{html.escape(name)}</td>"
                      f"<td>{s1:.3f} [{s1_lo:.3f}, {s1_hi:.3f}]</td>"
                      f"<td>{st:.3f} [{st_lo:.3f}, {st_hi:.3f}]</td></tr>\n")
    parts.append("</table>\n")
    return "".join(parts)


# --------------------------------------------------------------------------
# 5. Geração do HTML
# --------------------------------------------------------------------------

def generate_ensemble_statistics_report(summaries: List[dict], pooled: List[dict],
                                         experiment_dir: str, out_path: Optional[str] = None,
                                         n_bootstrap: int = 1000,
                                         rng: Optional[np.random.Generator] = None) -> str:
    """
    Gera o relatório e devolve o caminho do arquivo HTML salvo (dentro de
    `experiment_dir` por padrão). SEMPRE recalcula as estatísticas do
    zero com `n_bootstrap` IC's — não aceita um `stats` pré-computado
    porque o `stats` tipicamente já disponível (ex. o que a aba da GUI
    mantém ao vivo) usa `n_bootstrap=0` por custo (ver ensemble_stats.py)
    e portanto nunca teria os IC's que este relatório precisa; aceitar o
    parâmetro mesmo assim seria uma API enganosa (pareceria reaproveitado,
    mas seria ignorado). Custo extra é aceitável aqui: só roda quando o
    usuário pede o relatório, não a cada atualização da aba.
    """
    if rng is None:
        rng = np.random.default_rng()
    import ensemble_stats as es
    stats_with_ci = es.compute_ensemble_stats(summaries, pooled, n_bootstrap=n_bootstrap, rng=rng)

    meta = _read_experiment_metadata(experiment_dir)
    vardecomp = _read_vardecomp_summary(experiment_dir)
    exp_name = html.escape(os.path.basename(os.path.normpath(experiment_dir)))

    fig = build_ensemble_charts_figure(stats_with_ci)
    fig_b64 = _fig_to_base64_png(fig)

    modules = meta.get("prebiotic_modules") or {}
    active_modules = ", ".join(k for k, v in modules.items() if v and k != "acoustic_mode") or "—"

    # run_nested_variance_experiment (--variance-decomposition) grava
    # vardecomp_summary.json em vez de experiment_metadata.json — mesmos
    # dados de proveniência (seed base, modo acústico), schema diferente.
    if vardecomp is not None:
        base_seed_txt = html.escape(str(vardecomp.get("base_seed", "n/a")))
        sweep_txt = t("ensemble_report_meta_vardecomp_value", outer=vardecomp["outer_n"], inner=vardecomp["inner_n"])
        acoustic_txt = html.escape(str(vardecomp.get("acoustic_mode", "off")))
    else:
        base_seed_txt = html.escape(str(meta.get("base_seed", "n/a")))
        sweep_txt = t("ensemble_report_yes") if meta.get("sensitivity_sweep") else t("ensemble_report_no")
        acoustic_txt = html.escape(str(modules.get("acoustic_mode", "off")))

    meta_rows = [
        (t("ensemble_report_meta_experiment"), exp_name),
        (t("ensemble_report_meta_generated"), datetime.now().isoformat(timespec="seconds")),
        (t("ensemble_report_meta_base_seed"), base_seed_txt),
        (t("ensemble_report_meta_n_runs"), str(len(summaries))),
        (t("ensemble_report_meta_sweep"), sweep_txt),
        (t("ensemble_report_meta_acoustic"), acoustic_txt),
        (t("ensemble_report_meta_molecule"), html.escape(str(summaries[0].get("molecule_class_label", "n/a")))
         if summaries else "n/a"),
        (t("ensemble_report_meta_basin"), html.escape(str(meta.get("basin", "n/a")))),
        (t("ensemble_report_meta_modules"), html.escape(active_modules)),
    ]

    table1_rows = _table1_rows(stats_with_ci)
    type_counts = stats_with_ci["top_hotspot_type_counts"]

    driver = _driver_table(summaries)

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{t("ensemble_report_title")} — {exp_name}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 980px;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
h1 {{ font-size: 1.5rem; border-bottom: 2px solid #2a78d6; padding-bottom: 0.3rem; }}
h2 {{ font-size: 1.15rem; margin-top: 2rem; color: #2a4d78; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.88rem; }}
th, td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: right; }}
th {{ background: #f0f3f7; text-align: center; }}
td:first-child, th:first-child {{ text-align: left; }}
.meta-table td:first-child {{ font-weight: 600; width: 220px; }}
.meta-table td {{ text-align: left; }}
.caption {{ font-size: 0.85rem; color: #444; margin-top: 0.4rem; }}
.note {{ font-size: 0.82rem; color: #666; font-style: italic; }}
.intro {{ background: #f7f9fc; border-left: 3px solid #2a78d6; padding: 0.6rem 1rem; }}
img {{ max-width: 100%; }}
.runs-table-wrap {{ max-height: 480px; overflow-y: auto; border: 1px solid #ddd; }}
.runs-table-wrap table {{ margin: 0; }}
.runs-table-wrap th {{ position: sticky; top: 0; background: #f0f3f7; }}
</style></head><body>
<h1>{t("ensemble_report_title")}</h1>
<p class="intro">{t("ensemble_report_intro")}</p>

<h2>{html.escape(exp_name)}</h2>
<table class="meta-table">
"""]
    for label, value in meta_rows:
        parts.append(f"<tr><td>{label}</td><td>{value}</td></tr>\n")
    parts.append("</table>\n")

    parts.append(f"<h2>{t('ensemble_report_table1_title')}</h2>\n<table>\n<tr>"
                  f"<th>{t('ensemble_report_table1_th_metric')}</th>"
                  f"<th>{t('ensemble_report_table1_th_n')}</th>"
                  f"<th>{t('ensemble_report_table1_th_mean')}</th>"
                  f"<th>{t('ensemble_report_table1_th_median')}</th>"
                  f"<th>{t('ensemble_report_table1_th_std')}</th>"
                  f"<th>{t('ensemble_report_table1_th_iqr')}</th>"
                  f"<th>{t('ensemble_report_table1_th_skew')}</th></tr>\n")
    for label, n, mean_txt, median_txt, std_txt, iqr_txt, skew_txt in table1_rows:
        parts.append(f"<tr><td>{html.escape(label)}</td><td>{n}</td><td>{mean_txt}</td>"
                      f"<td>{median_txt}</td><td>{std_txt}</td><td>{iqr_txt}</td><td>{skew_txt}</td></tr>\n")
    parts.append("</table>\n")
    parts.append(f'<p class="note">{t("ensemble_report_table1_note", n_bootstrap=n_bootstrap)}</p>\n')

    type_txt = " | ".join(f"{html.escape(k)}: {v}" for k, v in type_counts.items())
    parts.append(f"<p><strong>{t('stat_top_type')}</strong> {html.escape(type_txt)}</p>\n")

    parts.append(f'<h2>{t("ensemble_report_figure_title")}</h2>\n'
                  f'<img src="data:image/png;base64,{fig_b64}" alt="ensemble charts">\n'
                  f'<p class="caption">{t("ensemble_report_figure_caption")}</p>\n')

    if vardecomp is not None:
        parts.append(_vardecomp_section_html(vardecomp))
        parts.append(_sobol_section_html(vardecomp))

    if driver is not None:
        parts.append(f"<h2>{t('ensemble_report_table3_title')}</h2>\n"
                      f'<p class="note">{t("ensemble_report_table3_note", response_name=driver["response_name"])}</p>\n'
                      f"<table>\n<tr><th>{t('ensemble_report_table3_th_predictor')}</th>"
                      f"<th>{t('ensemble_report_table3_th_coef')}</th>"
                      f"<th>{t('ensemble_report_table3_th_p')}</th>"
                      f"<th>{t('ensemble_report_table3_th_p_holm')}</th>"
                      f"<th>{t('ensemble_report_table3_th_vif')}</th></tr>\n")
        for name in driver["predictor_names"]:
            parts.append(f"<tr><td>{html.escape(name)}</td>"
                          f"<td>{driver['coefficients'][name]:+.3f}</td>"
                          f"<td>{driver['p_values'][name]:.2e}</td>"
                          f"<td>{driver['p_values_holm'][name]:.2e}</td>"
                          f"<td>{driver['vif'][name]:.2f}</td></tr>\n")
        parts.append("</table>\n")

    parts.append(f"<h2>{t('ensemble_report_table2_title')}</h2>\n"
                  '<div class="runs-table-wrap"><table>\n<tr>'
                  f"<th>{t('col_run')}</th><th>{t('col_seed')}</th><th>{t('col_n_vents')}</th>"
                  f"<th>{t('col_top_enrich')}</th><th>{t('col_mean_enrich')}</th>"
                  f"<th>{t('col_n_up')}</th><th>{t('col_n_down')}</th>"
                  f"<th>{t('col_top_type')}</th></tr>\n")
    for s in summaries:
        ps = s["prebiotic_summary"]
        top_enrich = ps["top_hotspot_enrichment_vs_control"]
        mean_enrich = ps["mean_enrichment_vs_control"]
        top_enrich_td = f"<td>{top_enrich:.3f}</td>" if top_enrich is not None else "<td>n/a</td>"
        mean_enrich_td = f"<td>{mean_enrich:.3f}</td>" if mean_enrich is not None else "<td>n/a</td>"
        parts.append(
            "<tr>"
            f"<td>{html.escape(os.path.basename(s['run_dir']))}</td>"
            f"<td>{s['seed']}</td><td>{s['n_vents']}</td>"
            f"{top_enrich_td}{mean_enrich_td}"
            f"<td>{ps['n_vents_increased_vs_control']}</td><td>{ps['n_vents_decreased_vs_control']}</td>"
            f"<td>{html.escape(str(ps['top_hotspot_vent_type']))}</td></tr>\n"
        )
    parts.append("</table></div>\n")

    parts.append("</body></html>\n")

    html_content = "".join(parts)
    if out_path is None:
        out_path = os.path.join(experiment_dir, "ensemble_statistics_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return out_path

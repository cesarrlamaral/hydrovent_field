"""
Estatísticas agregadas de um ensemble de runs (pooling de hotspots,
descritivas por run). Módulo puro, sem dependência de Tkinter, para que
a lógica possa ser reaproveitada por qualquer ferramenta que precise
processar os mesmos dados de um experimento.
"""

from __future__ import annotations

import csv
import os
import warnings
from collections import Counter
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats


def load_pooled_hotspots(summaries: list[dict]) -> list[dict]:
    records = []
    for s in summaries:
        path = s.get("hotspots_csv_path")
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["final_concentration_uM"] = float(row["final_concentration_uM"])
                row["temperature_c"] = float(row["temperature_c"])
                row["enrichment_vs_control"] = (
                    float(row["enrichment_vs_control"]) if row.get("enrichment_vs_control") else None
                )
                records.append(row)
    return records


_CONTINUOUS_STAT_NAMES = ("mean", "std", "median", "q1", "q3", "iqr", "mad", "mad_scaled",
                          "skewness", "kurtosis", "mean_median_gap_over_iqr")


def _bootstrap_point_estimates(boot: np.ndarray, arr_size: int) -> dict:
    """`boot`: (n_bootstrap, arr_size) — uma reamostragem por linha.
    Recalcula, TOTALMENTE VETORIZADO (sem laço Python por reamostragem —
    importante porque arrays pooled de um ensemble grande podem ter
    dezenas de milhares de pontos, e bootstrap tipicamente usa milhares
    de reamostragens), as MESMAS estatísticas contínuas de `describe()`,
    linha a linha."""
    mean = boot.mean(axis=1)
    median = np.median(boot, axis=1)
    std = boot.std(axis=1)
    q1, q3 = np.percentile(boot, [25, 75], axis=1)
    iqr = q3 - q1
    mad = np.median(np.abs(boot - median[:, None]), axis=1)
    mad_scale = 1.0 / scipy_stats.norm.ppf(0.75)

    if arr_size < 3:
        skewness = np.full(boot.shape[0], np.nan)
    else:
        # mesma guarda de RuntimeWarning de `describe()`: reamostragens
        # que calham em variância zero (plausível com valores repetidos/
        # amostra pequena) produzem 0/0 — suprimido aqui de propósito,
        # não escondendo um problema real (o resultado NaN é o correto).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = scipy_stats.skew(boot, axis=1, bias=False)
    if arr_size < 4:
        kurtosis = np.full(boot.shape[0], np.nan)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            kurtosis = scipy_stats.kurtosis(boot, axis=1, bias=False)

    gap = np.where(iqr > 0, (mean - median) / np.where(iqr > 0, iqr, 1.0), 0.0)

    return {
        "mean": mean, "std": std, "median": median, "q1": q1, "q3": q3, "iqr": iqr,
        "mad": mad, "mad_scaled": mad * mad_scale,
        "skewness": skewness, "kurtosis": kurtosis, "mean_median_gap_over_iqr": gap,
    }


def describe(arr: np.ndarray, n_bootstrap: int = 0, rng: Optional[np.random.Generator] = None) -> dict:
    """
    Descritivas de `arr`. Além de n/mean/std/min/median/max (chaves
    PRÉ-EXISTENTES, semântica inalterada — GUI e report.py já leem essas
    diretamente), adiciona estatísticas ROBUSTAS: distribuições deste
    projeto são conhecidamente caudal-pesadas (altura de chaminé de longa
    cauda, evento raro de Gor'kov — ver docs/PHYSICS_MODEL.md §7.8.1), e
    mean/std sozinhos podem enganar sobre o formato real da distribuição.

    - `q1`/`q3`/`iqr`: quartis e amplitude interquartil (percentil,
      definição elementar) — medida de espalhamento robusta a outliers,
      ao contrário de `std`.
    - `mad`/`mad_scaled`: desvio absoluto mediano em torno da mediana
      (`median(|x - median(x)|)`) — medida de espalhamento robusta
      clássica (Rousseeuw & Croux, 1993, "Alternatives to the median
      absolute deviation," JASA 88(424), 1273-1283). `mad_scaled` aplica
      o fator 1/Φ⁻¹(3/4) ≈ 1,4826 (Φ⁻¹ = quantil da normal padrão) que
      torna o MAD um estimador CONSISTENTE do desvio-padrão sob
      normalidade — comparável diretamente a `std`, mas sem o peso
      desproporcional de outliers que `std` tem (derivação: sob X~N(μ,σ²),
      E[|X-mediana|] = σ·Φ⁻¹(3/4), então dividir por essa constante
      recupera σ; um fato matemático verificável, não um número
      arbitrário).
    - `skewness`/`kurtosis`: coeficiente de assimetria e curtose em
      excesso (normal=0), estimador ajustado de Fisher-Pearson
      (`scipy.stats.skew`/`kurtosis`, `bias=False` — correção de viés de
      amostra finita, ver documentação do scipy). Quantificam
      diretamente O QUANTO a distribuição se desvia de simétrica/normal
      — não é preciso adivinhar a partir de mean/std.
    - `mean_median_gap_over_iqr`: `(mean-median)/iqr` — o quanto a média
      está sendo "puxada" da mediana por assimetria/outliers, na escala
      do espalhamento típico dos dados. Perto de 0 → mean≈mediana
      (distribuição aproximadamente simétrica); afastado de 0 → mean e
      mediana contam histórias diferentes, prefira mediana/IQR na
      interpretação. Diagnóstico AUTOEXPLICATIVO (razão de duas
      quantidades já reportadas), não depende de um limiar externo
      memorizado.

    `skewness`/`kurtosis` retornam NaN para n<3/n<4 respectivamente —
    abaixo disso o valor calculado é matematicamente degenerado (ex.:
    assimetria de exatamente 2 pontos é SEMPRE 0 por simetria, não
    porque a distribuição real seja simétrica) em vez de informativo;
    NaN é mais honesto que um zero enganoso.

    `n_bootstrap`: se >0 (padrão 0 — compatível com todo chamador
    existente, nenhuma mudança de comportamento sem pedir), adiciona um
    IC 95% por bootstrap de casos (Efron & Tibshirani, 1993, "An
    Introduction to the Bootstrap," Chapman & Hall) para CADA estatística
    contínua (`mean_ci95`, `median_ci95`, `std_ci95`, `q1_ci95`,
    `q3_ci95`, `iqr_ci95`, `mad_ci95`, `mad_scaled_ci95`,
    `skewness_ci95`, `kurtosis_ci95`, `mean_median_gap_over_iqr_ci95`) —
    até aqui só a fração de eventos raros tinha IC (`_wilson_ci95` em
    report.py, para a proporção binária), toda estatística CONTÍNUA era
    reportada como número nu, sem incerteza. Totalmente vetorizado (uma
    única matriz `(n_bootstrap, n)` de reamostragens, sem laço Python) —
    necessário porque arrays pooled de ensembles grandes têm dezenas de
    milhares de pontos.
    """
    if arr.size == 0:
        result = {
            "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0,
            "q1": 0.0, "q3": 0.0, "iqr": 0.0, "mad": 0.0, "mad_scaled": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "mean_median_gap_over_iqr": 0.0,
        }
        if n_bootstrap > 0:
            for name in _CONTINUOUS_STAT_NAMES:
                result[f"{name}_ci95"] = (0.0, 0.0)
        return result
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q1, q3 = (float(v) for v in np.percentile(arr, [25, 75]))
    iqr = q3 - q1
    mad = float(np.median(np.abs(arr - median)))
    mad_scale = 1.0 / scipy_stats.norm.ppf(0.75)  # = 1/Phi^-1(3/4) ~ 1.4826

    # variância zero (todos os valores idênticos) torna assimetria/curtose
    # matematicamente 0/0 — scipy já devolve NaN corretamente nesse caso,
    # mas emite RuntimeWarning de "cancelamento catastrófico" a cada
    # chamada; a guarda evita o warning sem mudar o resultado (mesmo NaN).
    zero_variance = np.std(arr) == 0
    skewness = float("nan") if (arr.size < 3 or zero_variance) else float(scipy_stats.skew(arr, bias=False))
    kurtosis = float("nan") if (arr.size < 4 or zero_variance) else float(scipy_stats.kurtosis(arr, bias=False))

    result = {
        "n": int(arr.size), "mean": mean, "std": float(np.std(arr)),
        "min": float(np.min(arr)), "median": median, "max": float(np.max(arr)),
        "q1": q1, "q3": q3, "iqr": iqr,
        "mad": mad, "mad_scaled": mad * mad_scale,
        "skewness": skewness, "kurtosis": kurtosis,
        "mean_median_gap_over_iqr": (mean - median) / iqr if iqr > 0 else 0.0,
    }

    if n_bootstrap > 0:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
        boot_estimates = _bootstrap_point_estimates(arr[idx], arr.size)
        for name in _CONTINUOUS_STAT_NAMES:
            vals = boot_estimates[name]
            # todo-NaN acontece de verdade (ex.: arr original já tem
            # variância zero -> skewness/kurtosis são NaN em TODA
            # reamostragem) — resultado correto é um IC NaN/NaN, o
            # warning "All-NaN slice" do numpy é esperado, não um erro.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            result[f"{name}_ci95"] = (float(lo), float(hi))
        result["n_bootstrap"] = n_bootstrap

    return result


def compute_ensemble_stats(summaries: list[dict], pooled: list[dict], n_bootstrap: int = 0,
                            rng: Optional[np.random.Generator] = None) -> dict:
    """
    `n_bootstrap`: repassado a cada `describe()` interno (padrão 0 —
    comportamento idêntico ao de antes desta mudança, nenhum consumidor
    existente é afetado sem pedir explicitamente). Custo REAL medido:
    ~8s para n_bootstrap=2000 num array pooled de 30 mil pontos (a
    reamostragem em si é rápida; ordenar 2000×30000 elementos para
    mediana/percentis é o que domina o tempo) — arrays por-run (ex.
    `n_vents`, tipicamente centenas a milhares de pontos) são bem mais
    rápidos. Por isso o padrão é 0 (desligado): ligar por padrão em
    TODO `compute_ensemble_stats()` deixaria a aba de estatísticas da
    GUI/geração de relatório sensivelmente mais lenta em ensembles
    grandes sem o usuário ter pedido.
    """
    max_concs = np.array([s["prebiotic_summary"]["max_concentration_uM"] for s in summaries])
    mean_concs = np.array([s["prebiotic_summary"]["mean_concentration_uM"] for s in summaries])
    n_vents = np.array([s["n_vents"] for s in summaries])
    top_types = [s["prebiotic_summary"]["top_hotspot_vent_type"] for s in summaries]
    pooled_concs = np.array([r["final_concentration_uM"] for r in pooled]) if pooled else np.array([])

    top_enrich_raw = [s["prebiotic_summary"]["top_hotspot_enrichment_vs_control"] for s in summaries]
    enrich_mask = np.array([v is not None for v in top_enrich_raw])
    top_enrich = np.array([v for v in top_enrich_raw if v is not None])
    n_vents_for_top_enrich = n_vents[enrich_mask] if enrich_mask.size else np.array([])
    pooled_enrich = np.array([
        r["enrichment_vs_control"] for r in pooled if r["enrichment_vs_control"] is not None
    ]) if pooled else np.array([])
    n_increased = sum(s["prebiotic_summary"]["n_vents_increased_vs_control"] for s in summaries)
    n_decreased = sum(s["prebiotic_summary"]["n_vents_decreased_vs_control"] for s in summaries)
    n_unchanged = sum(s["prebiotic_summary"]["n_vents_unchanged_vs_control"] for s in summaries)

    return {
        "max_concentration": describe(max_concs, n_bootstrap, rng),
        "mean_concentration": describe(mean_concs, n_bootstrap, rng),
        "n_vents": describe(n_vents, n_bootstrap, rng),
        "pooled_concentration": describe(pooled_concs, n_bootstrap, rng),
        "top_hotspot_enrichment": describe(top_enrich, n_bootstrap, rng),
        "pooled_enrichment": describe(pooled_enrich, n_bootstrap, rng),
        "n_vents_increased_vs_control": n_increased,
        "n_vents_decreased_vs_control": n_decreased,
        "n_vents_unchanged_vs_control": n_unchanged,
        "top_hotspot_type_counts": dict(Counter(top_types)),
        "max_concs_array": max_concs,
        "pooled_concs_array": pooled_concs,
        "n_vents_array": n_vents,
        "pooled_enrich_array": pooled_enrich,
        "top_enrich_array": top_enrich,
        "n_vents_for_top_enrich_array": n_vents_for_top_enrich,
    }

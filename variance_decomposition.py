"""
Decomposição de variância estocástica (campo de fumarolas, seed) vs.
paramétrica (incerteza sobre entrainment_alpha / raio-densidade do
agregado acústico) via um desenho ANINHADO balanceado: N_outer pontos de
parâmetro amostrados por Hipercubo Latino conjunto (`fumarola_field.
joint_latin_hypercube`), cada um com N_inner réplicas de campo (seeds
distintas, parâmetro FIXO dentro do grupo).

Motivação: o `--sensitivity-sweep` original (ver PHYSICS_MODEL.md §7.8)
varia seed E parâmetro físico juntos, run a run — o spread resultante no
ensemble é uma mistura de "quanto varia por causa do campo de fumarolas
ser aleatório" e "quanto varia por causa de não sabermos o valor real do
parâmetro", sem forma de separar as duas fontes. Esse módulo resolve isso
com a decomposição clássica de ANOVA de um fator aleatório aninhado
(lei da variância total, `Var(Y) = E[Var(Y|theta)] + Var(E[Y|theta])`) —
ver Searle, S.R., Casella, G., & McCulloch, C.E. (1992), "Variance
Components," Wiley, cap. 3 (estimadores de método dos momentos para
ANOVA de um fator aleatório balanceada).

Módulo puro (sem Tkinter, sem I/O de simulação) — a orquestração das runs
(gerar o desenho aninhado, executar, agrupar por ponto externo) vive em
`fumarola_field.run_nested_variance_experiment`, que chama as funções
daqui só para a parte estatística.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def default_response_value(summary: dict) -> Optional[float]:
    """
    Extrai um valor de resposta escalar de uma `summary` de run (mesmo
    dict retornado por `fumarola_field.execute_run` / salvo em
    `metadata.json`). Prioriza a profundidade do poço de Gor'kov da classe
    de agregado near-field (`trap_depth_over_kT`) quando o modo acústico
    particle_trap/both estava ativo — é a métrica já usada em
    `report._relevance_drivers`/`_aggregate_acoustic_stats` para o mesmo
    propósito, mantém consistência com o resto do projeto. Cai para o
    enriquecimento do hotspot dominante vs. controle (`prebiotic_summary
    .top_hotspot_enrichment_vs_control`) quando não há dado acústico —
    ainda uma métrica física real (não um substituto arbitrário).
    """
    diag = summary.get("acoustic_diagnostics")
    if diag:
        agg = (diag.get("particle_classes") or {}).get("near_field_fe_oxyhydroxide_aggregate")
        if agg and "trap_depth_over_kT" in agg:
            return float(agg["trap_depth_over_kT"])
    val = (summary.get("prebiotic_summary") or {}).get("top_hotspot_enrichment_vs_control")
    return float(val) if val is not None else None


def _method_of_moments_components(outer_groups: List[np.ndarray]) -> dict:
    """Estimadores de método dos momentos (ANOVA de um fator aleatório,
    desenho balanceado: mesmo N_inner em todo grupo externo)."""
    k = len(outer_groups)
    n = len(outer_groups[0])
    if k < 2:
        raise ValueError("nested_variance_decomposition precisa de pelo menos 2 grupos externos (N_outer >= 2)")
    if n < 2:
        raise ValueError("nested_variance_decomposition precisa de pelo menos 2 réplicas internas (N_inner >= 2)")
    if any(len(g) != n for g in outer_groups):
        raise ValueError("desenho não-balanceado: todo grupo externo precisa do mesmo N_inner")

    arr = np.array(outer_groups, dtype=float)  # shape (k, n)
    group_means = arr.mean(axis=1)
    grand_mean = float(group_means.mean())

    # MSW (mean square within) = média das variâncias amostrais intra-grupo
    # (ddof=1) — estimador não-viesado de sigma^2_estocastico.
    group_vars = arr.var(axis=1, ddof=1)
    msw = float(group_vars.mean())

    # MSB (mean square between) = n * variância amostral das médias de
    # grupo (ddof=1) — E[MSB] = sigma^2_estocastico + n*sigma^2_parametrico
    # (ANOVA de um fator aleatório balanceada, ver Searle et al. 1992).
    msb = float(n * group_means.var(ddof=1))

    sigma2_stochastic = msw
    # Estimador de método dos momentos pode sair negativo quando o sinal
    # paramétrico está no ou abaixo do ruído de amostragem (finito N) —
    # convenção padrão: grampear em 0, não é um erro de cálculo, é o
    # próprio estimador dizendo "não detectável nesta amostra".
    sigma2_parametric_raw = (msb - msw) / n
    sigma2_parametric = max(0.0, sigma2_parametric_raw)

    total = sigma2_stochastic + sigma2_parametric
    if total > 0:
        frac_stochastic = sigma2_stochastic / total
        frac_parametric = sigma2_parametric / total
    else:
        frac_stochastic = frac_parametric = 0.0

    return {
        "n_outer": k, "n_inner": n,
        "grand_mean": grand_mean,
        "group_means": group_means.tolist(),
        "group_stds": np.sqrt(group_vars).tolist(),
        "within_group_variance": sigma2_stochastic,
        "between_group_variance": sigma2_parametric,
        "between_group_variance_raw": sigma2_parametric_raw,
        "between_group_variance_was_clipped": sigma2_parametric_raw < 0.0,
        "total_variance_anova": total,
        "raw_pooled_variance": float(arr.var(ddof=1)),
        "stochastic_fraction": frac_stochastic,
        "parametric_fraction": frac_parametric,
    }


def nested_variance_decomposition(outer_groups: List[np.ndarray], n_bootstrap: int = 2000,
                                   rng: Optional[np.random.Generator] = None) -> dict:
    """
    Decompõe a variância total de `outer_groups` (lista de N_outer arrays,
    cada um com N_inner valores de resposta — mesmo parâmetro físico
    fixo dentro do array, seeds de campo distintas entre os elementos)
    em componente estocástica (campo) e paramétrica (incerteza sobre o
    parâmetro), via ANOVA de um fator aleatório balanceada (método dos
    momentos, Searle, Casella & McCulloch 1992).

    IC 95% em `*_fraction_ci95` via bootstrap ANINHADO (cluster bootstrap
    de 2 estágios: reamostra QUAIS grupos externos entram, e dentro de
    cada grupo reamostrado, reamostra as réplicas internas — preserva a
    estrutura hierárquica do desenho; ver Davison & Hinkley, 1997, "Bootstrap
    Methods and Their Application," cap. 3.8, para bootstrap de dados
    agrupados/hierárquicos). Com N_outer/N_inner modestos (a faixa
    tipicamente viável neste projeto, dezenas, não milhares — cada réplica
    é uma simulação física completa) esses componentes são eles mesmos
    incertos; reportar só o ponto estimado sem IC seria enganoso.
    """
    result = _method_of_moments_components(outer_groups)
    k, n = result["n_outer"], result["n_inner"]
    arr = np.array(outer_groups, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    boot_stoch = np.empty(n_bootstrap)
    boot_param = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        outer_idx = rng.integers(0, k, size=k)
        resampled = []
        for oi in outer_idx:
            inner_idx = rng.integers(0, n, size=n)
            resampled.append(arr[oi, inner_idx])
        try:
            comp = _method_of_moments_components(resampled)
        except ValueError:
            # degenerado (variância zero em algum grupo reamostrado) —
            # descartado da distribuição bootstrap, não tratado como 0
            # artificial (evitaria enviesar o IC para baixo).
            boot_stoch[b] = np.nan
            boot_param[b] = np.nan
            continue
        boot_stoch[b] = comp["stochastic_fraction"]
        boot_param[b] = comp["parametric_fraction"]

    boot_stoch = boot_stoch[~np.isnan(boot_stoch)]
    boot_param = boot_param[~np.isnan(boot_param)]
    result["stochastic_fraction_ci95"] = (
        (float(np.percentile(boot_stoch, 2.5)), float(np.percentile(boot_stoch, 97.5)))
        if boot_stoch.size else (float("nan"), float("nan"))
    )
    result["parametric_fraction_ci95"] = (
        (float(np.percentile(boot_param, 2.5)), float(np.percentile(boot_param, 97.5)))
        if boot_param.size else (float("nan"), float("nan"))
    )
    result["n_bootstrap_valid"] = int(boot_stoch.size)
    return result

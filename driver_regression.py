"""
Regressão multivariada por transformação de postos (rank-transform
regression) — generaliza a correlação de Spearman um-parâmetro-por-vez
(usada em `report._relevance_drivers`) para um modelo que controla todos
os preditores SIMULTANEAMENTE, separando efeito próprio de confundimento
entre preditores correlacionados entre si.

Motivação concreta deste projeto: `report._relevance_drivers` correlaciona
cada parâmetro varrido (α, raio/densidade do agregado) E o nº de vents do
campo (não varrido de propósito — emerge da geração estocástica do campo,
correlacionado com a seed) CONTRA `gorkov_trap_depth_over_kT`, um
parâmetro de cada vez. O achado "nº de vents tem ρ=+0,13 (p≈3×10⁻⁵),
plausivelmente efeito de múltiplas comparações — mais vents por campo,
mais chances de um deles calhar num raio favorável" (docs/PHYSICS_MODEL.md
§7.8.1) ficou como suspeita não testada: correlação marginal não
distingue "nº de vents afeta a profundidade de poço por si só" de
"nº de vents e a profundidade de poço são ambos afetados por uma terceira
variável" (aqui, plausivelmente o raio do agregado, se por acaso houver
alguma associação espúria entre nº de vents e o raio sorteado numa
amostra finita). Este módulo resolve isso via regressão múltipla.

Método: transformação de postos (Iman, R.L., & Conover, W.J., 1979,
"The use of the rank transform in regression," Technometrics, 21(4),
499-509) — cada preditor e a resposta são substituídos por seus POSTOS
(rank, com postos médios para empates), depois ajustados por mínimos
quadrados ordinários. Preserva a robustez da correlação de Spearman a
não-normalidade/não-linearidade monotônica (resposta deste projeto é
conhecidamente assimétrica — cauda longa de eventos raros, ver §7.8.1),
mas agora com coeficientes PARCIAIS (controlando os demais preditores),
erros-padrão, testes t e IC por bootstrap, em vez de uma correlação
marginal isolada por parâmetro.

Módulo puro (sem Tkinter/relatório) — implementado do zero em numpy/scipy
(sem statsmodels), mesma filosofia de dependência enxuta do resto do
projeto.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.stats import rankdata, t as t_dist


def _standardized_ranks(X: np.ndarray) -> np.ndarray:
    """Postos (rankdata, postos médios em empates) de cada coluna,
    padronizados (média 0, desvio-padrão 1) — coeficientes ficam na
    mesma escala entre preditores, comparáveis diretamente."""
    ranks = np.apply_along_axis(rankdata, 0, X)
    std = ranks.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    return (ranks - ranks.mean(axis=0)) / std


def _ols(X: np.ndarray, y: np.ndarray):
    """OLS clássico com intercepto: retorna beta, erros-padrão, t, p, R²."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    y_hat = Xd @ beta
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    dof = max(n - p, 1)
    sigma2 = rss / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    se = np.where(se == 0, np.nan, se)
    tstat = beta / se
    pval = 2 * t_dist.sf(np.abs(tstat), df=dof)
    return beta, se, tstat, pval, r2, dof


def _holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Correção de Holm-Bonferroni (Holm, S., 1979, "A simple sequentially
    rejective multiple test procedure," Scandinavian Journal of
    Statistics, 6(2), 65-70) — menos conservadora que Bonferroni simples,
    ainda controla o erro familywise. Aplicada só aos p-valores desta
    MESMA regressão (k preditores) — não uma correção sistemática de
    todos os p-valores já reportados no projeto (isso é um escopo maior,
    item separado)."""
    order = np.argsort(pvals)
    k = len(pvals)
    adjusted = np.empty(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (k - rank) * pvals[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def rank_transform_regression(X: np.ndarray, y: np.ndarray, predictor_names: List[str],
                               n_bootstrap: int = 2000, rng: Optional[np.random.Generator] = None) -> dict:
    """
    `X`: (n, k) valores brutos dos preditores (escalas arbitrárias — a
    transformação de postos cuida disso). `y`: (n,) resposta.

    Retorna coeficiente padronizado, erro-padrão, t, p (bruto e corrigido
    por Holm), IC 95% por bootstrap (reamostragem de casos/linhas — Efron
    & Tibshirani, 1993, "An Introduction to the Bootstrap," Chapman &
    Hall) e VIF (fator de inflação de variância, `1/(1-R²_j)` de
    regredir o posto padronizado do preditor j contra os DEMAIS
    preditores — Fox, J., & Monette, G., 1992, "Generalized collinearity
    diagnostics," JASA, 87(417), 178-183) por preditor, para diagnosticar
    multicolinearidade entre os próprios preditores (independente de y).
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k + 1:
        raise ValueError(f"n ({n}) precisa ser maior que k+1 ({k + 1}) graus de liberdade")

    Xr = _standardized_ranks(X)
    yr = _standardized_ranks(y.reshape(-1, 1)).ravel()

    beta, se, tstat, pval, r2, dof = _ols(Xr, yr)
    # beta[0] é o intercepto (sempre ~0 após padronização, descartado do
    # relatório por preditor — mantido em beta_full só por completude).
    coef = beta[1:]
    coef_se = se[1:]
    coef_t = tstat[1:]
    coef_p = pval[1:]
    coef_p_holm = _holm_correction(coef_p)

    vif = np.empty(k)
    for j in range(k):
        others = np.delete(Xr, j, axis=1)
        if others.shape[1] == 0:
            vif[j] = 1.0
            continue
        _, _, _, _, r2_j, _ = _ols(others, Xr[:, j])
        vif[j] = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf

    boot_coef = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        Xrb = _standardized_ranks(Xb)
        yrb = _standardized_ranks(yb.reshape(-1, 1)).ravel()
        try:
            beta_b, _, _, _, _, _ = _ols(Xrb, yrb)
        except np.linalg.LinAlgError:
            boot_coef[b] = np.nan
            continue
        boot_coef[b] = beta_b[1:]

    ci_lo = np.nanpercentile(boot_coef, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_coef, 97.5, axis=0)

    return {
        "predictor_names": predictor_names,
        "n": n, "dof": dof, "r_squared": r2,
        "coefficients": {predictor_names[j]: float(coef[j]) for j in range(k)},
        "standard_errors": {predictor_names[j]: float(coef_se[j]) for j in range(k)},
        "t_stats": {predictor_names[j]: float(coef_t[j]) for j in range(k)},
        "p_values": {predictor_names[j]: float(coef_p[j]) for j in range(k)},
        "p_values_holm": {predictor_names[j]: float(coef_p_holm[j]) for j in range(k)},
        "ci95": {predictor_names[j]: (float(ci_lo[j]), float(ci_hi[j])) for j in range(k)},
        "vif": {predictor_names[j]: float(vif[j]) for j in range(k)},
        "n_bootstrap_valid": int(np.sum(~np.isnan(boot_coef[:, 0]))),
    }

"""
Testes de `driver_regression.py` — regressão multivariada por
transformação de postos (Iman & Conover 1979), generalização da
correlação de Spearman um-a-um para controlar todos os preditores
simultaneamente. Ver docs/PHYSICS_MODEL.md §7.8.4.

Rodar com: pytest tests/test_driver_regression.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import driver_regression as dr


# --------------------------------------------------------------------------
# 1. Caso central: confundimento — o motivo de existir este módulo
# --------------------------------------------------------------------------

def test_confounded_predictor_loses_significance_after_controlling_for_real_driver():
    """CASO CENTRAL deste módulo: x_confound é correlacionado com x1 (o
    driver real de y) mas NÃO tem efeito próprio nenhum sobre y. A
    correlação de Spearman marginal (um-a-um, o método antigo) mostra
    x_confound como fortemente associado a y só por transitividade — a
    regressão multivariada deve corretamente atribuir a ele um
    coeficiente parcial perto de zero e não-significativo, distinguindo
    efeito real de confundimento. Isto reproduz exatamente o tipo de
    pergunta suspeitada no projeto para `n_vents` (docs/PHYSICS_MODEL.md
    §7.8.1: "plausivelmente um efeito de múltiplas comparações")."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.uniform(0, 1, n)
    x_confound = x1 + rng.normal(0, 0.15, n)
    x_irrelevant = rng.uniform(0, 1, n)
    y = 3 * x1 + rng.normal(0, 0.3, n)

    X = np.column_stack([x1, x_confound, x_irrelevant])
    result = dr.rank_transform_regression(X, y, ["x1_real", "x_confound", "x_irrelevant"], rng=rng)

    assert result["p_values"]["x1_real"] < 0.001
    assert result["p_values"]["x_confound"] > 0.3
    assert result["p_values"]["x_irrelevant"] > 0.3
    assert abs(result["coefficients"]["x_confound"]) < 0.15
    assert abs(result["coefficients"]["x1_real"]) > 0.7

    # a correlação MARGINAL (Spearman simples), em contraste, mostraria
    # x_confound como fortemente associado — confirma que o teste está
    # de fato testando o cenário de confundimento, não um caso trivial.
    from scipy.stats import spearmanr
    rho_confound, p_confound = spearmanr(x_confound, y)
    assert p_confound < 1e-10, "x_confound deveria parecer 'significativo' na correlação marginal"


def test_vif_flags_collinear_predictors_but_not_independent_ones():
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.uniform(0, 1, n)
    x_collinear = x1 + rng.normal(0, 0.05, n)  # quase colinear com x1
    x_independent = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)  # y independente de tudo — só testando VIF aqui

    X = np.column_stack([x1, x_collinear, x_independent])
    result = dr.rank_transform_regression(X, y, ["x1", "x_collinear", "x_independent"], rng=rng)

    assert result["vif"]["x1"] > 5.0
    assert result["vif"]["x_collinear"] > 5.0
    assert result["vif"]["x_independent"] < 2.0


# --------------------------------------------------------------------------
# 2. Recuperação de um efeito real conhecido
# --------------------------------------------------------------------------

def test_recovers_true_driver_among_independent_predictors():
    rng = np.random.default_rng(2)
    n = 400
    x1 = rng.uniform(0, 1, n)  # forte
    x2 = rng.uniform(0, 1, n)  # fraco
    x3 = rng.uniform(0, 1, n)  # nulo
    y = 5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)

    X = np.column_stack([x1, x2, x3])
    result = dr.rank_transform_regression(X, y, ["x1", "x2", "x3"], rng=rng)

    assert abs(result["coefficients"]["x1"]) > abs(result["coefficients"]["x2"])
    assert abs(result["coefficients"]["x2"]) > abs(result["coefficients"]["x3"])
    assert result["p_values"]["x1"] < 0.001
    assert result["p_values"]["x3"] > 0.05


def test_robust_to_nonlinear_monotonic_relationship():
    """Vantagem central da transformação de postos sobre OLS bruto:
    robustez a relações MONOTÔNICAS não-lineares (ex. cúbica) — mesma
    justificativa de usar Spearman em vez de Pearson no projeto."""
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = x1 ** 3 + rng.normal(0, 0.05, n)  # monotônica em x1, mas NÃO linear

    X = np.column_stack([x1, x2])
    result = dr.rank_transform_regression(X, y, ["x1", "x2"], rng=rng)

    assert result["p_values"]["x1"] < 0.001
    assert result["p_values"]["x2"] > 0.1
    # não 1.0: x^3 tem derivada quase nula perto de x1=0 (ponto de
    # inflexão), então o ruído (std=0.05) inverte a ordem de postos de
    # pontos vizinhos ali — R² alto mas imperfeito é o resultado
    # ESPERADO, não uma falha do método.
    assert result["r_squared"] > 0.8


# --------------------------------------------------------------------------
# 3. Correção de Holm e propriedades gerais
# --------------------------------------------------------------------------

def test_holm_correction_is_never_smaller_than_raw_pvalue():
    pvals = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
    adjusted = dr._holm_correction(pvals)
    assert np.all(adjusted >= pvals - 1e-12)
    assert np.all(adjusted <= 1.0)


def test_holm_correction_is_monotonic_in_sorted_order():
    """Propriedade estrutural do procedimento de Holm: depois de
    ordenados, os p-valores ajustados nunca diminuem."""
    pvals = np.array([0.001, 0.004, 0.02, 0.04, 0.5])
    adjusted = dr._holm_correction(pvals)
    order = np.argsort(pvals)
    adjusted_sorted = adjusted[order]
    assert np.all(np.diff(adjusted_sorted) >= -1e-12)


def test_result_contains_all_expected_keys():
    rng = np.random.default_rng(4)
    n = 100
    X = rng.uniform(0, 1, size=(n, 2))
    y = X[:, 0] + rng.normal(0, 0.1, n)
    result = dr.rank_transform_regression(X, y, ["a", "b"], n_bootstrap=200, rng=rng)
    for key in ("coefficients", "standard_errors", "t_stats", "p_values", "p_values_holm", "ci95", "vif"):
        assert set(result[key]) == {"a", "b"}


def test_ci95_bounds_are_ordered():
    rng = np.random.default_rng(5)
    n = 150
    X = rng.uniform(0, 1, size=(n, 2))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.2, n)
    result = dr.rank_transform_regression(X, y, ["a", "b"], n_bootstrap=300, rng=rng)
    for name in ("a", "b"):
        lo, hi = result["ci95"][name]
        assert lo <= hi


def test_reproducible_with_same_seed():
    n = 100
    rng_data = np.random.default_rng(6)
    X = rng_data.uniform(0, 1, size=(n, 2))
    y = X[:, 0] + rng_data.normal(0, 0.1, n)

    r1 = dr.rank_transform_regression(X, y, ["a", "b"], n_bootstrap=200, rng=np.random.default_rng(11))
    r2 = dr.rank_transform_regression(X, y, ["a", "b"], n_bootstrap=200, rng=np.random.default_rng(11))
    assert r1["coefficients"] == r2["coefficients"]
    assert r1["ci95"] == r2["ci95"]


def test_raises_when_too_few_observations_for_predictors():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="graus de liberdade"):
        dr.rank_transform_regression(X, y, ["a", "b"])

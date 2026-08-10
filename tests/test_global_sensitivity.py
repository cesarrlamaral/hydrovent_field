"""
Testes de `global_sensitivity.py` — índices de Sobol' (Saltelli/Jansen)
sobre um surrogate de Processo Gaussiano treinado nos dados do desenho
aninhado. Ver docs/PHYSICS_MODEL.md §7.8.3.

O estimador de Sobol'/Saltelli (`sobol_indices_from_function`) é testado
DIRETAMENTE contra funções analíticas com decomposição de variância
conhecida por construção (aditivas, sem interação; produto, com
interação) — sem envolver o GP, pra isolar bugs do estimador Monte Carlo
dos de ajuste do surrogate. O GP (`GaussianProcessSurrogate`) e o
pipeline completo (`fit_surrogate_and_compute_sobol`) são testados
separadamente.

Rodar com: pytest tests/test_global_sensitivity.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import global_sensitivity as gs


# --------------------------------------------------------------------------
# 1. Estimador de Sobol'/Saltelli contra funções analíticas — Var(Uniform
#    (0,1)) = 1/12 é elementar (derivável em uma linha, não um "número
#    mágico" memorizado de uma fonte externa: Var = E[X^2]-E[X]^2 =
#    integral_0^1 x^2 dx - (1/2)^2 = 1/3 - 1/4 = 1/12).
# --------------------------------------------------------------------------

def test_sobol_additive_function_matches_analytic_variance_ratio():
    """f = x1 + 2*x2 (não depende de x3), xi ~ U(0,1) independentes.
    Var(f) = Var(x1) + 4*Var(x2) = 1/12 + 4/12 = 5/12 (sem termo de
    interação — soma pura). S_i = S_Ti = Var_i/Var(f) exatamente."""
    def f(X):
        return X[:, 0] + 2 * X[:, 1] + 0 * X[:, 2]

    bounds = [(0.0, 1.0)] * 3
    result = gs.sobol_indices_from_function(f, bounds, n=8192, rng=np.random.default_rng(0))

    v1, v2 = 1 / 12, 4 / 12
    total = v1 + v2
    expected_s1, expected_s2 = v1 / total, v2 / total

    assert result["first_order"][0] == pytest.approx(expected_s1, abs=0.01)
    assert result["first_order"][1] == pytest.approx(expected_s2, abs=0.01)
    assert result["first_order"][2] == pytest.approx(0.0, abs=0.01)
    # sem interação: S_i == S_Ti para cada variável.
    assert result["total_order"][0] == pytest.approx(result["first_order"][0], abs=0.01)
    assert result["total_order"][1] == pytest.approx(result["first_order"][1], abs=0.01)
    # soma dos índices de primeira ordem == 1 (toda a variância é explicada
    # por efeitos principais, nenhuma por interação).
    assert sum(result["first_order"].values()) == pytest.approx(1.0, abs=0.02)


def test_sobol_interaction_function_shows_total_exceeds_first_order():
    """f = x1*x2 (produto — interação pura, sem efeito aditivo próprio).
    Propriedade qualitativa robusta (não depende de um valor analítico
    exato memorizado): com interação genuína, o índice de efeito TOTAL
    de cada variável excede o de primeira ordem, e a soma dos índices de
    primeira ordem fica ESTRITAMENTE abaixo de 1 (a variância "faltante"
    é o termo de interação, contabilizado só no índice total)."""
    def f(X):
        return X[:, 0] * X[:, 1]

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result = gs.sobol_indices_from_function(f, bounds, n=8192, rng=np.random.default_rng(1))

    assert result["total_order"][0] > result["first_order"][0] + 0.05
    assert result["total_order"][1] > result["first_order"][1] + 0.05
    assert sum(result["first_order"].values()) < 0.95
    # simetria: x1 e x2 entram identicamente em x1*x2 sobre o mesmo domínio.
    assert result["first_order"][0] == pytest.approx(result["first_order"][1], abs=0.02)


def test_sobol_matrices_use_joint_sequence_not_independent_ones():
    """Regressão do bug real encontrado nesta sessão: gerar A e B como
    duas instâncias INDEPENDENTES de qmc.Sobol (em vez de uma sequência
    conjunta de dimensão 2d dividida em A/B) quebra a estrutura exigida
    pelo estimador de Saltelli/Jansen — mediu S1~0.09 em vez do valor
    analítico correto 0.2 na função aditiva acima. Este teste fixa
    exatamente esse caso numérico como regressão permanente."""
    def f(X):
        return X[:, 0] + 2 * X[:, 1]

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result = gs.sobol_indices_from_function(f, bounds, n=8192, rng=np.random.default_rng(0))
    assert result["first_order"][0] == pytest.approx(0.2, abs=0.02)
    assert result["first_order"][1] == pytest.approx(0.8, abs=0.02)


def test_sobol_indices_stay_within_valid_range_even_for_near_constant_function():
    """Regressão de um caso real encontrado nesta sessão: com poucos
    pontos de treino (outer_n pequeno, d=3), o GP pode convergir para uma
    função quase constante (sinal abaixo do ruído) — var_y do MC de Sobol'
    fica perto de zero e a razão v_i/var_y pode escapar de [0,1] por
    ruído puro. Índices de Sobol' verdadeiros NUNCA saem de [0,1] — o
    estimador deve grampear, não vazar um valor como 1.24."""
    rng = np.random.default_rng(123)

    def f_near_constant(X):
        # variação real, mas MUITO menor que o ruído de ponto flutuante
        # acumulado — simula um surrogate que "achou" que não há sinal.
        return 5.0 + 1e-10 * (X[:, 0] + X[:, 1] + X[:, 2]) + rng.normal(0, 1e-12, size=X.shape[0])

    bounds = [(0.0, 1.0)] * 3
    result = gs.sobol_indices_from_function(f_near_constant, bounds, n=512, rng=rng)
    for i in range(3):
        assert 0.0 <= result["first_order"][i] <= 1.0
        assert 0.0 <= result["total_order"][i] <= 1.0


def test_sobol_reproducible_with_same_seed():
    def f(X):
        return np.sin(X[:, 0]) + X[:, 1] ** 2

    bounds = [(-3.0, 3.0), (0.0, 1.0)]
    r1 = gs.sobol_indices_from_function(f, bounds, n=2048, rng=np.random.default_rng(42))
    r2 = gs.sobol_indices_from_function(f, bounds, n=2048, rng=np.random.default_rng(42))
    assert r1["first_order"] == r2["first_order"]
    assert r1["total_order"] == r2["total_order"]


# --------------------------------------------------------------------------
# 2. Processo Gaussiano — sanidade do ajuste (não estimador de Sobol')
# --------------------------------------------------------------------------

def test_gp_predicts_training_points_closely_with_low_noise():
    rng = np.random.default_rng(2)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform([0, 0], [1, 1], size=(20, 2))
    y = np.sin(3 * X[:, 0]) + 0.3 * X[:, 1]
    gp = gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=1e-6, rng=rng)
    y_pred = gp.predict(X)
    assert np.max(np.abs(y_pred - y)) < 0.05


def test_gp_loo_cv_r2_is_high_for_smooth_function_low_noise():
    rng = np.random.default_rng(3)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform([0, 0], [1, 1], size=(25, 2))
    y_true = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    noise_std = 0.02
    y = y_true + rng.normal(0, noise_std, size=25)

    gp = gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=noise_std ** 2, rng=rng)
    assert gp.loo_cv_r2() > 0.9


def test_gp_loo_cv_r2_matches_held_out_test_set():
    """A fórmula fechada de LOO-CV (Rasmussen & Williams 2006, eq. 5.12)
    deveria concordar com um R² calculado por um conjunto de teste
    genuinamente separado — testa a fórmula fechada contra o método
    'força bruta' de referência, não só que o número parece razoável."""
    rng = np.random.default_rng(4)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform([0, 0], [1, 1], size=(25, 2))
    y_true_train = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    noise_std = 0.02
    y = y_true_train + rng.normal(0, noise_std, size=25)

    gp = gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=noise_std ** 2, rng=rng)
    loo_r2 = gp.loo_cv_r2()

    X_test = rng.uniform([0, 0], [1, 1], size=(300, 2))
    y_test_true = np.sin(3 * X_test[:, 0]) + 0.5 * X_test[:, 1] ** 2
    y_pred = gp.predict(X_test)
    ss_res = np.sum((y_test_true - y_pred) ** 2)
    ss_tot = np.sum((y_test_true - y_test_true.mean()) ** 2)
    held_out_r2 = 1 - ss_res / ss_tot

    assert loo_r2 == pytest.approx(held_out_r2, abs=0.1)


def test_gp_fit_is_reproducible_with_same_seed():
    rng_data = np.random.default_rng(5)
    bounds = [(0.0, 1.0)]
    X = rng_data.uniform(0, 1, size=(15, 1))
    y = np.sin(3 * X[:, 0]) + rng_data.normal(0, 0.05, size=15)

    gp1 = gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=0.05 ** 2, rng=np.random.default_rng(7))
    gp2 = gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=0.05 ** 2, rng=np.random.default_rng(7))
    assert gp1.sigma_f2 == pytest.approx(gp2.sigma_f2)
    np.testing.assert_allclose(gp1.lengthscales, gp2.lengthscales)


def test_gp_handles_no_overflow_warning_during_optimization():
    """Regressão: reinicializações do otimizador podem visitar
    log-comprimentos-de-escala extremos e disparar overflow em exp()
    sem o clamp defensivo — checa que nenhum RuntimeWarning escapa."""
    import warnings

    rng = np.random.default_rng(6)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X = rng.uniform([0, 0], [1, 1], size=(10, 2))
    y = rng.normal(0, 1, size=10)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gs.GaussianProcessSurrogate(bounds).fit(X, y, noise_variance=0.1, rng=rng)


# --------------------------------------------------------------------------
# 3. Pipeline completo: ajusta o surrogate nos dados do desenho aninhado
#    e calcula Sobol' — dados sintéticos com sensibilidade CONHECIDA
# --------------------------------------------------------------------------

def test_full_pipeline_identifies_dominant_parameter():
    """x1 domina a resposta (amplitude ~1), x2 contribui pouco
    (amplitude ~0.2) — o índice de primeira ordem deveria refletir essa
    assimetria clara, mesmo com ruído estocástico substancial por grupo."""
    rng = np.random.default_rng(8)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    n_outer, n_inner = 30, 15
    outer_params = rng.uniform([0, 0], [1, 1], size=(n_outer, 2))
    sigma_within = 0.3
    outer_groups = [
        rng.normal(np.sin(3 * p[0]) + 0.2 * p[1], sigma_within, size=n_inner)
        for p in outer_params
    ]

    result = gs.fit_surrogate_and_compute_sobol(
        outer_params, outer_groups, sigma_within ** 2, bounds, ["x1", "x2"],
        n_mc=2048, n_bootstrap=50, rng=rng)

    assert result["first_order"]["x1"] > result["first_order"]["x2"]
    assert result["first_order"]["x1"] > 0.7
    lo, hi = result["first_order_ci95"]["x1"]
    assert lo < hi  # IC não-degenerado


def test_full_pipeline_single_dimension_attributes_all_variance_to_it():
    """Com um único parâmetro varrido (caso real do projeto quando
    acoustic_mode='off'/'streaming'), toda a variância explicada pelo
    surrogate É necessariamente do único parâmetro — S1 deveria ficar
    perto de 1."""
    rng = np.random.default_rng(9)
    bounds = [(0.0, 1.0)]
    n_outer, n_inner = 25, 15
    outer_params = rng.uniform(0, 1, size=(n_outer, 1))
    sigma_within = 0.1
    outer_groups = [
        rng.normal(2.0 * p[0], sigma_within, size=n_inner) for p in outer_params
    ]

    result = gs.fit_surrogate_and_compute_sobol(
        outer_params, outer_groups, sigma_within ** 2, bounds, ["alpha"],
        n_mc=2048, n_bootstrap=50, rng=rng)

    assert result["first_order"]["alpha"] > 0.85
    assert result["total_order"]["alpha"] > 0.85

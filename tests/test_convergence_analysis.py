"""
Testes de `convergence_analysis.py` — curva de convergência de Monte
Carlo (traço acumulado + IC vs. N) e projeção analítica de largura de IC
num N maior, sem rodar nenhuma simulação nova. Ver docs/PHYSICS_MODEL.md
§10.5.

Rodar com: pytest tests/test_convergence_analysis.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import convergence_analysis as ca


# --------------------------------------------------------------------------
# 1. IC de Wilson — valores conhecidos + cobertura empírica
# --------------------------------------------------------------------------

def test_wilson_ci_matches_documented_project_value():
    """k=7,n=1000 é o achado real já documentado em docs/PHYSICS_MODEL.md
    §7.8.1 (IC 95% [0,34%, 1,44%]) — reproduzido aqui como regressão."""
    lo, hi = ca.wilson_ci95(7, 1000)
    assert lo == pytest.approx(0.0034, abs=0.0002)
    assert hi == pytest.approx(0.0144, abs=0.0002)


def test_wilson_ci_zero_successes_matches_known_textbook_value():
    lo, hi = ca.wilson_ci95(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(0.0362, abs=0.001)


def test_wilson_ci_empirical_coverage_near_nominal():
    rng = np.random.default_rng(0)
    coverages = []
    for true_p in (0.01, 0.05, 0.2, 0.5):
        for n in (50, 200, 1000):
            covered = sum(
                1 for _ in range(500)
                if (lambda k: ca.wilson_ci95(k, n)[0] <= true_p <= ca.wilson_ci95(k, n)[1])(rng.binomial(n, true_p))
            )
            coverages.append(covered / 500)
    # Wilson tem cobertura conhecidamente OSCILANTE em torno do nominal
    # (Brown, Cai & DasGupta 2001) — checa a MÉDIA sobre várias
    # combinações (p,n), não um único caso isolado.
    assert np.mean(coverages) == pytest.approx(0.95, abs=0.03)


def test_wilson_ci_from_rate_matches_integer_version():
    lo1, hi1 = ca.wilson_ci95(7, 1000)
    lo2, hi2 = ca.wilson_ci95_from_rate(7 / 1000, 1000)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)


# --------------------------------------------------------------------------
# 2. Traço binomial — largura do IC diminui, fração converge pra taxa real
# --------------------------------------------------------------------------

def test_running_binomial_trace_ci_narrows_overall():
    rng = np.random.default_rng(1)
    successes = (rng.random(2000) < 0.03).astype(int)
    trace = ca.running_binomial_fraction_trace(successes, n_points=20)
    assert trace["ci_width"][0] > trace["ci_width"][-1]
    # amplamente decrescente (não necessariamente estrito ponto-a-ponto,
    # mas a largura no início deveria dominar a largura no fim por uma
    # margem grande, não só levemente).
    assert trace["ci_width"][0] > 5 * trace["ci_width"][-1]


def test_running_binomial_trace_converges_to_true_rate():
    rng = np.random.default_rng(2)
    true_rate = 0.1
    successes = (rng.random(5000) < true_rate).astype(int)
    trace = ca.running_binomial_fraction_trace(successes, n_points=20)
    assert trace["fraction"][-1] == pytest.approx(true_rate, abs=0.02)
    assert trace["ci_lo"][-1] <= true_rate <= trace["ci_hi"][-1]


def test_running_binomial_trace_includes_full_n():
    successes = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1])
    trace = ca.running_binomial_fraction_trace(successes, n_points=5)
    assert trace["n"][-1] == 10
    assert trace["n_total"] == 10
    assert trace["k_total"] == 4


def test_running_binomial_trace_rejects_empty_input():
    with pytest.raises(ValueError):
        ca.running_binomial_fraction_trace(np.array([]))


# --------------------------------------------------------------------------
# 3. Traço de média contínua — reaproveita ensemble_stats.describe
# --------------------------------------------------------------------------

def test_running_mean_trace_converges_to_true_mean():
    rng = np.random.default_rng(3)
    true_mean = 5.0
    values = rng.normal(true_mean, 1.0, size=1000)
    trace = ca.running_mean_trace(values, n_points=15, n_bootstrap=300, rng=rng)
    assert trace["mean"][-1] == pytest.approx(true_mean, abs=0.2)
    assert trace["ci_lo"][-1] <= true_mean <= trace["ci_hi"][-1]


def test_running_mean_trace_ci_generally_narrows():
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, size=1000)
    trace = ca.running_mean_trace(values, n_points=15, n_bootstrap=300, rng=rng)
    # não estritamente monotônico ponto-a-ponto (dado real tem ruído),
    # mas a largura média da segunda metade da curva deveria ser bem
    # menor que a da primeira metade.
    mid = len(trace["ci_width"]) // 2
    assert trace["ci_width"][mid:].mean() < trace["ci_width"][:mid].mean()


def test_running_mean_trace_rejects_empty_input():
    with pytest.raises(ValueError):
        ca.running_mean_trace(np.array([]))


# --------------------------------------------------------------------------
# 4. Projeção analítica — validada CONTRA dados reais maiores, não só a
#    fórmula confiando em si mesma
# --------------------------------------------------------------------------

def test_binomial_prediction_matches_real_larger_sample():
    """Gera uma amostra GRANDE real (n=20000) com uma taxa conhecida,
    usa só os primeiros 1000 pontos pra prever o IC em n=20000, e
    compara com o IC calculado de FATO sobre os 20000 pontos reais —
    valida que a extrapolação é utilizável, não só matematicamente
    plausível."""
    rng = np.random.default_rng(5)
    true_rate = 0.01
    full = (rng.random(20000) < true_rate).astype(int)
    k_1000 = int(full[:1000].sum())

    predicted = ca.predict_binomial_ci_at_n(k_1000, 1000, 20000)
    actual_ci = ca.wilson_ci95(int(full.sum()), 20000)

    # a largura PREVISTA deveria estar na mesma ordem de grandeza da
    # largura REAL observada com os 20000 pontos de verdade.
    actual_width = actual_ci[1] - actual_ci[0]
    assert predicted["ci_width_target"] == pytest.approx(actual_width, rel=0.5)
    assert predicted["width_ratio"] < 1.0  # IC mais largo N sempre estreita


def test_mean_ci_half_width_prediction_follows_inverse_sqrt_n():
    assert ca.predict_mean_ci_half_width_at_n(1.0, 100, 400) == pytest.approx(0.5)
    assert ca.predict_mean_ci_half_width_at_n(1.0, 100, 100) == pytest.approx(1.0)
    assert ca.predict_mean_ci_half_width_at_n(1.0, 100, 10000) == pytest.approx(0.1)


def test_mean_ci_prediction_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        ca.predict_mean_ci_half_width_at_n(1.0, 0, 100)

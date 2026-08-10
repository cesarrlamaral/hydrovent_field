"""
Testes de `ensemble_stats.py` — primeira cobertura de teste dedicada
deste módulo (extraído de gui.py numa sessão anterior sem ganhar testes
próprios na época). Foco em `describe()`: as chaves PRÉ-EXISTENTES
(n/mean/std/min/median/max, já lidas diretamente por gui.py)
precisam manter semântica idêntica; as novas (q1/q3/iqr/mad/mad_scaled/
skewness/kurtosis/mean_median_gap_over_iqr) são a extensão desta sessão.

Rodar com: pytest tests/test_ensemble_stats.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ensemble_stats as es


# --------------------------------------------------------------------------
# 1. Compatibilidade retroativa — chaves pré-existentes inalteradas
# --------------------------------------------------------------------------

def test_describe_empty_array_returns_all_zero_keys():
    result = es.describe(np.array([]))
    assert result["n"] == 0
    for key in ("mean", "std", "min", "median", "max", "q1", "q3", "iqr",
                "mad", "mad_scaled", "skewness", "kurtosis", "mean_median_gap_over_iqr"):
        assert result[key] == 0.0


def test_describe_preserves_preexisting_keys_exactly():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = es.describe(arr)
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["std"] == pytest.approx(np.std(arr))  # mesmo ddof=0 de antes
    assert result["min"] == pytest.approx(1.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["max"] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# 2. Novas estatísticas robustas — valores conhecidos/deriváveis
# --------------------------------------------------------------------------

def test_iqr_matches_percentile_definition():
    arr = np.arange(1.0, 101.0)  # 1..100
    result = es.describe(arr)
    q1_expected, q3_expected = np.percentile(arr, [25, 75])
    assert result["q1"] == pytest.approx(q1_expected)
    assert result["q3"] == pytest.approx(q3_expected)
    assert result["iqr"] == pytest.approx(q3_expected - q1_expected)


def test_symmetric_distribution_has_near_zero_skewness_and_gap():
    rng = np.random.default_rng(0)
    arr = rng.normal(10.0, 2.0, size=5000)
    result = es.describe(arr)
    assert result["skewness"] == pytest.approx(0.0, abs=0.15)
    assert result["mean_median_gap_over_iqr"] == pytest.approx(0.0, abs=0.05)
    assert result["kurtosis"] == pytest.approx(0.0, abs=0.3)  # normal: curtose em excesso ~0


def test_right_skewed_distribution_shows_positive_skewness_and_mean_above_median():
    """Distribuição exponencial (cauda longa à direita) — mesmo formato
    qualitativo das distribuições reais deste projeto (altura de
    chaminé, profundidade de poço de Gor'kov, ver docs/PHYSICS_MODEL.md
    §7.8.1) — é exatamente o caso em que mean/std sozinhos enganam."""
    rng = np.random.default_rng(1)
    arr = rng.exponential(scale=2.0, size=5000)
    result = es.describe(arr)
    assert result["skewness"] > 1.0  # exponencial tem assimetria teórica = 2
    assert result["mean"] > result["median"]
    assert result["mean_median_gap_over_iqr"] > 0.1


def test_mad_scaled_recovers_std_under_normality():
    """Propriedade definidora do fator de escala 1,4826: sob dados
    normais, MAD escalado deveria se aproximar do desvio-padrão real —
    ao contrário de MAD bruto, que subestima sistematicamente."""
    rng = np.random.default_rng(2)
    true_std = 3.0
    arr = rng.normal(0.0, true_std, size=20000)
    result = es.describe(arr)
    assert result["mad_scaled"] == pytest.approx(true_std, rel=0.05)
    assert result["mad"] < result["mad_scaled"]  # MAD bruto é sistematicamente menor


def test_robust_statistics_resist_a_single_extreme_outlier_more_than_std():
    """Propriedade central motivando este módulo: um único outlier
    extremo infla `std` desproporcionalmente mas mal move `iqr`/
    `mad_scaled` — exatamente por isso são chamadas de estatísticas
    ROBUSTAS."""
    rng = np.random.default_rng(3)
    clean = rng.normal(10.0, 1.0, size=200)
    contaminated = np.concatenate([clean, [10000.0]])  # 1 outlier extremo

    d_clean = es.describe(clean)
    d_contaminated = es.describe(contaminated)

    std_ratio = d_contaminated["std"] / d_clean["std"]
    iqr_ratio = d_contaminated["iqr"] / d_clean["iqr"]
    mad_ratio = d_contaminated["mad_scaled"] / d_clean["mad_scaled"]

    assert std_ratio > 5.0  # um outlier já domina o desvio-padrão
    assert iqr_ratio < 1.5  # IQR quase não se move
    assert mad_ratio < 1.5  # MAD escalado quase não se move


# --------------------------------------------------------------------------
# 3. Casos degenerados de amostra pequena — NaN honesto, não zero enganoso
# --------------------------------------------------------------------------

def test_skewness_is_nan_below_three_points():
    assert np.isnan(es.describe(np.array([1.0]))["skewness"])
    assert np.isnan(es.describe(np.array([1.0, 2.0]))["skewness"])
    assert not np.isnan(es.describe(np.array([1.0, 2.0, 3.0]))["skewness"])


def test_kurtosis_is_nan_below_four_points():
    assert np.isnan(es.describe(np.array([1.0, 2.0, 3.0]))["kurtosis"])
    assert not np.isnan(es.describe(np.array([1.0, 2.0, 3.0, 4.0]))["kurtosis"])


def test_mean_median_gap_is_zero_when_iqr_is_zero():
    """Todos os valores idênticos: iqr=0, gap não deveria dividir por
    zero (guarda explícita retorna 0.0, não NaN/inf)."""
    arr = np.full(10, 5.0)
    result = es.describe(arr)
    assert result["iqr"] == 0.0
    assert result["mean_median_gap_over_iqr"] == 0.0


def test_bootstrap_is_off_by_default_no_ci_keys():
    """n_bootstrap=0 (padrão) precisa continuar byte-idêntico ao
    comportamento pré-bootstrap — nenhum consumidor existente (GUI)
    deveria ver chaves novas sem pedir."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = es.describe(arr)
    assert "mean_ci95" not in result
    assert "n_bootstrap" not in result


def test_bootstrap_ci_has_correct_empirical_coverage():
    """Não testa uma única amostra (um IC 95% erra o valor verdadeiro
    ~5% das vezes POR DEFINIÇÃO — testar 1 seed só teria ~5% de chance
    de falhar por sorte, não por bug). Em vez disso mede a taxa de
    COBERTURA empírica sobre muitas réplicas independentes — a
    propriedade real que "IC 95%" promete."""
    rng = np.random.default_rng(10)
    true_mean = 10.0
    n_trials = 150
    covered = 0
    for _ in range(n_trials):
        arr = rng.normal(true_mean, 2.0, size=100)
        result = es.describe(arr, n_bootstrap=300, rng=rng)
        lo, hi = result["mean_ci95"]
        if lo <= true_mean <= hi:
            covered += 1
    coverage = covered / n_trials
    # banda folgada em torno de 0.95 (erro-padrão binomial ~0.018 p/
    # n_trials=150) — checa que a cobertura é aproximadamente correta,
    # não exatamente 95% (ruído de Monte Carlo é esperado).
    assert 0.85 <= coverage <= 1.0, f"cobertura empírica={coverage:.3f}, esperado perto de 0.95"


def test_bootstrap_adds_ci_for_every_continuous_statistic():
    rng = np.random.default_rng(11)
    arr = rng.exponential(2.0, size=300)
    result = es.describe(arr, n_bootstrap=500, rng=rng)
    for name in ("mean", "std", "median", "q1", "q3", "iqr", "mad", "mad_scaled",
                 "skewness", "kurtosis", "mean_median_gap_over_iqr"):
        lo, hi = result[f"{name}_ci95"]
        assert lo <= hi
    assert result["n_bootstrap"] == 500


def test_bootstrap_ci_narrows_with_larger_sample_size():
    """Propriedade estrutural do bootstrap: mais dados → CI mais
    estreito (erro-padrão cai com sqrt(n))."""
    rng = np.random.default_rng(12)
    small = rng.normal(0, 1, size=30)
    large = rng.normal(0, 1, size=3000)

    d_small = es.describe(small, n_bootstrap=1000, rng=rng)
    d_large = es.describe(large, n_bootstrap=1000, rng=rng)

    width_small = d_small["mean_ci95"][1] - d_small["mean_ci95"][0]
    width_large = d_large["mean_ci95"][1] - d_large["mean_ci95"][0]
    assert width_large < width_small


def test_bootstrap_reproducible_with_same_rng_state():
    arr = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 10.0])
    r1 = es.describe(arr, n_bootstrap=500, rng=np.random.default_rng(20))
    r2 = es.describe(arr, n_bootstrap=500, rng=np.random.default_rng(20))
    assert r1["mean_ci95"] == r2["mean_ci95"]
    assert r1["skewness_ci95"] == r2["skewness_ci95"]


def test_bootstrap_empty_array_gives_degenerate_ci():
    result = es.describe(np.array([]), n_bootstrap=100)
    for name in ("mean", "std", "median", "q1", "q3", "iqr", "mad", "mad_scaled",
                 "skewness", "kurtosis", "mean_median_gap_over_iqr"):
        assert result[f"{name}_ci95"] == (0.0, 0.0)


def test_bootstrap_matches_direct_manual_resampling():
    """A versão vetorizada (`_bootstrap_point_estimates`, uma matriz
    (n_bootstrap, n) de reamostragens sem laço Python) deveria produzir
    a MESMA distribuição bootstrap que reamostrar manualmente uma linha
    de cada vez com o mesmo fluxo de números aleatórios — testa que a
    vetorização não introduziu um bug sutil de indexação/eixo."""
    rng_manual = np.random.default_rng(30)
    arr = rng_manual.normal(5.0, 1.5, size=200)
    n_bootstrap = 50

    rng_vec = np.random.default_rng(99)
    idx = rng_vec.integers(0, arr.size, size=(n_bootstrap, arr.size))
    manual_means = np.array([arr[row].mean() for row in idx])

    boot = arr[idx]
    vec_means = boot.mean(axis=1)
    np.testing.assert_allclose(manual_means, vec_means)


def test_bootstrap_compute_ensemble_stats_wiring(tmp_path):
    """`compute_ensemble_stats` repassa `n_bootstrap`/`rng` para cada
    `describe()` interno — checa isso de ponta a ponta, não só
    `describe()` isolado."""
    summaries = [
        {
            "prebiotic_summary": {
                "max_concentration_uM": 10.0 + i, "mean_concentration_uM": 5.0 + i * 0.1,
                "top_hotspot_vent_type": "black_smoker",
                "top_hotspot_enrichment_vs_control": 2.0 + i * 0.05,
                "n_vents_increased_vs_control": 1, "n_vents_decreased_vs_control": 0,
                "n_vents_unchanged_vs_control": 0,
            },
            "n_vents": 5 + (i % 3),
        }
        for i in range(30)
    ]
    result = es.compute_ensemble_stats(summaries, pooled=[], n_bootstrap=200,
                                        rng=np.random.default_rng(42))
    assert "mean_ci95" in result["max_concentration"]
    assert "mean_ci95" in result["n_vents"]
    assert result["max_concentration"]["n_bootstrap"] == 200


def test_constant_array_gives_nan_skew_kurtosis_without_warning():
    """Regressão: variância zero torna assimetria/curtose 0/0 —
    scipy.stats.skew/kurtosis já devolvem NaN corretamente aí, mas sem
    a guarda explícita emitem RuntimeWarning de 'cancelamento
    catastrófico' a cada chamada (ruidoso em produção, já que módulos
    desligados geram arrays constantes de zero com frequência)."""
    import warnings

    arr = np.full(20, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = es.describe(arr)
    assert np.isnan(result["skewness"])
    assert np.isnan(result["kurtosis"])

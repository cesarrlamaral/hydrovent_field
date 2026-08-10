"""
Testes de `variance_decomposition.py` — decomposição de variância
estocástica (campo) vs. paramétrica via ANOVA de um fator aleatório
balanceada. Ver docs/PHYSICS_MODEL.md §7.8.2.

Rodar com: pytest tests/test_variance_decomposition.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import variance_decomposition as vd


# --------------------------------------------------------------------------
# 1. Recuperação de componentes de variância CONHECIDOS (dados sintéticos)
# --------------------------------------------------------------------------

def _make_synthetic_groups(k, n, sigma_between, sigma_within, rng):
    """k grupos externos, n réplicas internas cada, gerados com
    componentes de variância REALMENTE conhecidos (efeito de grupo ~
    N(0, sigma_between^2), ruído interno ~ N(0, sigma_within^2))."""
    group_effects = rng.normal(0.0, sigma_between, size=k)
    groups = [rng.normal(effect, sigma_within, size=n) for effect in group_effects]
    return groups


def test_recovers_known_variance_components_when_both_sources_present():
    """Não usa tolerância pontual arbitrária (estimador de variância tem
    ruído de amostragem real mesmo com k=40 grupos, ~sqrt(2/(k-1))~23% de
    erro-padrão relativo — uma tolerância apertada seria flaky por
    sorte de seed). Em vez disso testa a garantia que a própria função
    oferece: a fração paramétrica VERDADEIRA cai dentro do IC 95% relatado."""
    rng = np.random.default_rng(0)
    sigma_between, sigma_within = 3.0, 1.0
    true_parametric_fraction = sigma_between ** 2 / (sigma_between ** 2 + sigma_within ** 2)
    groups = _make_synthetic_groups(k=40, n=25, sigma_between=sigma_between,
                                     sigma_within=sigma_within, rng=rng)
    result = vd.nested_variance_decomposition(groups, n_bootstrap=500, rng=rng)

    lo, hi = result["parametric_fraction_ci95"]
    assert lo <= true_parametric_fraction <= hi
    # ordem de grandeza correta (folga generosa, ver docstring acima) —
    # ambos os componentes claramente detectados, nenhum colapsado a 0.
    assert result["within_group_variance"] == pytest.approx(sigma_within ** 2, rel=0.6)
    assert result["between_group_variance"] == pytest.approx(sigma_between ** 2, rel=0.6)
    assert result["parametric_fraction"] > 0.5


def test_pure_stochastic_case_has_near_zero_parametric_fraction():
    """Sem efeito de grupo nenhum (sigma_between=0): toda a variância
    observada entre médias de grupo é ruído de amostragem finita — o
    estimador de método dos momentos deve reconhecer isso (fração
    paramétrica perto de 0, não inflada)."""
    rng = np.random.default_rng(1)
    groups = _make_synthetic_groups(k=30, n=20, sigma_between=0.0, sigma_within=2.0, rng=rng)
    result = vd.nested_variance_decomposition(groups, n_bootstrap=200, rng=rng)

    assert result["parametric_fraction"] < 0.15
    assert result["stochastic_fraction"] > 0.85


def test_pure_parametric_case_has_zero_stochastic_variance():
    """Sem ruído interno nenhum (réplicas idênticas dentro do grupo):
    toda a variância é do efeito de grupo — fração estocástica deve ser
    exatamente 0."""
    rng = np.random.default_rng(2)
    group_effects = rng.normal(0.0, 5.0, size=10)
    groups = [np.full(6, effect) for effect in group_effects]
    result = vd.nested_variance_decomposition(groups, n_bootstrap=200, rng=rng)

    assert result["within_group_variance"] == pytest.approx(0.0, abs=1e-12)
    assert result["stochastic_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_between_group_variance_is_clipped_at_zero_not_negative():
    """Método dos momentos pode dar (MSB-MSW)/n < 0 quando não há sinal
    paramétrico detectável — convenção padrão é grampear em 0, e o motivo
    fica registrado em `between_group_variance_was_clipped`, não escondido."""
    rng = np.random.default_rng(3)
    # sigma_between=0 pequena amostra: MSB pode cair abaixo de MSW por acaso.
    groups = _make_synthetic_groups(k=5, n=4, sigma_between=0.0, sigma_within=3.0, rng=rng)
    result = vd.nested_variance_decomposition(groups, n_bootstrap=100, rng=rng)

    assert result["between_group_variance"] >= 0.0
    if result["between_group_variance_raw"] < 0:
        assert result["between_group_variance_was_clipped"] is True
        assert result["between_group_variance"] == 0.0


# --------------------------------------------------------------------------
# 2. Validações de entrada
# --------------------------------------------------------------------------

def test_requires_at_least_two_outer_groups():
    with pytest.raises(ValueError, match="N_outer"):
        vd.nested_variance_decomposition([np.array([1.0, 2.0, 3.0])])


def test_requires_at_least_two_inner_replicates():
    with pytest.raises(ValueError, match="N_inner"):
        vd.nested_variance_decomposition([np.array([1.0]), np.array([2.0])])


def test_requires_balanced_design():
    with pytest.raises(ValueError, match="balanceado"):
        vd.nested_variance_decomposition([np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])])


# --------------------------------------------------------------------------
# 3. Reprodutibilidade do IC via bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_ci_is_reproducible_with_same_rng_state():
    groups = _make_synthetic_groups(k=15, n=10, sigma_between=2.0, sigma_within=1.0,
                                     rng=np.random.default_rng(9))
    r1 = vd.nested_variance_decomposition(groups, n_bootstrap=300, rng=np.random.default_rng(123))
    r2 = vd.nested_variance_decomposition(groups, n_bootstrap=300, rng=np.random.default_rng(123))
    assert r1["stochastic_fraction_ci95"] == r2["stochastic_fraction_ci95"]
    assert r1["parametric_fraction_ci95"] == r2["parametric_fraction_ci95"]


# --------------------------------------------------------------------------
# 4. Extrator de resposta padrão
# --------------------------------------------------------------------------

def test_default_response_value_prefers_gorkov_trap_depth():
    summary = {
        "acoustic_diagnostics": {
            "particle_classes": {
                "near_field_fe_oxyhydroxide_aggregate": {"trap_depth_over_kT": 0.042}
            }
        },
        "prebiotic_summary": {"top_hotspot_enrichment_vs_control": 7.5},
    }
    assert vd.default_response_value(summary) == pytest.approx(0.042)


def test_default_response_value_falls_back_to_enrichment_without_acoustics():
    summary = {
        "acoustic_diagnostics": None,
        "prebiotic_summary": {"top_hotspot_enrichment_vs_control": 3.3},
    }
    assert vd.default_response_value(summary) == pytest.approx(3.3)


def test_default_response_value_returns_none_when_nothing_available():
    summary = {"acoustic_diagnostics": None, "prebiotic_summary": {"top_hotspot_enrichment_vs_control": None}}
    assert vd.default_response_value(summary) is None

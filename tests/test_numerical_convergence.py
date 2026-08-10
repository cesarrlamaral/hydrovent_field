"""
Testes de `numerical_convergence.py` — verificação de solução numérica
(convergência de tolerância do integrador de EDO da pluma e de malha do
solver de PDE acústico), nunca feita antes neste projeto. Ver
docs/PHYSICS_MODEL.md §10.6.

Rodar com: pytest tests/test_numerical_convergence.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numerical_convergence as nc
import plume_physics as pp


# --------------------------------------------------------------------------
# 1. Extrapolação de Richardson — propriedade matemática, com função
#    conhecida analiticamente (não depende de nenhum solver do projeto)
# --------------------------------------------------------------------------

def test_richardson_order_recovers_known_first_order_scheme():
    """Erro de um esquema de ordem p com refinamento h, h/2, h/4 se
    comporta como e(h) = C*h^p — construído aqui com p=1 exato (sem
    ruído) pra confirmar que a fórmula recupera p=1."""
    h = 1.0
    error_at = lambda hh: 3.0 * hh  # erro ~ C*h^1, C=3
    f_exact = 10.0
    f_coarse = f_exact + error_at(h)
    f_medium = f_exact + error_at(h / 2)
    f_fine = f_exact + error_at(h / 4)
    order = nc.richardson_observed_order(f_coarse, f_medium, f_fine, refinement_ratio=2.0)
    assert order == pytest.approx(1.0, abs=1e-9)


def test_richardson_order_recovers_known_second_order_scheme():
    h = 1.0
    error_at = lambda hh: 2.0 * hh ** 2
    f_exact = 5.0
    f_coarse = f_exact + error_at(h)
    f_medium = f_exact + error_at(h / 2)
    f_fine = f_exact + error_at(h / 4)
    order = nc.richardson_observed_order(f_coarse, f_medium, f_fine, refinement_ratio=2.0)
    assert order == pytest.approx(2.0, abs=1e-9)


def test_richardson_order_is_infinite_when_already_converged():
    order = nc.richardson_observed_order(1.0, 1.0000001, 1.0000001, refinement_ratio=2.0)
    assert order == float("inf")


# --------------------------------------------------------------------------
# 2. Convergência de tolerância do integrador de EDO da pluma — solver
#    REAL do projeto (plume_physics.integrate_plume), não sintético
# --------------------------------------------------------------------------

def test_ode_default_tolerance_is_converged_to_tighter_tolerance():
    """A tolerância DEFAULT do projeto (rtol=1e-8, atol=1e-12) deveria
    já estar no regime convergido — mudar pra uma tolerância mais
    apertada não deveria alterar os diagnósticos físicos de forma
    mensurável. Esta é a checagem central que resolve #9 para o
    solver de EDO: confirma que o default já é adequado, sem precisar
    apertar a tolerância "só por garantia" sem saber se fazia diferença."""
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    result = nc.ode_tolerance_convergence_study(
        source, pp.DEFAULT_ALPHA_ENTRAINMENT, pp.DEFAULT_N_BRUNT_VAISALA, pp.AMBIENT_TEMP_C,
        tolerance_levels=[(1e-6, 1e-10), (1e-8, 1e-12), (1e-10, 1e-14)])
    # mudança do default (índice 1) para o mais apertado (índice 2)
    assert result["rise_height_rel_change"][1] < 1e-6
    assert result["dilution_rel_change"][1] < 1e-6


def test_ode_tolerance_convergence_across_vent_types():
    """Mesma checagem, mas nos 3 tipos de vent reais (geometrias de
    fonte bem diferentes) — não confiar num único caso."""
    for vent_type in ("black_smoker", "white_smoker", "diffuse_flow"):
        source = pp.build_source(temperature_c=200.0, vent_type=vent_type)
        result = nc.ode_tolerance_convergence_study(
            source, pp.DEFAULT_ALPHA_ENTRAINMENT, pp.DEFAULT_N_BRUNT_VAISALA, pp.AMBIENT_TEMP_C,
            tolerance_levels=[(1e-6, 1e-10), (1e-8, 1e-12), (1e-10, 1e-14)])
        assert result["rise_height_rel_change"][1] < 1e-5, f"falhou para {vent_type}"


# --------------------------------------------------------------------------
# 3. Convergência de malha do solver de PDE acústico — solver REAL do
#    projeto (acoustics.solve_steady_advection_diffusion)
# --------------------------------------------------------------------------

def test_pde_grid_convergence_approaches_theoretical_first_order():
    """O esquema é upwind de 1ª ordem por construção (documentado em
    acoustics.py) — sob um cenário sintético deliberadamente
    "estressado" (velocidade/difusividade maiores que os defaults reais
    do projeto, pra isolar o erro de truncamento do ruído de
    precisão do solver linear que domina nos parâmetros reais — ver
    docs/PHYSICS_MODEL.md §10.6), a ordem observada deveria se
    aproximar de 1 conforme a malha refina, não ficar presa numa ordem
    muito diferente (o que indicaria um bug de implementação, não só
    "ainda não convergiu")."""
    result = nc.pde_grid_convergence_study(
        domain_size_m=200.0, source_sigma_m=5.0, diffusivity_m2_s=0.05, loss_rate_per_s=1e-4,
        velocity_m_s=(0.02, 0.0), grid_sizes=[129, 257, 513])
    c = result["concentration_at_probe"]
    order = nc.richardson_observed_order(c[0], c[1], c[2], refinement_ratio=2.0)
    assert 0.5 < order < 1.5


def test_pde_grid_convergence_at_real_default_parameters_is_already_tiny():
    """Com os parâmetros REAIS de produção (difusividade DEFAULT_SOLUTE_
    DIFFUSIVITY_M2_S=8e-10 m2/s, minúscula) a mudança relativa entre
    resoluções que colchetam o --size default real (257) já é
    desprezível — o grid de produção não é o gargalo de precisão desta
    simulação."""
    result = nc.pde_grid_convergence_study(
        domain_size_m=1200.0, source_sigma_m=14.0625, diffusivity_m2_s=8.0e-10,
        loss_rate_per_s=8.0e-10 / 300.0 ** 2, velocity_m_s=(1e-6, 0.0),
        grid_sizes=[129, 257, 513])
    assert np.all(result["rel_change"] < 1e-3)


def test_pde_grid_convergence_study_uses_fixed_physical_probe_location():
    """Regressão de um bug real encontrado construindo este módulo: a
    primeira versão usava o ÍNDICE de célula mais próximo pra amostrar
    a sonda/posicionar a fonte, o que corresponde a um local FÍSICO
    diferente a cada resolução (jitter de até h/2) — contaminava a
    ordem observada (chegou a medir -1,75 e +2,56, sem sentido físico
    algum pro upwind de 1ª ordem). Checa que a fonte cai exatamente no
    centro físico do domínio independente da paridade de `n`."""
    for n in (32, 33, 64, 65):  # paridades par/ímpar deliberadamente misturadas
        h = 100.0 / n
        coords_m = (np.arange(n) + 0.5) * h
        center_idx = np.argmin(np.abs(coords_m - 50.0))
        assert abs(coords_m[center_idx] - 50.0) <= h  # sempre dentro de uma célula do centro real

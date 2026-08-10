"""
Testes de validação do módulo de termoforese calibrado (prebiotic.py,
`module_thermophoresis`). A classe "nucleotideos" usa uma fórmula
calibrada com dados reais medidos por Baaske et al. (2007, PNAS
104(22):9346-9351) — estes testes reproduzem os exemplos numéricos
publicados no artigo. A fórmula (Eq. 1 do artigo, k=0.42) foi
verificada por leitura direta do texto primário completo em
2026-08-06 (ver docs/PHYSICS_MODEL.md §8.2) — não é mais uma
reconstrução por regressão. As outras três classes (aminoácidos,
lipídeos, açúcares) continuam com a fórmula ilustrativa antiga, sem
medição equivalente — testado como regressão para garantir que a
generalização da fórmula não alterou o comportamento delas.

Rodar com: pytest tests/test_prebiotic.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prebiotic as pb


class _FakeVent:
    def __init__(self, temperature_c):
        self.temperature_c = temperature_c


def _params_with_fixed_delta_t(base_params: dict, delta_t_k: float, **overrides) -> dict:
    """Fixa ΔT em um valor exato (independente da temperatura do vent) para testar a fórmula isoladamente."""
    p = dict(base_params)
    p["pore_delta_t_min_k"] = delta_t_k
    p["pore_delta_t_max_k"] = delta_t_k
    p.update(overrides)
    return p


# --------------------------------------------------------------------------
# 1. Benchmarks contra os exemplos numéricos publicados por Baaske et al. (2007)
# --------------------------------------------------------------------------
# Exemplos do artigo, ΔT=30K: nucleotídeo único, razão de aspecto 10:1,
# S_T=0.015/K (condição de sal baixo, 1.7 mM) -> ~7x; razão de aspecto
# 125:1, mesmo S_T -> ~10^10x (Fig. 2a / Table 2 do artigo). Tolerância
# ampla mantida mesmo com k=0.42 verificado, porque os valores "~7x" e
# "~10^10x" no próprio artigo já são arredondamentos da simulação
# numérica por elementos finitos (a Eq. 1 analítica é uma aproximação
# confirmada pelos autores, não um valor exato ponto a ponto).

def test_thermophoresis_matches_baaske_short_pore_example():
    base = pb.MOLECULE_CLASSES["nucleotideos"]
    params = _params_with_fixed_delta_t(base, 30.0, soret_coefficient_per_k=0.015, pore_aspect_ratio=10.0)
    vent = _FakeVent(temperature_c=400.0)  # temp_frac=1, irrelevante já que delta_t está fixo
    result = pb.module_thermophoresis(vent, True, params)

    assert 4.0 < result < 12.0, f"esperado ~7x (Baaske et al. 2007, r=10/ΔT=30K/S_T=0.015), obtido {result:.2f}x"


def test_thermophoresis_matches_baaske_long_pore_example():
    base = pb.MOLECULE_CLASSES["nucleotideos"]
    params = _params_with_fixed_delta_t(base, 30.0, soret_coefficient_per_k=0.015, pore_aspect_ratio=125.0)
    vent = _FakeVent(temperature_c=400.0)
    result = pb.module_thermophoresis(vent, True, params)

    assert 1e8 < result < 1e12, f"esperado ~10^10x (Baaske et al. 2007, r=125/ΔT=30K/S_T=0.015), obtido {result:.3e}x"


def test_thermophoresis_scales_correctly_with_aspect_ratio():
    """Verificação de consistência interna: dobrar a razão de aspecto deve elevar o expoente ao quadrado (log-linear)."""
    base = pb.MOLECULE_CLASSES["nucleotideos"]
    vent = _FakeVent(temperature_c=400.0)

    p10 = _params_with_fixed_delta_t(base, 20.0, pore_aspect_ratio=10.0)
    p20 = _params_with_fixed_delta_t(base, 20.0, pore_aspect_ratio=20.0)
    r10 = pb.module_thermophoresis(vent, True, p10)
    r20 = pb.module_thermophoresis(vent, True, p20)

    # log(enhancement) é linear em pore_aspect_ratio -> log(r20) deve ser 2x log(r10)
    assert abs(np.log(r20) - 2 * np.log(r10)) < 1e-9


# --------------------------------------------------------------------------
# 2. Regressão: as outras 3 classes continuam com a fórmula ilustrativa antiga
# --------------------------------------------------------------------------

def test_thermophoresis_unchanged_for_uncalibrated_classes():
    vent = _FakeVent(temperature_c=400.0)  # temp_frac=1 -> delta_t = pore_delta_t_max_k = 20.0
    for key in ("aminoacidos", "lipideos", "acucares"):
        params = pb.MOLECULE_CLASSES[key]
        assert params["thermophoresis_convection_coupled"] is False
        result = pb.module_thermophoresis(vent, True, params)
        expected = float(np.exp(params["soret_coefficient_per_k"] * params["pore_delta_t_max_k"]))
        assert abs(result - expected) < 1e-9, f"{key}: fórmula deveria ser exp(S_T*ΔT) sem acoplamento de convecção"


def test_thermophoresis_disabled_returns_neutral_factor():
    vent = _FakeVent(temperature_c=400.0)
    for key in pb.MOLECULE_CLASSES:
        assert pb.module_thermophoresis(vent, False, pb.MOLECULE_CLASSES[key]) == 1.0


# --------------------------------------------------------------------------
# 3. Sanidade dos defaults calibrados de "nucleotideos"
# --------------------------------------------------------------------------

def test_nucleotide_class_uses_measured_soret_coefficient_not_illustrative_guess():
    params = pb.MOLECULE_CLASSES["nucleotideos"]
    assert params["thermophoresis_convection_coupled"] is True
    # 0.006/K é o valor medido (170 mM de sal) de Baaske et al. (2007) — não mais o
    # coeficiente ilustrativo antigo (0.12/K) que não tinha nenhuma medição por trás.
    assert abs(params["soret_coefficient_per_k"] - 0.006) < 1e-9
    assert params["pore_aspect_ratio"] > 0


# --------------------------------------------------------------------------
# 4. Gradiente de prótons — referência biológica real (Nernst, Sojo et al. 2016)
# --------------------------------------------------------------------------
# gradient_frac agora compara o potencial de Nernst do ΔpH simulado
# contra a força próton-motriz real necessária para fixação de carbono
# em organismos extantes (~3 unidades de pH / ~177.6 mV) — não mais um
# auto-normalizador do próprio modelo (MAX_DELTA_PH).

class _FakeVentProton:
    def __init__(self, vent_type, ph):
        self.vent_type = vent_type
        self.chemistry = {"pH": ph}


def test_nernst_reference_matches_sojo_et_al_2016():
    """~3 unidades de pH a 59.2 mV/unidade (25°C) -> ~177.6 mV, o valor citado como referência biológica."""
    assert abs(pb.REFERENCE_PROTON_MOTIVE_FORCE_MV - 177.6) < 0.1


def test_proton_gradient_frac_equals_one_at_reference_ph_units():
    """No ΔpH exatamente igual à referência biológica (3 unidades), gradient_frac deve valer exatamente 1."""
    vent = _FakeVentProton("white_smoker", ph=pb.SEAWATER_PH - pb.REFERENCE_PROTON_GRADIENT_PH_UNITS)
    params = dict(pb.DEFAULT_PARAMS)
    params["proton_max_factor"] = 10.0
    params["proton_vent_type_weight"] = {"white_smoker": 1.0}
    result = pb.module_proton_gradient(vent, True, params)
    assert abs(result - (1.0 + 10.0 * 1.0 * 1.0)) < 1e-9


def test_proton_gradient_scales_linearly_with_delta_ph():
    """O potencial de Nernst é linear em ΔpH -> dobrar ΔpH deve dobrar (fator-1)/peso."""
    params = dict(pb.DEFAULT_PARAMS)
    params["proton_max_factor"] = 10.0
    params["proton_vent_type_weight"] = {"white_smoker": 1.0}

    v1 = _FakeVentProton("white_smoker", ph=pb.SEAWATER_PH - 1.5)
    v2 = _FakeVentProton("white_smoker", ph=pb.SEAWATER_PH - 3.0)
    f1 = pb.module_proton_gradient(v1, True, params) - 1.0
    f2 = pb.module_proton_gradient(v2, True, params) - 1.0
    assert abs(f2 - 2 * f1) < 1e-9


def test_proton_gradient_disabled_returns_neutral_factor():
    vent = _FakeVentProton("black_smoker", ph=3.2)
    assert pb.module_proton_gradient(vent, False, pb.DEFAULT_PARAMS) == 1.0

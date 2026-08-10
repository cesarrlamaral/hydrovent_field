"""
Testes de validação do modelo de pluma hidrotermal (plume_physics.py,
reaction_kinetics.py). Cada teste ancora um benchmark citável da
literatura (ver docs/PHYSICS_MODEL.md) e declara explicitamente a
tolerância e a razão dela — nenhuma tolerância é "porque passou".

Rodar com: pytest tests/test_plume_physics.py -v
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plume_physics as pp
import reaction_kinetics as rk


# --------------------------------------------------------------------------
# 1. Limite analítico não-estratificado: Q(z) ~ z^(5/3)
# --------------------------------------------------------------------------
# Resultado clássico de plumas turbulentas puras não-estratificadas (sem
# consumo de flutuabilidade), citado universalmente na literatura de
# plumas (Morton, Taylor & Turner, 1956; Turner, J.S. (1973), Buoyancy
# Effects in Fluids, Cambridge Univ. Press). Testado no regime
# assintótico (longe da fonte), onde o transiente inicial já não
# domina.

def test_unstratified_scaling_matches_classical_5_3_power_law():
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source, n_freq=0.0, z_max=2000.0, n_points=400)

    mask = (profile.z > 500) & (profile.z < 1900)
    log_z = np.log(profile.z[mask])
    log_q = np.log(profile.q[mask])
    slope = np.polyfit(log_z, log_q, 1)[0]

    assert abs(slope - 5.0 / 3.0) < 0.05, (
        f"expoente de escala Q~z^slope = {slope:.4f}, esperado ~5/3=1.6667 "
        "(regime assintótico não-estratificado, MTT 1956)"
    )


# --------------------------------------------------------------------------
# 2. Altura de flutuabilidade neutra vs. forma fechada MTT
# --------------------------------------------------------------------------
# Forma fechada derivada diretamente de Morton, Taylor & Turner (1956),
# eqs. (10)/(14) (locus x1=2.125, onde a flutuabilidade se anula —
# mesmo evento B(z)=0 usado pelo integrador): z ~= 0.7326 * alpha^-0.5 *
# B0^0.25 * N^-0.75. Verificada nesta sessão por três vias independentes
# (reprodução numérica exata da Tabela 1 do artigo, verificação
# algébrica das eqs. 7/8/10, e conferência do coeficiente dimensional
# 0.410 de eq. 14) — ver docs/PHYSICS_MODEL.md §2.2. O locus x1=2.125
# (flutuabilidade nula) é exatamente o mesmo evento B(z)=0 que a
# integração numérica usa como parada, então — diferente da fórmula
# "2.98" anterior, sem dependência em alpha e sem locus correspondente
# claro — aqui não há diferença estrutural esperada entre as duas
# formulações; a concordância observada é <1%. Tolerância de 10% cobre
# o overshoot residual de momento antes do evento disparar e o erro
# numérico do integrador (~0.1%, ver teste de conservação de massa
# abaixo), não uma diferença de modelo.

def test_rise_height_reconciles_with_mtt_closed_form():
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source)

    analytic_height = (
        0.7326 * pp.DEFAULT_ALPHA_ENTRAINMENT ** -0.5
        * source.b0 ** 0.25 * pp.DEFAULT_N_BRUNT_VAISALA ** -0.75
    )
    ratio = profile.rise_height_m / analytic_height

    assert 0.9 < ratio < 1.1, (
        f"altura numérica={profile.rise_height_m:.1f}m vs. forma fechada MTT="
        f"{analytic_height:.1f}m, razão={ratio:.3f} (esperado 0.9-1.1)"
    )
    assert profile.reached_neutral_buoyancy


# --------------------------------------------------------------------------
# 3. Diluição em campo próximo: Mottl & McConachy (1990)
# --------------------------------------------------------------------------
# Mottl, M.J., & McConachy, T.F. (1990). "Chemical processes in buoyant
# hydrothermal plumes on the East Pacific Rise near 21°N." GCA 54,
# 1911-1927. Usando Li como traçador conservativo, mediram razões de
# mistura de 10^2 a 10^4 g água do mar / g fluido de vent nos primeiros
# 22 m acima de black smokers 273-350°C. A tolerância aqui é alargada
# meia ordem de grandeza para baixo (limite inferior 30 em vez de 100)
# porque o raio/velocidade de saída do orifício no modelo (não medidos,
# só plausíveis) e o alpha usado (default 0.1, dentro da faixa medida
# 0.07-0.18) deslocam o resultado dentro dessa faixa de incerteza
# documentada — ver o teste de sensibilidade a alpha mais abaixo.

def test_mottl_mcconachy_dilution_at_22m():
    source = pp.build_source(temperature_c=310.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source)
    dilution_22m = pp.dilution_at_height(profile, 22.0)

    assert 30.0 < dilution_22m < 2e4, (
        f"diluição em z=22m = {dilution_22m:.1f}, esperado dentro de "
        "10^2-10^4 (Mottl & McConachy 1990) com folga de meia ordem de "
        "grandeza para a incerteza de alpha/geometria do orifício"
    )


def test_dilution_at_22m_within_literature_band_across_alpha_uncertainty_range():
    """Varrendo alpha pela faixa medida em campo (0.07-0.18, Rona et al.
    2006), pelo menos um extremo deve cair dentro da banda 10^2-10^4
    citada por Mottl & McConachy (1990) — evidência de que o modelo é
    consistente com o benchmark dentro da incerteza paramétrica
    conhecida, não só com o valor default."""
    source = pp.build_source(temperature_c=310.0, vent_type="black_smoker")
    alpha_lo, alpha_hi = pp.ALPHA_ENTRAINMENT_RANGE
    dilutions = []
    for alpha in (alpha_lo, pp.DEFAULT_ALPHA_ENTRAINMENT, alpha_hi):
        profile = pp.integrate_plume(source, alpha=alpha)
        dilutions.append(pp.dilution_at_height(profile, 22.0))

    assert any(1e2 <= d <= 1e4 for d in dilutions), (
        f"nenhum valor de alpha na faixa medida {pp.ALPHA_ENTRAINMENT_RANGE} "
        f"produziu diluição em 10^2-10^4 em z=22m; valores obtidos: {dilutions}"
    )


# --------------------------------------------------------------------------
# 4. Diluição na altura de flutuabilidade neutra: ~10^4:1
# --------------------------------------------------------------------------
# Lupton, J.E., Delaney, J.R., Johnson, H.P., & Tivey, M.K. (1985).
# "Entrainment and vertical transport of deep-ocean water by buoyant
# hydrothermal plumes." Nature 316, 621-623: plumas hidrotermais são
# tipicamente misturas de 1 parte fluido de vent para 10^4 partes água
# do mar ambiente na altura de flutuabilidade neutra. Tolerância de uma
# ordem de grandeza para cada lado é generosa mas honesta: este é um
# valor "típico" de campo, não uma constante universal.

def test_dilution_at_neutral_buoyancy_is_order_1e4():
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source)
    dilution_top = float(profile.dilution[-1])

    assert 1e3 <= dilution_top <= 1e5, (
        f"diluição na altura de flutuabilidade neutra = {dilution_top:.0f}, "
        "esperado ordem de grandeza ~10^4 (Lupton et al. 1985)"
    )


# --------------------------------------------------------------------------
# 5. Diluição média nos primeiros 150m: Rudnicki & Elderfield (1993), TAG
# --------------------------------------------------------------------------
# Rudnicki, M.D., & Elderfield, H. (1993). "A chemical model of the
# buoyant and neutrally buoyant plume above the TAG vent field." GCA 57,
# 2939-2957. Diluição média no ponto de precipitação de Fe-oxi-hidróxido,
# calculada a partir de dados de CTD dentro dos primeiros 150 m de
# ascensão: Ē_Fe ~= 570. Tolerância de fator 3 para cada lado: é um
# único ponto derivado de campo (não uma lei constitutiva), e nosso
# campo é procedural/sintético, não uma réplica geométrica do TAG.

def test_tag_average_dilution_first_150m():
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source, z_max=500.0)
    mask = profile.z <= 150.0
    assert mask.sum() > 5, "resolução insuficiente nos primeiros 150m para uma média confiável"
    avg_dilution_150m = float(np.mean(profile.dilution[mask]))

    assert 190.0 <= avg_dilution_150m <= 1710.0, (
        f"diluição média nos primeiros 150m = {avg_dilution_150m:.0f}, "
        "esperado próximo a 570 (fator 3, Rudnicki & Elderfield 1993, TAG)"
    )


# --------------------------------------------------------------------------
# 6. Teste adversarial/falseável: assimetria de cinética Fe(II) Atlântico x Pacífico
# --------------------------------------------------------------------------
# Field, M.P., & Sherrell, R.M. (2000). GCA 64, 619-628: água profunda
# do Pacífico tem pH e O2 mais baixos que a do Atlântico, produzindo
# oxidação de Fe(II) sistematicamente mais lenta (~1 ordem de grandeza
# ou mais) na mesma diluição. Este é um teste que o modelo PODE falhar
# se a parametrização por bacia estiver incorreta — não foi ajustado
# para passar por construção além da escolha das meias-vidas de campo
# citadas em reaction_kinetics.py.

def test_atlantic_fe_oxidation_faster_than_pacific_by_at_least_one_order_of_magnitude():
    t_c = 50.0  # temperatura arbitrária de teste, mesma para as duas bacias
    k_atlantic = rk.k_fe2(t_c, basin="atlantic")
    k_pacific = rk.k_fe2(t_c, basin="pacific")

    assert k_atlantic > k_pacific, "cinética de Fe(II) do Atlântico deveria ser mais rápida que a do Pacífico"
    assert k_atlantic / k_pacific >= 10.0, (
        f"razão k_atlantic/k_pacific = {k_atlantic / k_pacific:.1f}, esperado >= 10x "
        "(Field & Sherrell 2000)"
    )


# --------------------------------------------------------------------------
# 7. Identidade de conservação de massa (auto-consistência numérica)
# --------------------------------------------------------------------------
# Checagem interna, independente de literatura: dQ/dz derivado
# numericamente do perfil resolvido deve bater com o fechamento de
# entranhamento 2*sqrt(pi)*alpha*sqrt(M) usado para construir o próprio
# sistema de EDOs. Tolerância apertada (1%) porque isto testa
# corretude de implementação/integração numérica, não incerteza física.

def test_entrainment_closure_mass_conservation_identity():
    source = pp.build_source(temperature_c=350.0, vent_type="black_smoker")
    profile = pp.integrate_plume(source)

    dq_dz_numeric = np.gradient(profile.q, profile.z)
    dq_dz_analytic = 2.0 * math.sqrt(math.pi) * profile.alpha * np.sqrt(np.maximum(profile.m, 0.0))

    interior = slice(5, -5)
    rel_err = np.abs(dq_dz_numeric[interior] - dq_dz_analytic[interior]) / np.abs(dq_dz_analytic[interior])

    assert rel_err.max() < 0.01, f"erro relativo máximo na identidade de entranhamento = {rel_err.max():.4f}, esperado < 1%"


# --------------------------------------------------------------------------
# 8. Velocidades de saída do orifício — validação contra medições de campo
# --------------------------------------------------------------------------
# black_smoker: 0.7-2.4 m/s, medição direta por flowmeter de turbina in
# situ, Converse, Holland & Edmond (1984), Earth Planet. Sci. Lett. 69,
# 159-175 (verificado por leitura direta do PDF primário completo em
# 2026-08-06 — a faixa "1-5 m/s" usada aqui antes dessa verificação era
# a estimativa de Macdonald et al. 1980 citada de segunda mão DENTRO do
# artigo de Converse, não a medição própria dele; corrigido). diffuse_flow:
# ~0.001-0.111 m/s combinando Mittelstaedt et al. (2012), G-cubed 13,
# Q0AF04 (0.009-0.111 m/s) e um estudo anterior de sensor duplo no
# mesmo edifício (0.0011-0.0049 m/s). white_smoker permanece sem
# medição encontrada — não testado aqui.

def test_black_smoker_exit_velocity_within_measured_field_range():
    assert 0.7 <= pp.EXIT_VELOCITY_BY_TYPE["black_smoker"] <= 2.4


def test_diffuse_flow_exit_velocity_within_measured_field_range():
    assert 0.0011 <= pp.EXIT_VELOCITY_BY_TYPE["diffuse_flow"] <= 0.111

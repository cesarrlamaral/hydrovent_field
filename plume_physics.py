"""
Modelo integral de pluma turbulenta flutuante (Morton, Taylor & Turner,
1956) aplicado a fumarolas hidrotermais, com transporte reativo de
traçadores acoplado (extensão de Rudnicki & Elderfield, 1992).

Substitui a antiga fórmula de altura única `3.8*F^0.25*N^-0.75`
(Speer & Rona, 1989) por uma integração numérica completa do sistema de
EDOs de entranhamento, da qual a diluição D(z)=Q(z)/Q0 e a altura de
ascensão saem como resultado, não como fórmula fechada. A fórmula
fechada permanece útil apenas como *teste de validação* (ver
tests/test_plume_physics.py) — não como implementação.

Todas as constantes e defaults têm citação e faixa de incerteza
documentadas em docs/PHYSICS_MODEL.md; leia esse arquivo antes de mudar
qualquer valor aqui.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

# --------------------------------------------------------------------------
# Constantes físicas e defaults calibrados (ver docs/PHYSICS_MODEL.md)
# --------------------------------------------------------------------------

RHO_SEAWATER = 1027.0            # kg/m^3
CP_SEAWATER = 3990.0             # J/(kg K)
G = 9.81                         # m/s^2
THERMAL_EXPANSION_COEF = 2.0e-4  # 1/K, aproximação para água do mar fria
AMBIENT_TEMP_C = 2.0             # °C, água do mar profunda ambiente

# α: coeficiente de entranhamento. Default "customarily used with
# integral theory" para plumas hidrotermais — Lavelle, J.W. (1997),
# JGR 102(C2):3405-3420. Faixa medida em campo (Grotto vent, Main
# Endeavour Field): 0.07-0.18 — Rona, P.A., Bemis, K.G., Jones, C.D.,
# Jackson, D.R., Mitsuzawa, K., & Silver, D. (2006), "Entrainment and
# bending in a major hydrothermal plume, Main Endeavour Field, Juan de
# Fuca Ridge," Geophys. Res. Lett. 33, L19313, doi:10.1029/2006GL027211
# (CORREÇÃO 2026-08-06, verificado por leitura direta do PDF primário
# completo, biblios/2006gl027211.pdf: a citação usada aqui antes desta
# verificação tinha autor principal, título e número de artigo errados
# — "Bemis, Jones & Jackson, 'Plume anomaly detected by acoustic
# Doppler current profiler,' L02613" — só o DOI estava correto; o valor
# numérico 0.07-0.18 já estava certo e bate exatamente com a Tabela 1
# do artigo real). Constante-α é conhecida por falhar nos primeiros
# ~2 m acima do orifício (Lemaréchal, Roullet & Gula, 2025, JGR Oceans
# 130(10)) — limitação documentada, não corrigida nesta fase.
DEFAULT_ALPHA_ENTRAINMENT = 0.1
ALPHA_ENTRAINMENT_RANGE = (0.07, 0.18)

# N: frequência de Brunt-Väisälä. Único valor oceânico de dorsal
# verificado nas fontes consultadas: Juan de Fuca Ridge, 2100-2350 m —
# Lavelle (1997). Não há valor publicado verificado para MAR/EPR nas
# fontes consultadas; usado aqui como fallback citável, não como
# universal — trate como parâmetro a calibrar por sítio quando possível.
DEFAULT_N_BRUNT_VAISALA = 7.9e-4  # s^-1

# Velocidades de saída por tipo de vent.
#
# black_smoker (1.5 m/s): validado dentro da faixa medida em campo por
# medição direta (flowmeter de turbina in situ, "Alvin"), 0.7-2.4 m/s
# — Converse, D.R., Holland, H.D., & Edmond, J.M. (1984), "Flow rates
# in the axial hot springs of the East Pacific Rise (21°N):
# implications for the heat budget and the formation of massive sulfide
# deposits," Earth Planet. Sci. Lett. 69, 159-175 (verificado por
# leitura direta do PDF primário completo em 2026-08-06,
# biblios/0012-821x2990080-3.pdf — CORREÇÃO: a faixa citada
# anteriormente aqui, "1-5 m/s," não é a medição própria de Converse et
# al.; é a estimativa de Macdonald et al. (1980, EPSL 48, 1-7) para o
# vent "National Geographic," citada de segunda mão DENTRO do artigo de
# Converse — os dois números não são intercambiáveis. O valor do
# modelo, 1.5 m/s, permanece dentro da faixa medida diretamente por
# Converse et al., 0.7-2.4 m/s, então a validação do valor não muda,
# só a atribuição da citação).
#
# diffuse_flow (0.05 m/s): validado dentro da faixa combinada medida em
# campo, ~0.001-0.111 m/s — Mittelstaedt, E., et al. (2012),
# "Quantifying diffuse and discrete venting at the Tour Eiffel vent
# site, Lucky Strike hydrothermal field," Geochem. Geophys. Geosyst. 13,
# Q0AF04 (0.009-0.111 m/s, velocimetria óptica); Sarrazin, J., Rodier,
# P., Tivey, M.K., Singh, H., Schultz, A., & Sarradin, P.-M. (2009),
# "A dual sensor device to estimate fluid flow velocity at diffuse
# hydrothermal vents," Deep-Sea Res. I 56(11), 2065-2074, relatou
# 0.0011-0.0049 m/s no mesmo edifício, em fraturas de baixa temperatura
# (4.5-16.4°C) — a discrepância de ~1 ordem de
# grandeza entre os dois métodos/condições não está resolvida, tratada
# aqui como a incerteza real da faixa, não escondida.
#
# white_smoker (0.6 m/s): NENHUMA medição de velocidade de saída
# específica para white smokers foi encontrada na pesquisa de
# literatura que fundamenta este modelo — permanece um valor plausível
# não citado (intermediário entre black smoker e diffuse flow por
# suposição, não por medição).
EXIT_VELOCITY_BY_TYPE = {
    "black_smoker": 1.5,
    "white_smoker": 0.6,
    "diffuse_flow": 0.05,
}

DEFAULT_VENT_RADIUS_M = 0.15


@dataclass
class PlumeSource:
    """Condições de contorno na base da pluma (z=0, orifício do vent)."""
    q0: float  # fluxo de volume inicial, m^3/s
    m0: float  # fluxo de momento inicial, m^4/s^2
    b0: float  # fluxo de flutuabilidade inicial, m^4/s^3


def build_source(temperature_c: float, vent_type: str,
                  vent_radius_m: float = DEFAULT_VENT_RADIUS_M,
                  ambient_temp_c: float = AMBIENT_TEMP_C) -> PlumeSource:
    """Deriva Q0, M0, B0 a partir da temperatura do fluido e geometria do
    orifício, usando a aproximação de Boussinesq (g' = g*alpha_T*deltaT)."""
    delta_t = max(temperature_c - ambient_temp_c, 1.0)
    area = math.pi * vent_radius_m ** 2
    w0 = EXIT_VELOCITY_BY_TYPE[vent_type]
    q0 = area * w0
    m0 = q0 * w0
    g_prime0 = G * THERMAL_EXPANSION_COEF * delta_t
    b0 = g_prime0 * q0
    return PlumeSource(q0=q0, m0=m0, b0=b0)


@dataclass
class PlumeProfile:
    """Perfil vertical resolvido da pluma, do orifício até a altura de
    flutuabilidade neutra (ou até o topo de integração, se a
    flutuabilidade nunca zerar — ver `reached_neutral_buoyancy`)."""
    z: np.ndarray                  # altura acima do orifício, m
    q: np.ndarray                  # fluxo de volume, m^3/s
    m: np.ndarray                  # fluxo de momento, m^4/s^2
    b: np.ndarray                  # fluxo de flutuabilidade, m^4/s^3
    t: np.ndarray                  # tempo desde a emissão, s
    w: np.ndarray                  # velocidade vertical, m/s
    radius: np.ndarray             # raio da pluma, m
    dilution: np.ndarray           # D(z) = Q(z)/Q0, adimensional
    temperature_c: np.ndarray      # temperatura local da pluma, °C
    q0: float
    rise_height_m: float           # altura de flutuabilidade neutra (ou topo de integração)
    reached_neutral_buoyancy: bool
    alpha: float
    n_freq: float


def _ode_rhs(z, y, alpha: float, n_freq: float):
    q, m, b, t = y
    m_safe = max(m, 1e-12)
    q_safe = max(q, 1e-12)
    dq_dz = 2.0 * math.sqrt(math.pi) * alpha * math.sqrt(max(m, 0.0))
    dm_dz = 2.0 * q * b / m_safe
    db_dz = -2.0 * (n_freq ** 2) * q
    dt_dz = q_safe / m_safe  # = 1/w
    return [dq_dz, dm_dz, db_dz, dt_dz]


def integrate_plume(source: PlumeSource,
                     alpha: float = DEFAULT_ALPHA_ENTRAINMENT,
                     n_freq: float = DEFAULT_N_BRUNT_VAISALA,
                     ambient_temp_c: float = AMBIENT_TEMP_C,
                     z_max: float = 500.0,
                     n_points: int = 200,
                     rtol: float = 1e-8,
                     atol: float = 1e-12) -> PlumeProfile:
    """Integra o sistema de EDOs de entranhamento MTT estratificado:

        dQ/dz = 2*sqrt(pi)*alpha*sqrt(M)
        dM/dz = 2*Q*B/M
        dB/dz = -2*N^2 * Q

    (derivado das formas de fluxo padrão Q=pi*b^2*w, M=pi*b^2*w^2,
    B=pi*b^2*w*g'; ver docs/PHYSICS_MODEL.md §2 para a derivação completa
    e a correção de 2026-08-08 do fator 2 que faltava em dM/dz e dB/dz,
    verificada por leitura direta de Morton, Taylor & Turner 1956, eqs.
    7(ii-iii) e 8). Integração até o evento B(z)=0 (altura de flutuabilidade neutra —
    a camada de pluma efetivamente observada em campo) ou até M(z)~0
    (momento esgotado), o que ocorrer primeiro. Para N=0 (não
    estratificado) nenhum evento dispara e a integração vai até z_max.

    `rtol`/`atol`: tolerâncias do RK45 adaptativo, expostas (defaults
    inalterados) para o estudo de convergência numérica em
    `numerical_convergence.py` — ver docs/PHYSICS_MODEL.md §10.6.
    """
    def event_neutral_buoyancy(z, y, *_args):
        return y[2]
    event_neutral_buoyancy.terminal = True
    event_neutral_buoyancy.direction = -1

    def event_momentum_exhausted(z, y, *_args):
        return y[1] - 1e-9
    event_momentum_exhausted.terminal = True
    event_momentum_exhausted.direction = -1

    y0 = [source.q0, source.m0, source.b0, 0.0]
    sol = solve_ivp(
        _ode_rhs, (0.0, z_max), y0, method="RK45", args=(alpha, n_freq),
        events=[event_neutral_buoyancy, event_momentum_exhausted],
        dense_output=True, max_step=max(z_max / n_points, 1e-3),
        rtol=rtol, atol=atol,
    )

    reached_neutral = len(sol.t_events[0]) > 0
    if reached_neutral:
        z_top = float(sol.t_events[0][0])
    elif len(sol.t_events[1]) > 0:
        z_top = float(sol.t_events[1][0])
    else:
        z_top = float(sol.t[-1])

    z_top = max(z_top, 1e-3)
    z_arr = np.linspace(0.0, z_top, n_points)
    q_arr, m_arr, b_arr, t_arr = sol.sol(z_arr)

    q_safe = np.maximum(q_arr, 1e-12)
    m_safe = np.maximum(m_arr, 1e-12)
    w_arr = m_arr / q_safe
    radius_arr = q_arr / np.sqrt(np.pi * m_safe)
    dilution_arr = q_arr / source.q0
    g_prime_arr = b_arr / q_safe
    delta_t_arr = g_prime_arr / (G * THERMAL_EXPANSION_COEF)
    temperature_arr = ambient_temp_c + delta_t_arr

    return PlumeProfile(
        z=z_arr, q=q_arr, m=m_arr, b=b_arr, t=t_arr, w=w_arr,
        radius=radius_arr, dilution=dilution_arr, temperature_c=temperature_arr,
        q0=source.q0, rise_height_m=z_top,
        reached_neutral_buoyancy=reached_neutral, alpha=alpha, n_freq=n_freq,
    )


def dilution_at_height(profile: PlumeProfile, z_query: float) -> float:
    """Interpola D(z)=Q(z)/Q0 no perfil já resolvido (interpolação linear
    entre os pontos amostrados). `z_query` é limitado a [0, z_top] do
    perfil — não extrapola além da altura de flutuabilidade neutra."""
    z_clamped = min(max(z_query, 0.0), float(profile.z[-1]))
    return float(np.interp(z_clamped, profile.z, profile.dilution))


def integrate_species_transport(profile: PlumeProfile, c0: float,
                                 k_fn: Optional[Callable[[float], float]] = None,
                                 prompt_removal_fraction: float = 0.0) -> np.ndarray:
    """Concentração C(z) de uma espécie química ao longo do perfil da pluma.

    Deriva da equação do traçador d(Q*C)/dz = -k(T(z))*Q*C/w(z) (extensão
    de Rudnicki & Elderfield, 1992, ao sistema MTT), cuja solução — usando
    dt/dz=1/w e concentração de fundo ambiente aproximada como zero (i.e.
    C representa o excesso acima do background) — é:

        C(z) = (C0_efetivo / D(z)) * exp(-∫[0,t(z)] k dt')

    onde D(z)=Q(z)/Q0 é a diluição conservativa (já resolvida em
    `profile`) e a exponencial captura a perda adicional por reação ao
    longo do tempo de residência na pluma. `k_fn` recebe a temperatura
    local da pluma em °C (derivada de B/Q, portanto internamente
    consistente com a física de flutuabilidade) e devolve uma constante
    de taxa pseudo-primeira-ordem em s^-1. `prompt_removal_fraction`
    modela perda quase instantânea próxima ao orifício (ex.: precipitação
    de sulfeto de Fe, Mottl & McConachy 1990) aplicada antes da diluição
    contínua.
    """
    c0_eff = c0 * (1.0 - prompt_removal_fraction)
    if k_fn is None:
        k_arr = np.zeros_like(profile.t)
    else:
        k_arr = np.array([k_fn(tc) for tc in profile.temperature_c])
    k_integral = cumulative_trapezoid(k_arr, profile.t, initial=0.0)
    dilution_safe = np.maximum(profile.dilution, 1e-12)
    return (c0_eff / dilution_safe) * np.exp(-k_integral)

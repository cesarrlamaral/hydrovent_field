"""
Cinética de reação para as espécies químicas transportadas na pluma
hidrotermal (H2S, Fe(II), Mn(II); CH4 é tratado como traçador
conservativo — nenhuma cinética de oxidação de CH4 foi encontrada na
pesquisa de literatura que fundamenta este módulo).

Abordagem: para cada espécie, ancoramos a *magnitude* da taxa numa
meia-vida medida em campo (num contexto de pluma hidrotermal real, na
temperatura de referência `T_REF_C` = água do mar ambiente) e a
*dependência com a temperatura* numa energia de ativação de laboratório
(lei de Arrhenius), quando disponível. Isso é uma aproximação
explícita: as meias-vidas de campo citadas são valores efetivos,
integrados ao longo do gradiente de temperatura da pluma, não medidas
numa única temperatura controlada — ver docs/PHYSICS_MODEL.md, seção
"Limitações".

Todas as constantes têm citação. Nenhum valor aqui deve ser alterado
sem atualizar também docs/PHYSICS_MODEL.md.
"""

from __future__ import annotations

import math

R_GAS = 8.314  # J/(mol K)
T_REF_C = 2.0  # °C — água do mar profunda ambiente, mesma referência de plume_physics.AMBIENT_TEMP_C


def _arrhenius_k(t_celsius: float, k_ref: float, ea_j_mol: float, t_ref_c: float = T_REF_C) -> float:
    t_k = t_celsius + 273.15
    t_ref_k = t_ref_c + 273.15
    return k_ref * math.exp(-ea_j_mol / R_GAS * (1.0 / t_k - 1.0 / t_ref_k))


# --------------------------------------------------------------------
# H2S
# --------------------------------------------------------------------
# Meia-vida base: Millero, F.J., Hubinger, S., Fernandez, M., & Garnett,
# S. (1987). "Oxidation of H2S in seawater as a function of temperature,
# pH, and ionic strength." Environ. Sci. Technol. 21(5), 439-443.
# t1/2 = 26 +/- 9 h em agua do mar, pH 8, 25 degC. Energia de ativacao
# (independente da forca ionica): 39 +/- 2 kJ/mol.
#
# Fator de realce em pluma: Radford-Knoery, J., German, C.R., Charlou,
# J.-L., Donval, J.-P., & Fouquet, Y. (2001). "Distribution and behavior
# of dissolved hydrogen sulfide in hydrothermal plumes." Limnol.
# Oceanogr. 46(2), 461-464. Reporta que a remocao de sulfeto em plumas
# reais e ~2 ordens de grandeza mais rapida que a cinetica de agua do
# mar de laboratorio (provavelmente oxidacao catalisada por
# metal/particula, ex. PNAS 2021 sobre Fe-catalyzed sulfide oxidation).
# Modelado aqui como fator explicito e ajustavel, nao escondido dentro
# de uma constante de taxa "base" inflada.

H2S_HALF_LIFE_SEAWATER_S = 26 * 3600.0  # Millero et al. 1987, pH 8, 25 degC
H2S_EA_J_MOL = 39_000.0
H2S_T_REF_C = 25.0
DEFAULT_H2S_PLUME_ENHANCEMENT = 100.0  # Radford-Knoery et al. 2001


def k_h2s(temperature_c: float, plume_enhancement: float = DEFAULT_H2S_PLUME_ENHANCEMENT) -> float:
    """Constante de taxa pseudo-primeira-ordem para oxidação de H2S, s^-1."""
    k_ref = math.log(2.0) / H2S_HALF_LIFE_SEAWATER_S
    k_lab = _arrhenius_k(temperature_c, k_ref, H2S_EA_J_MOL, t_ref_c=H2S_T_REF_C)
    return k_lab * plume_enhancement


# --------------------------------------------------------------------
# Fe(II)
# --------------------------------------------------------------------
# Energia de ativacao de laboratorio: Millero, F.J., Sotolongo, S., &
# Izaguirre, M. (1987). "The oxidation kinetics of Fe(II) in seawater."
# Geochim. Cosmochim. Acta 51, 793-801. Ea = 29 +/- 2 kJ/mol.
#
# Meias-vidas efetivas de campo, na pluma proxima, usadas para ancorar a
# magnitude por bacia oceanica (a assimetria Atlantico/Pacifico e um
# resultado citado explicitamente por Field & Sherrell 2000: agua
# profunda do Pacifico tem pH e O2 mais baixos -> oxidacao mais lenta):
#   Atlantico (TAG, pluma flutuante): t1/2 = 2.1 min
#     Rudnicki, M.D., & Elderfield, H. (1993). GCA 57, 2939-2957.
#   Pacifico (EPR 9 45'N): t1/2 = 3.3 h
#     Field, M.P., & Sherrell, R.M. (2000). GCA 64, 619-628.
#
# Precipitacao imediata de sulfeto de Fe proxima ao orificio (perda
# instantanea, aplicada antes da oxidacao continua): fracao ~68%
# calculada em EPR 9 45'N (Field & Sherrell 2000 -- CORRIGIDO
# 2026-08-08, verificado por leitura direta do PDF primario completo:
# "we calculate a Fe loss of ~68% (~5.6 uM) of the total Fe vented" via
# balanco de massa Fe/Mn; o valor usado aqui antes desta verificacao,
# 65%, nao aparece em lugar nenhum do artigo -- possivel confusao com
# a citacao de Mottl & McConachy dentro do proprio Field & Sherrell,
# "40-90% of vent fluid Fe forms sulfides... (Mottl and McConachy,
# 1990)"); faixa mais ampla 40-90% em Mottl & McConachy (1990), GCA 54,
# 1911-1927.

FE_EA_J_MOL = 29_000.0
FE_PROMPT_SULFIDE_FRACTION_DEFAULT = 0.68
FE_PROMPT_SULFIDE_FRACTION_RANGE = (0.4, 0.9)

BASIN_PARAMS = {
    "atlantic": {
        "fe_half_life_s": 2.1 * 60.0,
        "label": "TAG, Mid-Atlantic Ridge — Rudnicki & Elderfield (1993)",
    },
    "pacific": {
        "fe_half_life_s": 3.3 * 3600.0,
        "label": "EPR 9°45'N — Field & Sherrell (2000)",
    },
}


def k_fe2(temperature_c: float, basin: str = "atlantic") -> float:
    """Constante de taxa pseudo-primeira-ordem para oxidação de Fe(II), s^-1.

    A magnitude difere por bacia oceânica (ver BASIN_PARAMS); a
    dependência com a temperatura usa a energia de ativação de
    laboratório de Millero et al. (1987) aplicada a ambas as bacias por
    falta de uma medição independente por bacia.
    """
    if basin not in BASIN_PARAMS:
        raise ValueError(f"basin desconhecida: {basin!r}. Use uma de {list(BASIN_PARAMS)}")
    k_ref = math.log(2.0) / BASIN_PARAMS[basin]["fe_half_life_s"]
    return _arrhenius_k(temperature_c, k_ref, FE_EA_J_MOL, t_ref_c=T_REF_C)


def fe_prompt_sulfide_fraction() -> float:
    """Fração de Fe(II) removida quase instantaneamente como sulfeto
    perto do orifício (ver citações acima). Retorna o valor default;
    o chamador pode usar FE_PROMPT_SULFIDE_FRACTION_RANGE para análise
    de sensibilidade."""
    return FE_PROMPT_SULFIDE_FRACTION_DEFAULT


# --------------------------------------------------------------------
# Mn(II)
# --------------------------------------------------------------------
# Cowen, J.P., Massoth, G.J., & Feely, R.A. (1990). "Scavenging rates of
# dissolved manganese in a hydrothermal vent plume." Deep-Sea Res.
# 37(10), 1619-1637. Metodo de radiotracador (54Mn):
#   k1 < 0.2 /ano na pluma flutuante
#   k1 ~= 2 /ano na pluma nao-flutuante, ~20 km do eixo da dorsal
# Nenhuma energia de ativacao foi encontrada na pesquisa de literatura
# para Mn(II); nao aplicamos escalonamento com temperatura aqui — essa
# e uma limitacao documentada, nao uma omissao silenciosa.

MN_SCAVENGING_RATE_PER_YEAR = {
    "buoyant_plume": 0.2,
    "non_buoyant_plume_20km": 2.0,
}

_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def k_mn2(stage: str = "buoyant_plume") -> float:
    """Constante de taxa de sequestro de Mn(II), s^-1. Sem dependência de
    temperatura (nenhuma Ea publicada encontrada) — ver docstring do módulo."""
    if stage not in MN_SCAVENGING_RATE_PER_YEAR:
        raise ValueError(f"stage desconhecido: {stage!r}. Use um de {list(MN_SCAVENGING_RATE_PER_YEAR)}")
    return MN_SCAVENGING_RATE_PER_YEAR[stage] / _SECONDS_PER_YEAR

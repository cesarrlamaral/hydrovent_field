"""
Módulos de concentração de moléculas prebióticas em hotspots hidrotermais.

Ferramenta de análise comparativa: para cada configuração de mecanismos
ativos (diluição, termoforese, adsorção mineral, gradiente de prótons) e
classe de molécula, calcula a concentração esperada em cada vent e o
fator de enriquecimento em relação a um controle fixo — a mesma síntese
de base, sujeita apenas à diluição da pluma, sem nenhum mecanismo
concentrador ativo (ver CONTROL_FLAGS). O resultado central de cada run
é esse enriquecimento relativo (quantas vezes a concentração aumenta ou
diminui em relação ao controle) — a métrica que a ferramenta reporta
com confiança — e não a concentração absoluta em µM, que depende de
constantes de ordem de grandeza (coeficiente de Soret, capacidade de
adsorção, ganho do gradiente de prótons) ainda não calibradas com dados
de campo e por isso deve ser lida como uma estimativa, não uma previsão.

A classe de molécula (aminoácidos, nucleotídeos, lipídeos ou açúcares —
ver MOLECULE_CLASSES) é selecionável e determina o perfil de parâmetros
usado: baseline de síntese, janela térmica de estabilidade, força de
termoforese e afinidade por superfícies minerais.

Cada função de módulo é independente e pode ser ligada/desligada, para
permitir estudos de sensibilidade (ablation studies) sobre quanto cada
mecanismo contribui para o enriquecimento final em um possível hotspot,
relevante para o "problema da concentração" em Astrobiologia: como
moléculas prebióticas diluídas no oceano atingem concentrações
suficientes para polimerizar.

Mecanismos modelados:

1. Diluição/advecção da pluma — dilui a concentração à medida que o
   fluido hidrotermal se mistura com água do mar (contraponto aos demais).
2. Termoforese em poros minerais — efeito Soret acoplado à convecção
   num poro alongado concentra moléculas ao longo de um gradiente
   térmico (Baaske et al. 2007; Braun & Libchaber 2002). Para a classe
   "nucleotideos" esta é uma fórmula CALIBRADA com dados reais medidos
   (S_T e a dependência com a razão de aspecto do poro, ver
   `module_thermophoresis`) — não mais um coeficiente ilustrativo.
   Aminoácidos/lipídeos/açúcares continuam com a fórmula ilustrativa
   antiga (sem medição equivalente encontrada na literatura consultada).
3. Adsorção em superfícies minerais — pirita/mackinawita (mundo
   ferro-enxofre de Wächtershäuser) adsorve e concentra orgânicos na
   superfície mineral via isoterma de Langmuir.
4. Gradiente de prótons em compartimentos alcalinos — micro-compartimentos
   minerais com paredes finas de FeS geram um gradiente de prótons
   análogo à quimiosmose primordial (Russell & Martin; campo Lost City).
   Aplicado com peso reduzido em black smokers, já que a hipótese é
   especificamente sobre fluido alcalino, não o fluido ácido dos black
   smokers clássicos. CALIBRADO (2026-08-05): o ΔpH é convertido num
   potencial transmembrana real via equação de Nernst e comparado contra
   a força próton-motriz biológica real necessária para fixação de
   carbono (~3 unidades de pH, Sojo et al. 2016) — não mais um
   auto-normalizador arbitrário. O mecanismo em si permanece contestado
   na literatura (Jackson, 2016) — ver `module_proton_gradient`.
5. Campo acústico (hipótese exploratória, ver acoustics.py) — streaming
   de contorno e/ou aprisionamento de partícula via potencial de
   Gor'kov, calculados sobre o CAMPO INTEIRO de fumarolas (não por
   fumarola isolada) e amostrados na posição de cada fumarola. Ao
   contrário dos módulos 1-4, o fator de cada fumarola não é calculado
   aqui — é recebido já pronto via `acoustic_factors` (ver
   `fumarola_field.execute_run` e `acoustics.acoustic_enrichment_field`),
   porque depende da posição de TODAS as fumarolas simultaneamente, não
   só da fumarola em questão.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fumarola_field import Vent

SEAWATER_TEMP_C = 2.0
SEAWATER_PH = 7.8
HYDROTHERMAL_ENDMEMBER_PH = 3.2
MAX_DELTA_PH = abs(HYDROTHERMAL_ENDMEMBER_PH - SEAWATER_PH)  # 4.6, ΔpH máximo deste modelo (diagnóstico, não normalizador — ver módulo 4)

# --- módulo 4: gradiente de prótons — referência biológica real (Nernst) ---
# Inclinação de Nernst ideal, 25°C: ΔV = 59.2 mV por unidade de ΔpH
# (eletroquímica padrão, ver p.ex. Sojo et al. 2016, que cita a mesma
# inclinação "~59 mV por unidade de pH").
NERNST_MV_PER_PH_AT_25C = 59.2

# Sojo, V., Herschy, B., Whicher, A., Camprubí, E., & Lane, N. (2016),
# "The Origin of Life in Alkaline Hydrothermal Vents," Astrobiology
# 16(2), 181-197 (verificado por leitura direta do PDF primário
# completo em 2026-08-06, biblios/ast.2015.1406.pdf): o artigo afirma
# explicitamente, no texto principal e na legenda da Fig. 1, que poros
# de vents alcalinos têm um gradiente de pH de 3 unidades através da
# barreira inorgânica, gerando uma força próton-motriz de ~200 mV, que
# é "exactly equivalent in both magnitude (about 3 pH units) and
# polarity" à força próton-motriz usada por células autotróficas
# extantes hoje — esta é a comparação direta e explícita que usamos
# como REFERÊNCIA BIOLÓGICA REAL (não mais um auto-normalizador do
# próprio modelo, que era o que MAX_DELTA_PH fazia antes desta
# calibração). NOTA (correção 2026-08-06): o mesmo artigo também discute
# um cenário Hadeano mais extremo, em que um oceano mais ácido (rico em
# CO2) poderia elevar o gradiente a até 6 unidades de pH (~400 mV) — mas
# o artigo apresenta esse número como um potencial adicional/máximo,
# NÃO como "equivalente" a nenhuma referência biológica; não deve ser
# confundido com os "3 unidades" usados aqui, que é o número que o
# próprio artigo compara diretamente à vida extante.
REFERENCE_PROTON_GRADIENT_PH_UNITS = 3.0
REFERENCE_PROTON_MOTIVE_FORCE_MV = NERNST_MV_PER_PH_AT_25C * REFERENCE_PROTON_GRADIENT_PH_UNITS  # ≈177.6 mV

# Crítica quantitativa séria a todo o mecanismo, com peso igual: Jackson,
# J.B. (2016), "Natural pH gradients in hydrothermal alkali vents were
# unlikely to have played a role in the origin of life," J. Mol. Evol.
# 83(1), 1-11 — argumenta que uma membrana inorgânica fina (~1 μm,
# >200x mais espessa que uma bicamada lipídica) com QUALQUER canal
# permeável a H+ (necessário para qualquer maquinário molecular
# realmente USAR o gradiente) deixaria o gradiente colapsar por difusão
# para ~0.004 unidades de pH — muito abaixo do necessário. Ou seja: o
# próprio mecanismo modelado aqui (um gradiente de pH em massa
# traduzido em potencial transmembrana utilizável) é uma hipótese
# seriamente contestada na literatura, não um fato estabelecido — ver
# docs/PHYSICS_MODEL.md, seção 8, para a discussão completa. Todos os
# quatro números desta crítica (1 μm, >200x, 0.004 unidades de pH,
# 24 J/mol vs. 24 kJ/mol necessários) foram verificados por leitura
# direta do PDF primário completo em 2026-08-06
# (biblios/s00239-016-9756-6.pdf) — sem divergência do que já estava
# documentado aqui.

# --------------------------------------------------------------------------
# Parâmetros dos módulos (ajustáveis), por classe de molécula prebiótica
# --------------------------------------------------------------------------
#
# Cada classe tem sua própria janela térmica de síntese/estabilidade, sua
# própria força de termoforese (moléculas maiores/mais polares respondem
# mais a gradientes térmicos, Baaske et al. 2007) e sua própria afinidade
# por superfícies minerais. Os parâmetros de física do poro/compartimento
# (ΔT local, peso por tipo de vent do gradiente de prótons) não dependem
# da molécula e ficam em SHARED_PARAMS. Assim como o restante do módulo,
# estes são valores ilustrativos de ordem de grandeza, não medições diretas.

SHARED_PARAMS = {
    # módulo 1: diluição/advecção
    "max_pure_temp_c": 400.0,         # mesma referência usada em mixing_chemistry

    # módulo 2: termoforese em poros
    "pore_delta_t_min_k": 5.0,        # gradiente térmico local mínimo (poro perto de diffuse flow)
    "pore_delta_t_max_k": 20.0,       # gradiente térmico local máximo (poro perto de black smoker)

    # Razão de aspecto (comprimento/largura) do segmento de poro, usada
    # SÓ pelas classes com "thermophoresis_convection_coupled": True
    # (ver módulo 2 abaixo e docs/PHYSICS_MODEL.md) — 10:1 é o segmento
    # único mais conservador testado experimentalmente por Baaske et al.
    # (2007) (faixa testada: 10:1 a 125:1, ou cascatas de segmentos
    # alcançando razões efetivas maiores). NÃO é uma medição da geometria
    # real de poros em paredes de chaminé hidrotermal — essa geometria
    # não foi encontrada na pesquisa de literatura consultada; usar a
    # geometria do aparato de laboratório de Baaske et al. como análogo
    # plausível é uma escolha explícita, não uma medição de campo.
    "pore_aspect_ratio": 10.0,
    # Constante da forma analítica exp(k·S_T·ΔT·razão_de_aspecto) —
    # VERIFICADO em 2026-08-06 contra o texto primário completo (PDF em
    # biblios/baaske2007.pdf): é a Eq. 1 de Baaske et al. (2007), PNAS
    # 104(22):9346-9351, p. 9348 ("c_BOTTOM/c_TOP = exp[0.42 × S_T × ΔT × r]",
    # solução analítica de Furry, Jones & Onsager 1939 / Debye 1939 para
    # coluna termogravitacional, confirmada pelos autores contra a
    # simulação numérica por elementos finitos). k=0.42, não mais um
    # valor reconstruído por regressão — lido diretamente da equação.
    # Ver tests/test_prebiotic.py para os testes de referência.
    "thermophoresis_convection_fit_k": 0.42,

    # módulo 4: gradiente de prótons
    "proton_vent_type_weight": {      # hipótese de Russell/Martin é sobre fluido alcalino, não black smoker ácido
        "black_smoker": 0.2,
        "white_smoker": 1.0,
        "diffuse_flow": 1.0,
    },
}

MOLECULE_CLASS_LABELS = {
    "aminoacidos": "Aminoácidos",
    "nucleotideos": "Nucleotídeos / bases nitrogenadas",
    "lipideos": "Lipídeos (ácidos graxos anfifílicos)",
    "acucares": "Açúcares (gliceraldeído, ribose)",
}

# versão em inglês, usada nas FIGURAS (matplotlib) geradas por
# fumarola_field.py — títulos/eixos/legendas de figura são sempre em
# inglês (material de publicação), diferente do resto da UI/CLI em
# português.
MOLECULE_CLASS_LABELS_EN = {
    "aminoacidos": "Amino Acids",
    "nucleotideos": "Nucleotides / Nitrogenous Bases",
    "lipideos": "Lipids (Amphiphilic Fatty Acids)",
    "acucares": "Sugars (Glyceraldehyde, Ribose)",
}

# parâmetros específicos de cada classe (baseline de síntese, janela
# térmica, resposta a termoforese e a adsorção mineral)
_CLASS_SPECIFIC_PARAMS = {
    "aminoacidos": {
        "baseline_max_uM": 2.0,           # ordem de grandeza de Lang et al. 2018
        "temp_optimum_c": 150.0,
        "temp_width_c": 80.0,
        "soret_coefficient_per_k": 0.03,  # S_T pequeno vs. DNA (~0.1-1 /K) — ilustrativo, ver módulo 2
        "thermophoresis_convection_coupled": False,  # sem medição de Baaske-tipo para aminoácidos
        "adsorption_max_factor": 50.0,
        "adsorption_half_sat": 2.0,       # mmol²/kg² de Fe*H2S
        "proton_max_factor": 20.0,
    },
    "nucleotideos": {
        # síntese abiótica mais rara/instável (baseline menor); mas
        # moléculas maiores e com grupo fosfato respondem mais forte à
        # termoforese e adsorvem mais fortemente em superfícies minerais
        # via o fosfato. soret_coefficient_per_k É UM VALOR MEDIDO (não
        # ilustrativo): 0.006 /K, nucleotídeo único, 170 mM de sal
        # monovalente — Baaske, P., Weinert, F.M., Duhr, S., Lemke, K.H.,
        # Russell, M.J., & Braun, D. (2007), "Extreme accumulation of
        # nucleotides in simulated hydrothermal pore systems," PNAS
        # 104(22), 9346-9351. Usamos a condição de sal MAIS ALTA testada
        # (170 mM) como análogo mais próximo à força iônica da água do
        # mar (~500-600 mM) que a condição diluída (1.7 mM, S_T=0.015/K)
        # também reportada — mas 170 mM ainda é ~3x mais diluído que a
        # água do mar real, e S_T DIMINUI com a salinidade nos dados
        # deles, então este valor provavelmente ainda SOBRESTIMA S_T em
        # condições reais de água do mar — extrapolação explícita, não
        # uma medição direta nas condições deste modelo. "Nucleotídeo"
        # também não é o mesmo que "base nitrogenada" livre (não medida
        # separadamente por Baaske et al.) — a classe combina os dois
        # rótulos, mas só o nucleotídeo completo foi medido. Valor
        # 0.006/K (Tabela 1 do artigo, 170 mM) confirmado por leitura
        # direta do PDF primário em 2026-08-06 (biblios/baaske2007.pdf),
        # não mais uma citação de segunda mão. Nota: os autores afirmam
        # na Discussão que "because thermophoretic drift is common for
        # molecules, the accumulation scheme applies similarly to nucleic
        # acids, amino acids, and lipids" — reivindicam generalidade do
        # MECANISMO (convecção acoplada a termodifusão), não medem S_T
        # para aminoácidos/lipídeos; por isso as outras 3 classes
        # continuam com a fórmula ilustrativa não acoplada.
        "baseline_max_uM": 0.5,
        "temp_optimum_c": 100.0,
        "temp_width_c": 55.0,
        "soret_coefficient_per_k": 0.006,
        "thermophoresis_convection_coupled": True,
        "adsorption_max_factor": 80.0,
        "adsorption_half_sat": 1.0,
        "proton_max_factor": 22.0,
    },
    "lipideos": {
        # anfifílicos: termoforese em solução é fraca (tendem a
        # autoagregar/particionar em interfaces em vez de responder a
        # gradiente térmico), mas adsorção/concentração em superfícies
        # minerais é muito forte; janela térmica larga (vesículas toleram
        # uma faixa ampla, mas se desestabilizam em temperatura muito alta)
        "baseline_max_uM": 1.0,
        "temp_optimum_c": 120.0,
        "temp_width_c": 100.0,
        "soret_coefficient_per_k": 0.015,  # ilustrativo, ver módulo 2 — sem medição de Baaske-tipo para lipídeos
        "thermophoresis_convection_coupled": False,
        "adsorption_max_factor": 90.0,
        "adsorption_half_sat": 1.5,
        "proton_max_factor": 15.0,
    },
    "acucares": {
        # frágeis termicamente (degradação tipo Maillard/caramelização em
        # temperatura alta), pouca afinidade por superfícies minerais
        "baseline_max_uM": 1.5,
        "temp_optimum_c": 80.0,
        "temp_width_c": 50.0,
        "soret_coefficient_per_k": 0.02,  # ilustrativo, ver módulo 2 — sem medição de Baaske-tipo para açúcares
        "thermophoresis_convection_coupled": False,
        "adsorption_max_factor": 15.0,
        "adsorption_half_sat": 3.0,
        "proton_max_factor": 10.0,
    },
}

MOLECULE_CLASSES = {
    key: {**SHARED_PARAMS, **specific} for key, specific in _CLASS_SPECIFIC_PARAMS.items()
}

DEFAULT_MOLECULE_CLASS = "aminoacidos"
DEFAULT_PARAMS = MOLECULE_CLASSES[DEFAULT_MOLECULE_CLASS]


@dataclass
class ModuleFlags:
    dilution: bool = True
    thermophoresis: bool = True
    mineral_adsorption: bool = True
    proton_gradient: bool = True
    # "off" | "streaming" | "particle_trap" | "both" — ver acoustics.py.
    # Não é bool como os demais porque não é um simples ligar/desligar:
    # são mecanismos físicos distintos e mutuamente não-exclusivos.
    acoustic_mode: str = "off"

    def as_dict(self) -> dict:
        return {
            "dilution": self.dilution,
            "thermophoresis": self.thermophoresis,
            "mineral_adsorption": self.mineral_adsorption,
            "proton_gradient": self.proton_gradient,
            "acoustic_mode": self.acoustic_mode,
        }


# controle de referência: síntese de base sujeita apenas à diluição da
# pluma, sem nenhum mecanismo concentrador ativo. Todo run é comparado
# contra este controle fixo (mesmos vents, mesma classe de molécula),
# nunca contra outro run, para que o fator de enriquecimento tenha um
# referencial estável e reprodutível
CONTROL_FLAGS = ModuleFlags(dilution=True, thermophoresis=False,
                             mineral_adsorption=False, proton_gradient=False)


def _temp_suitability(temperature_c: float, params: dict) -> float:
    """Janela térmica favorável à síntese/estabilidade de aminoácidos (pico ~150°C)."""
    t_opt = params["temp_optimum_c"]
    width = params["temp_width_c"]
    return float(np.exp(-((temperature_c - t_opt) / width) ** 2))


def synthesis_baseline_uM(vent: "Vent", params: dict = DEFAULT_PARAMS) -> float:
    """
    Concentração de referência de aminoácidos no fluido hidrotermal (µM),
    antes de qualquer mecanismo de concentração local. Escala com a
    disponibilidade de precursores reduzidos (H2S, CH4, proxy de química
    redutora) e com a adequação térmica (síntese favorecida termicamente,
    mas aminoácidos se degradam em temperaturas muito altas).
    """
    reducing = vent.chemistry.get("H2S", 0.0) + vent.chemistry.get("CH4", 0.0)
    reducing_frac = np.clip(reducing / 7.7, 0.0, 1.0)  # normalizado pelo fim de membro puro (H2S+CH4 max)
    temp_frac = _temp_suitability(vent.temperature_c, params)
    return float(params["baseline_max_uM"] * reducing_frac * temp_frac)


def module_dilution(vent: "Vent", enabled: bool, params: dict = DEFAULT_PARAMS) -> float:
    """
    Fração de fluido hidrotermal não diluído remanescente no hotspot,
    aproximada por 1/D(z=1m), onde D é a diluição turbulenta real da
    pluma resolvida pelo modelo integral de Morton-Taylor-Turner
    (plume_physics.py, via fumarola_field.simulate_plume) — substitui o
    proxy anterior baseado só em temperatura. Desligado = hotspot
    hipotético que capta fluido praticamente puro, sem diluição
    (controle idealizado).

    Aproximação explícita: o regime físico de interesse para hipóteses
    prebióticas (mistura difusiva em poros de parede de chaminé, ex.
    Lost City / Russell & Martin) é distinto do regime de pluma
    turbulenta em coluna d'água livre modelado aqui — usar D(1m) como
    proxy é uma simplificação documentada, não uma medição desse
    processo. Ver docs/PHYSICS_MODEL.md, seção "Limitações".
    """
    if not enabled:
        return 1.0
    dilution = max(getattr(vent, "dilution_near_field_1m", 1.0), 1.0)
    return float(np.clip(1.0 / dilution, 0.0, 1.0))


def module_thermophoresis(vent: "Vent", enabled: bool, params: dict = DEFAULT_PARAMS) -> float:
    """
    Fator de enriquecimento por termoforese em um poro mineral com
    gradiente térmico local. ΔT local escala com a temperatura do fluido
    (fumarolas mais quentes geram gradientes locais mais íngremes nas
    paredes da chaminé).

    Duas fórmulas, dependendo de `params["thermophoresis_convection_coupled"]`:

    - **True** (só a classe "nucleotideos", que tem medição real —
      Baaske et al. 2007): `enhancement = exp(k * S_T * ΔT * razão_de_aspecto)`
      — o mecanismo real é acumulação por convecção termo-difusiva
      ACOPLADA à geometria alongada do poro (fluido circula ao longo do
      poro por convecção enquanto moléculas migram através dele por
      termodifusão), não um simples equilíbrio Soret estático. É esse
      acoplamento com a razão de aspecto (comprimento/largura) que
      produz os fatores de 10⁸-10¹⁵× relatados no artigo — muitas ordens
      de grandeza maiores que `exp(S_T*ΔT)` sozinho. É a Eq. 1 do artigo
      primário (p. 9348), com k=0.42 lido diretamente do texto completo
      (verificado 2026-08-06, ver docs/PHYSICS_MODEL.md) — não mais um
      valor reconstruído por regressão. Ressalvas que permanecem:
      extrapolação de salinidade (170 mM testado vs. ~500-600 mM da água
      do mar real) e geometria do poro não medida em chaminés reais
      (usamos a razão de aspecto do aparato de laboratório como análogo).
    - **False** (aminoácidos, lipídeos, açúcares — sem medição
      equivalente encontrada): `enhancement = exp(S_T * ΔT)`, a mesma
      fórmula ilustrativa de sempre, inalterada (razão de aspecto e k
      reduzem-se a 1, sem efeito).
    """
    if not enabled:
        return 1.0
    temp_frac = np.clip(vent.temperature_c / 400.0, 0.0, 1.0)
    delta_t = params["pore_delta_t_min_k"] + temp_frac * (
        params["pore_delta_t_max_k"] - params["pore_delta_t_min_k"]
    )
    if params.get("thermophoresis_convection_coupled", False):
        aspect_ratio = params["pore_aspect_ratio"]
        k = params["thermophoresis_convection_fit_k"]
    else:
        aspect_ratio = 1.0
        k = 1.0
    return float(np.exp(k * params["soret_coefficient_per_k"] * delta_t * aspect_ratio))


def module_mineral_adsorption(vent: "Vent", enabled: bool, params: dict = DEFAULT_PARAMS) -> float:
    """
    Enriquecimento por adsorção em superfícies de sulfeto mineral
    (pirita/mackinawita), via isoterma de Langmuir. A capacidade de
    formação mineral é aproximada pelo produto Fe×H2S (proxy de
    precipitação de FeS), modulada pela mesma janela térmica de
    estabilidade usada na síntese.
    """
    if not enabled:
        return 1.0
    fe = vent.chemistry.get("Fe", 0.0)
    h2s = vent.chemistry.get("H2S", 0.0)
    mineral_potential = fe * h2s
    langmuir = mineral_potential / (mineral_potential + params["adsorption_half_sat"])
    temp_factor = _temp_suitability(vent.temperature_c, params)
    return float(1.0 + params["adsorption_max_factor"] * langmuir * temp_factor)


def module_proton_gradient(vent: "Vent", enabled: bool, params: dict = DEFAULT_PARAMS) -> float:
    """
    Enriquecimento por compartimentalização quimiosmótica primordial em
    paredes minerais finas (Russell & Martin; campo Lost City), proporcional
    à magnitude do gradiente de pH fluido-água do mar. Aplicado com peso
    reduzido em black smokers: a hipótese é especificamente sobre fluido
    hidrotermal alcalino, não o fluido ácido de black smokers clássicos —
    este modelo usa apenas a MAGNITUDE do gradiente como proxy (não
    distingue direção ácido/alcalino), então o peso por tipo de vent
    compensa essa simplificação (aproximação inalterada nesta calibração).

    O que MUDOU: `gradient_frac` não é mais normalizado pelo ΔpH máximo
    deste próprio modelo (MAX_DELTA_PH, um auto-normalizador arbitrário)
    — agora o ΔpH é convertido num potencial transmembrana real via
    equação de Nernst (59.2 mV/unidade de pH a 25°C) e comparado contra
    uma REFERÊNCIA BIOLÓGICA REAL: a força próton-motriz necessária para
    fixação de carbono em organismos extantes, equivalente a ~3 unidades
    de pH / ~177.6 mV (Sojo et al., 2016). `gradient_frac=1` significa
    "esta fumarola produz, em magnitude, o mesmo potencial que a vida
    moderna usa para fixar carbono" — não mais um valor arbitrário de
    0-1 sem referência externa. Sem teto superior artificial: fumarolas
    com ΔpH maior que a referência dão gradient_frac>1, uma comparação
    quantitativa genuína, não um artefato de normalização.

    IMPORTANTE: o mecanismo em si (gradiente de pH em massa fluido→oceano
    sustentando um potencial USÁVEL por maquinário molecular através de
    uma membrana inorgânica fina) é uma hipótese seriamente contestada
    na literatura (Jackson, 2016, argumenta que o gradiente colapsaria
    por difusão para ~0.004 unidades de pH em qualquer canal permeável a
    H+) — esta calibração melhora a REFERÊNCIA de comparação, não
    resolve essa disputa. Ver docs/PHYSICS_MODEL.md, seção 8.
    """
    if not enabled:
        return 1.0
    delta_ph = abs(vent.chemistry.get("pH", SEAWATER_PH) - SEAWATER_PH)
    nernst_potential_mv = NERNST_MV_PER_PH_AT_25C * delta_ph
    gradient_frac = max(nernst_potential_mv / REFERENCE_PROTON_MOTIVE_FORCE_MV, 0.0)
    weight = params["proton_vent_type_weight"].get(vent.vent_type, 0.5)
    return float(1.0 + params["proton_max_factor"] * gradient_frac * weight)


def module_acoustic(vent: "Vent", mode: str, acoustic_factors: Optional[Dict[int, float]]) -> float:
    """
    Fator de enriquecimento acústico pré-computado para o campo inteiro
    (ver acoustics.acoustic_enrichment_field), amostrado na posição
    desta fumarola. `acoustic_factors` é None quando mode="off" ou
    quando o chamador não forneceu o campo (ex.: comparação de
    controle) — nesse caso o fator neutro (1.0) é usado.
    """
    if mode == "off" or acoustic_factors is None:
        return 1.0
    return float(acoustic_factors.get(vent.id, 1.0))


def _concentration_uM(vent: "Vent", flags: ModuleFlags, baseline: float, params: dict,
                       acoustic_factors: Optional[Dict[int, float]] = None) -> tuple[float, dict]:
    f_dilution = module_dilution(vent, flags.dilution, params)
    f_thermo = module_thermophoresis(vent, flags.thermophoresis, params)
    f_adsorption = module_mineral_adsorption(vent, flags.mineral_adsorption, params)
    f_proton = module_proton_gradient(vent, flags.proton_gradient, params)
    f_acoustic = module_acoustic(vent, flags.acoustic_mode, acoustic_factors)
    return baseline * f_dilution * f_thermo * f_adsorption * f_proton * f_acoustic, {
        "factor_dilution": f_dilution,
        "factor_thermophoresis": f_thermo,
        "factor_mineral_adsorption": f_adsorption,
        "factor_proton_gradient": f_proton,
        "factor_acoustic": f_acoustic,
    }


def compute_hotspot(vent: "Vent", flags: ModuleFlags, params: dict = DEFAULT_PARAMS,
                     molecule_class: str = DEFAULT_MOLECULE_CLASS,
                     control_flags: ModuleFlags = CONTROL_FLAGS,
                     acoustic_factors: Optional[Dict[int, float]] = None) -> dict:
    """
    Calcula a concentração final da classe de molécula escolhida no hotspot
    de um vent, módulo a módulo, e o fator de enriquecimento em relação ao
    controle (mesmo baseline de síntese, apenas diluído — ver CONTROL_FLAGS).
    `acoustic_factors` nunca é aplicado ao controle (CONTROL_FLAGS.acoustic_mode
    é sempre "off"), então o enriquecimento reportado já inclui o efeito
    acústico quando `flags.acoustic_mode != "off"`.
    """
    baseline = synthesis_baseline_uM(vent, params)
    final_uM, factors = _concentration_uM(vent, flags, baseline, params, acoustic_factors)
    control_uM, _ = _concentration_uM(vent, control_flags, baseline, params, acoustic_factors=None)

    if control_uM > 1e-12:
        enrichment = final_uM / control_uM
    else:
        enrichment = None  # sem síntese de base neste vent: comparação não é significativa

    return {
        "vent_id": vent.id,
        "vent_type": vent.vent_type,
        "temperature_c": vent.temperature_c,
        "molecule_class": molecule_class,
        "baseline_uM": round(baseline, 6),
        "factor_dilution": round(factors["factor_dilution"], 4),
        "factor_thermophoresis": round(factors["factor_thermophoresis"], 4),
        "factor_mineral_adsorption": round(factors["factor_mineral_adsorption"], 4),
        "factor_proton_gradient": round(factors["factor_proton_gradient"], 4),
        "factor_acoustic": round(factors["factor_acoustic"], 6),
        "final_concentration_uM": round(final_uM, 6),
        "control_concentration_uM": round(control_uM, 6),
        "enrichment_vs_control": round(enrichment, 4) if enrichment is not None else None,
    }


def compute_field_hotspots(vents: List["Vent"], flags: ModuleFlags, params: dict = DEFAULT_PARAMS,
                            molecule_class: str = DEFAULT_MOLECULE_CLASS,
                            control_flags: ModuleFlags = CONTROL_FLAGS,
                            acoustic_factors: Optional[Dict[int, float]] = None) -> dict:
    """
    Calcula hotspots para todos os vents do campo e resume o resultado. A
    métrica primária é `enrichment_vs_control`: quantas vezes a concentração
    de cada vent aumenta (>1) ou diminui (<1) em relação ao controle fixo
    (CONTROL_FLAGS) com os mesmos vents e a mesma classe de molécula.
    """
    records = [compute_hotspot(v, flags, params, molecule_class, control_flags, acoustic_factors)
               for v in vents]

    enrichments = [r["enrichment_vs_control"] for r in records if r["enrichment_vs_control"] is not None]
    records.sort(key=lambda r: r["enrichment_vs_control"] if r["enrichment_vs_control"] is not None else -1,
                 reverse=True)

    concentrations = [r["final_concentration_uM"] for r in records]
    n_increased = sum(1 for e in enrichments if e > 1.0)
    n_decreased = sum(1 for e in enrichments if e < 1.0)
    n_unchanged = sum(1 for e in enrichments if e == 1.0)

    summary = {
        "molecule_class": molecule_class,
        "molecule_class_label": MOLECULE_CLASS_LABELS.get(molecule_class, molecule_class),
        "modules_enabled": flags.as_dict(),
        "n_vents": len(records),
        "max_concentration_uM": concentrations[0] if concentrations else 0.0,
        "mean_concentration_uM": float(np.mean(concentrations)) if concentrations else 0.0,
        "median_concentration_uM": float(np.median(concentrations)) if concentrations else 0.0,
        "top_hotspot_vent_id": records[0]["vent_id"] if records else None,
        "top_hotspot_vent_type": records[0]["vent_type"] if records else None,
        "top_hotspot_enrichment_vs_control": records[0]["enrichment_vs_control"] if records else None,
        "mean_enrichment_vs_control": float(np.mean(enrichments)) if enrichments else None,
        "median_enrichment_vs_control": float(np.median(enrichments)) if enrichments else None,
        "n_vents_increased_vs_control": n_increased,
        "n_vents_decreased_vs_control": n_decreased,
        "n_vents_unchanged_vs_control": n_unchanged,
    }
    return {"records": records, "summary": summary}

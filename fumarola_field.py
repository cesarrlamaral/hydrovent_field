"""
Gerador procedural de campos de fumarolas hidrotermais oceânicas.

Simula um trecho de dorsal meso-oceânica (heightmap via diamond-square),
esculpe o vale axial de rifte, distribui clusters de fumarolas ao longo
do eixo, classifica cada fumarola (black smoker / white smoker / diffuse
flow), modela a diluição química da pluma hidrotermal por mistura com
água do mar, estima a altura de ascensão da pluma flutuante (teoria de
Morton-Taylor-Turner aplicada por Speer & Rona, 1989) e atribui zonação
faunística quimiossintética em torno de cada orifício.

Uso:
    python fumarola_field.py --seed 42 --size 257 --n-clusters 6 --spreading-rate 60
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import concurrent.futures
import json
import csv
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy.stats import qmc
import matplotlib
# Backend não-interativo: este módulo só salva PNGs em disco (nenhum
# plt.show() em lugar nenhum do projeto), o que também é obrigatório para
# gerar imagens dentro de processos de trabalho (--parallel) — um backend
# GUI (Tk/Qt) não pode ser criado fora do processo/thread principal.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

import plume_physics as pp
import reaction_kinetics as rk
import acoustics as ac
import variance_decomposition as vd
import global_sensitivity as gs
from prebiotic import (
    ModuleFlags,
    compute_field_hotspots,
    MOLECULE_CLASSES,
    MOLECULE_CLASS_LABELS,
    MOLECULE_CLASS_LABELS_EN,
    DEFAULT_MOLECULE_CLASS,
)

# pasta outputs/ sempre ancorada na pasta do projeto, independente de onde o
# script é chamado (evita criar outputs/ no diretório de trabalho atual)
DEFAULT_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# paleta "abissal": azul-petróleo quase negro (fundo do vale) subindo a
# teal/ciano nas cristas, evocando a coluna d'água do oceano profundo
ABYSSAL_CMAP = LinearSegmentedColormap.from_list(
    "abyssal",
    ["#01040a", "#03122b", "#06375c", "#0c6478", "#3fb6a8", "#a4e8d8"],
)


# --------------------------------------------------------------------------
# 1. Terreno: dorsal meso-oceânica via diamond-square + vale axial
# --------------------------------------------------------------------------

def diamond_square(size: int, roughness: float, rng: np.random.Generator) -> np.ndarray:
    """Gera um heightmap fractal. `size` deve ser 2**n + 1."""
    n = size - 1
    if n & (n - 1) != 0:
        raise ValueError("size deve ser (potência de 2) + 1, ex: 65, 129, 257")

    grid = np.zeros((size, size))
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = n
    scale = 1.0
    while step > 1:
        half = step // 2

        # diamond step
        for y in range(half, size, step):
            for x in range(half, size, step):
                avg = (
                    grid[y - half, x - half] + grid[y - half, x + half]
                    + grid[y + half, x - half] + grid[y + half, x + half]
                ) / 4.0
                grid[y, x] = avg + rng.uniform(-1, 1) * scale

        # square step
        for y in range(0, size, half):
            for x in range((y + half) % step, size, step):
                pts = []
                if y - half >= 0:
                    pts.append(grid[y - half, x])
                if y + half < size:
                    pts.append(grid[y + half, x])
                if x - half >= 0:
                    pts.append(grid[y, x - half])
                if x + half < size:
                    pts.append(grid[y, x + half])
                grid[y, x] = sum(pts) / len(pts) + rng.uniform(-1, 1) * scale

        step = half
        scale *= roughness

    # normaliza para [0, 1]
    grid -= grid.min()
    grid /= grid.max()
    return grid


def carve_axial_valley(terrain: np.ndarray, axis_wander: float, depth: float,
                        width_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Esculpe um vale de rifte (graben axial) serpenteando pela grade,
    típico de dorsais lentas/intermediárias (ex: MAR). Retorna o terreno
    modificado e a posição y do eixo para cada coluna x (para uso na
    distribuição dos vents).
    """
    size = terrain.shape[0]
    xs = np.arange(size)

    # eixo com leve serpenteio (soma de senoides de baixa frequência)
    axis_y = size / 2 + axis_wander * (
        0.6 * np.sin(2 * np.pi * xs / size * 1.3 + rng.uniform(0, 2 * np.pi))
        + 0.4 * np.sin(2 * np.pi * xs / size * 2.7 + rng.uniform(0, 2 * np.pi))
    )

    width = width_frac * size
    yy, xx = np.mgrid[0:size, 0:size]
    dist_to_axis = np.abs(yy - axis_y[xx])

    valley = -depth * np.exp(-(dist_to_axis ** 2) / (2 * (width / 2) ** 2))
    terrain = terrain + valley
    terrain -= terrain.min()
    terrain /= terrain.max()
    return terrain, axis_y


# --------------------------------------------------------------------------
# 2. Modelo de fumarola individual
# --------------------------------------------------------------------------

VENT_TYPES = {
    # nome: (faixa de temperatura °C, cor de referência, fluido dominante)
    "black_smoker": {"temp_range": (300, 405), "color": "#ff3b1f", "chem_scale": 1.0},
    "white_smoker": {"temp_range": (100, 300), "color": "#ffd166", "chem_scale": 0.55},
    "diffuse_flow": {"temp_range": (5, 100), "color": "#ff9f4a", "chem_scale": 0.15},
}

SEAWATER = {"temp": 2.0, "H2S": 0.0, "CH4": 0.0, "Fe": 0.0, "Mn": 0.0, "pH": 7.8}

# concentrações de fim de membro hidrotermal (mmol/kg), valores plausíveis
# para dorsais meso-oceânicas (ordem de grandeza de Von Damm, 1995)
HYDROTHERMAL_ENDMEMBER = {"H2S": 6.5, "CH4": 1.2, "Fe": 2.0, "Mn": 0.8, "pH": 3.2}


@dataclass
class Vent:
    id: int
    cluster_id: int
    x: float
    y: float
    depth_m: float
    vent_type: str
    temperature_c: float
    chemistry: dict = field(default_factory=dict)
    plume_rise_m: float = 0.0
    chimney_height_m: float = 0.0
    fauna_zones: dict = field(default_factory=dict)
    # diluição turbulenta real D(z=1m) do modelo de pluma (plume_physics.py) —
    # usada por prebiotic.py como proxy físico de campo próximo, distinto da
    # concentração em `chemistry` (avaliada na altura de flutuabilidade neutra)
    dilution_near_field_1m: float = 1.0

    def to_record(self):
        rec = asdict(self)
        rec.pop("chemistry")
        rec.pop("fauna_zones")
        rec.update({f"chem_{k}": v for k, v in self.chemistry.items()})
        rec.update({f"fauna_{k}_m": v for k, v in self.fauna_zones.items()})
        return rec


def classify_and_sample_temperature(rng: np.random.Generator, spreading_rate: float) -> tuple[str, float]:
    """
    Sorteia o tipo de fumarola e sua temperatura. Dorsais rápidas (>80 mm/ano)
    tendem a ter sistemas magmáticos mais rasos e maior proporção de black
    smokers de alta temperatura; dorsais lentas favorecem sistemas
    tectonicamente controlados com mais white smokers / diffuse flow.
    """
    fast_bias = np.clip((spreading_rate - 20) / 100, 0, 1)
    weights = np.array([0.25 + 0.35 * fast_bias, 0.40 - 0.15 * fast_bias, 0.35 - 0.20 * fast_bias])
    weights /= weights.sum()
    vtype = rng.choice(list(VENT_TYPES.keys()), p=weights)
    lo, hi = VENT_TYPES[vtype]["temp_range"]
    # distribuição triangular: mais denso perto do centro da faixa
    temp = rng.triangular(lo, (lo + hi) / 2, hi)
    return vtype, float(temp)


def simulate_plume(vent_type: str, temperature_c: float, rng: np.random.Generator,
                    alpha: float = pp.DEFAULT_ALPHA_ENTRAINMENT,
                    n_freq: float = pp.DEFAULT_N_BRUNT_VAISALA,
                    basin: str = "atlantic", keep_profile: bool = False):
    """
    Integra o modelo de pluma turbulenta flutuante estratificada
    (Morton-Taylor-Turner, 1956; ver plume_physics.py) e o transporte
    reativo por espécie (reaction_kinetics.py) para este vent. Devolve
    `(chem, near_field_chem, rise_m, dilution_1m, profile)`:

    - `chem`: concentração de cada espécie na altura de flutuabilidade
      neutra (a camada de pluma efetivamente observável em campo — não
      mais um proxy de temperatura). H2S/Fe/Mn seguem cinética de reação
      real com citação (ver reaction_kinetics.py); CH4 é tratado como
      traçador conservativo (nenhuma cinética de oxidação de CH4 foi
      encontrada na literatura consultada). pH usa mistura conservativa
      de [H+] (fisicamente correto, ao contrário de interpolar pH
      linearmente) — sem química de tamponamento carbonático, uma
      simplificação documentada em docs/PHYSICS_MODEL.md.
    - `near_field_chem`: concentração de ORIGEM de cada espécie (`c0`,
      antes de qualquer diluição/reação pela pluma) — usada por
      `fauna_zonation` (habitat real da fauna quimiossintética é a
      abertura/fluxo difuso, não a altura de flutuabilidade neutra, já
      diluída por ordens de grandeza em relação à origem; usar `chem`
      ali era um bug real, os limiares de H2S nunca eram atingidos por
      nenhum vent gerado — achado testando a camada de fauna da
      visualização artística, 2026-08-07).
    - `rise_m`: altura de flutuabilidade neutra, resultado da integração
      (não mais uma fórmula fechada de ponto único).
    - `dilution_1m`: diluição turbulenta real D(z=1m), usada por
      prebiotic.py como proxy de campo próximo (ver módulo prebiotic.py
      e docs/PHYSICS_MODEL.md — regime físico diferente da mistura
      difusiva em poros de parede de chaminé, que não é modelada aqui).
    - `profile`: o `PlumeProfile` completo, retido apenas se
      `keep_profile=True` (usado por --export-plume-profiles).

    A variabilidade geoquímica vento-a-vento no fluido de fim de membro
    (±15%, `rng.uniform`) reflete a variabilidade real documentada entre
    fumarolas (Von Damm, 1995), aplicada à concentração de origem antes
    do transporte — não é ruído de visualização.
    """
    scale = VENT_TYPES[vent_type]["chem_scale"]
    source = pp.build_source(temperature_c, vent_type)
    profile = pp.integrate_plume(source, alpha=alpha, n_freq=n_freq)

    species_kinetics = {
        "H2S": (rk.k_h2s, 0.0),
        "Fe": (lambda t: rk.k_fe2(t, basin=basin), rk.fe_prompt_sulfide_fraction()),
        "Mn": (lambda t: rk.k_mn2("buoyant_plume"), 0.0),
        "CH4": (None, 0.0),
    }
    chem = {}
    # Concentração de origem (antes do transporte/diluição turbulenta pela
    # pluma) — o habitat real da fauna quimiossintética é a abertura/fluxo
    # difuso, não a altura de flutuabilidade neutra (que já está diluída
    # por vários fatores em relação à origem). Usada por `fauna_zonation`
    # (ver comentário lá) em vez de `chem`, que é fisicamente errado pra
    # esse propósito específico apesar de ser o valor certo pra tudo mais
    # neste módulo (hotspots, prebiotic.py).
    near_field_chem = {}
    species_profiles = {}
    for species, (k_fn, prompt_frac) in species_kinetics.items():
        c0 = HYDROTHERMAL_ENDMEMBER[species] * scale * rng.uniform(0.85, 1.15)
        near_field_chem[species] = round(float(c0), 6)
        c_arr = pp.integrate_species_transport(profile, c0, k_fn=k_fn, prompt_removal_fraction=prompt_frac)
        chem[species] = round(float(c_arr[-1]), 6)
        if keep_profile:
            species_profiles[species] = c_arr

    # pH: mistura conservativa de [H+] (aditivo), não interpolação linear de pH
    h_plus_sw = 10 ** (-SEAWATER["pH"])
    h_plus_end = 10 ** (-HYDROTHERMAL_ENDMEMBER["pH"])
    dilution_ref = max(float(profile.dilution[-1]), 1.0)
    h_plus = h_plus_sw + (h_plus_end - h_plus_sw) * scale / dilution_ref
    chem["pH"] = round(float(-math.log10(max(h_plus, 1e-14))), 2)
    dilution_1m = pp.dilution_at_height(profile, 1.0)

    plume_export = {"profile": profile, "species": species_profiles} if keep_profile else None
    return chem, near_field_chem, profile.rise_height_m, dilution_1m, plume_export


def sample_chimney_height(vent_type: str, temperature_c: float, rng: np.random.Generator) -> float:
    """
    Altura real do edifício da chaminé, em metros. A maioria das chaminés
    ativas fica na faixa de poucos metros a ~12 m; ocasionalmente, em
    sistemas de sulfeto maciço estáveis por longos períodos sem colapsar,
    surgem estruturas excepcionais de dezenas de metros — documentado em
    "Godzilla" (campo Mothra, Endeavour), que atingiu ~45 m antes de
    desabar em 1995, e nas chaminés carbonáticas de Lost City (até ~60 m,
    um sistema hidrotermal distinto dos black smokers clássicos).
    """
    temp_frac = np.clip(temperature_c / 400.0, 0.05, 1.0)
    height = 2.0 + temp_frac * 10.0  # típico: ~2-12 m

    if vent_type == "black_smoker" and rng.random() < 0.06:
        height = rng.uniform(25, 48)  # estrutura excepcional tipo "Godzilla"
    elif vent_type == "white_smoker" and rng.random() < 0.02:
        height = rng.uniform(15, 28)

    return float(height)


def fauna_zonation(vent_type: str, near_field_chem: dict, rng: np.random.Generator) -> dict:
    """
    Zonação simplificada de comunidades quimiossintéticas em função de
    temperatura/sulfeto, seguindo o padrão observado em campos do
    Pacífico Leste / Atlântico (tapetes bacterianos > vermes tubulares >
    mexilhões > camarões/caranguejos > fauna periférica).

    `near_field_chem` DEVE ser a concentração de ORIGEM (`simulate_plume`'s
    `near_field_chem`, não `chem`) — a fauna vive na abertura/fluxo difuso,
    não na altura de flutuabilidade neutra da pluma (diluída por ordens de
    grandeza em relação à origem). Passar `chem` aqui era um bug real: os
    limiares de H2S abaixo (calibrados pra faixa de origem, ~0-6.5 mM)
    nunca eram atingidos por nenhum vent gerado com a `chem` diluída,
    então `tubeworm`/`mussel_bed` nunca apareciam em nenhum campo real.
    """
    h2s = near_field_chem.get("H2S", 0)
    zones = {}
    zones["bacterial_mat"] = round(rng.uniform(0.2, 1.0), 2)
    if h2s > 1.5:
        zones["tubeworm"] = round(rng.uniform(0.5, 3.0), 2)
    if h2s > 0.8:
        zones["mussel_bed"] = round(rng.uniform(1.0, 5.0), 2)
    if vent_type == "black_smoker" and rng.random() < 0.6:
        zones["shrimp_swarm"] = round(rng.uniform(0.3, 2.5), 2)
    zones["peripheral_fauna"] = round(rng.uniform(3.0, 10.0), 2)
    return zones


# --------------------------------------------------------------------------
# 3. Distribuição espacial do campo de fumarolas
# --------------------------------------------------------------------------

def generate_vent_field(terrain: np.ndarray, axis_y: np.ndarray, n_clusters: int,
                         vents_per_cluster: tuple[int, int], spreading_rate: float,
                         local_relief_m: float, ocean_depth_baseline_m: float,
                         rng: np.random.Generator,
                         alpha: float = pp.DEFAULT_ALPHA_ENTRAINMENT,
                         n_freq: float = pp.DEFAULT_N_BRUNT_VAISALA,
                         basin: str = "atlantic",
                         export_profiles: bool = False) -> tuple[List["Vent"], dict]:
    """
    `local_relief_m` é a mesma amplitude real de relevo usada em todas as
    renderizações (a área pesquisada tem tipicamente 50-300 m de variação
    local); `ocean_depth_baseline_m` é a profundidade ambiente do oceano
    na crista da dorsal (tipicamente 2000-3000 m), somada ao relevo local
    para dar a profundidade absoluta de cada fumarola. Usar um único valor
    de relevo em toda a pipeline evita que a profundidade/física de cada
    vent seja calculada sobre uma amplitude de terreno diferente da que
    aparece nas visualizações e, portanto, de qualquer modelo de
    gradiente construído sobre esses dados.

    `alpha` (coef. de entranhamento) e `n_freq` (frequência de
    Brunt-Väisälä) são tratados como constantes em todo o campo — ver
    docs/PHYSICS_MODEL.md para os defaults citados e sua incerteza.
    `basin` seleciona a cinética de oxidação de Fe(II) (assimetria
    Atlântico/Pacífico documentada por Field & Sherrell, 2000).
    Retorna `(vents, profiles)`; `profiles` mapeia `vent.id -> PlumeProfile`
    e só é preenchido quando `export_profiles=True`.
    """
    size = terrain.shape[0]
    vents: List[Vent] = []
    profiles: dict = {}
    vent_id = 0

    cluster_xs = np.sort(rng.choice(np.arange(int(size * 0.05), int(size * 0.95)),
                                     size=n_clusters, replace=False))

    for cid, cx in enumerate(cluster_xs):
        cy = axis_y[int(cx)]
        n_vents = int(rng.integers(vents_per_cluster[0], vents_per_cluster[1] + 1))
        cluster_spread = rng.uniform(1.5, 4.5)

        for _ in range(n_vents):
            x = np.clip(cx + rng.normal(0, cluster_spread), 0, size - 1)
            y = np.clip(cy + rng.normal(0, cluster_spread), 0, size - 1)
            xi, yi = int(x), int(y)
            depth_m = float(ocean_depth_baseline_m + terrain[yi, xi] * local_relief_m)

            vtype, temp = classify_and_sample_temperature(rng, spreading_rate)
            chem, near_field_chem, rise, dilution_1m, plume_export = simulate_plume(
                vtype, temp, rng, alpha=alpha, n_freq=n_freq,
                basin=basin, keep_profile=export_profiles)
            chimney_height_m = sample_chimney_height(vtype, temp, rng)
            fauna = fauna_zonation(vtype, near_field_chem, rng)

            vents.append(Vent(
                id=vent_id, cluster_id=cid, x=float(x), y=float(y), depth_m=depth_m,
                vent_type=vtype, temperature_c=round(temp, 1), chemistry=chem,
                chimney_height_m=round(chimney_height_m, 1),
                plume_rise_m=round(rise, 1), fauna_zones=fauna,
                dilution_near_field_1m=round(dilution_1m, 4),
            ))
            if plume_export is not None:
                profiles[vent_id] = plume_export
            vent_id += 1

    return vents, profiles


# --------------------------------------------------------------------------
# 4. Visualização
# --------------------------------------------------------------------------

def _bathymetry_depth_m(terrain: np.ndarray, local_relief_m: float, ocean_depth_baseline_m: float) -> np.ndarray:
    """Converte o heightmap normalizado em profundidade absoluta (m), mesma fórmula usada em `generate_vent_field`."""
    return ocean_depth_baseline_m + terrain * local_relief_m


def _add_bathymetry_contours(ax, terrain: np.ndarray, local_relief_m: float, ocean_depth_baseline_m: float,
                              extent: Optional[tuple] = None, n_levels: int = 6,
                              color: str = "#2b2b28", label_unit: str = "m") -> Line2D:
    """
    Desenha isóbatas (linhas de profundidade constante, em metros abaixo do
    nível do mar) sobre o eixo, derivadas da mesma profundidade
    base + amplitude de relevo usadas para calcular `Vent.depth_m` em
    `generate_vent_field` — garante que as linhas batimétricas mostradas
    correspondem exatamente à física usada no resto da simulação, não a um
    relevo estilizado independente. Linhas e rótulos usam contorno branco
    (path_effects) para permanecerem legíveis independente do colormap de
    fundo (terreno sombreado, campo de enriquecimento, etc.). Retorna um
    proxy `Line2D` para uso em legendas (o objeto `QuadContourSet` do
    matplotlib não é aceito diretamente por `ax.legend`).
    """
    depth_m = _bathymetry_depth_m(terrain, local_relief_m, ocean_depth_baseline_m)
    levels = np.linspace(float(depth_m.min()), float(depth_m.max()), n_levels)
    size_y, size_x = terrain.shape
    halo = [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
    if extent is not None:
        x_coords = np.linspace(extent[0], extent[1], size_x)
        y_coords = np.linspace(extent[2], extent[3], size_y)
        cs = ax.contour(x_coords, y_coords, depth_m, levels=levels, colors=color,
                         linewidths=0.7, alpha=0.9, linestyles="solid")
    else:
        cs = ax.contour(depth_m, levels=levels, origin="lower", colors=color,
                         linewidths=0.7, alpha=0.9, linestyles="solid")
    cs.set_path_effects(halo)
    labels = ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:.0f}")
    for lbl in labels:
        lbl.set_path_effects(halo)
    return Line2D([0], [0], color=color, linewidth=0.7, label=f"isobath (depth, {label_unit})")


def plot_field(terrain: np.ndarray, vents: List[Vent], out_path: str, title: str,
                local_relief_m: float, ocean_depth_baseline_m: float):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- mapa 2D com relevo sombreado ---
    ax = axes[0]
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(terrain, cmap=ABYSSAL_CMAP, vert_exag=3, blend_mode="soft")
    ax.imshow(shaded, origin="lower")
    bathy_proxy = _add_bathymetry_contours(ax, terrain, local_relief_m, ocean_depth_baseline_m)

    for vtype, style in VENT_TYPES.items():
        pts = [v for v in vents if v.vent_type == vtype]
        if not pts:
            continue
        xs = [v.x for v in pts]
        ys = [v.y for v in pts]
        sizes = [20 + v.temperature_c * 0.6 for v in pts]
        ax.scatter(xs, ys, s=sizes, c=style["color"], edgecolors="white",
                   linewidths=0.6, label=vtype.replace("_", " "), zorder=5)

    ax.set_title("Hydrothermal vent field over the mid-ocean ridge")
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid, rift axis ~ center)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [bathy_proxy], labels=labels + [bathy_proxy.get_label()],
              loc="upper right", fontsize=8, framealpha=0.85, title="Legend",
              title_fontsize=8)

    # --- dispersão de temperatura x altura de pluma ---
    ax2 = axes[1]
    for vtype, style in VENT_TYPES.items():
        pts = [v for v in vents if v.vent_type == vtype]
        if not pts:
            continue
        ax2.scatter([v.temperature_c for v in pts], [v.plume_rise_m for v in pts],
                    c=style["color"], edgecolors="white", linewidths=0.6,
                    label=vtype.replace("_", " "), s=40)
    ax2.set_xlabel("Fluid temperature (°C)")
    ax2.set_ylabel("Plume rise height to neutral buoyancy (m)")
    ax2.set_title("Buoyant plume model (Speer & Rona, 1989)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# rampa divergente (azul = abaixo do controle, vermelho = acima do
# controle), deliberadamente distinta do matiz frio do ABYSSAL_CMAP do
# terreno, já que os dois aparecem juntos na mesma cena — ColorBrewer RdBu
ENRICHMENT_CMAP = plt.get_cmap("RdBu_r")


def _format_fold(x: float) -> str:
    """Formata um fator multiplicativo (ex: enriquecimento vs. controle) sem notação científica."""
    if x >= 100:
        return f"{x:.0f}x"
    if x >= 10:
        return f"{x:.1f}x"
    if x >= 0.01:
        return f"{x:.2f}x"
    return f"{x:.4f}x"


def plot_hotspots(terrain: np.ndarray, vents: List[Vent], hotspot_records: List[dict],
                   out_path: str, title: str, local_relief_m: float, ocean_depth_baseline_m: float,
                   top_n: int = 12, molecule_label: str = "amino acids"):
    """
    Visualiza o enriquecimento de concentração da classe de molécula
    escolhida em relação ao controle (mesma síntese de base, sujeita
    apenas à diluição da pluma, sem mecanismo concentrador ativo — ver
    CONTROL_FLAGS em prebiotic.py): mapa espacial (cor e tamanho = fator
    de enriquecimento, rampa divergente centrada em 1x) e ranking dos
    principais hotspots. Vents sem síntese de base (baseline ~0, onde a
    razão vs. controle não é significativa) aparecem em cinza neutro.
    """
    by_id = {r["vent_id"]: r for r in hotspot_records}
    enrichments = np.array([
        by_id[v.id]["enrichment_vs_control"] if by_id[v.id]["enrichment_vs_control"] is not None else np.nan
        for v in vents
    ])
    valid = ~np.isnan(enrichments)
    log2_enrich = np.full_like(enrichments, np.nan)
    log2_enrich[valid] = np.log2(enrichments[valid])
    max_abs_log2 = max(float(np.nanmax(np.abs(log2_enrich))) if valid.any() else 1.0, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_log2, vcenter=0.0, vmax=max_abs_log2)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- mapa espacial ---
    ax = axes[0]
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(terrain, cmap=ABYSSAL_CMAP, vert_exag=3, blend_mode="soft")
    ax.imshow(shaded, origin="lower", alpha=0.55)  # backdrop discreto: o dado é o enriquecimento, não o relevo
    bathy_proxy = _add_bathymetry_contours(ax, terrain, local_relief_m, ocean_depth_baseline_m)

    xs = np.array([v.x for v in vents])
    ys = np.array([v.y for v in vents])
    sizes = np.full(len(vents), 40.0)
    if valid.any():
        sizes[valid] = 25 + 220 * (np.abs(log2_enrich[valid]) / max_abs_log2) ** 0.7

    if (~valid).any():
        ax.scatter(xs[~valid], ys[~valid], s=sizes[~valid], c="#8a8a86",
                   edgecolors="#2c2c2a", linewidths=0.5, zorder=4, label="no baseline synthesis (n/a)")

    sc = ax.scatter(xs[valid], ys[valid], s=sizes[valid], c=log2_enrich[valid], cmap=ENRICHMENT_CMAP, norm=norm,
                     edgecolors="#2c2c2a", linewidths=0.6, zorder=5)

    if valid.any():
        top_idx = int(np.nanargmax(enrichments))
        ax.scatter([xs[top_idx]], [ys[top_idx]], s=[sizes[top_idx] * 1.6], facecolors="none",
                   edgecolors="#0b0b0b", linewidths=1.6, zorder=6, label="leading hotspot (highest enrichment)")
        ax.annotate(f"leading hotspot\n{_format_fold(enrichments[top_idx])} control",
                    (xs[top_idx], ys[top_idx]), xytext=(12, 12), textcoords="offset points",
                    fontsize=8, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#c3c2b7", alpha=0.9))

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(f"{molecule_label} enrichment vs. control")
    tick_log2 = np.linspace(-max_abs_log2, max_abs_log2, 5)
    cbar.set_ticks(tick_log2)
    cbar.set_ticklabels([_format_fold(2 ** t) for t in tick_log2])

    ax.set_title("Enrichment vs. control (dilution, no concentrating mechanism)")
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")
    enrich_proxy = Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a4b8f",
                          markeredgecolor="#2c2c2a", markersize=8,
                          label="vent (color/size = enrichment, see colorbar)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[enrich_proxy] + handles + [bathy_proxy],
              labels=[enrich_proxy.get_label()] + labels + [bathy_proxy.get_label()],
              loc="lower right", fontsize=7, framealpha=0.85, title="Legend", title_fontsize=7)

    # --- ranking dos principais hotspots ---
    ax2 = axes[1]
    ranked = [r for r in hotspot_records if r["enrichment_vs_control"] is not None]
    top_records = ranked[:top_n][::-1]  # já vem ordenado desc.; inverte p/ maior no topo do barh
    labels = [f"#{r['vent_id']} {r['vent_type'].replace('_', ' ')}" for r in top_records]
    values = [r["enrichment_vs_control"] for r in top_records]
    colors = ENRICHMENT_CMAP(norm(np.log2(values))) if values else []
    ax2.barh(labels, values, color=colors, edgecolor="#2c2c2a", linewidth=0.5)
    ax2.axvline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax2.set_xlabel("Enrichment vs. control (x)")
    ax2.set_title(f"Top {min(top_n, len(ranked))} hotspots")
    ax2.grid(axis="x", alpha=0.3)
    ax2.tick_params(axis="y", labelsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


MODULE_GRADIENT_LABELS = {
    "dilution": "Plume dilution / advection",
    "thermophoresis": "Pore thermophoresis",
    "mineral_adsorption": "Mineral-surface adsorption",
    "proton_gradient": "Proton gradient (alkaline compartmentalization)",
}


def _idw_gradient_field(terrain: np.ndarray, vents: List[Vent], values_by_id: dict,
                         sigma_cells: float) -> np.ndarray:
    """
    Interpola os fatores POR FUMAROLA (escalares — ver docstring de
    `plot_module_gradient_map`) sobre a grade inteira via média
    ponderada por kernel gaussiano centrado em cada fumarola. É uma
    ferramenta de VISUALIZAÇÃO (mostra onde o efeito de cada módulo se
    concentra espacialmente, dado a posição das fumarolas), não um
    campo físico contínuo resolvido — ao contrário do campo acústico
    (acoustics.py), que de fato resolve pressão/velocidade em cada
    célula da grade. `sigma_cells` é a largura do kernel (escolha de
    visualização, não um comprimento de difusão físico).
    """
    size = terrain.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    num = np.zeros((size, size))
    den = np.zeros((size, size))
    for v in vents:
        val = values_by_id.get(v.id)
        if val is None:
            continue
        w = np.exp(-((xx - v.x) ** 2 + (yy - v.y) ** 2) / (2 * sigma_cells ** 2))
        num += w * val
        den += w
    return num / np.maximum(den, 1e-12)


def plot_module_gradient_map(terrain: np.ndarray, vents: List[Vent], hotspot_records: List[dict],
                              factor_key: str, module_label: str, out_path: str, title: str,
                              local_relief_m: float, ocean_depth_baseline_m: float):
    """
    Vista de topo do campo de fumarolas sobreposta por um mapa de
    gradiente do fator de enriquecimento de UM módulo prebiótico
    específico — pensado para explicar, etapa a etapa, o que cada
    mecanismo de concentração gera espacialmente (um mapa por módulo
    ativo).

    IMPORTANTE (ler antes de interpretar a figura): os quatro módulos
    clássicos (diluição, termoforese, adsorção mineral, gradiente de
    prótons) calculam apenas um FATOR ESCALAR por fumarola — nenhum
    deles resolve um campo físico contínuo entre fumarolas. O "mapa de
    gradiente" aqui é uma INTERPOLAÇÃO por kernel gaussiano entre os
    valores discretos de cada fumarola (`_idw_gradient_field`) — uma
    ferramenta de visualização para ver onde o efeito se concentra no
    espaço, não uma previsão espacial do modelo. Isto é diferente do
    módulo acústico (ver `plot_acoustic_field`), que resolve um campo
    físico real ponto a ponto.
    """
    by_id = {r["vent_id"]: r[factor_key] for r in hotspot_records}
    values = np.array([by_id.get(v.id, 1.0) for v in vents])
    size = terrain.shape[0]
    n_vents = max(len(vents), 1)
    sigma_cells = max(size / (2.0 * np.sqrt(n_vents)), 3.0)
    field = _idw_gradient_field(terrain, vents, by_id, sigma_cells)

    log2_field = np.log2(np.clip(field, 1e-6, None))
    log2_vals = np.log2(np.clip(values, 1e-6, None))
    max_abs = max(float(np.max(np.abs(log2_field))), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8, 7))
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(terrain, cmap=ABYSSAL_CMAP, vert_exag=3, blend_mode="soft")
    ax.imshow(shaded, origin="lower", alpha=0.5)
    im = ax.imshow(log2_field, origin="lower", cmap=ENRICHMENT_CMAP, norm=norm, alpha=0.75)
    bathy_proxy = _add_bathymetry_contours(ax, terrain, local_relief_m, ocean_depth_baseline_m)

    xs = [v.x for v in vents]
    ys = [v.y for v in vents]
    sizes = 40 + 110 * (np.abs(log2_vals) / max_abs) ** 0.7
    ax.scatter(xs, ys, s=sizes, c=log2_vals, cmap=ENRICHMENT_CMAP, norm=norm,
               edgecolors="#2c2c2a", linewidths=0.6, zorder=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(f"{module_label} factor (interpolated, log2, 0=1x)")
    tick_log2 = np.linspace(-max_abs, max_abs, 5)
    cbar.set_ticks(tick_log2)
    cbar.set_ticklabels([_format_fold(2 ** t) for t in tick_log2])

    ax.set_title(f"Gradient map (interpolated between vents) — {module_label}", fontsize=11)
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")
    vent_proxy = Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a4b8f",
                        markeredgecolor="#2c2c2a", markersize=8,
                        label="actual vent position (color/size = local factor)")
    # (sem swatch de cor fixa pro fundo interpolado: um retângulo de cor
    # única não pode representar honestamente uma rampa DIVERGENTE — o
    # fundo pode ser azul, branco ou vermelho dependendo do valor local; a
    # barra de cores já explica isso corretamente, ver `cbar.set_label`
    # acima. Um Patch fixo aqui pintava vermelho mesmo quando o campo
    # inteiro era azul — bug real corrigido 2026-08-07.)
    ax.legend(handles=[vent_proxy, bathy_proxy], loc="lower right",
              fontsize=6.5, framealpha=0.85, title="Legend", title_fontsize=6.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_acoustic_field(terrain: np.ndarray, vents: List[Vent], acoustic_result: dict,
                         domain_size_m: float, out_path: str, title: str,
                         local_relief_m: float, ocean_depth_baseline_m: float):
    """
    Visualiza o campo de enriquecimento acústico (hipótese exploratória,
    ver acoustics.py): mapa espacial do fator de enriquecimento sobre o
    domínio inteiro (não só nas fumarolas — o ponto da hipótese é que
    regiões de interferência podem NÃO coincidir com uma fumarola), com
    os picos locais detectados marcados distintamente das posições das
    fumarolas, e um painel de diagnóstico mostrando se o mecanismo é
    fisicamente relevante nas condições simuladas (ver
    `diagnostics["gorkov_trap_physically_relevant"]`).
    """
    size = terrain.shape[0]
    meters_per_cell = domain_size_m / (size - 1)
    field = acoustic_result["enrichment_field"]
    diag = acoustic_result["diagnostics"]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(terrain, cmap=ABYSSAL_CMAP, vert_exag=3, blend_mode="soft")
    ax.imshow(shaded, origin="lower", alpha=0.45,
              extent=[0, domain_size_m, 0, domain_size_m])
    bathy_proxy = _add_bathymetry_contours(ax, terrain, local_relief_m, ocean_depth_baseline_m,
                                           extent=(0, domain_size_m, 0, domain_size_m))

    vmax = max(float(np.nanmax(field)), 1.0 + 1e-9)
    im = ax.imshow(field, origin="lower", cmap="magma", alpha=0.75,
                    extent=[0, domain_size_m, 0, domain_size_m], vmin=1.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(f"Acoustic enrichment ({acoustic_result['mode']})")

    vent_x_m = [v.x * meters_per_cell for v in vents]
    vent_y_m = [v.y * meters_per_cell for v in vents]
    ax.scatter(vent_x_m, vent_y_m, s=30, c="#3fb6a8", edgecolors="white",
               linewidths=0.5, marker="^", zorder=5, label="vents")

    peaks = acoustic_result["peaks"][:8]
    if peaks:
        ax.scatter([p["x_m"] for p in peaks], [p["y_m"] for p in peaks], s=70,
                   facecolors="none", edgecolors="#ffffff", linewidths=1.4,
                   zorder=6, label="detected interference peaks")

    ax.set_title("Acoustic enrichment field over the domain")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [bathy_proxy], labels=labels + [bathy_proxy.get_label()],
              loc="upper right", fontsize=7, framealpha=0.85, title="Legend", title_fontsize=7)

    ax2 = axes[1]
    ax2.axis("off")
    relevant = diag.get("gorkov_trap_physically_relevant")
    lines = [
        f"Mode: {acoustic_result['mode']}",
        f"Sound speed: {diag.get('sound_speed_m_s', float('nan')):.1f} m/s",
        f"Absorption @100Hz: {diag.get('absorption_np_per_m_at_100hz', float('nan')):.3e} Np/m",
    ]
    if "streaming_speed_max_m_s" in diag:
        lines.append(f"Max. streaming speed: {diag['streaming_speed_max_m_s']:.3e} m/s")
    if "gorkov_trap_depth_over_kT" in diag:
        lines.append(f"Trap depth / kT: {diag['gorkov_trap_depth_over_kT']:.3e}")
        lines.append("Physically relevant trap (>kT): " + ("YES" if relevant else "NO"))
    lines.append("")
    lines.append("Exploratory hypothesis, no experimental validation —")
    lines.append("see docs/PHYSICS_MODEL.md, acoustic model section,")
    lines.append("before citing any result from this panel.")
    ax2.text(0.02, 0.95, "\n".join(lines), transform=ax2.transAxes, va="top", fontsize=10,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fcfcfb", ec="#c3c2b7", alpha=0.9))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _chimney_mesh(base_x: float, base_y: float, base_z: float, height: float, base_radius: float,
                   rng: np.random.Generator, n_theta: int = 16, n_h: int = 12):
    """
    Malha procedural de uma chaminé hidrotermal: base afunilando rapidamente
    para uma torre esguia, com um leve alargamento ("flange") perto do topo
    (comum em chaminés de sulfeto maciço) e irregularidade de superfície via
    ruído harmônico, para lembrar a textura rugosa dos depósitos minerais.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    hfrac = np.linspace(0, 1, n_h)
    Theta, Hf = np.meshgrid(theta, hfrac)

    radius = base_radius * (1 - Hf) ** 1.6 + base_radius * 0.12
    flange = 1 + 0.3 * np.clip((Hf - 0.82) / 0.18, 0, 1)
    radius *= flange

    phase1, phase2 = rng.uniform(0, 2 * np.pi, 2)
    bumps = (0.35 * np.sin(3 * Theta + phase1)
             + 0.25 * np.sin(5 * Theta - 4 * Hf * np.pi + phase2)
             + 0.15 * rng.normal(0, 1, Theta.shape))
    radius = np.clip(radius * (1 + 0.22 * bumps), base_radius * 0.08, None)

    Xc = base_x + radius * np.cos(Theta)
    Yc = base_y + radius * np.sin(Theta)
    Zc = base_z + Hf * height
    return Xc, Yc, Zc


def _plume_smoke(base_x: float, base_y: float, start_z: float, top_z: float,
                  vent_color: str, rng: np.random.Generator, n_points: int = 90):
    """
    Nuvem de pontos representando a pluma hidrotermal subindo em turbulência
    até o nível de flutuabilidade neutra: alarga e dilui (cor e opacidade
    tendem à água do mar) à medida que sobe, imitando a mistura turbulenta
    com a coluna d'água.
    """
    if top_z <= start_z:
        return None

    hfrac = rng.uniform(0, 1, n_points) ** 1.4
    z = start_z + hfrac * (top_z - start_z)
    spread = 0.15 + hfrac * 1.6
    r = spread * np.sqrt(rng.uniform(0, 1, n_points))
    theta = rng.uniform(0, 2 * np.pi, n_points)
    x = base_x + r * np.cos(theta)
    y = base_y + r * np.sin(theta)

    base_rgba = np.array(mcolors.to_rgba(vent_color))
    dilute_rgba = np.array(mcolors.to_rgba("#cfe9e6"))
    t = hfrac[:, np.newaxis]
    colors = base_rgba[np.newaxis, :] * (1 - t) + dilute_rgba[np.newaxis, :] * t
    colors[:, 3] = np.clip(0.5 * (1 - 0.55 * hfrac), 0.05, 0.5)
    sizes = 6 + hfrac * 45
    return x, y, z, colors, sizes


def plot_field_3d(terrain: np.ndarray, vents: List[Vent], out_path: str, title: str,
                   local_relief_m: float, z_exag: float = 25.0, view: tuple[float, float] = (28, -60),
                   chimney_scale: float = 1.0, seed: int = 0):
    """
    Renderiza o terreno como superfície 3D com chaminés hidrotermais
    modeladas como pequenos edifícios cônicos irregulares (não apenas
    marcadores) e a pluma como uma nuvem de fumaça turbulenta subindo até
    a altura de flutuabilidade neutra estimada por `plume_rise_height`.
    `z_exag` controla a exageração vertical do relevo; `chimney_scale`
    ajusta a altura visual das chaminés (elas são desenhadas fora de
    escala real, como marcos visuais sobre o terreno).
    """
    size = terrain.shape[0]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    # desativa o cálculo automático de profundidade do matplotlib (que mistura
    # incorretamente a superfície semi-transparente com os marcadores) e passa
    # a respeitar a ordem de desenho / zorder explícito de cada artista
    ax.computed_zorder = False

    xs = np.arange(size)
    ys = np.arange(size)
    X, Y = np.meshgrid(xs, ys)
    Z = terrain * z_exag

    stride = max(1, size // 120)
    ax.plot_surface(X, Y, Z, cmap=ABYSSAL_CMAP, linewidth=0, antialiased=True,
                     alpha=0.85, rstride=stride, cstride=stride, zorder=1)

    z_lift = 0.015 * z_exag  # embute levemente a base da chaminé no relevo, evitando "flutuar"

    for v in vents:
        style = VENT_TYPES[v.vent_type]
        rng_v = np.random.default_rng((seed * 1_000_003 + v.id * 7919 + 12345) % (2 ** 32 - 1))

        base_x, base_y = v.x, v.y
        base_z = terrain[int(v.y), int(v.x)] * z_exag - z_lift

        temp_frac = np.clip(v.temperature_c / 400.0, 0.05, 1.0)
        chimney_height = z_exag * 0.055 * (0.5 + temp_frac) * chimney_scale
        chimney_base_r = 0.25 + chimney_height * 0.20

        Xc, Yc, Zc = _chimney_mesh(base_x, base_y, base_z, chimney_height, chimney_base_r, rng_v)
        ax.plot_surface(Xc, Yc, Zc, color=style["color"], linewidth=0, antialiased=True,
                         shade=True, zorder=6)

        chimney_top_z = base_z + chimney_height
        plume_top_z = base_z + v.plume_rise_m / local_relief_m * z_exag
        n_smoke = int(np.clip(40 + v.temperature_c * 0.2, 40, 130))
        smoke = _plume_smoke(base_x, base_y, chimney_top_z, plume_top_z, style["color"], rng_v, n_smoke)
        if smoke is not None:
            sx, sy, sz, scolors, ssizes = smoke
            ax.scatter(sx, sy, sz, s=ssizes, c=scolors, edgecolors="none",
                       depthshade=False, zorder=7)

    legend_handles = [Patch(facecolor=style["color"], edgecolor="white", label=vtype.replace("_", " "))
                       for vtype, style in VENT_TYPES.items()]

    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")
    ax.set_zlabel(f"exaggerated elevation (real relief ≈ {local_relief_m:.0f} m variation)")
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=view[0], azim=view[1])
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_field_3d_true_scale(terrain: np.ndarray, vents: List[Vent], out_path: str, title: str,
                              domain_size_m: float, local_relief_m: float, seed: int = 0,
                              view: tuple[float, float] = (12, -50)):
    """
    Renderiza o mesmo campo em proporção física verdadeira (x, y e z em
    metros, caixa com aspecto 1:1:1) — sem nenhum exagero vertical. Isso
    mostra como um campo de fumarolas realmente aparenta: o fundo
    oceânico é quase plano em relação à sua extensão horizontal, e as
    chaminés (poucos metros) e mesmo as plumas (dezenas a centenas de
    metros) ficam bem menores que o exagero artificial usado em
    `plot_field_3d`. `local_relief_m` é a variação de relevo real da
    área pesquisada (tipicamente 50-300 m para um único campo de
    fumarolas, bem menor que a batimetria regional).
    """
    size = terrain.shape[0]
    meters_per_cell = domain_size_m / (size - 1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    xs = np.arange(size) * meters_per_cell
    ys = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, ys)
    Z = terrain * local_relief_m

    stride = max(1, size // 150)
    ax.plot_surface(X, Y, Z, cmap=ABYSSAL_CMAP, linewidth=0, antialiased=True,
                     alpha=0.9, rstride=stride, cstride=stride, zorder=1)

    max_plume_top = local_relief_m
    for v in vents:
        style = VENT_TYPES[v.vent_type]
        rng_v = np.random.default_rng((seed * 1_000_003 + v.id * 7919 + 12345) % (2 ** 32 - 1))

        base_x, base_y = v.x * meters_per_cell, v.y * meters_per_cell
        base_z = terrain[int(v.y), int(v.x)] * local_relief_m

        chimney_height_m = v.chimney_height_m
        chimney_base_r_m = 0.6 + chimney_height_m * 0.16

        Xc, Yc, Zc = _chimney_mesh(base_x, base_y, base_z, chimney_height_m, chimney_base_r_m, rng_v)
        ax.plot_surface(Xc, Yc, Zc, color=style["color"], linewidth=0, antialiased=True,
                         shade=True, zorder=6)

        chimney_top_z = base_z + chimney_height_m
        plume_top_z = base_z + v.plume_rise_m  # já em metros reais, sem conversão
        max_plume_top = max(max_plume_top, plume_top_z)

        n_smoke = int(np.clip(30 + v.temperature_c * 0.12, 30, 90))
        smoke = _plume_smoke(base_x, base_y, chimney_top_z, plume_top_z, style["color"], rng_v, n_smoke)
        if smoke is not None:
            sx, sy, sz, scolors, ssizes = smoke
            ax.scatter(sx, sy, sz, s=ssizes, c=scolors, edgecolors="none",
                       depthshade=False, zorder=7)

    legend_handles = [Patch(facecolor=style["color"], edgecolor="white", label=vtype.replace("_", " "))
                       for vtype, style in VENT_TYPES.items()]

    z_extent = max_plume_top * 1.05
    ax.set_box_aspect((domain_size_m, domain_size_m, z_extent))
    ax.set_zlim(0, z_extent)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("true elevation (m) — no exaggeration, 1:1:1 aspect ratio")
    ax.set_title(title + "\n(true vertical scale — note how flat the terrain appears)", fontsize=11)
    ax.view_init(elev=view[0], azim=view[1])
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Exportação
# --------------------------------------------------------------------------

def export_data(vents: List[Vent], out_dir: str, basename: str):
    records = [v.to_record() for v in vents]

    json_path = os.path.join(out_dir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, f"{basename}.csv")
    if records:
        fieldnames = sorted({k for r in records for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    return json_path, csv_path


def export_plume_profiles(profiles: dict, csv_path: str):
    """Grava os perfis de pluma completos em formato longo (uma linha por
    altura amostrada por vento): vent_id, z_m, dilution, temperature_c,
    t_since_vent_s, C_H2S, C_Fe, C_Mn, C_CH4. Só chamado quando
    --export-plume-profiles está ativo (custo de integração ODE por
    vento não é desprezível em ensembles grandes — ver
    docs/PHYSICS_MODEL.md)."""
    rows = []
    for vent_id, bundle in profiles.items():
        profile = bundle["profile"]
        species = bundle["species"]
        for i in range(len(profile.z)):
            row = {
                "vent_id": vent_id,
                "z_m": round(float(profile.z[i]), 3),
                "dilution": float(profile.dilution[i]),
                "temperature_c": round(float(profile.temperature_c[i]), 3),
                "t_since_vent_s": round(float(profile.t[i]), 3),
            }
            for sp_name, sp_arr in species.items():
                row[f"C_{sp_name}"] = float(sp_arr[i])
            rows.append(row)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# --------------------------------------------------------------------------
# 5b. Varredura de sensibilidade (Hipercubo Latino)
# --------------------------------------------------------------------------

def latin_hypercube_1d(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """
    Amostragem por Hipercubo Latino em 1D — McKay, M.D., Beckman, R.J., &
    Conover, W.J. (1979), "A comparison of three methods for selecting
    values of input variables in the analysis of output from a computer
    code," Technometrics 21(2), 239-245. Estratifica [low, high] em `n`
    intervalos iguais e sorteia um ponto dentro de cada um, depois
    embaralha a ordem — importante quando combinado com outro parâmetro
    amostrado independentemente (ex.: alpha e tamanho do agregado
    acústico), para não correlacionar artificialmente os dois ao
    emparelhar um valor por run.
    """
    edges = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return low + u * (high - low)


def joint_latin_hypercube(n: int, bounds: List[tuple], rng: np.random.Generator) -> np.ndarray:
    """
    Hipercubo Latino CONJUNTO em d dimensões (generalização multi-D do
    mesmo método de McKay, Beckman & Conover 1979 — ver `latin_hypercube_1d`),
    via `scipy.stats.qmc.LatinHypercube`. Cada dimensão sai marginalmente
    estratificada (mesma garantia de `latin_hypercube_1d`), mas a atribuição
    de estrato por dimensão é escolhida com otimização de discrepância
    centrada (`optimization="random-cd"`) em vez de uma permutação puramente
    aleatória — reduz a chance de correlação espúria residual entre
    dimensões que uma permutação independente ingênua pode deixar por sorte
    numa amostra pequena, sem introduzir nenhuma correlação FÍSICA real
    (as margens continuam as mesmas faixas documentadas por parâmetro).

    `bounds`: lista de `(low, high)`, uma tupla por dimensão. Retorna um
    array `(n, d)` já escalado para as faixas reais (não [0,1]).
    """
    d = len(bounds)
    optimization = "random-cd" if n * d > 1 else None
    sampler = qmc.LatinHypercube(d=d, seed=rng, optimization=optimization)
    sample = sampler.random(n=n)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    return qmc.scale(sample, lows, highs)


# --------------------------------------------------------------------------
# 6. Gerenciamento de runs / outputs
# --------------------------------------------------------------------------

def make_timestamped_dir(parent: str, prefix: str, when: Optional[datetime] = None) -> str:
    """
    Cria (e retorna o caminho de) uma subpasta dentro de `parent`, nomeada
    "{prefix}_DDDDDD_HHHHHH" (data AAMMDD + hora HHMMSS). Em caso de colisão
    (duas pastas no mesmo segundo, comum em ensembles rápidos), desambigua
    com um sufixo incremental.
    """
    when = when or datetime.now()
    base_name = when.strftime(f"{prefix}_%y%m%d_%H%M%S")
    os.makedirs(parent, exist_ok=True)

    candidate = base_name
    suffix = 2
    while os.path.exists(os.path.join(parent, candidate)):
        candidate = f"{base_name}_{suffix}"
        suffix += 1

    new_dir = os.path.join(parent, candidate)
    os.makedirs(new_dir)
    return new_dir


def make_experiment_dir(outputs_root: str, when: Optional[datetime] = None) -> str:
    """Cria a pasta de um experimento (uma execução do programa): "experimento_DDDDDD_HHHHHH"."""
    return make_timestamped_dir(outputs_root, "experimento", when)


def make_run_dir(experiment_dir: str, when: Optional[datetime] = None) -> str:
    """Cria a subpasta de uma run dentro da pasta do experimento: "run_DDDDDD_HHHHHH"."""
    return make_timestamped_dir(experiment_dir, "run", when)


def execute_run(args: argparse.Namespace, seed: int, run_dir: str, make_images: bool) -> dict:
    """Executa uma run completa (terreno + campo de fumarolas) e grava tudo em `run_dir`."""
    rng = np.random.default_rng(seed)

    terrain = diamond_square(args.size, args.roughness, rng)
    terrain, axis_y = carve_axial_valley(terrain, axis_wander=args.size * 0.12,
                                          depth=0.4, width_frac=0.12, rng=rng)

    vents, plume_profiles = generate_vent_field(
        terrain, axis_y, n_clusters=args.n_clusters,
        vents_per_cluster=(args.vents_min, args.vents_max),
        spreading_rate=args.spreading_rate, local_relief_m=args.local_relief_m,
        ocean_depth_baseline_m=args.ocean_depth_baseline_m, rng=rng,
        alpha=args.entrainment_alpha, n_freq=args.stratification_n, basin=args.basin,
        export_profiles=args.export_plume_profiles,
    )

    json_path, csv_path = export_data(vents, run_dir, args.basename)

    plume_profiles_csv_path = None
    if plume_profiles:
        plume_profiles_csv_path = os.path.join(run_dir, f"{args.basename}_plume_profiles.csv")
        export_plume_profiles(plume_profiles, plume_profiles_csv_path)

    acoustic_mode = getattr(args, "acoustic_mode", "off")
    acoustic_result = None
    if acoustic_mode != "off":
        radius_um = getattr(args, "acoustic_particle_radius_um", None)
        acoustic_result = ac.acoustic_enrichment_field(
            vents, terrain, domain_size_m=args.domain_size_m, local_relief_m=args.local_relief_m,
            ocean_depth_baseline_m=args.ocean_depth_baseline_m, mode=acoustic_mode, rng=rng,
            particle_radius_m=(radius_um * 1e-6) if radius_um is not None else None,
            particle_density_kg_m3=getattr(args, "acoustic_particle_density", None),
            aggregate_radius_m=getattr(args, "aggregate_radius_m", None),
            aggregate_density_kg_m3=getattr(args, "aggregate_density_kg_m3", None),
            cross_vent_coherence=getattr(args, "acoustic_cross_vent_coherence", "incoherent"),
        )

    module_flags = ModuleFlags(
        dilution=not args.no_dilution,
        thermophoresis=not args.no_thermophoresis,
        mineral_adsorption=not args.no_mineral_adsorption,
        proton_gradient=not args.no_proton_gradient,
        acoustic_mode=acoustic_mode,
    )
    molecule_class = getattr(args, "molecule_class", DEFAULT_MOLECULE_CLASS)
    molecule_params = dict(MOLECULE_CLASSES[molecule_class])
    pore_aspect_ratio = getattr(args, "pore_aspect_ratio", None)
    if pore_aspect_ratio is not None:
        molecule_params["pore_aspect_ratio"] = pore_aspect_ratio
    molecule_label = MOLECULE_CLASS_LABELS[molecule_class]
    # versão em inglês só para texto DENTRO das figuras (títulos/eixos/
    # legendas) — o resto (metadata.json, GUI) continua em português via
    # molecule_label acima.
    molecule_label_en = MOLECULE_CLASS_LABELS_EN[molecule_class]
    acoustic_factors = acoustic_result["per_vent_factor"] if acoustic_result else None
    hotspots = compute_field_hotspots(vents, module_flags, params=molecule_params, molecule_class=molecule_class,
                                       acoustic_factors=acoustic_factors)
    hotspots_csv_path = os.path.join(run_dir, "prebiotic_hotspots.csv")
    if hotspots["records"]:
        with open(hotspots_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(hotspots["records"][0].keys()))
            writer.writeheader()
            writer.writerows(hotspots["records"])

    png_path = png_3d_path = png_true_path = png_hotspots_path = png_acoustic_path = None
    png_artistic_path = None
    png_module_paths = {key: None for key in MODULE_GRADIENT_LABELS}
    if make_images:
        png_path = os.path.join(run_dir, f"{args.basename}.png")
        plot_field(terrain, vents, png_path,
                   title=f"Seed={seed} | spreading_rate={args.spreading_rate} mm/yr | "
                         f"{len(vents)} vents in {args.n_clusters} clusters",
                   local_relief_m=args.local_relief_m, ocean_depth_baseline_m=args.ocean_depth_baseline_m)

        png_hotspots_path = os.path.join(run_dir, f"{args.basename}_hotspots.png")
        modules_on = ", ".join(k for k, v in hotspots["summary"]["modules_enabled"].items() if v) or "none"
        plot_hotspots(terrain, vents, hotspots["records"], png_hotspots_path,
                      title=f"Enrichment vs. control ({molecule_label_en}) | seed={seed} | modules: {modules_on}",
                      molecule_label=molecule_label_en,
                      local_relief_m=args.local_relief_m, ocean_depth_baseline_m=args.ocean_depth_baseline_m)

        if not args.no_3d:
            png_3d_path = os.path.join(run_dir, f"{args.basename}_3d.png")
            plot_field_3d(terrain, vents, png_3d_path,
                          title=f"3D hydrothermal vent field | seed={seed} | {len(vents)} vents",
                          local_relief_m=args.local_relief_m, z_exag=args.z_exag,
                          view=(args.view_elev, args.view_azim),
                          chimney_scale=args.chimney_scale, seed=seed)

        if args.true_scale:
            png_true_path = os.path.join(run_dir, f"{args.basename}_3d_truescale.png")
            plot_field_3d_true_scale(terrain, vents, png_true_path,
                                      title=f"Hydrothermal vent field | seed={seed} | {len(vents)} vents",
                                      domain_size_m=args.domain_size_m, local_relief_m=args.local_relief_m,
                                      seed=seed)

        if getattr(args, "artistic_render", False):
            # Import tardio: pyvista é uma dependência OPCIONAL, pesada,
            # só usada por esta visualização não-científica (ver
            # artistic_render.py) — não deve travar o resto do projeto
            # para quem não a usa.
            import artistic_render as ar
            # Uma única visualização artística (sem fauna — a opção
            # com/sem fauna existiu numa versão anterior, mas o usuário
            # relatou que as duas versões "não faziam a menor diferença"
            # visualmente; removida em vez de mantida como opção morta).
            png_artistic_path = os.path.join(run_dir, f"{args.basename}_artistic.png")
            ar.render_artistic_scene(terrain, vents, png_artistic_path,
                                      domain_size_m=args.domain_size_m,
                                      local_relief_m=args.local_relief_m, seed=seed)

        if acoustic_result is not None:
            png_acoustic_path = os.path.join(run_dir, f"{args.basename}_acoustic.png")
            plot_acoustic_field(terrain, vents, acoustic_result, args.domain_size_m, png_acoustic_path,
                                 title=f"Exploratory acoustic field ({acoustic_mode}) | seed={seed}",
                                 local_relief_m=args.local_relief_m, ocean_depth_baseline_m=args.ocean_depth_baseline_m)

        for module_key, module_label in MODULE_GRADIENT_LABELS.items():
            if not getattr(module_flags, module_key):
                continue
            path = os.path.join(run_dir, f"{args.basename}_module_{module_key}.png")
            plot_module_gradient_map(terrain, vents, hotspots["records"], f"factor_{module_key}",
                                      module_label, path,
                                      title=f"{molecule_label_en} | seed={seed} | module: {module_label}",
                                      local_relief_m=args.local_relief_m,
                                      ocean_depth_baseline_m=args.ocean_depth_baseline_m)
            png_module_paths[module_key] = path

    counts = {t: sum(1 for v in vents if v.vent_type == t) for t in VENT_TYPES}
    summary = {
        "run_dir": run_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "size": args.size,
        "roughness": args.roughness,
        "n_clusters": args.n_clusters,
        "spreading_rate": args.spreading_rate,
        "local_relief_m": args.local_relief_m,
        "ocean_depth_baseline_m": args.ocean_depth_baseline_m,
        "n_vents": len(vents),
        "vent_type_counts": counts,
        "json_path": json_path,
        "csv_path": csv_path,
        "plume_profiles_csv_path": plume_profiles_csv_path,
        "entrainment_alpha": args.entrainment_alpha,
        "stratification_n": args.stratification_n,
        "basin": args.basin,
        "png_2d_path": png_path,
        "png_3d_path": png_3d_path,
        "png_truescale_path": png_true_path,
        "png_artistic_path": png_artistic_path,
        "png_hotspots_path": png_hotspots_path,
        "png_acoustic_path": png_acoustic_path,
        "png_module_dilution_path": png_module_paths["dilution"],
        "png_module_thermophoresis_path": png_module_paths["thermophoresis"],
        "png_module_mineral_adsorption_path": png_module_paths["mineral_adsorption"],
        "png_module_proton_gradient_path": png_module_paths["proton_gradient"],
        "hotspots_csv_path": hotspots_csv_path,
        "molecule_class": molecule_class,
        "molecule_class_label": molecule_label,
        "pore_aspect_ratio": molecule_params["pore_aspect_ratio"],
        "prebiotic_modules": hotspots["summary"]["modules_enabled"],
        "prebiotic_summary": {k: v for k, v in hotspots["summary"].items() if k != "modules_enabled"},
        "acoustic_mode": acoustic_mode,
        "acoustic_diagnostics": acoustic_result["diagnostics"] if acoustic_result else None,
        "acoustic_top_peaks": acoustic_result["peaks"][:5] if acoustic_result else None,
        "sensitivity_sweep": getattr(args, "sensitivity_sweep", False),
        "experiment_mode": getattr(args, "experiment_mode", "exploratory"),
    }

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def append_runs_index(outputs_root: str, summaries: List[dict]):
    """Acrescenta uma linha por run a um índice cumulativo `runs_index.csv` na raiz de outputs."""
    index_path = os.path.join(outputs_root, "runs_index.csv")
    rows = [{
        "experiment_dir": os.path.basename(os.path.dirname(s["run_dir"])),
        "run_dir": os.path.basename(s["run_dir"]),
        "timestamp": s["timestamp"],
        "seed": s["seed"],
        "molecule_class": s["molecule_class"],
        "n_vents": s["n_vents"],
        "n_black_smoker": s["vent_type_counts"].get("black_smoker", 0),
        "n_white_smoker": s["vent_type_counts"].get("white_smoker", 0),
        "n_diffuse_flow": s["vent_type_counts"].get("diffuse_flow", 0),
        "n_clusters": s["n_clusters"],
        "spreading_rate": s["spreading_rate"],
        "entrainment_alpha": s["entrainment_alpha"],
        "stratification_n": s["stratification_n"],
        "basin": s["basin"],
        "dilution_on": s["prebiotic_modules"]["dilution"],
        "thermophoresis_on": s["prebiotic_modules"]["thermophoresis"],
        "mineral_adsorption_on": s["prebiotic_modules"]["mineral_adsorption"],
        "proton_gradient_on": s["prebiotic_modules"]["proton_gradient"],
        "acoustic_mode": s["prebiotic_modules"]["acoustic_mode"],
        "max_concentration_uM": s["prebiotic_summary"]["max_concentration_uM"],
        "mean_concentration_uM": s["prebiotic_summary"]["mean_concentration_uM"],
        "top_hotspot_vent_type": s["prebiotic_summary"]["top_hotspot_vent_type"],
        "top_hotspot_enrichment_vs_control": s["prebiotic_summary"]["top_hotspot_enrichment_vs_control"],
        "mean_enrichment_vs_control": s["prebiotic_summary"]["mean_enrichment_vs_control"],
        "n_vents_increased_vs_control": s["prebiotic_summary"]["n_vents_increased_vs_control"],
        "n_vents_decreased_vs_control": s["prebiotic_summary"]["n_vents_decreased_vs_control"],
    } for s in summaries]

    file_exists = os.path.exists(index_path)
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return index_path


def interactive_menu() -> tuple[int, bool, bool]:
    """Menu de terminal: single run ou ensemble, nº de runs (1-10000) e
    execução paralela (só perguntado para ensembles com mais de 1 run)."""
    print("=== Simulação de Campo de Fumarolas ===")
    print("[1] Simulação única (single run)")
    print("[2] Conjunto de simulações (ensemble)")

    while True:
        choice = input("Escolha uma opção [1/2]: ").strip()
        if choice in ("1", "2"):
            break
        print("Opção inválida. Digite 1 ou 2.")

    if choice == "1":
        return 1, True, False

    while True:
        raw = input("Quantas runs deseja executar? (1-10000): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 10000:
            n_runs = int(raw)
            break
        print("Valor inválido. Digite um número inteiro entre 1 e 10000.")

    make_images = False
    if n_runs > 1:
        resp = input("Gerar imagens (2D/3D) para cada run? Mais lento em ensembles grandes. (s/N): ").strip().lower()
        make_images = resp == "s"
    else:
        make_images = True

    parallel = False
    if n_runs > 1:
        n_workers = _default_parallel_workers()
        resp = input(f"Execução paralela ({n_workers} processos, mais rápida em ensembles grandes, "
                      f"log fora de ordem)? Padrão sequencial. (s/N): ").strip().lower()
        parallel = resp == "s"

    return n_runs, make_images, parallel


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None,
                         help="semente RNG base; se omitida, é sorteada aleatoriamente. Em ensembles, cada run "
                              "deriva sua própria seed a partir desta (via SeedSequence), permitindo reproduzir "
                              "o conjunto inteiro")
    parser.add_argument("--size", type=int, default=257, help="tamanho da grade, 2^n+1 (ex: 65,129,257,513)")
    parser.add_argument("--roughness", type=float, default=0.55, help="rugosidade do terreno fractal (0-1)")
    parser.add_argument("--n-clusters", type=int, default=6, help="nº de clusters de fumarolas ao longo do eixo")
    parser.add_argument("--vents-min", type=int, default=2, help="mín. de fumarolas por cluster")
    parser.add_argument("--vents-max", type=int, default=9, help="máx. de fumarolas por cluster")
    parser.add_argument("--spreading-rate", type=float, default=60.0,
                         help="taxa de espalhamento da dorsal em mm/ano (lenta <40, intermediária 40-90, rápida >90)")
    parser.add_argument("--ocean-depth-baseline-m", type=float, default=2500.0,
                         help="profundidade ambiente do oceano na crista da dorsal, em metros (tipicamente 2000-3000m); "
                              "somada ao relevo local para dar a profundidade absoluta de cada fumarola")
    parser.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR,
                         help="pasta raiz onde as subpastas de cada run são criadas (padrão: outputs/ dentro da pasta do projeto)")
    parser.add_argument("--basename", type=str, default="fumarola_field", help="nome base dos arquivos dentro de cada run")
    parser.add_argument("--no-3d", action="store_true", help="pula a renderização 3D (quando imagens estão ativas)")
    parser.add_argument("--z-exag", type=float, default=25.0, help="exagero vertical do relevo na cena 3D")
    parser.add_argument("--view-elev", type=float, default=55.0, help="elevação da câmera na cena 3D")
    parser.add_argument("--view-azim", type=float, default=-50.0, help="azimute da câmera na cena 3D")
    parser.add_argument("--chimney-scale", type=float, default=1.0,
                         help="multiplicador da altura visual das chaminés na cena 3D")
    parser.add_argument("--true-scale", action="store_true",
                         help="também renderiza uma cena 3D em escala vertical verdadeira (sem exagero, metros reais, 1:1:1)")
    parser.add_argument("--artistic-render", action="store_true",
                         help="renderiza uma visualização artística NÃO-científica (PyVista, materiais/"
                              "iluminação fotográficos calibrados contra uma foto real de referência) — "
                              "geometria vem dos mesmos dados procedurais, cores/câmera são escolha "
                              "estética; requer o pacote opcional pyvista (pip install pyvista)")
    parser.add_argument("--domain-size-m", type=float, default=1200.0,
                         help="extensão horizontal real da área simulada, em metros (cena de escala verdadeira)")
    parser.add_argument("--local-relief-m", type=float, default=150.0,
                         help="variação real de relevo da área pesquisada, em metros (tipicamente 50-300m); "
                              "usada de forma consistente na profundidade de cada fumarola e em ambas as cenas 3D")
    parser.add_argument("--runs", type=int, default=None,
                         help="nº de runs a executar sem passar pelo menu interativo (1-10000); "
                              "se omitido, um menu pergunta single/ensemble")
    parser.add_argument("--ensemble-images", action="store_true",
                         help="gera imagens para cada run de um ensemble não-interativo (--runs > 1); "
                              "por padrão ensembles só salvam dados (JSON/CSV/metadata), pois é bem mais rápido")
    parser.add_argument("--parallel", action="store_true",
                         help="distribui as runs de um ensemble (--runs > 1) entre múltiplos processos "
                              "(um por núcleo de CPU) em vez do laço sequencial padrão; mesmas seeds/"
                              "parâmetros por run, portanto reprodutibilidade idêntica, só a ordem de "
                              "conclusão no log/índice pode variar")
    parser.add_argument("--workers", type=int, default=None,
                         help=f"nº de processos usados com --parallel (padrão: núcleos lógicos - 1 = "
                              f"{_default_parallel_workers()} nesta máquina)")
    parser.add_argument("--no-dilution", action="store_true",
                         help="desliga o módulo de diluição/advecção da pluma (concentração assume fluido não-diluído)")
    parser.add_argument("--no-thermophoresis", action="store_true",
                         help="desliga o módulo de termoforese em poros minerais (efeito Soret)")
    parser.add_argument("--no-mineral-adsorption", action="store_true",
                         help="desliga o módulo de adsorção em superfícies minerais (mundo ferro-enxofre)")
    parser.add_argument("--no-proton-gradient", action="store_true",
                         help="desliga o módulo de gradiente de prótons em compartimentos alcalinos (Russell & Martin)")
    parser.add_argument("--acoustic-mode", type=str, default="off",
                         choices=["off", "streaming", "particle_trap", "both"],
                         help="modelo acústico exploratório de concentração prebiótica via campo sonoro dos "
                              "vents (hipótese original, sem validação experimental — ver acoustics.py e "
                              "docs/PHYSICS_MODEL.md); 'streaming'=advecção por streaming de contorno sobre "
                              "soluto dissolvido, 'particle_trap'=aprisionamento de partícula via potencial "
                              "de Gor'kov, 'both'=ambos compostos")
    parser.add_argument("--acoustic-particle-radius-um", type=float, default=None,
                         help="raio (μm) de UMA partícula customizada para --acoustic-mode particle_trap/both; "
                              "se omitido (padrão), usa a população citada de duas classes (colóide fino de "
                              "sulfeto + agregado de Fe-oxi-hidróxido de campo próximo, González-Santana et "
                              "al. 2020) em vez de um único tamanho — ver acoustics.py PARTICLE_CLASSES. Só "
                              "use este flag para EXPLORAR um tamanho hipotético específico.")
    parser.add_argument("--acoustic-particle-density", type=float, default=None,
                         help="densidade (kg/m3) da partícula customizada; ver --acoustic-particle-radius-um")
    parser.add_argument("--acoustic-cross-vent-coherence", type=str, default="incoherent",
                         choices=["incoherent", "coherent"],
                         help="como combinar o campo acústico ENTRE fumarolas diferentes (só afeta "
                              "--acoustic-mode != off): 'incoherent' (padrão, fisicamente motivado — "
                              "soma em potência, sem suposição de sincronismo de fase entre fumarolas "
                              "independentes) ou 'coherent' (teste de limite superior idealizado: todas "
                              "as fumarolas compartilham uma única frequência tonal sorteada e são "
                              "somadas em fase, produzindo franjas de interferência genuínas ENTRE "
                              "fumarolas — cenário de melhor caso, não uma previsão; usado para testar "
                              "se a suposição de incoerência afeta a conclusão do modelo, ver "
                              "docs/PHYSICS_MODEL.md)")
    parser.add_argument("--molecule-class", type=str, default=DEFAULT_MOLECULE_CLASS,
                         choices=list(MOLECULE_CLASSES.keys()),
                         help="classe de molécula prebiótica a modelar: " +
                              ", ".join(f"{k} ({v})" for k, v in MOLECULE_CLASS_LABELS.items()) +
                              f" (padrão: {DEFAULT_MOLECULE_CLASS})")
    parser.add_argument("--pore-aspect-ratio", type=float, default=None,
                         help="razão de aspecto (comprimento/largura) do poro para o módulo de termoforese "
                              "calibrado (só afeta a classe nucleotideos, que usa a fórmula acoplada à "
                              "convecção — ver prebiotic.py); se omitido, usa o padrão 10:1 (segmento único "
                              "mais conservador testado por Baaske et al. 2007, faixa testada 10-125:1)")
    parser.add_argument("--entrainment-alpha", type=float, default=pp.DEFAULT_ALPHA_ENTRAINMENT,
                         help="coeficiente de entranhamento alpha do modelo de pluma MTT (padrão: "
                              f"{pp.DEFAULT_ALPHA_ENTRAINMENT}, faixa medida em campo "
                              f"{pp.ALPHA_ENTRAINMENT_RANGE} — Rona et al. 2006; ver docs/PHYSICS_MODEL.md)")
    parser.add_argument("--stratification-n", type=float, default=pp.DEFAULT_N_BRUNT_VAISALA,
                         help="frequência de Brunt-Väisälä N (s^-1) da coluna d'água estratificada (padrão: "
                              f"{pp.DEFAULT_N_BRUNT_VAISALA}, único valor oceânico de dorsal verificado nas fontes "
                              "consultadas — Juan de Fuca, Lavelle 1997; não calibrado para MAR/EPR, ver docs/PHYSICS_MODEL.md)")
    parser.add_argument("--basin", type=str, default="atlantic", choices=list(rk.BASIN_PARAMS.keys()),
                         help="bacia oceânica de referência para a cinética de oxidação de Fe(II) (assimetria "
                              "Atlântico/Pacífico documentada por Field & Sherrell, 2000; padrão: atlantic)")
    parser.add_argument("--export-plume-profiles", action="store_true",
                         help="grava o perfil vertical completo da pluma (diluição, temperatura, concentração por "
                              "espécie) de cada vento em <basename>_plume_profiles.csv. Desligado por padrão: a "
                              "integração ODE por vento tem custo não desprezível em ensembles grandes")
    parser.add_argument("--sensitivity-sweep", action="store_true",
                         help="varre parâmetros com faixa de incerteza DOCUMENTADA (não escolhas ilustrativas sem "
                              f"faixa citável) via Hipercubo Latino CONJUNTO ao longo do ensemble: entrainment_alpha em "
                              f"{pp.ALPHA_ENTRAINMENT_RANGE} (Rona et al. 2006) e, se --acoustic-mode "
                              "for particle_trap/both, o raio/densidade do agregado acústico em 14-20 μm / "
                              "2400-3600 kg/m3 (González-Santana et al. 2020). Cada run usa um ponto amostrado "
                              "distinto (seed do campo E parâmetro variam juntos — não separa as duas fontes de "
                              "variância, ver --variance-decomposition para isso); --seed continua controlando "
                              "a reprodutibilidade completa da varredura.")
    parser.add_argument("--variance-decomposition", action="store_true",
                         help="modo alternativo a --runs/--sensitivity-sweep: desenho ANINHADO (N_outer pontos de "
                              "parâmetro via Hipercubo Latino conjunto x N_inner réplicas de campo por ponto, "
                              "parâmetro FIXO dentro de cada grupo) que separa formalmente quanto da variância "
                              "do resultado vem da aleatoriedade do campo de fumarolas (estocástica) vs. da "
                              "incerteza sobre entrainment_alpha/raio-densidade do agregado (paramétrica), via "
                              "ANOVA de um fator aleatório (Searle, Casella & McCulloch 1992) com IC 95% por "
                              "bootstrap aninhado — ver variance_decomposition.py e docs/PHYSICS_MODEL.md §7.8.2. "
                              "Total de runs = --outer-samples x --inner-replicates.")
    parser.add_argument("--outer-samples", type=int, default=20,
                         help="N_outer para --variance-decomposition: nº de pontos de parâmetro distintos "
                              "(resolução da componente PARAMÉTRICA; padrão 20)")
    parser.add_argument("--inner-replicates", type=int, default=10,
                         help="N_inner para --variance-decomposition: nº de seeds de campo por ponto de "
                              "parâmetro (resolução da componente ESTOCÁSTICA; padrão 10)")
    args = parser.parse_args()

    if args.variance_decomposition:
        if args.runs is not None or args.sensitivity_sweep:
            parser.error("--variance-decomposition é um modo alternativo a --runs/--sensitivity-sweep "
                          "(deriva seu próprio nº de runs de --outer-samples x --inner-replicates)")
        if args.outer_samples < 2 or args.inner_replicates < 2:
            parser.error("--outer-samples e --inner-replicates precisam ser >= 2 (ANOVA de um fator precisa "
                          "de pelo menos 2 grupos com pelo menos 2 réplicas cada)")
        run_nested_variance_experiment(args, args.outer_samples, args.inner_replicates,
                                        make_images=args.ensemble_images,
                                        parallel=args.parallel, n_workers=args.workers)
        return

    if args.runs is not None:
        if not (1 <= args.runs <= 10000):
            parser.error("--runs deve estar entre 1 e 10000")
        n_runs = args.runs
        make_images = True if n_runs == 1 else args.ensemble_images
        parallel = args.parallel
    else:
        n_runs, make_images, parallel = interactive_menu()

    run_experiment(args, n_runs, make_images, parallel=parallel, n_workers=args.workers)


def _derive_run_seeds_and_sweep(base_seed: int, n_runs: int, sensitivity_sweep: bool, acoustic_mode: str):
    """
    Deriva deterministicamente as seeds de cada run (e, se ativa, as amostras
    da varredura de sensibilidade) a partir de `base_seed` e `n_runs` — a
    MESMA sequência sempre, dado o mesmo `base_seed`/`n_runs`/`sensitivity_sweep`/
    `acoustic_mode`. Extraído de `run_experiment` para ser reutilizado por
    `resume_experiment` sem duplicar (e arriscar divergir) a lógica de
    derivação de seeds — é isso que torna a retomada de um experimento
    interrompido cientificamente idêntica a tê-lo rodado sem interrupção.
    """
    seed_sequence = np.random.SeedSequence(base_seed)
    # +1 filho reservado para a RNG da varredura de sensibilidade — deriva
    # da MESMA seed base, então a varredura inteira também é reproduzível
    # com --seed, sem consumir/perturbar as seeds dos campos de fumarola.
    all_children = seed_sequence.spawn(n_runs + 1)
    child_seeds = all_children[:n_runs]
    run_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in child_seeds]

    alpha_samples = agg_radius_samples = agg_density_samples = None
    if sensitivity_sweep:
        sweep_rng = np.random.default_rng(all_children[n_runs])
        bounds = [pp.ALPHA_ENTRAINMENT_RANGE]
        sweep_acoustic = acoustic_mode in ("particle_trap", "both")
        if sweep_acoustic:
            bounds += [(14e-6, 20e-6), (2400.0, 3600.0)]
        # Amostragem CONJUNTA (não mais 3 chamadas 1D sequenciais e
        # independentes) — ver `joint_latin_hypercube`. Cada dimensão
        # continua com sua própria faixa/margem exatamente como antes;
        # só o DESENHO (como as dimensões são combinadas por run) mudou.
        joint = joint_latin_hypercube(n_runs, bounds, sweep_rng)
        alpha_samples = joint[:, 0]
        if sweep_acoustic:
            agg_radius_samples = joint[:, 1]
            agg_density_samples = joint[:, 2]
    return run_seeds, alpha_samples, agg_radius_samples, agg_density_samples


def find_run_dirs(experiment_dir: str) -> List[str]:
    """Lista as subpastas run_* de um experimento, em ordem cronológica (o nome é timestamped)."""
    if not os.path.isdir(experiment_dir):
        return []
    names = sorted(d for d in os.listdir(experiment_dir)
                    if d.startswith("run_") and os.path.isdir(os.path.join(experiment_dir, d)))
    return [os.path.join(experiment_dir, d) for d in names]


def load_run_summary(run_dir: str) -> Optional[dict]:
    """Carrega o metadata.json de uma run, se existir e for JSON válido; None caso contrário
    (run incompleta — interrompida por crash antes de terminar — ou corrompida)."""
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_aborted_experiments(outputs_root: str) -> List[dict]:
    """
    Varre `outputs_root` por pastas "experimento_*" que têm
    `experiment_metadata.json` mas menos runs completas (com `metadata.json`
    válido) do que `n_runs` planejado — ou seja, foram interrompidas (crash,
    fechamento da GUI, etc.) antes de terminar. Experimentos de uma versão
    anterior do programa, que não salvava os `args` completos no metadata,
    são listados mas marcados como não retomáveis (`resumable=False`).
    """
    aborted = []
    if not os.path.isdir(outputs_root):
        return aborted
    for name in sorted(os.listdir(outputs_root)):
        exp_dir = os.path.join(outputs_root, name)
        meta_path = os.path.join(exp_dir, "experiment_metadata.json")
        if not (name.startswith("experimento_") and os.path.isdir(exp_dir) and os.path.exists(meta_path)):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        n_runs = meta.get("n_runs")
        if not n_runs:
            continue
        n_completed = sum(1 for rd in find_run_dirs(exp_dir) if load_run_summary(rd) is not None)
        if n_completed < n_runs:
            aborted.append({
                "experiment_dir": exp_dir,
                "n_completed": n_completed,
                "n_runs": n_runs,
                "timestamp": meta.get("timestamp"),
                "base_seed": meta.get("base_seed"),
                "resumable": "args" in meta,
            })
    return aborted


def _default_parallel_workers() -> int:
    """Nº de processos padrão para `--parallel`: núcleos lógicos - 1, deixando
    um núcleo livre para a UI/SO permanecerem responsivos durante um
    ensemble longo. Nunca menor que 1."""
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def _sweep_run_args(args: argparse.Namespace, i: int, sensitivity_sweep: bool,
                     alpha_samples, agg_radius_samples, agg_density_samples) -> argparse.Namespace:
    """Retorna os args efetivos da run `i` (1-based), aplicando os valores
    amostrados pelo sensitivity sweep (LHS) quando ativo — sem cópia se o
    sweep está desligado (mesmo objeto `args`, comportamento anterior).
    Compartilhada por `run_experiment`/`resume_experiment`/execução
    paralela para que as três rotas derivem os args de cada run de forma
    idêntica."""
    if not sensitivity_sweep:
        return args
    run_args = argparse.Namespace(**vars(args))
    run_args.entrainment_alpha = float(alpha_samples[i - 1])
    if agg_radius_samples is not None:
        run_args.aggregate_radius_m = float(agg_radius_samples[i - 1])
        run_args.aggregate_density_kg_m3 = float(agg_density_samples[i - 1])
    return run_args


def _log_run_result(label_index: int, n_total: int, run_dir: str, run_seed: int, summary: dict,
                     progress_cb=None) -> None:
    ps = summary["prebiotic_summary"]
    top_enrich = ps["top_hotspot_enrichment_vs_control"]
    top_enrich_txt = f"{top_enrich:.2f}x" if top_enrich is not None else "n/a"
    print(f"[{label_index}/{n_total}] {os.path.basename(run_dir)} | seed={run_seed} | "
          f"{summary['n_vents']} fumarolas | tipos={summary['vent_type_counts']} | "
          f"hotspot líder={top_enrich_txt} vs. controle ({ps['top_hotspot_vent_type']}) | "
          f"aumentaram={ps['n_vents_increased_vs_control']} diminuíram={ps['n_vents_decreased_vs_control']}")
    if progress_cb:
        progress_cb(label_index, n_total, summary)


def _execute_jobs_parallel(jobs: list, make_images: bool, progress_cb, n_total: int,
                            n_workers: Optional[int] = None) -> list:
    """Executa `jobs` (lista de `(label_index, run_dir, run_seed, run_args)`,
    já com as pastas de run CRIADAS previamente pelo chamador — criação de
    pasta usa checagem de colisão via `os.path.exists`+`os.makedirs`, não
    é atômica entre processos, então TEM que acontecer sequencialmente no
    processo principal antes do dispatch) num `ProcessPoolExecutor`, um
    processo por núcleo. Cada worker roda `execute_run` de forma
    independente (sem estado compartilhado — processos separados, não
    threads). Retorna as summaries na MESMA ORDEM de `jobs`, não na ordem
    de conclusão, para que o resultado agregado seja idêntico
    independentemente de paralelismo (as seeds já garantiam reprodutibilidade
    por índice; isto preserva também a ordem da lista retornada)."""
    results: list = [None] * len(jobs)
    workers = max(1, min(n_workers or _default_parallel_workers(), len(jobs)))
    print(f"Execução paralela ativa: {workers} processo(s) de trabalho "
          f"(de {os.cpu_count()} núcleos lógicos disponíveis).")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for idx, (label_index, run_dir, run_seed, run_args) in enumerate(jobs):
            fut = executor.submit(execute_run, run_args, run_seed, run_dir, make_images)
            future_map[fut] = (idx, label_index, run_dir, run_seed)
        for fut in concurrent.futures.as_completed(future_map):
            idx, label_index, run_dir, run_seed = future_map[fut]
            summary = fut.result()
            results[idx] = summary
            _log_run_result(label_index, n_total, run_dir, run_seed, summary, progress_cb)
    return results


def resume_experiment(experiment_dir: str, progress_cb=None,
                       parallel: bool = False, n_workers: Optional[int] = None) -> dict:
    """
    Retoma um experimento interrompido a partir do ponto em que parou.
    Reconstrói os `args` originais (salvos em `experiment_metadata.json`) e
    recomputa as seeds/amostras da varredura de sensibilidade EXATAMENTE
    como na execução original (mesma derivação a partir de base_seed+n_runs
    — ver `_derive_run_seeds_and_sweep`), então executa só as runs que
    faltam. Runs já completas (com `metadata.json` válido) não são
    refeitas nem duplicadas no índice cumulativo.
    """
    meta_path = os.path.join(experiment_dir, "experiment_metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if "args" not in meta:
        raise ValueError(
            "Este experimento foi criado por uma versão anterior do programa, que não salvava "
            "os parâmetros completos (\"args\") em experiment_metadata.json — não é possível "
            "retomá-lo com segurança (os parâmetros de terreno/física não são recuperáveis)."
        )

    args_dict = dict(meta["args"])
    base_seed = meta["base_seed"]
    args_dict["seed"] = base_seed  # garante reprodutibilidade exata, não sorteia uma nova
    args = argparse.Namespace(**args_dict)

    n_runs = meta["n_runs"]
    make_images = meta["make_images"]
    sensitivity_sweep = meta.get("sensitivity_sweep", False)
    acoustic_mode = getattr(args, "acoustic_mode", "off")

    run_seeds, alpha_samples, agg_radius_samples, agg_density_samples = _derive_run_seeds_and_sweep(
        base_seed, n_runs, sensitivity_sweep, acoustic_mode)

    completed_summaries = [
        summary for rd in find_run_dirs(experiment_dir)
        if (summary := load_run_summary(rd)) is not None
    ]
    n_completed = len(completed_summaries)

    if n_completed >= n_runs:
        print(f"Experimento em {experiment_dir} já está completo ({n_completed}/{n_runs} runs) — nada a retomar.")
        return {"experiment_dir": experiment_dir, "summaries": completed_summaries,
                "index_path": os.path.join(args.outputs_dir, "runs_index.csv"), "base_seed": base_seed}

    print(f"\nRetomando experimento em {os.path.abspath(experiment_dir)}")
    print(f"Runs já completas: {n_completed}/{n_runs} — continuando a partir da run {n_completed + 1}\n")

    if parallel and (n_runs - n_completed) > 1:
        jobs = []
        for i in range(n_completed, n_runs):
            run_dir = make_run_dir(experiment_dir)
            run_args = _sweep_run_args(args, i + 1, sensitivity_sweep, alpha_samples,
                                        agg_radius_samples, agg_density_samples)
            jobs.append((i + 1, run_dir, run_seeds[i], run_args))
        new_summaries = _execute_jobs_parallel(jobs, make_images, progress_cb, n_runs, n_workers)
    else:
        new_summaries = []
        for i in range(n_completed, n_runs):
            run_dir = make_run_dir(experiment_dir)
            run_args = _sweep_run_args(args, i + 1, sensitivity_sweep, alpha_samples,
                                        agg_radius_samples, agg_density_samples)
            summary = execute_run(run_args, run_seeds[i], run_dir, make_images)
            new_summaries.append(summary)
            _log_run_result(i + 1, n_runs, run_dir, run_seeds[i], summary, progress_cb)

    index_path = append_runs_index(args.outputs_dir, new_summaries)
    print(f"\nRetomada concluída: {len(new_summaries)} run(s) adicional(is) gravada(s) em {os.path.abspath(experiment_dir)}")
    print(f"Índice cumulativo: {index_path}")

    return {"experiment_dir": experiment_dir, "summaries": completed_summaries + new_summaries,
            "index_path": index_path, "base_seed": base_seed}


def run_experiment(args: argparse.Namespace, n_runs: int, make_images: bool, progress_cb=None,
                    parallel: bool = False, n_workers: Optional[int] = None) -> dict:
    """
    Orquestra um experimento completo (1-10000 runs): deriva as seeds a partir
    de uma seed base, cria a pasta do experimento, executa cada run e grava
    o índice cumulativo. Reaproveitada pelo CLI (`main`) e pela GUI.

    `progress_cb(i, n_runs, summary)`, se fornecido, é chamado após cada run
    (além do `print` normal), para que uma interface possa acompanhar o progresso.

    `parallel=True` distribui as runs entre múltiplos processos (um por
    núcleo de CPU, ver `_default_parallel_workers`/`n_workers`) em vez do
    laço sequencial padrão — mesma seed/parâmetros por run índice-a-índice,
    portanto reprodutibilidade idêntica, só a ordem de conclusão no log
    pode variar. Runs curtas ou N pequeno não valem o overhead de criar
    processos; o padrão (`parallel=False`) permanece o laço sequencial
    original, inalterado.
    """
    base_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32 - 1))
    sensitivity_sweep = getattr(args, "sensitivity_sweep", False)
    run_seeds, alpha_samples, agg_radius_samples, agg_density_samples = _derive_run_seeds_and_sweep(
        base_seed, n_runs, sensitivity_sweep, getattr(args, "acoustic_mode", "off"))

    experiment_dir = make_experiment_dir(args.outputs_dir)

    print(f"\nSeed base: {base_seed} (repita com --seed {base_seed} --runs {n_runs} para reproduzir este conjunto)")
    print(f"Executando {n_runs} run(s) | imagens: {'sim' if make_images else 'não (apenas dados)'}"
          + (" | varredura de sensibilidade: ATIVA (alpha" +
             (" + agregado acústico" if agg_radius_samples is not None else "") + ")"
             if sensitivity_sweep else ""))
    print(f"Pasta do experimento: {os.path.abspath(experiment_dir)}\n")

    experiment_metadata = {
        "experiment_dir": experiment_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_seed": base_seed,
        "n_runs": n_runs,
        "make_images": make_images,
        "n_clusters": args.n_clusters,
        "spreading_rate": args.spreading_rate,
        "local_relief_m": args.local_relief_m,
        "ocean_depth_baseline_m": args.ocean_depth_baseline_m,
        "molecule_class": getattr(args, "molecule_class", DEFAULT_MOLECULE_CLASS),
        "prebiotic_modules": ModuleFlags(
            dilution=not args.no_dilution,
            thermophoresis=not args.no_thermophoresis,
            mineral_adsorption=not args.no_mineral_adsorption,
            proton_gradient=not args.no_proton_gradient,
            acoustic_mode=getattr(args, "acoustic_mode", "off"),
        ).as_dict(),
        "sensitivity_sweep": sensitivity_sweep,
        "parallel": parallel,
        # Args completos (com a seed já resolvida), para permitir retomar o
        # experimento exatamente como configurado caso seja interrompido
        # (ver `resume_experiment`). Experimentos gerados antes desta versão
        # não têm esta chave e por isso não são retomáveis.
        "args": {**vars(args), "seed": base_seed},
    }
    with open(os.path.join(experiment_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)

    if parallel and n_runs > 1:
        jobs = []
        for i, run_seed in enumerate(run_seeds, start=1):
            run_dir = make_run_dir(experiment_dir)
            run_args = _sweep_run_args(args, i, sensitivity_sweep, alpha_samples,
                                        agg_radius_samples, agg_density_samples)
            jobs.append((i, run_dir, run_seed, run_args))
        summaries = _execute_jobs_parallel(jobs, make_images, progress_cb, n_runs, n_workers)
    else:
        summaries = []
        for i, run_seed in enumerate(run_seeds, start=1):
            run_dir = make_run_dir(experiment_dir)
            run_args = _sweep_run_args(args, i, sensitivity_sweep, alpha_samples,
                                        agg_radius_samples, agg_density_samples)
            summary = execute_run(run_args, run_seed, run_dir, make_images)
            summaries.append(summary)
            _log_run_result(i, n_runs, run_dir, run_seed, summary, progress_cb)

    index_path = append_runs_index(args.outputs_dir, summaries)

    print(f"\nConcluído: {n_runs} run(s) gravada(s) em {os.path.abspath(experiment_dir)}")
    print(f"Índice cumulativo: {index_path}")

    return {"experiment_dir": experiment_dir, "summaries": summaries, "index_path": index_path,
            "base_seed": base_seed}


def run_nested_variance_experiment(args: argparse.Namespace, outer_n: int, inner_n: int,
                                    make_images: bool = False, parallel: bool = False,
                                    n_workers: Optional[int] = None, response_extractor=None,
                                    gsa_n_mc: int = 4096, gsa_n_bootstrap: int = 200) -> dict:
    """
    Desenho ANINHADO para separar variância estocástica (aleatoriedade do
    campo de fumarolas, seed) de variância paramétrica (incerteza sobre
    entrainment_alpha e, se acústica particle_trap/both estiver ativa,
    raio/densidade do agregado near-field): `outer_n` pontos de parâmetro
    amostrados por `joint_latin_hypercube`, cada um executado com `inner_n`
    seeds de campo distintas e o parâmetro FIXO dentro do grupo. Ver
    `variance_decomposition.py` para a estatística (ANOVA de um fator
    aleatório) e docs/PHYSICS_MODEL.md §7.8.2 para a motivação completa —
    o `--sensitivity-sweep` original varia seed e parâmetro JUNTOS run a
    run, então não consegue separar as duas fontes.

    Deriva seeds deterministicamente a partir de `args.seed`: um filho de
    `SeedSequence` por ponto externo (+1 reservado para a RNG do LHS
    conjunto externo) e `inner_n` netos por ponto externo para as seeds de
    campo daquele grupo — reprodutível por completo com o mesmo `--seed`.

    Usa uma pasta própria `vardecomp_*` (não `experimento_*`) porque o
    esquema de metadados é diferente do ensemble plano — deliberadamente
    fora do alcance do menu "abrir/retomar experimento" da GUI, que
    assume a estrutura de `run_experiment`.
    """
    base_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32 - 1))
    acoustic_mode = getattr(args, "acoustic_mode", "off")
    sweep_acoustic = acoustic_mode in ("particle_trap", "both")

    seed_sequence = np.random.SeedSequence(base_seed)
    outer_children = seed_sequence.spawn(outer_n + 1)
    outer_param_rng = np.random.default_rng(outer_children[outer_n])

    bounds = [pp.ALPHA_ENTRAINMENT_RANGE]
    if sweep_acoustic:
        bounds += [(14e-6, 20e-6), (2400.0, 3600.0)]
    outer_params = joint_latin_hypercube(outer_n, bounds, outer_param_rng)

    experiment_dir = make_timestamped_dir(args.outputs_dir, "vardecomp")
    n_total = outer_n * inner_n
    print(f"\nSeed base: {base_seed} (repita com --seed {base_seed} --variance-decomposition "
          f"--outer-samples {outer_n} --inner-replicates {inner_n} para reproduzir este desenho)")
    print(f"Desenho aninhado: {outer_n} pontos de parâmetro x {inner_n} réplicas de campo = {n_total} runs totais.")

    jobs = []  # (outer_idx, inner_idx, run_dir, seed, run_args)
    for oi in range(outer_n):
        inner_children = outer_children[oi].spawn(inner_n)
        inner_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in inner_children]
        run_args = argparse.Namespace(**vars(args))
        run_args.entrainment_alpha = float(outer_params[oi, 0])
        if sweep_acoustic:
            run_args.aggregate_radius_m = float(outer_params[oi, 1])
            run_args.aggregate_density_kg_m3 = float(outer_params[oi, 2])
        for ii in range(inner_n):
            run_dir = make_run_dir(experiment_dir)
            jobs.append((oi, ii, run_dir, inner_seeds[ii], run_args))

    if parallel and n_total > 1:
        flat_jobs = [(idx + 1, run_dir, seed, run_args)
                     for idx, (oi, ii, run_dir, seed, run_args) in enumerate(jobs)]
        flat_summaries = _execute_jobs_parallel(flat_jobs, make_images, None, n_total, n_workers)
    else:
        flat_summaries = []
        for idx, (oi, ii, run_dir, seed, run_args) in enumerate(jobs, start=1):
            summary = execute_run(run_args, seed, run_dir, make_images)
            flat_summaries.append(summary)
            _log_run_result(idx, n_total, run_dir, seed, summary, None)

    index_path = append_runs_index(args.outputs_dir, flat_summaries)

    extractor = response_extractor or vd.default_response_value
    outer_groups = []
    flat_rows = []
    for oi in range(outer_n):
        group_vals = []
        for (job_oi, ii, run_dir, seed, run_args), summary in zip(jobs, flat_summaries):
            if job_oi != oi:
                continue
            val = extractor(summary)
            if val is None:
                raise ValueError(
                    f"response_extractor não encontrou valor de resposta para outer={oi} inner={ii} "
                    f"({run_dir}) — verifique se os módulos necessários (acústico particle_trap/both, ou "
                    "algum módulo prebiótico clássico) estão ativos em `args`.")
            group_vals.append(val)
            flat_rows.append({
                "outer_idx": oi, "inner_idx": ii, "seed": seed,
                "entrainment_alpha": run_args.entrainment_alpha,
                "aggregate_radius_m": getattr(run_args, "aggregate_radius_m", None),
                "aggregate_density_kg_m3": getattr(run_args, "aggregate_density_kg_m3", None),
                "response_value": val,
                "run_dir": run_dir,
            })
        outer_groups.append(np.array(group_vals, dtype=float))

    decomp_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    decomposition = vd.nested_variance_decomposition(outer_groups, rng=decomp_rng)

    param_names = ["entrainment_alpha"] + (["aggregate_radius_m", "aggregate_density_kg_m3"]
                                            if sweep_acoustic else [])
    # Reaproveita EXATAMENTE os dados já coletados para o desenho aninhado
    # (outer_params/outer_groups) e a componente estocástica já estimada
    # (decomposition["within_group_variance"], usada como ruído de medição
    # conhecido da média de cada grupo) — nenhuma simulação física a mais
    # é rodada para os índices de Sobol'. Ver global_sensitivity.py.
    gsa_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    global_sensitivity = gs.fit_surrogate_and_compute_sobol(
        outer_params, outer_groups, decomposition["within_group_variance"], bounds, param_names,
        n_mc=gsa_n_mc, n_bootstrap=gsa_n_bootstrap, rng=gsa_rng)

    csv_path = os.path.join(experiment_dir, "vardecomp_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    result = {
        "experiment_dir": experiment_dir, "base_seed": base_seed,
        "outer_n": outer_n, "inner_n": inner_n, "acoustic_mode": acoustic_mode,
        "swept_parameters": param_names,
        "decomposition": decomposition, "global_sensitivity": global_sensitivity,
        "csv_path": csv_path, "index_path": index_path,
    }
    summary_path = os.path.join(experiment_dir, "vardecomp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nDecomposição concluída ({outer_n} x {inner_n} = {n_total} runs) em {os.path.abspath(experiment_dir)}")
    print(f"Fração estocástica (campo de fumarolas): {decomposition['stochastic_fraction']:.3f} "
          f"(IC95% [{decomposition['stochastic_fraction_ci95'][0]:.3f}, "
          f"{decomposition['stochastic_fraction_ci95'][1]:.3f}])")
    print(f"Fração paramétrica ({'+'.join(param_names)}): "
          f"{decomposition['parametric_fraction']:.3f} "
          f"(IC95% [{decomposition['parametric_fraction_ci95'][0]:.3f}, "
          f"{decomposition['parametric_fraction_ci95'][1]:.3f}])")
    print(f"\nSensibilidade global (índices de Sobol' sobre surrogate GP, "
          f"LOO CV R²={global_sensitivity['loo_cv_r2']:.3f}"
          f"{' — AVISO: ajuste fraco, índices pouco confiáveis' if global_sensitivity['loo_cv_r2_warning'] else ''}):")
    for name in param_names:
        s1 = global_sensitivity["first_order"][name]
        st = global_sensitivity["total_order"][name]
        print(f"  {name}: S1={s1:.3f}, ST={st:.3f}")
    print(f"Resumo: {summary_path}")

    return result


if __name__ == "__main__":
    main()

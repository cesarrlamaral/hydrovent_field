"""
Verificação de solução numérica dos dois solvers do projeto — nunca
checado antes: os resultados físicos mudariam se a tolerância do
integrador de EDO da pluma (`plume_physics.integrate_plume`, RK45
adaptativo) fosse mais apertada, ou se a malha do solver de PDE acústico
(`acoustics.solve_steady_advection_diffusion`, diferenças finitas
upwind de 1ª ordem) fosse mais fina? Isso é robustez NUMÉRICA pura —
"o método resolve corretamente a equação que eu escrevi", ortogonal a
"a equação em si está calibrada contra dado real" (validação física, já
feita em outras seções do projeto).

Método padrão de verificação de solução (não específico deste projeto):
refinar o parâmetro numérico (tolerância ou malha) em 2-3 níveis
sucessivos com razão de refinamento constante, e checar que a saída
converge — idealmente estimando a ORDEM de convergência observada via
extrapolação de Richardson e comparando com a ordem teórica do método
(RK45 é de ordem alta/adaptativo; o upwind de 1ª ordem tem ordem
teórica exatamente 1, uma previsão testável).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def richardson_observed_order(f_coarse: float, f_medium: float, f_fine: float,
                               refinement_ratio: float = 2.0) -> float:
    """Ordem de convergência OBSERVADA a partir de três soluções sucessivas
    (malha/tolerância grossa, média, fina) com razão de refinamento
    CONSTANTE `refinement_ratio` entre os passos grosso->médio e
    médio->fino (extrapolação de Richardson clássica — ver ex. LeVeque,
    R.J., 2007, "Finite Difference Methods for Ordinary and Partial
    Differential Equations," SIAM, cap. 1; fórmula padrão de verificação
    de solução, não específica de nenhum solver deste projeto):

        p = log(|f_coarse - f_medium| / |f_medium - f_fine|) / log(r)

    `p` próximo da ordem teórica do método confirma que o refinamento
    está no regime assintótico esperado (não é ruído numérico nem um
    problema mal-condicionado). Retorna `inf` quando `f_medium==f_fine`
    dentro da precisão de máquina (já convergido ao ponto de não haver
    mais diferença mensurável — o caso ideal, não um erro)."""
    num = f_coarse - f_medium
    den = f_medium - f_fine
    if den == 0:
        return float("inf") if num != 0 else float("nan")
    ratio = abs(num / den)
    if ratio <= 0:
        return float("nan")
    return math.log(ratio) / math.log(refinement_ratio)


# --------------------------------------------------------------------------
# 1. Convergência de tolerância do integrador de EDO (plume_physics.py)
# --------------------------------------------------------------------------

def ode_tolerance_convergence_study(source, alpha: float, n_freq: float, ambient_temp_c: float,
                                     tolerance_levels: Sequence[Tuple[float, float]],
                                     probe_z_m: float = 1.0) -> dict:
    """Roda `plume_physics.integrate_plume` em cada `(rtol, atol)` de
    `tolerance_levels` (da mais FROUXA pra mais APERTADA), extrai dois
    diagnósticos físicos escalares — `rise_height_m` (altura de
    flutuabilidade neutra) e a diluição interpolada em `probe_z_m`
    (mesma grandeza `dilution_near_field_1m` usada pelo resto do
    projeto) — e reporta a mudança relativa entre níveis sucessivos.
    Mudança relativa desprezível entre o nível atual (default do
    projeto) e um mais apertado confirma que o default já está no
    regime convergido."""
    import plume_physics as pp

    rise_heights, dilutions_at_probe = [], []
    for rtol, atol in tolerance_levels:
        profile = pp.integrate_plume(source, alpha=alpha, n_freq=n_freq,
                                      ambient_temp_c=ambient_temp_c, rtol=rtol, atol=atol)
        rise_heights.append(profile.rise_height_m)
        dilutions_at_probe.append(float(np.interp(probe_z_m, profile.z, profile.dilution)))

    rise_heights = np.array(rise_heights)
    dilutions_at_probe = np.array(dilutions_at_probe)

    def _rel_changes(arr):
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = np.abs(np.diff(arr)) / np.where(np.abs(arr[:-1]) > 0, np.abs(arr[:-1]), 1.0)
        return rc

    return {
        "tolerance_levels": list(tolerance_levels),
        "rise_height_m": rise_heights,
        "dilution_at_probe": dilutions_at_probe,
        "rise_height_rel_change": _rel_changes(rise_heights),
        "dilution_rel_change": _rel_changes(dilutions_at_probe),
    }


# --------------------------------------------------------------------------
# 2. Convergência de malha do solver de PDE (acoustics.py)
# --------------------------------------------------------------------------

def pde_grid_convergence_study(domain_size_m: float, source_sigma_m: float,
                                diffusivity_m2_s: float, loss_rate_per_s: float,
                                velocity_m_s: Tuple[float, float],
                                grid_sizes: Sequence[int],
                                probe_frac: Tuple[float, float] = (0.65, 0.5)) -> dict:
    """Roda `acoustics.solve_steady_advection_diffusion` num problema
    SINTÉTICO controlado — domínio físico fixo (`domain_size_m`), fonte
    gaussiana de largura física fixa (`source_sigma_m`, escalada pra
    `sigma_cells` a cada resolução — sem isso a fonte mudaria de
    largura FÍSICA a cada refinamento, o que testaria um problema
    diferente a cada malha, não uma convergência de verdade), campo de
    velocidade UNIFORME fixo (caso mais simples com comportamento
    qualitativo conhecido — advecção domina a favor do vento, decai
    contra) — em `grid_sizes` crescentes (mesmo domínio físico, mais
    células = malha mais fina), reportando a concentração interpolada
    num ponto de sonda físico FIXO (`probe_frac` do domínio,
    downstream do centro por padrão) a cada resolução.

    Como o esquema é upwind de 1ª ordem (documentado em
    `acoustics.solve_steady_advection_diffusion` — necessário para
    estabilidade em Peclet de malha alto), a ordem de convergência
    OBSERVADA (`richardson_observed_order`) deveria ficar perto de 1,
    não de uma ordem mais alta — uma previsão testável sobre o próprio
    método numérico escolhido, não um número arbitrário."""
    import acoustics as ac
    from scipy.ndimage import map_coordinates

    # Coordenadas físicas (metros), não índices de célula, em TODO lugar —
    # usar índice de célula pra posicionar fonte/sonda faria a fonte e o
    # ponto de amostragem mudarem de posição FÍSICA a cada resolução
    # (jitter de até h/2), contaminando a comparação de convergência com
    # ruído de posicionamento em vez de medir só o erro de discretização.
    # Achado real ao construir este estudo: a primeira versão (índice de
    # célula) dava ordem observada oscilando entre -1,75 e +2,56 — sem
    # sentido físico nenhum pro upwind de 1ª ordem — resolvido trocando
    # posicionamento de fonte E amostragem da sonda para coordenada física
    # exata (bilinear via `map_coordinates`).
    probe_x_m = probe_frac[0] * domain_size_m
    probe_y_m = probe_frac[1] * domain_size_m
    center_m = domain_size_m / 2.0

    concentrations = []
    for n in grid_sizes:
        h = domain_size_m / n
        # centro de cada célula i está em (i+0.5)*h metros (convenção de
        # grade centrada em célula), consistente em qualquer resolução.
        coords_m = (np.arange(n) + 0.5) * h
        yy_m, xx_m = np.meshgrid(coords_m, coords_m, indexing="ij")
        source_mask = np.exp(-((xx_m - center_m) ** 2 + (yy_m - center_m) ** 2)
                              / (2 * source_sigma_m ** 2))
        u_x = np.full((n, n), velocity_m_s[0])
        u_y = np.full((n, n), velocity_m_s[1])
        c_field = ac.solve_steady_advection_diffusion(u_x, u_y, source_mask, h,
                                                        diffusivity_m2_s, loss_rate_per_s)
        probe_i = probe_y_m / h - 0.5
        probe_j = probe_x_m / h - 0.5
        value = map_coordinates(c_field, [[probe_i], [probe_j]], order=1, mode="nearest")[0]
        concentrations.append(float(value))

    concentrations = np.array(concentrations)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_change = np.abs(np.diff(concentrations)) / np.where(
            np.abs(concentrations[:-1]) > 0, np.abs(concentrations[:-1]), 1.0)
    return {
        "grid_sizes": list(grid_sizes),
        "cell_size_m": [domain_size_m / n for n in grid_sizes],
        "concentration_at_probe": concentrations,
        "rel_change": rel_change,
    }

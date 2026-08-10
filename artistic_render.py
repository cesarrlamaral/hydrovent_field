"""
Renderização artística (não-científica) de um campo de fumarolas gerado
proceduralmente, via PyVista/VTK — texturas fotográficas reais (CC0, Poly
Haven) para rocha/crosta mineral, luz posicional com sombras reais, plumas
como nuvens de pontos.

IMPORTANTE — isto NÃO é uma figura científica: ao contrário de
`plot_field_3d`/`plot_field_3d_true_scale` (fumarola_field.py), que usam
cores/proporções com significado científico direto (legenda por tipo de
fumarola, escala verdadeira opcional), esta função existe só para gerar
uma imagem de apresentação/divulgação com aparência fotográfica — cores
de rocha/mineral e câmera/luz são escolhas estéticas calibradas contra
uma foto real de referência (assets/splash_vent_field.png), não valores
medidos. A GEOMETRIA (posição/altura de cada chaminé, altura de cada
pluma) continua vindo inteiramente dos mesmos dados procedurais/físicos
do resto do projeto — nada aqui é gerado por IA generativa nem inventa
dados além do que a simulação já calculou (ver conversa sobre a escolha
de abordagem determinística vs. estilização por IA).

Módulo deliberadamente independente de fumarola_field.py (não importa
nada de lá) para evitar import circular, já que fumarola_field.py chama
este módulo (não o contrário) — mesmo padrão de acoustics.py/
plume_physics.py, módulos "folha" importados por fumarola_field.py.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage

try:
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkObject as _vtkObject
except ImportError as exc:  # pragma: no cover - mensagem amigável se a dependência opcional faltar
    raise ImportError(
        "A visualização artística requer o pacote opcional 'pyvista' "
        "(pip install pyvista). Não é uma dependência obrigatória do "
        "restante do projeto."
    ) from exc

# Nesta GPU/driver (confirmado: AMD Radeon RX 7600), atores no estilo
# "points" (usados pelas plumas/neve marinha) disparam uma falha de
# compilação de UM shader específico do VTK — inofensiva (o fallback
# renderiza corretamente; verificado visualmente à exaustão nesta sessão),
# mas MUITO verborrágica: cada disparo despeja o código-fonte inteiro do
# shader no stderr, então um ensemble de poucas dezenas de runs já inunda
# o console com dezenas de milhares de linhas — alarmante de ver mesmo
# sem indicar problema real. `GlobalWarningDisplayOff` desliga só a JANELA
# de log de erro/aviso do VTK (não afeta o pipeline de renderização nem
# suprime exceções Python de verdade, que não passam por aqui).
_vtkObject.GlobalWarningDisplayOff()


# Paleta calibrada visualmente contra assets/splash_vent_field.png (fundo
# oceânico quase negro/azul-petróleo, chaminés com crosta mineral
# ferruginosa na base clareando para tons pálidos no topo, fumaça escura
# em black smokers / clara e sutil em white smoker e diffuse flow).
_BG_BOTTOM = "#01050a"
_BG_TOP = "#0a2530"
_HAZE_RGB = np.array([0.012, 0.045, 0.065])

# Compositing de fumaça (ver bloco no fim de `render_artistic_scene`):
# distância de cor (0-255) até a cor-chave magenta pura acima da qual um
# pixel é considerado 100% fumaça — só precisa ser bem menor que a
# distância de QUALQUER cor real da paleta de fumaça até o magenta (a
# mais próxima, fumaça quase-preta de black_smoker, já fica a ~320 dessa
# métrica), então o valor só importa mesmo pra suavizar pixels de borda
# anti-serrilhados. `_SMOKE_BLUR_SIGMA_FRAC` é o desvio-padrão do
# desfoque gaussiano como fração da largura da imagem — é isso que dá a
# impressão de neblina/difusão real em vez de pontos nítidos.
_SMOKE_KEY_DIST_SCALE = 50.0
_SMOKE_BLUR_SIGMA_FRAC = 0.0045

# Texturas PBR reais (CC0/domínio público, Poly Haven — polyhaven.com,
# nenhuma atribuição exigida) aplicadas como detalhe fotográfico de
# superfície (rachaduras, veios, manchas) sobre a malha — ver
# `_load_texture`/uso em `render_artistic_scene`. As cores abaixo NÃO são
# mais o albedo final (como na primeira versão deste módulo): são um
# TINGIMENTO multiplicado sobre a textura (por isso claras, perto de
# branco — um tingimento escuro sobre uma textura já escura ficaria preto).
# A textura em si é que carrega o detalhe/"rugosidade" real.
_TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "textures")
_ROCK_TEXTURE_PATH = os.path.join(_TEXTURES_DIR, "dark_rock_diff_1k.jpg")
_RUST_TEXTURE_PATH = os.path.join(_TEXTURES_DIR, "rust_coarse_01_diff_1k.jpg")
_ROCK_TILE_SIZE_M = 10.0
_CHIMNEY_TILE_SIZE_M = 1.4

_ROCK_TINT = np.array([0.55, 0.62, 0.68])
_RUST_TINT = np.array([0.95, 0.62, 0.30])

# Tingimento base→topo por TIPO REAL de fumarola (v.vent_type — já usado
# pra cor da fumaça, mas até agora ignorado pra cor da própria chaminé,
# fazendo toda chaminé sair com a mesma cor de ferrugem independente do
# tipo). Baseado no mineral dominante de cada tipo: black smoker precipita
# sulfeto quase preto no orifício ativo (topo) sobre uma base mais velha/
# oxidada (ferrugem); white smoker precipita anidrita/sílica pálida;
# diffuse flow tem pouca ou nenhuma estrutura mineral, rocha desgastada.
_CHIMNEY_TINTS_BY_TYPE = {
    "black_smoker": {
        "base": np.array([0.88, 0.55, 0.26]),
        "top": np.array([0.30, 0.28, 0.30]),
        "texture_gamma": 1.1,
        "texture_path": None,  # None = usa _RUST_TEXTURE_PATH (padrão)
    },
    "white_smoker": {
        "base": np.array([0.82, 0.66, 0.48]),
        "top": np.array([0.93, 0.91, 0.86]),
        "texture_gamma": 4.5,
        "texture_path": None,
    },
    "diffuse_flow": {
        "base": np.array([0.60, 0.62, 0.60]),
        "top": np.array([0.68, 0.70, 0.68]),
        "texture_gamma": 2.6,
        # Textura de ROCHA (a mesma do terreno), não a de ferrugem: tingir
        # sozinho não neutraliza o tom quente forte da foto de ferrugem
        # (multiplicar não consegue "esfriar" uma textura, só escurecer na
        # direção da cor do tingimento) — confirmado visualmente com um
        # tingimento cinza que ainda saía castanho. Diffuse flow tem pouca
        # ou nenhuma precipitação mineral mesmo, então faz sentido usar a
        # rocha "nua" em vez de uma crosta.
        "texture_path": "rock",
    },
}
# Tingimento (multiplicação de cor) sozinho só consegue ESCURECER uma
# textura na direção da cor do tingimento — nunca clarear nem mudar o tom
# fundamentalmente (multiplicar um pixel laranja por um tingimento quase-
# branco continua laranja, só um pouco mais escuro). Como as três chaminés
# usam a MESMA foto de ferrugem, só o tingimento não bastava pra diferenciar
# tipos de verdade (achado testando com um vent isolado de cada tipo lado a
# lado — saíam quase idênticos). `texture_gamma` acima clareia de verdade a
# imagem-base (mesmo mecanismo de `_load_texture`) antes do tingimento —
# white_smoker fica bem mais clara/pálida, diffuse_flow moderadamente.

# Manchas minerais adicionais no terreno (além da ferrugem por proximidade
# real a cada vent, já modelada via `stain`): anidrita branca e óxido de
# manganês escuro, comuns em campos reais mas ausentes até agora — sem
# elas o chão só tinha UM gradiente cinza-azulado→ferrugem, monocromático
# demais.
_WHITE_MINERAL_TINT = np.array([0.95, 0.93, 0.87])
_DARK_MINERAL_TINT = np.array([0.22, 0.22, 0.25])

# Camada de fauna quimiossintética (opt-in via `include_fauna`, ver
# `render_artistic_scene`/`fumarola_field.py` — gera SEMPRE as duas
# versões, com e sem esta camada). Raios reais vêm de `v.fauna_zones`
# (`fauna_zonation()` em fumarola_field.py, já calculado pra todo vent,
# nunca usado na cena 3D até agora). Cores calibradas contra fotos reais
# de campos hidrotermais: tapete bacteriano (Beggiatoa-like) branco-
# amarelado cobrindo o substrato bem perto do orifício; mexilhões
# formam uma crosta escura azul-petróleo em faixa ao redor, não até a
# própria abertura (fica coberta pelo tapete).
_BACTERIAL_MAT_TINT = np.array([0.88, 0.85, 0.60])
_MUSSEL_BED_TINT = np.array([0.14, 0.16, 0.20])
_TUBEWORM_BASE_TINT = np.array([0.90, 0.89, 0.85])
_TUBEWORM_TIP_TINT = np.array([0.55, 0.08, 0.10])
_SHRIMP_TINT = np.array([0.85, 0.72, 0.68])
# Fatores de escala VISUAL (não físicos) aplicados ao RAIO de renderização
# da fauna — ver uso em `_terrain_base_tint` (manchas de cor no terreno) e
# no loop de vents em `render_artistic_scene` (tufos de verme/camarão).
# Usuário reportou que as versões com/sem fauna "não fazem a menor
# diferença": os raios reais (~0.2-5m) ficavam pequenos demais pra ler
# contra o quadro mais aberto (ver `_PHOTO_NEIGHBORHOOD_R`) e, no caso de
# tufos/camarão especificamente, o raio de zona NEM estava sendo escalado
# antes (só as manchas de cor do terreno eram) — bug real, não só falta
# de contraste. Dois fatores em vez de um: tapete/mexilhões já tinham o
# maior raio real (até 5m) — um fator agressivo os deixaria enormes/
# artificiais; tufos/camarão têm raio real bem menor (até 2.5-3m) e
# precisam de mais escala pra virar um aglomerado reconhecível no quadro.
_FAUNA_COLOR_SCALE_M = 2.4
_FAUNA_OBJECT_SCALE_M = 4.5


_texture_cache: dict = {}


def _load_texture(path: str, gamma: float = 1.0):
    """Carrega e cacheia uma `pv.Texture` (evita reler o JPG do disco a
    cada chamada de `render_artistic_scene` na mesma sessão do processo).
    `gamma`>1 clareia a imagem (correção `(v/255)**(1/gamma)`, não um
    multiplicador linear) antes de criar a textura — a foto CC0 de rocha
    escolhida (dark_rock, Poly Haven) é escura o bastante (fiel à rocha
    vulcânica real) que mesmo com iluminação forte ficava quase preta na
    cena. Gamma em vez de multiplicar direto: levanta tons médios sem
    saturar desproporcionalmente as poucas áreas já claras (veios), que
    com multiplicação linear simples dominavam a cor final."""
    key = (path, round(gamma, 3))
    if key not in _texture_cache:
        if gamma != 1.0:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            arr = (np.asarray(img).astype(np.float32) / 255.0) ** (1.0 / gamma)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            _texture_cache[key] = pv.Texture(arr)
        else:
            _texture_cache[key] = pv.read_texture(path)
    return _texture_cache[key]

_SMOKE_PALETTE = {
    "black_smoker": {"rgb": np.array([0.10, 0.10, 0.12]), "alpha0": 0.85},
    "white_smoker": {"rgb": np.array([0.75, 0.76, 0.78]), "alpha0": 0.55},
    "diffuse_flow": {"rgb": np.array([0.65, 0.78, 0.80]), "alpha0": 0.22},
}


def _fog_blend(rgb: np.ndarray, dist_from_camera: np.ndarray, fog_scale_m: float) -> np.ndarray:
    """Mistura `rgb` (N,3) em direção à cor de neblina da água conforme a
    distância da câmera aumenta — aproximação determinística de
    atenuação/dispersão de luz na água (não é um fog volumétrico real
    renderizado por raymarching, é uma estimativa fechada suficiente para
    uma única imagem estática com câmera fixa)."""
    # Expoente reduzido (era 1.3): 1.3 mantinha o meio-campo quase tão
    # visível quanto o primeiro plano e então escurecia de repente perto
    # do limite — bom pra um quadro fechado numa chaminé só, mas escondia
    # outras chaminés/relevo do MESMO cluster que deveriam aparecer
    # esmaecendo aos poucos ao fundo (sensação de "campo se estendendo"),
    # não sumindo de vez cedo demais.
    t = np.clip(dist_from_camera / fog_scale_m, 0.0, 1.0)[:, np.newaxis] ** 1.0
    return rgb * (1 - t) + _HAZE_RGB[np.newaxis, :] * t


def _random_bump_field(Theta: np.ndarray, Hf: np.ndarray, rng: np.random.Generator,
                        n_bumps: int, amp_range: Tuple[float, float],
                        outward_bias: float = 0.75) -> np.ndarray:
    """Soma `n_bumps` "protuberâncias" gaussianas 2D (em Theta ao redor /
    Hf na altura) com centro, tamanho e sinal aleatórios — ao contrário de
    modular um cone monotônico com uma senoide regular (visual de "coluna
    estriada"), isto cria bojos/estrangulamentos LOCALIZADOS e assimétricos,
    mais parecido com o crescimento mineral irregular real de uma chaminé.
    A maioria dos bojos é pra FORA (`outward_bias`), mas uma fração é pra
    dentro (estrangulamentos), imitando erosão/quebra parcial da estrutura."""
    field = np.zeros_like(Theta)
    for _ in range(n_bumps):
        theta_center = rng.uniform(0, 2 * np.pi)
        h_center = rng.uniform(0.0, 0.9)
        theta_sigma = rng.uniform(0.45, 1.1)
        h_sigma = rng.uniform(0.06, 0.20)
        sign = 1.0 if rng.uniform() < outward_bias else -1.0
        amp = sign * rng.uniform(*amp_range)
        dtheta = np.angle(np.exp(1j * (Theta - theta_center)))  # distância angular com wrap-around
        d2 = (dtheta / theta_sigma) ** 2 + ((Hf - h_center) / h_sigma) ** 2
        field += amp * np.exp(-0.5 * d2)
    return field


def _chimney_mesh_and_hfrac(base_x: float, base_y: float, base_z: float, height: float,
                             base_radius: float, rng: np.random.Generator,
                             n_theta: int = 32, n_h: int = 28):
    """Malha de chaminé com irregularidade de superfície MULTI-ESCALA:
    (1) bojos/estrangulamentos localizados grandes (`_random_bump_field`,
    não um cone monotônico modulado por senoide regular — a versão
    anterior deste módulo tinha um visual de "coluna estriada" regular
    demais, sinalizado pelo usuário como pouco realista); (2) ruído
    fractal de várias oitavas para textura de superfície de escala média;
    (3) leve inclinação do eixo (chaminés reais raramente são
    perfeitamente verticais). Reimplementada aqui (não importa
    fumarola_field.py, evita import circular) — também devolve a fração
    de altura Hf de cada vértice (gradiente de cor rocha-ferruginosa-
    clara) e a fração angular Theta/(2π) (coordenada U de textura)."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    hfrac = np.linspace(0, 1, n_h)
    Theta, Hf = np.meshgrid(theta, hfrac)

    radius = base_radius * (1 - Hf) ** 1.6 + base_radius * 0.12
    flange = 1 + 0.3 * np.clip((Hf - 0.82) / 0.18, 0, 1)
    radius *= flange

    # Bojos grandes (2-4, escala ~0.15-0.5x o raio da base) — dominam a
    # silhueta, quebrando a simetria de revolução monotônica.
    n_bumps = rng.integers(2, 5)
    bump_field = _random_bump_field(Theta, Hf, rng, n_bumps, amp_range=(0.15, 0.5))

    # Ruído fractal (4 oitavas, frequência dobrando e amplitude caindo a
    # cada oitava) para rugosidade de escala média — substitui os 2
    # harmônicos fixos da versão anterior, que criavam um padrão
    # perceptualmente regular/repetitivo.
    fine_noise = np.zeros_like(Theta)
    freq, amp = 3.0, 0.28
    for _ in range(4):
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        fine_noise += amp * np.sin(freq * Theta + p1) * np.sin(freq * 1.7 * Hf * np.pi + p2)
        freq *= 2.15
        amp *= 0.55
    fine_noise += 0.10 * rng.normal(0, 1, Theta.shape)  # jitter fino por-vértice

    radius = np.clip(radius * (1 + bump_field) * (1 + fine_noise), base_radius * 0.10, None)

    # Leve inclinação do eixo — cresce linearmente com a altura, direção e
    # magnitude aleatórias por chaminé (até ~12% da altura no topo).
    lean_dir = rng.uniform(0, 2 * np.pi)
    lean_amount = rng.uniform(0.03, 0.12) * height

    Xc = base_x + radius * np.cos(Theta) + Hf * lean_amount * np.cos(lean_dir)
    Yc = base_y + radius * np.sin(Theta) + Hf * lean_amount * np.sin(lean_dir)
    Zc = base_z + Hf * height
    return Xc, Yc, Zc, Hf, Theta / (2 * np.pi)


def _plume_points(base_x: float, base_y: float, start_z: float, top_z: float,
                   vent_type: str, rng: np.random.Generator, n_points: int):
    """Nuvem de pontos da pluma com paleta fotorrealista (fumaça escura em
    black smokers, pálida em white smoker/diffuse flow). Dispersão radial
    crescente com a altura (mistura turbulenta, mesma forma funcional da
    versão científica `fumarola_field._plume_smoke`) SOMADA a uma
    "serpentina" de baixa frequência no centro da coluna (soma de
    senos/cossenos com fases aleatórias) — sem isso, uma pluma reta com só
    dispersão radial simétrica lê como uma nuvem de confete em volta de um
    eixo, não como fumaça real, que forma vórtices/curvas visíveis ao
    subir. CORES recalculadas aqui porque a paleta científica (vermelho/
    amarelo, por tipo, pensada pra legenda) não corresponde à cor real."""
    if top_z <= start_z or n_points <= 0:
        return None

    height_total = top_z - start_z
    hfrac = rng.uniform(0, 1, n_points) ** 1.3
    z = start_z + hfrac * height_total

    wobble_x = np.zeros(n_points)
    wobble_y = np.zeros(n_points)
    for i, wavenumber in enumerate((1.3, 2.6, 4.3)):
        amp = 0.9 / (i + 1)
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, 2)
        wobble_x += amp * np.sin(wavenumber * hfrac * 2 * np.pi + phase_x)
        wobble_y += amp * np.cos(wavenumber * hfrac * 2 * np.pi + phase_y)
    wobble_scale = 0.25 + hfrac * 1.3  # a serpentina também cresce com a altura

    spread = 0.15 + hfrac ** 0.8 * 1.7
    r = spread * np.sqrt(rng.uniform(0, 1, n_points))
    theta = rng.uniform(0, 2 * np.pi, n_points)
    x = base_x + r * np.cos(theta) + wobble_x * wobble_scale
    y = base_y + r * np.sin(theta) + wobble_y * wobble_scale

    palette = _SMOKE_PALETTE[vent_type]
    dilute_rgb = np.array([0.75, 0.85, 0.87])
    t = hfrac[:, np.newaxis]
    rgb = palette["rgb"][np.newaxis, :] * (1 - t) + dilute_rgb[np.newaxis, :] * t
    alpha = np.clip(palette["alpha0"] * (1 - 0.5 * hfrac), 0.05, 1.0)
    # Tamanho varia com a altura E com um fator aleatório por ponto (não só
    # monotonicamente crescente) — mistura pontos finos com "flocos"
    # maiores, quebrando o visual uniforme de antes.
    sizes = (4.0 + hfrac * 10.0) * rng.uniform(0.6, 1.6, n_points)
    return x, y, z, rgb, alpha, sizes


def _terrain_micro_roughness(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                              amplitude: float = 1.8, base_freq: float = 0.35) -> np.ndarray:
    """Ruído fractal (4 oitavas) de ALTA frequência espacial (feições a
    cada ~1-4m, bem menor que o relevo de dezenas/centenas de metros da
    heightmap diamond-square) somado à elevação real do terreno — imita a
    rugosidade de blocos/detritos de basalto do fundo oceânico real, que
    nenhuma textura plana (só cor, sem deslocar geometria) consegue dar."""
    field = np.zeros_like(X)
    freq, amp = base_freq, amplitude
    for _ in range(4):
        phase_x, phase_y = rng.uniform(0, 1000, 2)
        field += amp * np.sin(freq * X + phase_x) * np.sin(freq * 1.3 * Y + phase_y)
        freq *= 2.3
        amp *= 0.5
    field += amplitude * 0.15 * rng.normal(0, 1, X.shape)
    return field


def _fractal_phases(rng: np.random.Generator, octaves: int) -> List[Tuple[float, float]]:
    """Sorteia as fases (x,y) de cada oitava UMA vez, pra poder reavaliar o
    MESMO campo fractal contínuo em conjuntos de pontos diferentes (ex.:
    malha grossa de fundo vs. o patch fino de terreno perto da câmera,
    ver `render_artistic_scene`) sem costura visível na fronteira — dois
    campos com fases sorteadas independentemente ficariam descontínuos
    onde um substitui o outro, mesmo cobrindo exatamente a mesma área."""
    return [tuple(rng.uniform(0, 1000, 2)) for _ in range(octaves)]


def _terrain_mineral_patches(X: np.ndarray, Y: np.ndarray, phases: List[Tuple[float, float]],
                              patch_scale_m: float = 9.0) -> np.ndarray:
    """Ruído fractal (3 oitavas) de baixa frequência espacial, em [-1,1] —
    usado como campo de mistura pra manchas minerais adicionais (anidrita
    branca perto de +1, óxido de manganês escuro perto de -1) no terreno,
    independente da ferrugem por proximidade real a vents (`stain`).
    Escala bem maior (`patch_scale_m`) que `_terrain_micro_roughness`:
    manchas de dezenas de metros, não rugosidade de blocos. `phases` vem
    de `_fractal_phases` (não sorteia aqui) para poder reavaliar o mesmo
    campo em pontos diferentes sem costura — ver ali."""
    field = np.zeros_like(X)
    freq, amp = 1.0 / patch_scale_m, 1.0
    for phase_x, phase_y in phases:
        field += amp * np.sin(freq * X + phase_x) * np.sin(freq * 1.4 * Y + phase_y)
        freq *= 2.1
        amp *= 0.55
    return np.clip(field, -1, 1)


_FINE_ROCK_BASE_WAVELENGTH_M = 2.2
_FINE_ROCK_OCTAVES = 5


def _fine_rock_noise(X: np.ndarray, Y: np.ndarray, phases: List[Tuple[float, float]]) -> np.ndarray:
    """Ruído fractal RIDGED (`1-|sin|`, cristas/fraturas nítidas nos
    cruzamentos de zero, em vez da ondulação suave e arredondada de
    `_terrain_micro_roughness`) em escala genuinamente pequena (~0.2-2m).
    Usado SÓ no patch de terreno de alta resolução perto da câmera
    (`_build_terrain_hero_patch`), onde a malha tem vértices o bastante
    (~0.3-0.4m/célula) pra representar essa frequência sem aliasing — a
    malha de fundo (~4-5m/célula pro tamanho de domínio/grade padrão) não
    tem essa resolução, e é exatamente por isso que o substrato ainda
    lia como "liso" de perto mesmo com ruído de textura: qualquer feição
    de 1-2m simplesmente não existe na geometria em si nessa resolução,
    só como jitter por-vértice sem coerência espacial (aliasing puro).
    Devolve valores SEMPRE >=0 (normalizado, não centrado em 0) pra
    garantir que o patch nunca afunde abaixo da altura da malha grossa
    logo abaixo dele. `phases` vem de `_fractal_phases` (mesmo padrão de
    `_terrain_mineral_patches`) — não sorteia aqui, pra poder reavaliar
    o MESMO campo em pontos avulsos (`_hero_patch_extra_height`, usada
    pra plantar chaminé/fauna/blocos exatamente na superfície real do
    patch, não na altura da malha grossa por baixo dele — sem isso,
    objetos pequenos como tufos de verme ficavam enterrados sob o
    patch, que pode ficar até ~0.5m mais alto que a malha grossa ali)."""
    field = np.zeros_like(X)
    freq = 2 * np.pi / _FINE_ROCK_BASE_WAVELENGTH_M
    amp = 1.0
    total_amp = 0.0
    for phase_x, phase_y in phases:
        raw = np.sin(freq * X + phase_x) * np.sin(freq * 1.37 * Y + phase_y)
        ridged = 1.0 - np.abs(raw)
        field += amp * ridged
        total_amp += amp
        freq *= 2.15
        amp *= 0.52
    return field / total_amp


def _hero_patch_extra_height(Xq, Yq, cx: float, cy: float, patch_radius: float,
                              noise_phases: List[Tuple[float, float]]):
    """Elevação extra (margem sempre positiva + ruído fino RIDGED,
    atenuado a zero na borda) que `_build_terrain_hero_patch` soma à
    altura bilinear-interpolada da malha grossa em qualquer (Xq,Yq) —
    função PURA reutilizável tanto pra construir a malha do patch quanto
    pra consultar a altura real do CHÃO (incluindo o patch) na hora de
    plantar chaminé/fauna/blocos ali perto (`_ground_height` em
    `render_artistic_scene`), sem o quê esses objetos ficariam mal
    posicionados em relação à superfície visível de verdade."""
    dist = np.sqrt((Xq - cx) ** 2 + (Yq - cy) ** 2)
    edge_feather = max(patch_radius * 0.3, 1.0)
    taper = np.clip((patch_radius - dist) / edge_feather, 0.0, 1.0)
    taper = taper * taper * (3 - 2 * taper)
    noise = _fine_rock_noise(Xq, Yq, noise_phases)
    return _HERO_PATCH_LIFT_MARGIN_M + noise * _HERO_PATCH_NOISE_AMPLITUDE_M * taper


def _terrain_base_tint(Xq: np.ndarray, Yq: np.ndarray, vents: List, vent_xy_m,
                        mineral_phases: List[Tuple[float, float]]) -> np.ndarray:
    """Cor-base do terreno (rocha↔ferrugem por proximidade real a cada
    vent + manchas minerais) como função PURA de posição (Xq,Yq de
    qualquer formato/resolução) — fatorado fora de `render_artistic_scene`
    pra ser chamado tanto pela malha grossa de fundo quanto pelo patch
    fino de terreno perto da câmera (`_build_terrain_hero_patch`) com o
    MESMO campo de manchas minerais (`mineral_phases` compartilhadas, ver
    `_fractal_phases`) — garante que os dois se encaixam sem costura de
    cor na fronteira, já que é uma função contínua da posição, não um
    array pré-amostrado numa grade específica. NÃO inclui o ruído de
    albedo por-vértice (`tint_noise`) nem a mistura com neblina — isso
    fica a cargo de cada chamador, pode divergir sem risco de costura
    (jitter fino demais pro olho notar diferença de fase).

    NÃO inclui mais tapete bacteriano/leito de mexilhões (removido —
    ver `_build_fauna_crust_mesh`): tingir esta cor multiplicada sobre a
    foto de rocha ESCURA nunca conseguia dar um branco/pálido vívido
    (multiplicar por uma textura escura só escurece mais, mesma
    limitação matemática já documentada pra diferenciação de chaminé
    por tipo) — o tapete saía cinza-escuro, indistinguível de sombra
    comum ("ou é tudo uma sombra preta", relato direto do usuário).
    Fauna com cor de verdade agora é uma malha PRÓPRIA sem textura."""
    stain = np.zeros_like(Xq)
    for (vx, vy), v in zip(vent_xy_m, vents):
        radius = 8.0 + v.chimney_height_m * 1.5
        dist = np.sqrt((Xq - vx) ** 2 + (Yq - vy) ** 2)
        stain += np.clip(1 - dist / radius, 0, 1) ** 1.5
    stain = np.clip(stain, 0, 1)

    tint = (_ROCK_TINT[np.newaxis, np.newaxis, :] * (1 - stain[..., np.newaxis])
            + _RUST_TINT[np.newaxis, np.newaxis, :] * stain[..., np.newaxis])

    patch_field = _terrain_mineral_patches(Xq, Yq, mineral_phases)
    white_w = np.clip((patch_field - 0.15) / 0.4, 0, 1)[..., np.newaxis]
    dark_w = np.clip((-patch_field - 0.15) / 0.4, 0, 1)[..., np.newaxis]
    tint = tint * (1 - white_w) + _WHITE_MINERAL_TINT[np.newaxis, np.newaxis, :] * white_w
    tint = tint * (1 - dark_w) + _DARK_MINERAL_TINT[np.newaxis, np.newaxis, :] * dark_w
    return tint


_HERO_PATCH_LIFT_MARGIN_M = 0.05
_HERO_PATCH_NOISE_AMPLITUDE_M = 0.45
_HERO_PATCH_MIN_CELL_M = 0.30
_HERO_PATCH_TARGET_HALF_N = 140  # ~280 divisões por eixo, custo previsível


def _build_terrain_hero_patch(Z: np.ndarray, meters_per_cell: float, cx: float, cy: float,
                               patch_radius: float, vents: List, vent_xy_m,
                               mineral_phases: List[Tuple[float, float]],
                               noise_phases: List[Tuple[float, float]],
                               rng: np.random.Generator, dist_from_cam, fog_scale_m: float):
    """Malha de terreno de ALTA resolução (~0.35m/célula, vs. os ~4-5m/
    célula da malha grossa de fundo para o tamanho de grade/domínio
    padrão do projeto) cobrindo só a vizinhança imediata da câmera
    (`patch_radius`, tipicamente algumas dezenas de metros). Esse é o
    motivo REAL do substrato ainda ler como "liso" de perto mesmo depois
    de somar ruído de textura à malha grossa: nenhum ruído, por mais
    detalhado matematicamente, consegue representar uma feição de 1-2m
    numa malha com vértices a 4-5m de distância um do outro (limite de
    Nyquist) — as oitavas de frequência mais alta simplesmente aliasavam
    em jitter sem coerência espacial, não em rugosidade visível.

    A altura-base de cada vértice do patch vem de interpolação BILINEAR
    da MESMA elevação Z real já calculada pra malha grossa
    (`scipy.ndimage.map_coordinates`, não uma heightmap nova) — o patch
    representa o mesmo terreno em resolução mais fina, não inventa
    relevo. Ruído fino RIDGED (`_fine_rock_noise`, sempre >=0) é somado
    por cima com uma margem de elevação sempre positiva
    (`_HERO_PATCH_LIFT_MARGIN_M`) — o patch nunca fica mais baixo que a
    malha grossa na mesma posição (x,y), o que evita z-fighting sem
    precisar recortar/remover a malha grossa por baixo: sempre mais alto
    ali, o patch simplesmente a oculta por completo pra uma câmera
    olhando de cima (convenção de câmera "estilo ROV" usada em toda a
    cena). O ruído extra (não a margem) é atenuado a zero na borda do
    patch (`taper`, suavizado tipo smoothstep), então a altura converge
    de volta pra malha grossa sem degrau visível na fronteira — a cor
    (`_terrain_base_tint`, reavaliada com as MESMAS `mineral_phases` da
    malha grossa) e a textura (mesmo ladrilhamento X/Y em metros) também
    são funções contínuas da posição, então não há costura de cor nem de
    textura, só de resolução (invisível, já que a malha grossa não
    aparece mais ali — está oculta debaixo do patch)."""
    # Célula adaptativa (não mais um valor fixo de 0.35m): o raio de
    # enquadramento agora pode variar bem mais (cluster inteiro vs. hub
    # apertado), e um tamanho de célula fixo faria a malha crescer O(n²)
    # com o raio sem limite — trava o número de divisões por eixo pra
    # manter o custo de render previsível, com piso pra nunca ficar mais
    # grossa que a resolução mínima necessária pra rugosidade de rocha.
    fine_cell = max(patch_radius / _HERO_PATCH_TARGET_HALF_N, _HERO_PATCH_MIN_CELL_M)
    half_n = max(8, int(patch_radius / fine_cell))
    offsets = np.arange(-half_n, half_n + 1) * fine_cell
    size_y, size_x = Z.shape
    xs = np.clip(cx + offsets, 0.0, (size_x - 1) * meters_per_cell)
    ys = np.clip(cy + offsets, 0.0, (size_y - 1) * meters_per_cell)
    Xp, Yp = np.meshgrid(xs, ys)

    col_idx = Xp / meters_per_cell
    row_idx = Yp / meters_per_cell
    base_z = scipy.ndimage.map_coordinates(Z, [row_idx, col_idx], order=1, mode="nearest")

    # `_hero_patch_extra_height` é a MESMA função usada por `_ground_height`
    # em `render_artistic_scene` pra plantar chaminé/fauna/blocos na
    # superfície real do patch (não na malha grossa por baixo, que pode
    # ficar até ~0.5m mais baixa aqui) — só o jitter fino por-vértice
    # abaixo (não reproduzido na consulta pontual, contribui <3cm) é
    # exclusivo da malha, não afeta onde os objetos são plantados.
    extra = _hero_patch_extra_height(Xp, Yp, cx, cy, patch_radius, noise_phases)
    Zp = base_z + extra + 0.03 * rng.uniform(0, 1, Xp.shape)

    tint = _terrain_base_tint(Xp, Yp, vents, vent_xy_m, mineral_phases)
    tint_noise = 1.0 + 0.10 * rng.normal(0, 1, Zp.shape)
    tint = np.clip(tint * tint_noise[..., np.newaxis], 0, 1)

    flat_dist = dist_from_cam(Xp.ravel(order="F"), Yp.ravel(order="F"), Zp.ravel(order="F"))
    tint_flat = _fog_blend(tint.transpose(1, 0, 2).reshape(-1, 3), flat_dist, fog_scale_m)
    uv = np.column_stack([
        Xp.ravel(order="F") / _ROCK_TILE_SIZE_M,
        Yp.ravel(order="F") / _ROCK_TILE_SIZE_M,
    ])

    mesh = pv.StructuredGrid(Xp, Yp, Zp)
    mesh.active_texture_coordinates = uv
    mesh["colors"] = tint_flat
    return mesh


def _make_boulder_mesh(center_x: float, center_y: float, ground_z: float, radius: float,
                        rng: np.random.Generator, n_theta: int = 14, n_phi: int = 10):
    """Malha de um bloco de rocha solto/detrito angular, assentado no
    chão (não um bloco flutuante centrado) — reaproveita a mesma técnica
    de bojos gaussianos + ruído fractal de `_chimney_mesh_and_hfrac`,
    aplicada a um esferoide achatado em vez de um cone (blocos no fundo
    real assentam e se espalham, não ficam esféricos/altos)."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0.08, np.pi - 0.08, n_phi)  # evita polos degenerados
    Theta, Phi = np.meshgrid(theta, phi)
    Phi_frac = Phi / np.pi

    bump_field = _random_bump_field(Theta, Phi_frac, rng, n_bumps=int(rng.integers(3, 6)),
                                     amp_range=(0.15, 0.45), outward_bias=0.6)
    # `fine` GRAMPEADO (não só o raio final `r`): jitter gaussiano sem
    # limite ocasionalmente somava com bojos pra dentro já bem negativos
    # e produzia um raio negativo/quase-zero num vértice isolado — o
    # piso em `r` evitava um buraco, mas o salto abrupto entre esse
    # vértice grampeado e seus vizinhos normais criava uma "lasca" fina
    # e achatada de geometria (achado visual real revisando renders —
    # aparecia até sem a camada de fauna, então é do bloco de detrito
    # em si, não de nada adicionado nesta rodada).
    fine = np.clip(0.18 * rng.normal(0, 1, Theta.shape), -0.4, 0.4)
    r = np.clip((1 + bump_field) * (1 + fine), 0.35, None)

    Xb = center_x + radius * r * np.sin(Phi) * np.cos(Theta)
    Yb = center_y + radius * r * np.sin(Phi) * np.sin(Theta)
    Zb_raw = radius * 0.5 * r * np.cos(Phi)  # achatado verticalmente (bloco, não esfera)
    Zb = Zb_raw - Zb_raw.min() + ground_z  # a base do bloco encosta no chão real
    return Xb, Yb, Zb, Theta / (2 * np.pi), Phi_frac


def _tubeworm_tuft_mesh(base_x: float, base_y: float, ground_height_fn,
                         rng: np.random.Generator, zone_radius_m: float):
    """Malha combinada (`pv.merge`, um único ator em vez de um `add_mesh`
    por verme) de 1-3 AGLOMERADOS de vermes tubulares (Riftia-like)
    dentro da zona real de ocorrência (`zone_radius_m` =
    `v.fauna_zones["tubeworm"]`, de `fauna_zonation()` em
    fumarola_field.py). Colônias reais crescem em tufos densos e
    discretos, não espalhadas uniformemente pela zona inteira — por isso
    sorteia posições de TUFO dentro do raio e concentra os vermes perto
    de cada uma.

    POUCOS tubos GROSSOS, não muitos finos: um enquadramento típico
    mostra dezenas de metros por quadro, então hastes de ~1-6cm de raio
    (fisicamente corretas) viravam ruído sub-pixel — "várias bolinhas"
    em vez de um tufo reconhecível (relato direto do usuário depois da
    primeira versão). Cada verme individual continua uma haste fina e
    levemente curva (`pv.Spline().tube()`), clara na base (tubo
    quitinoso branco) escurecendo pra vermelho-escuro na ponta (plumas
    branquiais expostas), mas agora só 3-7 por tufo, bem mais grossas —
    e cada tufo ganha uma base arredondada própria (reaproveita a
    técnica de bojos gaussianos de `_make_boulder_mesh`, cor de tubo)
    pra ancorar visualmente os vermes ao chão em vez de "flutuarem".

    `ground_height_fn(x,y)` é chamado UMA VEZ POR TUFO (não uma altura
    única passada de fora) — bug real achado testando: com
    `_FAUNA_OBJECT_SCALE_M` grande, o raio de zona escalado pode chegar
    a dezenas de metros, então um tufo podia cair vários metros longe
    do vent original, onde o terreno real já está numa altura bem
    diferente da altura do próprio vent — usar essa altura errada fazia
    o tufo "flutuar" acima ou afundar abaixo do chão de verdade ali.
    Devolve `None` se `zone_radius_m<=0` (dict sem a chave)."""
    if zone_radius_m <= 0:
        return None
    n_clusters = int(rng.integers(1, 4))
    meshes = []
    for _ in range(n_clusters):
        # Fração reduzida (era 0.2-0.85) — tufos ficam mais colados à
        # base da própria chaminé, tanto por realismo (colônias reais
        # crescem bem perto da margem de fluxo difuso) quanto pra evitar
        # que o raio de zona já escalado (`_FAUNA_OBJECT_SCALE_M`) jogue
        # o tufo muito longe, sobre um trecho de terreno com relevo
        # bem diferente do da própria chaminé.
        cluster_dist = zone_radius_m * rng.uniform(0.1, 0.45)
        cluster_th = rng.uniform(0, 2 * np.pi)
        clx = base_x + cluster_dist * np.cos(cluster_th)
        cly = base_y + cluster_dist * np.sin(cluster_th)
        cluster_base_z = ground_height_fn(clx, cly)
        cluster_spread = rng.uniform(0.18, 0.4)

        # Base arredondada (mesma técnica dos blocos de detrito, cor
        # clara de tubo quitinoso) — ancora o tufo, evita a leitura de
        # "hastes flutuando no vazio" da primeira versão.
        base_radius = cluster_spread * rng.uniform(0.7, 1.1)
        Xb, Yb, Zb, Ub, Vb = _make_boulder_mesh(clx, cly, cluster_base_z, base_radius, rng, n_theta=10, n_phi=7)
        base_mesh = pv.StructuredGrid(Xb, Yb, Zb)
        base_mesh["colors"] = np.clip(
            _TUBEWORM_BASE_TINT[np.newaxis, :] * (1 + 0.08 * rng.normal(0, 1, (Xb.size, 1))), 0, 1)
        meshes.append(base_mesh)

        n_worms = int(rng.integers(3, 8))
        for _ in range(n_worms):
            r = cluster_spread * rng.uniform(0, 1) ** 0.5
            th = rng.uniform(0, 2 * np.pi)
            wx = clx + r * np.cos(th)
            wy = cly + r * np.sin(th)
            h = rng.uniform(0.45, 1.3)
            lean_dir = rng.uniform(0, 2 * np.pi)
            lean_amount = rng.uniform(0.05, 0.2) * h
            curve_amp = rng.uniform(-0.15, 0.15)
            t = np.linspace(0, 1, 6)
            perp = lean_dir + np.pi / 2
            px = wx + lean_amount * t * np.cos(lean_dir) + curve_amp * np.sin(t * np.pi) * np.cos(perp)
            py = wy + lean_amount * t * np.sin(lean_dir) + curve_amp * np.sin(t * np.pi) * np.sin(perp)
            pz = cluster_base_z + h * t
            pts = np.column_stack([px, py, pz])
            spline = pv.Spline(pts, 10)
            spline["hfrac"] = np.linspace(0, 1, spline.n_points)
            # Raio real de um tubo individual é ~5-15mm; engrossado bem
            # mais que na primeira versão (~3-6cm) — poucos tubos
            # GROSSOS (~10-22cm) leem como uma forma real à distância
            # típica de enquadramento, muitos finos só viravam poeira.
            tube_radius = rng.uniform(0.10, 0.22)
            tube = spline.tube(radius=tube_radius, n_sides=7)
            # Vermelho concentrado perto da ponta (expoente >1), não um
            # gradiente linear — na vida real só a pluma branquial na
            # extremidade é vermelha, o tubo em si é claro quase até lá.
            hf = np.clip(tube["hfrac"], 0, 1) ** 2.2
            color = (_TUBEWORM_BASE_TINT[np.newaxis, :] * (1 - hf[:, np.newaxis])
                     + _TUBEWORM_TIP_TINT[np.newaxis, :] * hf[:, np.newaxis])
            tube["colors"] = color
            meshes.append(tube)
    return pv.merge(meshes) if meshes else None


def _shrimp_swarm_points(base_x: float, base_y: float, ground_height_arr_fn,
                          rng: np.random.Generator, zone_radius_m: float):
    """Pontos de um enxame de camarões (Rimicaris-like) sobre o substrato
    perto do orifício, dentro do raio real `zone_radius_m` =
    `v.fauna_zones["shrimp_swarm"]` — em fotos reais lê como uma
    salpicadura DENSA de pontos pálidos sobre a rocha escura, então
    (ao contrário da fumaça) pontos opacos nítidos são apropriados aqui,
    não um artefato a esconder. A primeira versão espalhava poucos
    pontos (150-400) pelo raio de zona inteiro (já escalado por
    `_FAUNA_OBJECT_SCALE_M`) — resultado esparso demais, lia como "umas
    bolinhas soltas" em vez de um enxame de verdade (relato direto do
    usuário). Concentra bem mais pontos (600-1400) num raio efetivo
    MENOR que o raio de zona (`r**1.8` em vez de `r**0.5` — concentra
    perto do centro), formando uma mancha densa reconhecível em vez de
    ruído espalhado. Altura de cada ponto vem de `ground_height_arr_fn`
    (VETORIZADA, chão real incluindo o hero patch) reavaliada em CADA
    ponto, não uma altura única do vent — o raio efetivo ainda chega a
    vários metros com `_FAUNA_OBJECT_SCALE_M` grande, e o terreno real
    pode variar nessa distância (mesmo bug de altura já corrigido nos
    tufos de verme). Devolve `None` se `zone_radius_m<=0` (dict sem a
    chave — só ~60% dos black smokers têm essa zona, ver
    `fauna_zonation`)."""
    if zone_radius_m <= 0:
        return None
    n_points = int(rng.integers(600, 1400))
    effective_radius = zone_radius_m * 0.55
    r = effective_radius * rng.uniform(0, 1, n_points) ** 1.8
    th = rng.uniform(0, 2 * np.pi, n_points)
    x = base_x + r * np.cos(th)
    y = base_y + r * np.sin(th)
    z = ground_height_arr_fn(x, y) + rng.uniform(0.02, 0.3, n_points)
    jitter = 1.0 + 0.15 * rng.normal(0, 1, (n_points, 1))
    colors = np.clip(_SHRIMP_TINT[np.newaxis, :] * jitter, 0, 1)
    return x, y, z, colors


def _build_fauna_crust_mesh(center_x: float, center_y: float, r_inner: float, r_outer: float,
                             ground_height_arr_fn, color: np.ndarray, rng: np.random.Generator,
                             n_theta: int = 22, n_r: int = 6):
    """Malha rasteira (disco se `r_inner=0`, anel caso contrário) pra
    tapete bacteriano/leito de mexilhões, acompanhando o CHÃO REAL
    (`ground_height_arr_fn`, inclui o hero patch fino perto da câmera)
    — cor SÓLIDA final, SEM textura fotográfica multiplicada por cima.

    Isto substitui a primeira versão (tingimento por-vértice misturado
    na cor do terreno ANTES de multiplicar pela foto de rocha): a foto
    escolhida é escura o bastante que nenhum tingimento a clareia até
    um branco/pálido vívido (mesma limitação matemática já documentada
    pra diferenciação de chaminé por tipo — multiplicar só ESCURECE),
    então o tapete saía cinza-escuro, indistinguível de sombra comum —
    "ou é tudo uma sombra preta", relato direto do usuário. Uma malha
    PRÓPRIA sem `texture=` usa a cor do vértice como cor final de
    verdade, sem esse teto.

    Contorno externo levemente irregular (2 harmônicos de fase
    aleatória) em vez de um círculo perfeito — evita a leitura
    "desenhada"/artificial de uma forma geométrica exata.

    A altura de cada vértice NÃO segue o relevo real bruto sem limite:
    perto de fumarolas grandes o terreno real (heightmap + patch fino de
    rugosidade) pode ter encostas bem íngremes por baixo de um disco de
    só 1-2m — seguir isso ao pé da letra criava "paredes" quase
    verticais de cor sólida entre vértices adjacentes, lendo como
    estilhaços/lascas planas flutuando (achado visual real, não só
    teórico, testando esta função). A altura de cada vértice é
    GRAMPEADA a um desvio pequeno em torno da altura do CENTRO do disco
    — a crosta ainda ondula um pouco com o micro-relevo local, mas nunca
    escala uma encosta inteira."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phase1, phase2 = rng.uniform(0, 2 * np.pi, 2)
    edge_wobble = 1.0 + 0.12 * np.sin(2 * theta + phase1) + 0.08 * np.sin(5 * theta + phase2)
    r_frac = np.linspace(0, 1, max(n_r, 2))
    Theta, Rf = np.meshgrid(theta, r_frac)
    Router = r_outer * edge_wobble[np.newaxis, :]
    # Piso pequeno no raio interno pro caso disco (`r_inner=0`) — sem
    # isso, TODOS os pontos de theta na primeira linha colapsam no exato
    # mesmo ponto (pólo degenerado de malha estruturada), o que podia
    # gerar normais/sombreamento instáveis com `smooth_shading=True`.
    r_inner_eff = r_inner if r_inner > 0 else max(r_outer * 0.04, 0.03)
    R = r_inner_eff + (Router - r_inner_eff) * Rf
    X = center_x + R * np.cos(Theta)
    Y = center_y + R * np.sin(Theta)
    center_h = float(ground_height_arr_fn(np.array([center_x]), np.array([center_y]))[0])
    raw_h = ground_height_arr_fn(X, Y)
    max_deviation_m = 0.18
    Z = center_h + np.clip(raw_h - center_h, -max_deviation_m, max_deviation_m) + 0.02

    noise = 1.0 + 0.12 * rng.normal(0, 1, X.shape)
    colors = np.clip(color[np.newaxis, np.newaxis, :] * noise[..., np.newaxis], 0, 1)

    mesh = pv.StructuredGrid(X, Y, Z)
    mesh["colors"] = colors.transpose(1, 0, 2).reshape(-1, 3)
    return mesh


def render_artistic_scene(terrain: np.ndarray, vents: List, out_path: str,
                           domain_size_m: float, local_relief_m: float, seed: int,
                           resolution: Tuple[int, int] = (2400, 1800),
                           view: Optional[Tuple[float, float, float]] = None,
                           include_fauna: bool = False) -> str:
    """
    Renderiza `out_path` (PNG) com aparência fotográfica de um campo de
    fumarolas real, a partir dos MESMOS dados procedurais/físicos
    (terreno, posição/altura de cada chaminé, altura de cada pluma) já
    calculados pelo resto do projeto — só a técnica de renderização, cor
    e iluminação mudam. Ver docstring do módulo para o porquê disso não
    ser (e não pretender ser) uma figura científica.

    `view`, se fornecido, é `(elev_deg, azim_deg, distance_mult)` para
    posicionar a câmera manualmente; por padrão usa um ângulo baixo
    "estilo ROV" olhando para o centróide das fumarolas.

    `include_fauna`: sobrepõe tapete bacteriano/leito de mexilhões
    (manchas de cor no terreno), tufos de vermes tubulares e enxames de
    camarão usando os raios REAIS de `v.fauna_zones` (calculados por
    `fauna_zonation()` em fumarola_field.py pra todo vent, nunca usados
    na cena 3D antes desta opção existir). `fumarola_field.py` chama
    esta função DUAS vezes por run quando `--artistic-render` está
    ativo (uma com, uma sem esta camada) — ver `execute_run`.
    """
    size = terrain.shape[0]
    meters_per_cell = domain_size_m / (size - 1)

    xs = np.arange(size) * meters_per_cell
    ys = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, ys)
    # A heightmap diamond-square já dá o relevo em escala de dezenas/
    # centenas de metros (vale axial, cristas) — mas nada na escala de
    # detritos/blocos soltos do fundo oceânico real (metros ou menos), por
    # isso a superfície lia como "lisa" de perto mesmo com textura de
    # rocha aplicada (uma imagem plana não desloca geometria de verdade).
    # `_terrain_micro_roughness` soma ruído de ALTA frequência espacial
    # (feições a cada ~1-4m) diretamente na elevação Z real.
    rough_rng = np.random.default_rng((seed * 3_141_592_653) % (2 ** 32 - 1))
    Z = terrain * local_relief_m + _terrain_micro_roughness(X, Y, rough_rng)

    vent_xy_m = [(v.x * meters_per_cell, v.y * meters_per_cell) for v in vents]
    max_chimney = max((v.chimney_height_m for v in vents), default=10.0)
    max_plume_top = max((v.plume_rise_m for v in vents), default=local_relief_m)

    # --- Câmera: enquadra um GRUPO COESO de fumarolas (como uma foto de
    # ROV), não o campo inteiro (domínio tipicamente ~1200m, clusters
    # espalhados aleatoriamente por ele — um enquadramento sem neblina
    # forte perderia o realismo de visibilidade subaquática real) nem
    # um `cluster_id` inteiro cru. Raio de vizinhança calibrado contra o
    # espalhamento REAL de um cluster: `cluster_spread` em
    # generate_vent_field é um desvio-padrão de ~1.5-4.5 células (~7-21m
    # pra meters_per_cell típico), então vents individuais de um mesmo
    # cluster caem até ~2-3 desvios-padrão do centro (~20-60m) — um raio
    # de 25m cortava fumarolas do MESMO cluster fora do quadro, fazendo
    # a cena ler como "só um detalhe" em vez de "um campo" (achado
    # direto do usuário: "parece só um detalhe de uma parte, não um
    # campo"). 55m captura a esmagadora maioria de um cluster inteiro.
    _PHOTO_NEIGHBORHOOD_R = 55.0
    if vents:
        all_xy = np.array([(v.x * meters_per_cell, v.y * meters_per_cell) for v in vents])
        n = len(vents)
        if n > 1:
            d2 = ((all_xy[:, None, :] - all_xy[None, :, :]) ** 2).sum(axis=2)
            neighbor_counts = (d2 <= _PHOTO_NEIGHBORHOOD_R ** 2).sum(axis=1)
            hub_idx = int(np.argmax(neighbor_counts))
            in_neighborhood = d2[hub_idx] <= _PHOTO_NEIGHBORHOOD_R ** 2
            main_vents_xy = all_xy[in_neighborhood].tolist()
            hub_vents = [v for v, keep in zip(vents, in_neighborhood) if keep]
            hub_max_chimney = max((v.chimney_height_m for v in hub_vents), default=max_chimney)
        else:
            main_vents_xy = all_xy.tolist()
            hub_vents = list(vents)
            hub_max_chimney = max_chimney
    else:
        main_vents_xy = []
        hub_vents = []
        hub_max_chimney = max_chimney

    # Estatística ROBUSTA de altura (não o máximo bruto) pra ENQUADRAR a
    # cena — achado real testando duas seeds: uma chaminé rara tipo
    # "Godzilla" (25-48m, ~6% de chance por black_smoker, ver
    # sample_chimney_height) dentro do hub inflava `cam_dist` sozinha
    # (fórmula original pesava `hub_max_chimney` com o mesmo peso do raio
    # do cluster inteiro), empurrando a câmera bem mais longe pra "caber"
    # essa única chaminé excepcional — o resultado deixava as OUTRAS
    # ~10+ chaminés típicas do mesmo hub minúsculas e mal iluminadas no
    # quadro. `hub_max_chimney` (bruto) continua usado só pra margens de
    # segurança (linha de visão da pluma, altura da luz de preenchimento),
    # não mais pra decidir o enquadramento em si.
    hub_chimney_heights = [v.chimney_height_m for v in hub_vents] or [hub_max_chimney]
    hub_typical_chimney = float(np.percentile(hub_chimney_heights, 70))

    if main_vents_xy:
        pts_xy = np.array(main_vents_xy)
        cx, cy = float(pts_xy[:, 0].mean()), float(pts_xy[:, 1].mean())
        centered = pts_xy - [cx, cy]
        cluster_radius = float(np.sqrt((centered ** 2).sum(axis=1)).max()) if len(pts_xy) > 1 else 10.0
        if len(pts_xy) >= 2:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_axis = eigvecs[:, np.argmax(eigvals)]
            azim_default = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])) + 90)
        else:
            azim_default = 35.0
    else:
        cx = cy = domain_size_m / 2
        cluster_radius = 20.0
        azim_default = 35.0
    cluster_radius = max(cluster_radius, 10.0)

    def _terrain_z_at(x_m: float, y_m: float) -> float:
        ix = int(np.clip(round(x_m / meters_per_cell), 0, size - 1))
        iy = int(np.clip(round(y_m / meters_per_cell), 0, size - 1))
        return float(Z[iy, ix])

    cz = _terrain_z_at(cx, cy)
    focal_point = (cx, cy, cz + hub_typical_chimney * 0.85)

    # Elevação/distância aumentadas (eram 16°/1.2x) — com o raio de
    # vizinhança maior acima, um cluster inteiro cabe no quadro, mas só
    # se a câmera também recuar/subir o bastante pra enxergá-lo por
    # inteiro em vez de ficar "com o nariz encostado" numa única
    # chaminé em primeiro plano. Usa `hub_typical_chimney` (percentil 70,
    # não o máximo bruto) — ver comentário acima sobre chaminés "Godzilla"
    # inflando a distância pra toda a cena.
    elev, azim, dist_mult = view if view else (18.0, azim_default, 1.3)
    cam_dist = (cluster_radius + hub_typical_chimney) * 1.2 * dist_mult
    elev_rad, azim_rad = np.radians(elev), np.radians(azim)
    cam_x = focal_point[0] + cam_dist * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = focal_point[1] + cam_dist * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = focal_point[2] + cam_dist * np.sin(elev_rad)

    # A pluma sobe bem acima da chaminé (ver plume_render_height mais
    # abaixo) — não basta a CÂMERA estar acima do terreno local; a LINHA
    # DE VISÃO até o topo da pluma precisa desobstruir qualquer crista do
    # vale axial pelo caminho, senão a fumaça fica genuinamente atrás da
    # encosta (oclusão real pelo terreno — confirmado testando com o
    # terreno desligado, a fumaça aparece perfeitamente). Faz ray-marching
    # real contra a heightmap em vez de só checar os dois extremos.
    smoke_target = (focal_point[0], focal_point[1], focal_point[2] + hub_max_chimney * 1.5 + 8.0)
    los_margin = 3.0

    def _line_of_sight_clear(p0, p1) -> bool:
        for t in np.linspace(0.05, 0.95, 14):
            px = p0[0] + (p1[0] - p0[0]) * t
            py = p0[1] + (p1[1] - p0[1]) * t
            pz = p0[2] + (p1[2] - p0[2]) * t
            if pz < _terrain_z_at(px, py) + los_margin:
                return False
        return True

    cam_z = max(cam_z, _terrain_z_at(cam_x, cam_y) + 6.0)
    for _ in range(40):
        if _line_of_sight_clear((cam_x, cam_y, cam_z), smoke_target):
            break
        cam_z += 5.0
    cam_pos = (cam_x, cam_y, cam_z)
    # `cam_dist` já cresceu (raio de vizinhança/multiplicador de câmera
    # maiores acima), então esta escala de neblina cresce proporcionalmente
    # junto — mantém a MESMA sensação de queda de luz relativa (nada
    # uniformemente meio-visível até o infinito, ainda escurece de
    # verdade), só que agora sobre uma área que corresponde ao quadro bem
    # mais aberto. Ver também o expoente em `_fog_blend` (afrouxado) pra
    # deixar meio-campo/outras chaminés visíveis em vez de sumirem de
    # repente no preto — é isso que dá a sensação de "campo" se estendendo
    # ao fundo, não só uma chaminé isolada num vazio.
    fog_scale_m = max(cam_dist * 2.2, domain_size_m * 0.12)

    def dist_from_cam(px, py, pz):
        return np.sqrt((px - cam_pos[0]) ** 2 + (py - cam_pos[1]) ** 2 + (pz - cam_pos[2]) ** 2)

    # --- Terreno: rocha escura com auréola ferruginosa ao redor de cada
    # fumarola (queda radial determinística, não uma textura pintada à
    # mão) + manchas minerais + leve ruído de albedo pra não parecer
    # plástico. `mineral_phases` sorteadas UMA vez aqui (não dentro de
    # `_terrain_base_tint`) pra poder reavaliar exatamente o mesmo campo
    # de manchas minerais no patch fino de terreno mais abaixo, sem
    # costura de cor na fronteira entre as duas malhas.
    mineral_rng = np.random.default_rng((seed * 777_777_001) % (2 ** 32 - 1))
    mineral_phases = _fractal_phases(mineral_rng, 3)
    terrain_tint = _terrain_base_tint(X, Y, vents, vent_xy_m, mineral_phases)

    terrain_rng = np.random.default_rng((seed * 2_654_435_761) % (2 ** 32 - 1))
    tint_noise = 1.0 + 0.10 * terrain_rng.normal(0, 1, Z.shape)
    terrain_tint = np.clip(terrain_tint * tint_noise[..., np.newaxis], 0, 1)

    # order='F': pv.StructuredGrid armazena os pontos em ordem Fortran
    # (column-major), diferente do reshape/ravel padrão do numpy (C-order)
    # — usar a ordem errada aqui não quebra a GEOMETRIA (X,Y,Z ficam
    # consistentes entre si de qualquer forma), mas desalinha o array de
    # COR de cada vértice, produzindo um gradiente visualmente embaralhado
    # (achado testando o gradiente da chaminé isoladamente).
    flat_dist = dist_from_cam(X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F"))
    # NÃO usar reshape(order="F") direto no array de cor (H,W,3): isso
    # entrelaça o eixo de canal na ordem de leitura Fortran e embaralha
    # R/G/B entre pixels. transpose(1,0,2)+reshape(-1,3) em C-order é o
    # jeito correto de igualar a ordem de `X.ravel(order="F")` mantendo
    # cada trinca RGB intacta (verificado numericamente à parte).
    terrain_tint_flat = _fog_blend(terrain_tint.transpose(1, 0, 2).reshape(-1, 3), flat_dist, fog_scale_m)
    terrain_uv = np.column_stack([
        X.ravel(order="F") / _ROCK_TILE_SIZE_M,
        Y.ravel(order="F") / _ROCK_TILE_SIZE_M,
    ])

    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    plotter.set_background(_BG_BOTTOM, top=_BG_TOP)

    grid = pv.StructuredGrid(X, Y, Z)
    grid.active_texture_coordinates = terrain_uv
    grid["colors"] = terrain_tint_flat
    # pbr=False (Phong clássico) é DELIBERADO, não um esquecimento: neste
    # VTK/driver, `pbr=True` ignora silenciosamente o parâmetro `texture=`
    # (só aplica os canais metallic/roughness, sem textura nenhuma —
    # confirmado isolando um plano de teste). Só o pipeline clássico
    # modula textura×cor-por-vértice corretamente, então é o único jeito
    # de ter textura real (o "mais vida"/rugosidade pedido) E manter o
    # tingimento de dados (manchas ferruginosas por proximidade real da
    # fumarola) ao mesmo tempo.
    plotter.add_mesh(grid, texture=_load_texture(_ROCK_TEXTURE_PATH, gamma=2.4),
                      scalars="colors", rgb=True, smooth_shading=True,
                      pbr=False, ambient=0.30, diffuse=1.0, specular=0.05)

    # --- Patch de terreno de alta resolução perto da câmera ---
    # A malha grossa acima (mesma resolução da heightmap científica,
    # tipicamente ~4-5m/célula) não tem vértices o bastante pra
    # representar rugosidade em escala de rocha (~1-2m) sem aliasing —
    # ver docstring de `_build_terrain_hero_patch`. Sobrepõe uma malha
    # bem mais fina só na vizinhança da câmera (onde a diferença
    # realmente aparece numa foto de perto), sempre ligeiramente mais
    # alta que a malha grossa por baixo (evita z-fighting sem precisar
    # recortar nada).
    hero_rng = np.random.default_rng((seed * 555_555_449) % (2 ** 32 - 1))
    hero_patch_radius = max(cluster_radius * 2.2, 20.0)
    # Sorteada ANTES de passar `hero_rng` adiante (que ainda vai desenhar
    # jitter/ruído de albedo dentro de `_build_terrain_hero_patch`) — ver
    # `_ground_height` mais abaixo, que reavalia esse MESMO campo de
    # ruído em pontos avulsos (base de chaminé/fauna/blocos) pra saber a
    # altura real da superfície do patch ali, não a da malha grossa por
    # baixo dele.
    hero_noise_phases = _fractal_phases(hero_rng, _FINE_ROCK_OCTAVES)
    hero_mesh = _build_terrain_hero_patch(Z, meters_per_cell, cx, cy, hero_patch_radius,
                                           vents, vent_xy_m, mineral_phases, hero_noise_phases,
                                           hero_rng, dist_from_cam, fog_scale_m)
    plotter.add_mesh(hero_mesh, texture=_load_texture(_ROCK_TEXTURE_PATH, gamma=2.4),
                      scalars="colors", rgb=True, smooth_shading=True,
                      pbr=False, ambient=0.30, diffuse=1.0, specular=0.05)

    def _ground_height(x_m: float, y_m: float) -> float:
        """Altura real do CHÃO em (x,y), incluindo o patch fino de
        terreno se `(x,y)` estiver na sua vizinhança — usada pra plantar
        chaminé/fauna/blocos na superfície de verdade em vez da malha
        grossa por baixo (que pode ficar até ~0.5m mais baixa perto da
        câmera). Ver `_hero_patch_extra_height`/docstring de
        `_build_terrain_hero_patch`."""
        coarse = _terrain_z_at(x_m, y_m)
        extra = float(_hero_patch_extra_height(
            np.array([x_m]), np.array([y_m]), cx, cy, hero_patch_radius, hero_noise_phases)[0])
        return coarse + extra

    def _ground_height_arr(Xa: np.ndarray, Ya: np.ndarray) -> np.ndarray:
        """Versão VETORIZADA de `_ground_height` (evita uma chamada
        Python por vértice) — usada pra malhas de área maior como a
        crosta de fauna (`_build_fauna_crust_mesh`), que teriam dezenas
        de vértices cada."""
        col_idx = Xa / meters_per_cell
        row_idx = Ya / meters_per_cell
        coarse = scipy.ndimage.map_coordinates(Z, [row_idx, col_idx], order=1, mode="nearest")
        extra = _hero_patch_extra_height(Xa, Ya, cx, cy, hero_patch_radius, hero_noise_phases)
        return coarse + extra

    # --- Detritos: blocos de rocha soltos ---
    # Mesmo com deslocamento de micro-relevo + textura no terreno, uma
    # superfície contínua ainda lê como "lisa" de longe — fragmentos
    # angulares SOLTOS e discretos (comuns no fundo basáltico real perto
    # de fumarolas: pedaços de crosta quebrada, blocos de talude) dão
    # silhuetas e sombras próprias que quebram essa leitura de superfície
    # única, do jeito que nenhuma textura plana consegue. Concentrados em
    # PILHAS na base de cada chaminé (padrão real: fragmentos quebrados
    # se acumulam ali, não ficam uniformemente espalhados) + um pouco de
    # detrito de fundo mais esparso pela vizinhança. Tamanho segue viés de
    # lei de potência (muitos fragmentos pequenos, poucos grandes) em vez
    # de uma faixa uniforme.
    boulder_rng = np.random.default_rng((seed * 998_244_353) % (2 ** 32 - 1))
    rock_texture = _load_texture(_ROCK_TEXTURE_PATH, gamma=2.4)

    def _random_boulder_size() -> float:
        return 0.25 + 2.3 * boulder_rng.uniform(0, 1) ** 2.2

    def _place_boulder(bx: float, by: float, b_radius: float) -> None:
        bz = _ground_height(bx, by)
        Xb, Yb, Zb, Ub, Vb = _make_boulder_mesh(bx, by, bz, b_radius, boulder_rng)
        b_dist = dist_from_cam(Xb.ravel(order="F"), Yb.ravel(order="F"), Zb.ravel(order="F"))
        b_tint = np.clip(_ROCK_TINT[np.newaxis, :]
                          * (1 + 0.12 * boulder_rng.normal(0, 1, (Xb.size, 1))), 0, 1)
        b_tint_flat = _fog_blend(b_tint, b_dist, fog_scale_m)
        b_uv = np.column_stack([Ub.ravel(order="F") * 3, Vb.ravel(order="F") * 2])
        boulder_mesh = pv.StructuredGrid(Xb, Yb, Zb)
        boulder_mesh.active_texture_coordinates = b_uv
        boulder_mesh["colors"] = b_tint_flat
        plotter.add_mesh(boulder_mesh, texture=rock_texture, scalars="colors", rgb=True,
                          smooth_shading=True, pbr=False, ambient=0.30, diffuse=1.0, specular=0.05)

    # Pilhas de talus: 2-6 blocos por chaminé do hub, raio proporcional à
    # base real da chaminé (chaminés maiores/mais antigas acumulam mais
    # material quebrado ao redor).
    for v in hub_vents:
        vx0, vy0 = v.x * meters_per_cell, v.y * meters_per_cell
        chimney_base_r = 0.6 + v.chimney_height_m * 0.16
        pile_radius = chimney_base_r * 2.5 + 1.0
        for _ in range(int(boulder_rng.integers(2, 7))):
            pr = pile_radius * boulder_rng.uniform(0, 1) ** 0.6
            pth = boulder_rng.uniform(0, 2 * np.pi)
            _place_boulder(vx0 + pr * np.cos(pth), vy0 + pr * np.sin(pth), _random_boulder_size())

    # Detrito de fundo mais esparso, espalhado por toda a vizinhança do hub.
    scatter_radius = cluster_radius * 1.6
    for _ in range(14):
        br = scatter_radius * boulder_rng.uniform(0, 1) ** 0.7
        bth = boulder_rng.uniform(0, 2 * np.pi)
        _place_boulder(cx + br * np.cos(bth), cy + br * np.sin(bth), _random_boulder_size())

    # --- Chaminés + plumas ---
    # Pontos de fumaça são coletados aqui (não desenhados no `plotter`
    # principal) e renderizados numa segunda passada separada, depois
    # compostos com desfoque gaussiano por cima da cena principal — ver
    # bloco de compositing logo após o loop de vents/neve marinha/luzes.
    smoke_draws: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for v in vents:
        vx, vy = v.x * meters_per_cell, v.y * meters_per_cell
        # `_ground_height` (não o `terrain[...]*local_relief_m` bruto de
        # antes) pra plantar a chaminé na superfície real, incluindo o
        # patch fino perto da câmera — sem isso, tufos de verme/enxames
        # de camarão (pequenos, ~0.2-0.9m) plantados na altura antiga
        # ficavam enterrados sob o patch, que pode estar ~0.5m mais alto
        # ali (achado testando a camada de fauna, chaminés em si eram
        # altas o bastante pro erro passar despercebido).
        base_z = _ground_height(vx, vy)
        rng_v = np.random.default_rng((seed * 1_000_003 + v.id * 7919 + 12345) % (2 ** 32 - 1))

        chimney_height_m = v.chimney_height_m
        if chimney_height_m > 0:
            chimney_base_r_m = 0.6 + chimney_height_m * 0.16
            Xc, Yc, Zc, Hf, Theta_over_2pi = _chimney_mesh_and_hfrac(
                vx, vy, base_z, chimney_height_m, chimney_base_r_m, rng_v)
            color_t = np.clip(Hf, 0, 1)
            type_tints = _CHIMNEY_TINTS_BY_TYPE.get(v.vent_type, _CHIMNEY_TINTS_BY_TYPE["black_smoker"])
            chim_tint = (type_tints["base"][np.newaxis, np.newaxis, :] * (1 - color_t[..., np.newaxis])
                         + type_tints["top"][np.newaxis, np.newaxis, :] * color_t[..., np.newaxis])
            chim_noise = 1.0 + 0.08 * rng_v.normal(0, 1, Hf.shape)
            chim_tint = np.clip(chim_tint * chim_noise[..., np.newaxis], 0, 1)
            chim_dist = dist_from_cam(Xc.ravel(order="F"), Yc.ravel(order="F"), Zc.ravel(order="F"))
            # Ver comentário equivalente no bloco do terreno acima sobre
            # por que transpose(1,0,2)+reshape (não reshape(order="F")
            # direto) é o jeito certo de casar a ordem Fortran de pontos
            # do StructuredGrid sem embaralhar os canais R/G/B.
            chim_tint_flat = _fog_blend(chim_tint.transpose(1, 0, 2).reshape(-1, 3), chim_dist, fog_scale_m)
            # UV: ao redor (Theta) e ao longo da altura real em metros
            # (Hf*chimney_height_m), não Hf bruto — senão o "ladrilho" da
            # textura esticaria/encolheria conforme a altura de cada
            # chaminé em vez de manter uma escala física consistente.
            n_theta_repeat = max(1, round(2 * np.pi * chimney_base_r_m / _CHIMNEY_TILE_SIZE_M))
            chim_uv = np.column_stack([
                (Theta_over_2pi.ravel(order="F")) * n_theta_repeat,
                (Hf.ravel(order="F") * chimney_height_m) / _CHIMNEY_TILE_SIZE_M,
            ])

            chim_texture_path = (_ROCK_TEXTURE_PATH if type_tints["texture_path"] == "rock"
                                  else _RUST_TEXTURE_PATH)
            chim_mesh = pv.StructuredGrid(Xc, Yc, Zc)
            chim_mesh.active_texture_coordinates = chim_uv
            chim_mesh["colors"] = chim_tint_flat
            plotter.add_mesh(chim_mesh,
                              texture=_load_texture(chim_texture_path, gamma=type_tints["texture_gamma"]),
                              scalars="colors", rgb=True, smooth_shading=True,
                              pbr=False, ambient=0.30, diffuse=1.0, specular=0.2, specular_power=8)

        chimney_top_z = base_z + chimney_height_m
        # A altura física real de ascensão da pluma (v.plume_rise_m, tipicamente
        # dezenas a ~100+m — muito maior que a chaminé) é a certa para as
        # figuras CIENTÍFICAS do projeto, mas numa foto de perto (o objetivo
        # desta renderização) só a porção próxima da chaminé é visível antes
        # de se perder na neblina de fundo — um enquadramento que capturasse
        # a pluma inteira deixaria a chaminé minúscula. Escolha artística
        # explícita: renderiza só até um teto proporcional à própria chaminé.
        plume_render_height = min(v.plume_rise_m, chimney_height_m * 0.7 + 3.0)
        plume_top_z = chimney_top_z + plume_render_height
        n_smoke = int(np.clip(30 + v.temperature_c * 0.12, 30, 90)) * 8
        smoke = _plume_points(vx, vy, chimney_top_z, plume_top_z, v.vent_type, rng_v, n_smoke)
        if smoke is not None:
            sx, sy, sz, srgb, salpha, ssizes = smoke
            sdist = dist_from_cam(sx, sy, sz)
            srgb = _fog_blend(srgb, sdist, fog_scale_m)
            # A diluição da pluma com a altura é pré-misturada na própria
            # COR rumo à cor de neblina de fundo (não opacity real — ver
            # nota grande sobre o compositing de duas passadas mais abaixo,
            # onde isto se torna alpha DE VERDADE via chroma-key, não mais
            # um substituto). Mantido aqui porque ainda contribui pra
            # "esmaecer" pontos mais altos/diluídos mesmo depois do blur.
            srgb = srgb * salpha[:, np.newaxis] + _HAZE_RGB[np.newaxis, :] * (1 - salpha[:, np.newaxis])
            # `style="points"` só aceita um point_size ÚNICO por ator (sem
            # tamanho real por-ponto nesta GPU/driver) — divide em 2 grupos
            # (fino/grosso, pelo tamanho já sorteado em `_plume_points`) e
            # guarda cada um como seu próprio desenho, pra pelo menos ter
            # DUAS escalas simultâneas em vez de uma só (mistura de
            # "flocos" grandes com névoa fina, como fumaça real). NÃO
            # desenhados no `plotter` principal agora — ver bloco de
            # compositing de fumaça mais abaixo (`smoke_draws`).
            size_cut = np.median(ssizes)
            for mask in (ssizes <= size_cut, ssizes > size_cut):
                if not np.any(mask):
                    continue
                smoke_draws.append((
                    np.column_stack([sx[mask], sy[mask], sz[mask]]),
                    srgb[mask],
                    float(np.mean(ssizes[mask])),
                ))

        # --- Fauna quimiossintética (opt-in via `include_fauna`) ---
        # Tufos de vermes tubulares e enxames de camarão, ambos com
        # raio real de `v.fauna_zones` — só existem no dict quando as
        # condições reais (H2S/tipo de vent) de `fauna_zonation()`
        # favorecem essa comunidade, então nem todo vent ganha as duas
        # (ou nenhuma) camada. Geometria/pontos OPACOS, adicionados
        # direto no `plotter` principal (sem o compositing de desfoque
        # da fumaça — não são translúcidos, são objetos discretos como
        # os blocos de rocha).
        if include_fauna:
            fz = v.fauna_zones
            # Tapete bacteriano/leito de mexilhões: malhas rasteiras
            # PRÓPRIAS sem textura fotográfica (`_build_fauna_crust_mesh`)
            # — não mais tingimento sobre a cor do terreno (essa
            # abordagem multiplicava a cor pela foto de rocha ESCURA, que
            # nunca clareia até um tom pálido vívido, só escurece mais;
            # o tapete saía cinza-escuro indistinguível de sombra comum,
            # "ou é tudo uma sombra preta" no relato direto do usuário).
            r_mat = fz.get("bacterial_mat", 0.0) * _FAUNA_COLOR_SCALE_M
            if r_mat > 0:
                mat_mesh = _build_fauna_crust_mesh(vx, vy, 0.0, r_mat, _ground_height_arr,
                                                    _BACTERIAL_MAT_TINT, rng_v)
                mat_dist = dist_from_cam(*mat_mesh.points.T)
                mat_mesh["colors"] = _fog_blend(mat_mesh["colors"], mat_dist, fog_scale_m)
                plotter.add_mesh(mat_mesh, scalars="colors", rgb=True, smooth_shading=True,
                                  pbr=False, ambient=0.38, diffuse=0.85, specular=0.05)

            r_mussel = fz.get("mussel_bed", 0.0) * _FAUNA_COLOR_SCALE_M
            if r_mussel > 0:
                band_center = r_mussel * 0.65
                band_half_width = max(r_mussel * 0.45, 0.3)
                mussel_mesh = _build_fauna_crust_mesh(
                    vx, vy, max(band_center - band_half_width, 0.0), band_center + band_half_width,
                    _ground_height_arr, _MUSSEL_BED_TINT, rng_v)
                mussel_dist = dist_from_cam(*mussel_mesh.points.T)
                mussel_mesh["colors"] = _fog_blend(mussel_mesh["colors"], mussel_dist, fog_scale_m)
                plotter.add_mesh(mussel_mesh, scalars="colors", rgb=True, smooth_shading=True,
                                  pbr=False, ambient=0.25, diffuse=0.8, specular=0.25, specular_power=12)

            # `_FAUNA_OBJECT_SCALE_M` aplicado ao RAIO DE ZONA aqui — antes
            # só as manchas de cor do terreno eram escaladas, o raio de
            # posicionamento dos tufos/camarão continuava no valor real
            # pequeno (~0.3-3m), tipicamente enterrado dentro do raio das
            # pilhas de detrito ao redor da própria chaminé (bug real por
            # trás do "com/sem fauna não faz diferença").
            tuft_radius = v.fauna_zones.get("tubeworm", 0.0) * _FAUNA_OBJECT_SCALE_M
            tuft = _tubeworm_tuft_mesh(vx, vy, _ground_height, rng_v, tuft_radius)
            if tuft is not None:
                tuft_dist = dist_from_cam(*tuft.points.T)
                tuft["colors"] = _fog_blend(tuft["colors"], tuft_dist, fog_scale_m)
                plotter.add_mesh(tuft, scalars="colors", rgb=True, smooth_shading=True,
                                  pbr=False, ambient=0.3, diffuse=0.9, specular=0.1)

            shrimp_radius = v.fauna_zones.get("shrimp_swarm", 0.0) * _FAUNA_OBJECT_SCALE_M
            shrimp = _shrimp_swarm_points(vx, vy, _ground_height_arr, rng_v, shrimp_radius)
            if shrimp is not None:
                shx, shy, shz, shcolors = shrimp
                shdist = dist_from_cam(shx, shy, shz)
                shcolors = _fog_blend(shcolors, shdist, fog_scale_m)
                shrimp_cloud = pv.PolyData(np.column_stack([shx, shy, shz]))
                shrimp_cloud["colors"] = shcolors
                plotter.add_mesh(shrimp_cloud, scalars="colors", rgb=True, style="points",
                                  render_points_as_spheres=False, point_size=3.5,
                                  opacity=1.0, lighting=False)

    # --- "Neve marinha": partículas retroespalhando a luz do "ROV",
    # amostradas a partir da MESMA seed do campo (reprodutível). Ao
    # contrário da versão anterior (uniforme por todo o domínio, a maioria
    # longe demais da câmera pra contribuir), amostra numa esfera CENTRADA
    # NA CÂMERA com densidade radial maior perto dela (r ~ uniform^0.6) —
    # é exatamente aí que o retroespalhamento real do facho de luz de um
    # ROV aparece numa foto.
    snow_rng = np.random.default_rng((seed * 40_503 + 777) % (2 ** 32 - 1))
    n_snow = 900
    snow_r = fog_scale_m * 0.8 * snow_rng.uniform(0, 1, n_snow) ** 0.6
    snow_theta = snow_rng.uniform(0, 2 * np.pi, n_snow)
    snow_phi = np.arccos(snow_rng.uniform(-1, 1, n_snow))
    snow_x = cam_pos[0] + snow_r * np.sin(snow_phi) * np.cos(snow_theta)
    snow_y = cam_pos[1] + snow_r * np.sin(snow_phi) * np.sin(snow_theta)
    snow_z = cam_pos[2] + snow_r * np.cos(snow_phi)
    snow_dist = snow_r
    # opacity=1.0 pelo mesmo motivo do bloco de fumaça acima (pontos
    # translúcidos não renderizam de forma confiável nesta GPU/driver) —
    # "fraquinho" pré-misturado na cor rumo à neblina em vez de alpha real.
    # Visibilidade cai com a distância real da câmera (mais próximo = mais
    # retroespalhamento visível), não um valor fixo pra toda partícula.
    snow_visibility = np.clip(0.55 * (1 - snow_dist / fog_scale_m), 0.03, 0.55)[:, np.newaxis]
    snow_rgb = (np.array([0.85, 0.90, 0.92])[np.newaxis, :] * snow_visibility
                + _HAZE_RGB[np.newaxis, :] * (1 - snow_visibility))
    snow_cloud = pv.PolyData(np.column_stack([snow_x, snow_y, snow_z]))
    snow_cloud["colors"] = snow_rgb
    # lighting=False: a cor por-ponto já É o resultado final (neblina/
    # queda de visibilidade por distância já pré-misturadas em Python) —
    # deixar o VTK aplicar luz de cena por cima só reintroduziria uma
    # segunda fonte de variação não controlada sobre um valor que já é
    # fisicamente motivado. Mesmo raciocínio se aplica à fumaça abaixo.
    plotter.add_mesh(snow_cloud, scalars="colors", rgb=True, style="points",
                      render_points_as_spheres=False, point_size=3.0, opacity=1.0,
                      lighting=False)

    # --- Iluminação: um "farol" posicional (câmera/ROV) + preenchimento
    # azulado fraco simulando o brilho ambiente da água profunda ---
    # Intensidade/ângulo do cone aumentados (eram 1.15/50°) — o quadro
    # ficou bem mais aberto nesta rodada (raio de vizinhança/distância de
    # câmera maiores, ver acima), então o farol precisa cobrir uma área
    # angular maior E chegar mais forte no assunto principal; com os
    # valores antigos, várias fumarolas do próprio cluster em quadro
    # ficavam praticamente invisíveis/no escuro (achado real testando
    # várias seeds depois da mudança de câmera).
    key_light = pv.Light(position=cam_pos, focal_point=focal_point, color="#eef0ee", intensity=1.9)
    key_light.positional = True
    key_light.cone_angle = 65
    plotter.add_light(key_light)

    fill_light = pv.Light(position=(focal_point[0], focal_point[1], focal_point[2] + max_chimney * 8),
                           focal_point=focal_point, color="#2f7085", intensity=0.38)
    plotter.add_light(fill_light)

    plotter.enable_shadows()
    plotter.camera_position = [cam_pos, focal_point, (0, 0, 1)]
    plotter.camera.view_angle = 40

    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
    plotter.reset_camera_clipping_range()
    base_img = plotter.screenshot(return_img=True)
    plotter.close()

    # --- Fumaça: segunda passada + desfoque gaussiano, compositado por
    # cima da cena principal ---
    # Pontos opacos sozinhos (`style="points"`, sem alpha real — ver notas
    # antigas sobre opacity<1 renderizando vazio nesta GPU quando há
    # terreno opaco na cena) davam uma nuvem de "bolinhas" nítidas e
    # artificiais, sem nenhuma impressão de neblina/difusão real. Em vez
    # de depender do pipeline de blending translúcido do VTK (já
    # confirmado instável aqui), a fumaça agora é renderizada numa cena
    # SEPARADA (só os pontos de fumaça, mesma câmera, fundo de cor-chave
    # magenta que não ocorre em nenhuma paleta real da cena), convertida
    # em máscara alfa por distância de cor até a cor-chave, desfocada
    # (gaussiana, alfa premultiplicado pra não sangrar a cor-chave nas
    # bordas) e composta sobre `base_img` em numpy/PIL puro — o desfoque
    # É o que dá a "impressão de haze" pedida, sem depender de
    # transparência real do VTK em nenhum momento.
    if smoke_draws:
        smoke_plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
        smoke_key_color = (255, 0, 255)
        smoke_plotter.set_background(tuple(c / 255.0 for c in smoke_key_color))
        for points_xyz, colors_rgb, point_size in smoke_draws:
            cloud = pv.PolyData(points_xyz)
            cloud["colors"] = colors_rgb
            smoke_plotter.add_mesh(cloud, scalars="colors", rgb=True, style="points",
                                    render_points_as_spheres=False,
                                    point_size=point_size, opacity=1.0, lighting=False)
        smoke_plotter.camera_position = [cam_pos, focal_point, (0, 0, 1)]
        smoke_plotter.camera.view_angle = 40
        smoke_plotter.reset_camera_clipping_range()
        smoke_img = smoke_plotter.screenshot(return_img=True)
        smoke_plotter.close()

        key = np.array(smoke_key_color, dtype=np.float64)
        smoke_f = smoke_img.astype(np.float64)
        # Distância de cor até a cor-chave: mais robusto que luminância
        # pura pra detectar cobertura (fumaça de black_smoker é quase
        # preta — luminância sozinha não a distinguiria do fundo escuro
        # da cena principal, mas continua bem longe do magenta puro).
        key_dist = np.linalg.norm(smoke_f - key[np.newaxis, np.newaxis, :], axis=2)
        alpha = np.clip(key_dist / _SMOKE_KEY_DIST_SCALE, 0.0, 1.0)
        # Alfa premultiplicado: remove o sangramento da cor-chave nos
        # pixels de borda anti-serrilhados (parcialmente misturados com
        # magenta pelo próprio antialiasing do VTK) ANTES de desfocar —
        # desfocar cor não-premultiplicada geraria uma franja magenta.
        alpha_safe = np.maximum(alpha, 1e-3)
        true_color = (smoke_f - key[np.newaxis, np.newaxis, :] * (1 - alpha)[..., np.newaxis]) / alpha_safe[..., np.newaxis]
        true_color = np.clip(true_color, 0, 255)
        premult = true_color * alpha[..., np.newaxis]

        sigma_px = max(1.5, resolution[0] * _SMOKE_BLUR_SIGMA_FRAC)
        premult_blur = scipy.ndimage.gaussian_filter(premult, sigma=(sigma_px, sigma_px, 0))
        alpha_blur = scipy.ndimage.gaussian_filter(alpha, sigma=sigma_px)

        base_f = base_img.astype(np.float64)
        composite = premult_blur + base_f * (1 - alpha_blur[..., np.newaxis])
        final_img = np.clip(composite, 0, 255).astype(np.uint8)
    else:
        final_img = base_img

    from PIL import Image
    Image.fromarray(final_img).save(out_path)
    return out_path

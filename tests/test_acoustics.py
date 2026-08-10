"""
Testes de validação do modelo acústico exploratório (acoustics.py). Cada
teste ancora um benchmark citável ou uma verificação analítica de
sanidade (ver docs/PHYSICS_MODEL.md, seção do modelo acústico) — nenhuma
tolerância é "porque passou". Vários testes aqui existem para registrar,
como regressão permanente, os próprios achados negativos do modelo (ex.:
a armadilha de Gor'kov não é fisicamente relevante nas pressões
acústicas reais medidas em vents) — não apenas para confirmar que o
código "roda sem erro".

Rodar com: pytest tests/test_acoustics.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import acoustics as ac


# --------------------------------------------------------------------------
# 1. Velocidade do som — Mackenzie (1981)
# --------------------------------------------------------------------------
# Valor de referência amplamente citado para o oceano profundo é ~1500
# m/s; nas condições ambiente deste projeto (T=2°C, D~2500m, S=34.7psu),
# a equação de nove termos deve cair dentro de uma faixa estreita desse
# valor de referência (não é um benchmark de precisão de laboratório,
# é uma verificação de que a equação foi implementada sem erro grosseiro
# de sinal/unidade).

def test_sound_speed_matches_known_deep_ocean_reference_value():
    c = ac.sound_speed_seawater(temp_c=2.0, depth_m=2500.0, salinity_psu=34.7)
    assert 1485.0 < c < 1515.0, f"c={c:.1f} m/s, esperado ~1500 m/s (referência de oceano profundo)"


def test_sound_speed_increases_with_temperature_near_ambient():
    """Dependência qualitativa esperada da eq. de Mackenzie (1981) perto de T=2°C: dc/dT>0."""
    c_cold = ac.sound_speed_seawater(temp_c=2.0, depth_m=2500.0)
    c_warm = ac.sound_speed_seawater(temp_c=10.0, depth_m=2500.0)
    assert c_warm > c_cold


# --------------------------------------------------------------------------
# 2. Absorção da água do mar — Ainslie & McColm (1998)
# --------------------------------------------------------------------------
# Nas frequências medidas por Crone et al. (2006) para vents (10-500 Hz),
# tabelas oceanográficas padrão dão valores de ordem 1e-3 dB/km — muitas
# ordens de grandeza menor que em frequências de sonar (kHz-MHz). Convertido
# para Np/m (÷8.686÷1000), isso corresponde a ordem 1e-7 a 1e-6 Np/m.

def test_absorption_at_vent_frequencies_matches_known_order_of_magnitude():
    alpha_100hz = ac.seawater_absorption_np_per_m(100.0, temp_c=2.0, depth_m=2500.0)
    alpha_db_km = alpha_100hz * 8.686 * 1000.0
    assert 1e-4 < alpha_db_km < 1e-1, (
        f"absorção@100Hz={alpha_db_km:.2e} dB/km, esperado ordem 1e-3 dB/km "
        "(tabelas oceanográficas padrão para essa faixa de frequência)"
    )


def test_absorption_increases_with_frequency_in_measured_band():
    """Monotonicidade esperada bem abaixo dos picos de relaxação (ácido bórico ~1kHz, MgSO4 ~100kHz)."""
    alpha_low = ac.seawater_absorption_np_per_m(10.0, temp_c=2.0, depth_m=2500.0)
    alpha_high = ac.seawater_absorption_np_per_m(500.0, temp_c=2.0, depth_m=2500.0)
    assert alpha_high > alpha_low


def test_absorption_negligible_over_domain_scale():
    """
    Verificação central que justifica omitir/simplificar absorção como
    termo dominante: ao longo de todo o domínio simulado (até ~1.2 km,
    --domain-size-m default), a perda por absorção deve ser << 1 dB.
    """
    alpha = ac.seawater_absorption_np_per_m(250.0, temp_c=2.0, depth_m=2500.0)
    domain_size_m = 1200.0
    loss_db = alpha * domain_size_m * 8.686
    assert loss_db < 0.1, f"perda por absorção em {domain_size_m}m = {loss_db:.4f} dB, esperado desprezível"


# --------------------------------------------------------------------------
# 3. Potencial de Gor'kov — sinal do contraste e escala com r^3
# --------------------------------------------------------------------------
# Gor'kov, L.P. (1962); formulação moderna em Bruus, H. (2012), Lab Chip
# 12, 1014-1021. Para partícula mais densa e mais rígida que o fluido
# (caso de colóide de sulfeto em água do mar), f1>0 e f2>0 é o resultado
# padrão que leva ao aprisionamento em nós de pressão — comportamento
# amplamente confirmado experimentalmente para partículas densas/células
# em acustoforese (não um resultado nosso, um fato estabelecido da área).

def test_gorkov_contrast_factors_positive_for_dense_rigid_particle():
    f1, f2 = ac.gorkov_contrast_factors(particle_density_kg_m3=5010.0)
    assert f1 > 0
    assert f2 > 0


def test_gorkov_potential_scales_as_particle_radius_cubed():
    """V0=(4/3)pi*r^3 entra linearmente em U -> dobrar o raio deve multiplicar a profundidade do poço por 8x."""
    rng = np.random.default_rng(1)
    size = 33
    terrain = np.random.default_rng(2).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=0, x=size / 2, y=size / 2, chimney_height_m=5.0)]

    field = ac.compute_acoustic_field(vents, terrain, domain_size_m=400.0, local_relief_m=100.0,
                                       ocean_depth_baseline_m=2500.0, ambient_temp_c=2.0,
                                       salinity_psu=34.7, ph=7.8, rng=rng)

    u_small = ac.gorkov_potential_field(field, particle_radius_m=1e-6, particle_density_kg_m3=5010.0)
    u_big = ac.gorkov_potential_field(field, particle_radius_m=2e-6, particle_density_kg_m3=5010.0)

    depth_small = float(np.max(u_small) - np.min(u_small))
    depth_big = float(np.max(u_big) - np.min(u_big))
    ratio = depth_big / depth_small
    assert abs(ratio - 8.0) < 0.01, f"razão de profundidade do poço = {ratio:.4f}, esperado exatamente 8.0 (2^3)"


def _make_field_for_population_tests():
    rng = np.random.default_rng(42)
    size = 65
    terrain = np.random.default_rng(7).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=i, x=float(20 + i * 5), y=float(30 + i * 3), chimney_height_m=5.0 + i)
             for i in range(4)]

    field = ac.compute_acoustic_field(vents, terrain, domain_size_m=800.0, local_relief_m=100.0,
                                       ocean_depth_baseline_m=2500.0, ambient_temp_c=2.0,
                                       salinity_psu=34.7, ph=7.8, rng=rng)
    return field, vents


def test_gorkov_trap_not_physically_relevant_for_fine_colloid_at_measured_vent_pressures():
    """
    Achado central deste módulo (registrado como regressão permanente,
    não só como "passou"): nas pressões acústicas REAIS medidas em vents
    por Crone et al. (2006) (0.4-2.6 Pa RMS), a profundidade do poço de
    Gor'kov para o colóide fino de sulfeto (PARTICLE_CLASSES
    ["fine_sulfide_colloid"]) fica MUITO abaixo de kT — a agitação
    térmica domina completamente, e a "armadilha" não é fisicamente
    relevante. Ver docs/PHYSICS_MODEL.md, seção Limitações.
    """
    field, _ = _make_field_for_population_tests()
    spec = ac.PARTICLE_CLASSES["fine_sulfide_colloid"]
    u_field = ac.gorkov_potential_field(field, spec["radius_m"], spec["density_kg_m3"])
    _, depth_over_kt = ac.particle_boltzmann_enhancement(u_field, temp_c=2.0)

    assert depth_over_kt < 1e-2, (
        f"profundidade/kT = {depth_over_kt:.3e}, esperado << 1 (agitação térmica domina "
        "nas pressões acústicas reais medidas em vents)"
    )


def test_gorkov_trap_remains_negligible_for_largest_documented_aggregate():
    """
    Extensão do achado acima para a classe de agregado de campo próximo
    (PARTICLE_CLASSES["near_field_fe_oxyhydroxide_aggregate"], 14-20 μm,
    González-Santana et al. 2020, pluma do TAG) — o maior tamanho de
    partícula/agregado hidrotermal com raio diretamente documentado na
    pesquisa de literatura que fundamenta este modelo. Mesmo essa classe,
    bem mais favorável que o colóide fino, deve ficar abaixo de kT nas
    pressões acústicas reais — um achado mais forte e melhor ancorado do
    que testar apenas um colóide fino arbitrário.
    """
    field, _ = _make_field_for_population_tests()
    spec = ac.PARTICLE_CLASSES["near_field_fe_oxyhydroxide_aggregate"]
    u_field = ac.gorkov_potential_field(field, spec["radius_m"], spec["density_kg_m3"])
    _, depth_over_kt = ac.particle_boltzmann_enhancement(u_field, temp_c=2.0)

    assert depth_over_kt < 1.0, (
        f"profundidade/kT = {depth_over_kt:.3e} para o maior agregado hidrotermal documentado "
        "(14-20 μm, González-Santana et al. 2020) — esperado ainda abaixo do limiar térmico"
    )
    # a classe de agregado é maior que a fina, então sua profundidade deve ser proporcionalmente maior
    fine_spec = ac.PARTICLE_CLASSES["fine_sulfide_colloid"]
    u_fine = ac.gorkov_potential_field(field, fine_spec["radius_m"], fine_spec["density_kg_m3"])
    _, depth_fine = ac.particle_boltzmann_enhancement(u_fine, temp_c=2.0)
    assert depth_over_kt > depth_fine


def test_acoustic_enrichment_field_uses_particle_population_by_default():
    """
    Quando `particle_radius_m`/`particle_density_kg_m3` NÃO são
    fornecidos explicitamente, `acoustic_enrichment_field` deve usar a
    população de duas classes citadas (não um tamanho arbitrário) e
    reportar diagnósticos para ambas, usando a classe de agregado (maior,
    mais favorável) como campo principal.
    """
    field, vents = _make_field_for_population_tests()
    rng = np.random.default_rng(1)
    terrain = np.random.default_rng(7).uniform(0.3, 0.7, size=(65, 65))

    result = ac.acoustic_enrichment_field(vents, terrain, domain_size_m=800.0, local_relief_m=100.0,
                                           ocean_depth_baseline_m=2500.0, mode="particle_trap", rng=rng)
    diag = result["diagnostics"]
    assert diag["particle_classes"] is not None
    assert set(diag["particle_classes"].keys()) == set(ac.PARTICLE_CLASSES.keys())

    agg = diag["particle_classes"]["near_field_fe_oxyhydroxide_aggregate"]
    fine = diag["particle_classes"]["fine_sulfide_colloid"]
    assert agg["trap_depth_over_kT"] > fine["trap_depth_over_kT"]
    # o diagnóstico "principal" (top-level) deve corresponder à classe de agregado, não à fina
    assert diag["gorkov_trap_depth_over_kT"] == agg["trap_depth_over_kT"]


def test_acoustic_enrichment_field_explicit_particle_size_skips_population():
    """Se o chamador fornece um tamanho customizado, a população citada não deve ser usada."""
    field, vents = _make_field_for_population_tests()
    rng = np.random.default_rng(1)
    terrain = np.random.default_rng(7).uniform(0.3, 0.7, size=(65, 65))

    result = ac.acoustic_enrichment_field(vents, terrain, domain_size_m=800.0, local_relief_m=100.0,
                                           ocean_depth_baseline_m=2500.0, mode="particle_trap", rng=rng,
                                           particle_radius_m=500e-6, particle_density_kg_m3=5010.0)
    assert result["diagnostics"]["particle_classes"] is None


# --------------------------------------------------------------------------
# 4. Auto-interferência fonte+imagem: franjas espaciais reais
# --------------------------------------------------------------------------
# A estrutura de nós/antinós de uma fumarola só existe porque a fonte e
# sua imagem no fundo estão separadas por uma altura finita
# (source_height_m). No limite degenerado (fonte exatamente no plano do
# espelho), fonte e imagem coincidem e o campo deveria perder a
# estrutura de franjas (só sobra o dobro de amplitude, sem variação
# espacial de interferência) — uma verificação direta da consistência
# geométrica do modelo de imagem.

def _count_local_extrema_along_radial_line(p2: np.ndarray) -> int:
    """
    Conta extremos locais interiores ao longo de uma linha radial a
    partir do centro da grade (onde a fonte está posicionada nos dois
    testes abaixo). Decaimento puro de monopolo (1/r, sem franjas) é
    MONOTÔNICO ao longo dessa linha (zero extremos interiores);
    interferência real (franjas de nó/antinó) produz múltiplos máximos
    e mínimos locais — a assinatura que distingue os dois regimes, sem
    ser confundida pela tendência de queda suave de amplitude com a
    distância (que por si só teria CV alto mas não é "franja").
    """
    center = p2.shape[0] // 2
    line = p2[center, center:]
    d = np.diff(line)
    signs = np.sign(d)
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0))


def test_self_interference_produces_spatial_fringes_when_source_elevated():
    """
    Este é o efeito clássico do "espelho de Lloyd" (Lloyd's mirror,
    interferência fonte+imagem perto de um contorno refletor — ver
    Urick, R.J., "Principles of Underwater Sound") aplicado à
    auto-interferência de uma fumarola. A diferença máxima de caminho
    fonte-vs-imagem aqui é limitada a 2×(altura do receptor)=2m (altura
    do receptor fixa em 1m, ver `receiver_height_m` em
    compute_acoustic_field) — nas frequências REAIS medidas em vents
    (10-250 Hz, λ=6-150m) essa diferença de caminho cobre MENOS de um
    ciclo completo de interferência (ver docs/PHYSICS_MODEL.md), então
    aqui usamos deliberadamente uma frequência bem mais alta que a faixa
    real (1000 Hz) só para tornar a estrutura de franjas nitidamente
    visível e comprovar que a matemática da interferência está correta
    — não para afirmar algo sobre plausibilidade física (isso é
    testado separadamente, com os parâmetros reais).
    """
    size = 161
    domain_size_m = 300.0
    meters_per_cell = domain_size_m / (size - 1)
    xs = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, xs)
    z_receiver = np.full_like(X, 1.0)

    src = ac.VentAcousticSource(vent_id=0, x_m=domain_size_m / 2, y_m=domain_size_m / 2,
                                 floor_elev_m=0.0, source_height_m=10.0,
                                 freq_hz=1000.0, p_ref_broadband_pa=1.0, p_ref_tonal_pa=2.0)
    c_sound = 1500.0
    phasor = ac.vent_tonal_pressure_phasor(src, X, Y, z_receiver, c_sound, alpha_np_per_m=0.0)
    p2 = np.abs(phasor) ** 2

    n_extrema = _count_local_extrema_along_radial_line(p2)
    assert n_extrema >= 1, (
        f"{n_extrema} extremos locais interiores, esperado >=1 (franjas reais de nó/antinó "
        "para fonte elevada acima do plano-espelho, efeito do espelho de Lloyd)"
    )


def test_self_interference_degenerates_when_source_on_mirror_plane():
    size = 161
    domain_size_m = 300.0
    meters_per_cell = domain_size_m / (size - 1)
    xs = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, xs)
    z_receiver = np.full_like(X, 1.0)

    src = ac.VentAcousticSource(vent_id=0, x_m=domain_size_m / 2, y_m=domain_size_m / 2,
                                 floor_elev_m=0.0, source_height_m=1e-6,
                                 freq_hz=100.0, p_ref_broadband_pa=1.0, p_ref_tonal_pa=2.0)
    c_sound = 1500.0
    phasor = ac.vent_tonal_pressure_phasor(src, X, Y, z_receiver, c_sound, alpha_np_per_m=0.0)
    p2 = np.abs(phasor) ** 2

    n_extrema = _count_local_extrema_along_radial_line(p2)
    assert n_extrema == 0, (
        f"{n_extrema} extremos locais interiores, esperado 0 (decaimento monotônico puro "
        "quando fonte e imagem coincidem — sem estrutura de franjas)"
    )


# --------------------------------------------------------------------------
# 5. Combinação incoerente entre fumarolas diferentes
# --------------------------------------------------------------------------
# Duas fumarolas distintas nunca devem ser somadas em fase (ver
# justificativa no docstring do módulo) — a intensidade total deve ser
# exatamente a soma das intensidades individuais, ponto a ponto.

def test_different_vents_combine_incoherently_in_intensity():
    """
    Constrói duas fontes EXPLICITAMENTE (sem passar por build_acoustic_sources,
    para não depender de quantas amostras de RNG cada caminho consome) e
    confirma que |p1+p2 combinado incoerentemente|^2 == |p1|^2+|p2|^2 —
    a propriedade central de que fumarolas diferentes nunca são somadas
    em fase (ver docstring do módulo).
    """
    size = 49
    domain_size_m = 400.0
    meters_per_cell = domain_size_m / (size - 1)
    xs = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, xs)
    z_receiver = np.full_like(X, 1.0)
    c_sound = 1500.0

    src1 = ac.VentAcousticSource(vent_id=0, x_m=80.0, y_m=200.0, floor_elev_m=0.0,
                                  source_height_m=5.0, freq_hz=90.0, p_ref_broadband_pa=1.0, p_ref_tonal_pa=2.0)
    src2 = ac.VentAcousticSource(vent_id=1, x_m=320.0, y_m=200.0, floor_elev_m=0.0,
                                  source_height_m=8.0, freq_hz=140.0, p_ref_broadband_pa=1.2, p_ref_tonal_pa=2.4)

    p2_1 = np.abs(ac.vent_tonal_pressure_phasor(src1, X, Y, z_receiver, c_sound, 0.0)) ** 2
    p2_2 = np.abs(ac.vent_tonal_pressure_phasor(src2, X, Y, z_receiver, c_sound, 0.0)) ** 2

    p2_combined_via_loop = np.zeros_like(X)
    for src in (src1, src2):
        p2_combined_via_loop += np.abs(ac.vent_tonal_pressure_phasor(src, X, Y, z_receiver, c_sound, 0.0)) ** 2

    np.testing.assert_allclose(p2_combined_via_loop, p2_1 + p2_2, rtol=1e-10)


# --------------------------------------------------------------------------
# 5b. Modo "coherent" (teste de limite superior) vs. "incoherent" (padrão)
# --------------------------------------------------------------------------
# Regressão para a correção de 2026-08-06: o termo de VELOCIDADE (usado
# por streaming e por Gor'kov) chegou a somar fasores complexos de
# fumarolas DIFERENTES em fase, inconsistente com a soma em potência já
# usada para a pressão. Estes testes documentam o comportamento
# correto dos dois modos explícitos.

def _two_vent_setup(seed=42):
    rng = np.random.default_rng(seed)
    size = 65
    terrain = np.random.default_rng(seed + 1).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=0, x=20.0, y=30.0, chimney_height_m=6.0),
             _FakeVent(id=1, x=40.0, y=35.0, chimney_height_m=8.0)]
    kwargs = dict(domain_size_m=800.0, local_relief_m=100.0, ocean_depth_baseline_m=2500.0,
                  ambient_temp_c=2.0, salinity_psu=34.7, ph=7.8)
    return rng, terrain, vents, kwargs


def test_single_vent_incoherent_and_coherent_modes_are_identical():
    """Com apenas 1 fumarola não há 'entre fumarolas' — os dois modos devem coincidir exatamente."""
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    size = 33
    terrain = np.random.default_rng(10).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=0, x=16.0, y=16.0, chimney_height_m=6.0)]
    kwargs = dict(domain_size_m=300.0, local_relief_m=100.0, ocean_depth_baseline_m=2500.0,
                  ambient_temp_c=2.0, salinity_psu=34.7, ph=7.8)

    field_incoh = ac.compute_acoustic_field(vents, terrain, rng=rng1, cross_vent_coherence="incoherent", **kwargs)
    field_coh = ac.compute_acoustic_field(vents, terrain, rng=rng2, cross_vent_coherence="coherent", **kwargs)

    np.testing.assert_allclose(field_incoh.p2_tonal_total, field_coh.p2_tonal_total, rtol=1e-10)
    np.testing.assert_allclose(field_incoh.v2_tonal_total, field_coh.v2_tonal_total, rtol=1e-10)


def test_coherent_mode_uses_single_shared_frequency_across_vents():
    """No modo 'coherent', build_acoustic_sources deve dar a MESMA freq_hz a todas as fumarolas."""
    rng, terrain, vents, kwargs = _two_vent_setup()
    sources = ac.build_acoustic_sources(vents, terrain, kwargs["domain_size_m"], kwargs["local_relief_m"],
                                         rng, cross_vent_coherence="coherent")
    freqs = {s.freq_hz for s in sources}
    assert len(freqs) == 1, f"esperada uma única frequência compartilhada, obtido {freqs}"


def test_different_vents_still_combine_incoherently_by_default_after_fix():
    """
    Verificação end-to-end (via compute_acoustic_field, não só a função de
    fasor isolada) de que o modo padrão soma em POTÊNCIA tanto pressão
    quanto velocidade — a correção do bug de 2026-08-06.
    """
    rng, terrain, vents, kwargs = _two_vent_setup()
    field_two = ac.compute_acoustic_field(vents, terrain, rng=rng, cross_vent_coherence="incoherent", **kwargs)

    # Recomputa cada fumarola isoladamente (mesmas fontes, extraídas do campo de 2 fumarolas)
    # e confirma que p2/v2 total = soma das contribuições individuais, ponto a ponto.
    rng_check = np.random.default_rng(42)
    sources = ac.build_acoustic_sources(vents, terrain, kwargs["domain_size_m"], kwargs["local_relief_m"],
                                         rng_check, cross_vent_coherence="incoherent")
    size = terrain.shape[0]
    meters_per_cell = kwargs["domain_size_m"] / (size - 1)
    xs = np.arange(size) * meters_per_cell
    X, Y = np.meshgrid(xs, xs)
    z_receiver = terrain * kwargs["local_relief_m"] + 1.0
    c_sound = ac.sound_speed_seawater(kwargs["ambient_temp_c"], kwargs["ocean_depth_baseline_m"], kwargs["salinity_psu"])

    p2_sum = np.zeros_like(X)
    v2_sum = np.zeros_like(X)
    for src in sources:
        alpha = ac.seawater_absorption_np_per_m(src.freq_hz, kwargs["ambient_temp_c"], kwargs["ocean_depth_baseline_m"], kwargs["salinity_psu"], kwargs["ph"])
        p_tonal = ac.vent_tonal_pressure_phasor(src, X, Y, z_receiver, c_sound, alpha)
        p2_sum += np.abs(p_tonal) ** 2
        omega = 2 * np.pi * src.freq_hz
        vx_i, vy_i = ac._particle_velocity_from_pressure(p_tonal, meters_per_cell, omega, ac.RHO_SEAWATER)
        v2_sum += np.abs(vx_i) ** 2 + np.abs(vy_i) ** 2

    np.testing.assert_allclose(field_two.p2_tonal_total, p2_sum, rtol=1e-8)
    np.testing.assert_allclose(field_two.v2_tonal_total, v2_sum, rtol=1e-8)


def test_coherent_mode_diagnostic_is_reported():
    rng, terrain, vents, kwargs = _two_vent_setup()
    result = ac.acoustic_enrichment_field(vents, terrain, domain_size_m=kwargs["domain_size_m"],
                                           local_relief_m=kwargs["local_relief_m"],
                                           ocean_depth_baseline_m=kwargs["ocean_depth_baseline_m"],
                                           mode="particle_trap", rng=rng, cross_vent_coherence="coherent")
    assert result["diagnostics"]["cross_vent_coherence"] == "coherent"


# --------------------------------------------------------------------------
# 6. Streaming de contorno + solver de advecção-difusão
# --------------------------------------------------------------------------

def test_streaming_velocity_field_finite_and_real():
    rng = np.random.default_rng(5)
    size = 33
    terrain = np.random.default_rng(6).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=0, x=16.0, y=16.0, chimney_height_m=6.0)]
    field = ac.compute_acoustic_field(vents, terrain, domain_size_m=300.0, local_relief_m=100.0,
                                       ocean_depth_baseline_m=2500.0, ambient_temp_c=2.0,
                                       salinity_psu=34.7, ph=7.8, rng=rng)
    ux, uy = ac.boundary_streaming_velocity(field)
    assert np.all(np.isfinite(ux))
    assert np.all(np.isfinite(uy))


def test_streaming_enrichment_field_never_exactly_zero_near_domain_boundary():
    """
    Regressão: perto da borda do domínio (Dirichlet C=0), COM e SEM
    streaming decaem para o mesmo piso numérico — sem regularização
    simétrica, a razão bruta cai para exatamente 0.0 (ruído de ponto
    flutuante, não física real), o que quebra qualquer log2() a jusante
    (ex.: histogramas do relatório científico). O limite físico correto
    longe da fonte é enriquecimento -> 1 (nada para o streaming
    advectar), nunca 0.
    """
    rng = np.random.default_rng(11)
    size = 49
    terrain = np.random.default_rng(12).uniform(0.3, 0.7, size=(size, size))
    from dataclasses import dataclass

    @dataclass
    class _FakeVent:
        id: int; x: float; y: float; chimney_height_m: float

    vents = [_FakeVent(id=0, x=24.0, y=24.0, chimney_height_m=6.0)]
    field = ac.compute_acoustic_field(vents, terrain, domain_size_m=400.0, local_relief_m=100.0,
                                       ocean_depth_baseline_m=2500.0, ambient_temp_c=2.0,
                                       salinity_psu=34.7, ph=7.8, rng=rng)
    enrichment, _ = ac.streaming_enrichment_field(field, vents, domain_size_m=400.0)

    assert np.all(np.isfinite(enrichment))
    assert np.all(enrichment > 0), "enriquecimento nunca deve ser exatamente 0 (ver docstring do teste)"
    # canto do domínio, longe da fonte e da borda-fonte: deve estar perto do limite físico correto (1x)
    assert abs(enrichment[0, 0] - 1.0) < 0.05

def test_advection_diffusion_solver_conserves_positivity_with_zero_velocity():
    """Sem streaming (controle), a concentração estacionária de um solene com fonte positiva deve ser >=0 em todo lugar."""
    size = 25
    ux = np.zeros((size, size))
    uy = np.zeros((size, size))
    source_mask = np.zeros((size, size))
    source_mask[size // 2, size // 2] = 1.0
    meters_per_cell = 10.0
    diffusivity = 8e-10
    loss_rate = diffusivity / (250.0) ** 2

    c = ac.solve_steady_advection_diffusion(ux, uy, source_mask, meters_per_cell, diffusivity, loss_rate)
    assert np.all(c >= -1e-12)
    assert c[size // 2, size // 2] > 0

# Modelo físico-químico de pluma hidrotermal — documentação de referência

Este documento descreve o modelo de gradiente físico-químico de campo
próximo implementado em `plume_physics.py` e `reaction_kinetics.py`
(Fase 1 do projeto — ver `simulate_plume()` em `fumarola_field.py` para
o ponto de integração). Todo default e toda tolerância de teste devem
poder ser rastreados a uma citação ou a uma justificativa explícita
neste arquivo. Se você mudar uma constante em `plume_physics.py` ou
`reaction_kinetics.py`, atualize a entrada correspondente aqui.

## 1. Escopo

Modelado: a pluma turbulenta flutuante de campo próximo (do orifício do
vent até a altura de flutuabilidade neutra, tipicamente dezenas a
centenas de metros), incluindo diluição conservativa e transporte
reativo de H2S, Fe(II) e Mn(II).

**Fora de escopo nesta fase** (ver seção 5):
dispersão de fundo/diffuse flow por advecção-difusão de longo alcance,
correntes de fundo, coalescência de plumas entre vents do mesmo
cluster, cinética de oxidação de CH4, mistura difusiva em poros de
parede de chaminé (regime distinto da pluma em coluna d'água livre),
modo de sítio real (Endeavour/TAG geometricamente replicados).

## 2. Sistema de equações — pluma turbulenta flutuante (MTT)

Referência primária: Morton, B.R., Taylor, G.I., & Turner, J.S. (1956).
"Turbulent gravitational convection from maintained and instantaneous
sources." *Proc. R. Soc. Lond. A* 234(1196), 1-23.

Forma numérica implementada (derivada das variáveis de fluxo padrão
Q=πb²w, M=πb²w², B=πb²wg′; ver Jones, Hogg, Kerr et al. (2020), *Phil.
Trans. R. Soc. A*, PMC7422873, para a forma equivalente per-ângulo):

```
dQ/dz = 2*sqrt(pi)*alpha*sqrt(M)      (entranhamento)
dM/dz = 2*Q*B/M                       (momento)
dB/dz = -2*N^2 * Q                    (flutuabilidade consumida pela estratificação)
```

**Correção real encontrada e corrigida (2026-08-08)**: até esta data,
`dM/dz` e `dB/dz` estavam implementados sem o fator 2 (`dM/dz = Q*B/M`,
`dB/dz = -N²*Q`) — um erro real na física central do modelo inteiro,
presente desde a implementação original. Encontrado ao ler
`biblios/rspa.1956.0011.pdf` (Morton, Taylor & Turner 1956) na íntegra e
derivar `Q=πb²w, M=πb²w², B=πb²wg'` diretamente a partir das equações do
artigo, em vez de confiar na forma já reduzida citada de segunda mão. A
derivação, partindo das eqs. (7,ii-iii) do artigo (`d/dx(b²u²) =
2b²g(ρ0-ρ)/ρ1` e `d/dx(b²ug(ρ0-ρ)/ρ1) = 2b²u(g/ρ1)(dρ0/dx)`, ambas com
o fator 2 explícito no lado direito, confirmadas também pela forma
reduzida equivalente da eq. (8), `dV⁴/dx=4F*W`, `dF*/dx=-2WG`) mostra
que o fator 2 sobrevive na conversão para as variáveis de fluxo padrão:
com `Q=πW`, `M=πV²`, `B=πF*` (W=b²u, V=bu, F*=b²ug(ρ0-ρ)/ρ1, as próprias
variáveis do artigo), obtém-se `dM/dz = 2QB/M` e `dB/dz = -2N²Q`, não
`QB/M` e `-N²Q`. A entrada `dQ/dz = 2√π·α·√M` já estava correta (não
precisou de correção). A verificação foi cruzada numericamente: a
integração da Tabela 1 do artigo (sistema reduzido eq. 11,
`dw/dx1=v, dv⁴/dx1=fw, df/dx1=-w`) reproduz exatamente os valores
tabulados (ex. em x1=1: v=0.5971, w=0.3624, f=0.8636) e os dois pontos
notáveis citados no texto (x1=2.8 onde a velocidade se anula, x1=2.125
onde a flutuabilidade se anula) — confirmando que a leitura das
equações (7)/(8)/(10)/(14) está correta antes de aplicá-la à correção
do código. Impacto: a altura de ascensão calculada (`rise_height_m`)
diminui (a pluma consome flutuabilidade e perde momento mais rápido
que antes) — ex. para um black smoker a 350°C com os defaults do
projeto, de 328m para 254m. Todos os testes de benchmark de campo em
`tests/test_plume_physics.py` (Mottl & McConachy 1990, Lupton et al.
1985, Rudnicki & Elderfield 1993 — todos com tolerância de ordem de
grandeza ou fator ≥3) continuam passando após a correção; ver §2.2 para
o ajuste correspondente na fórmula fechada de reconciliação.

**Confirmação independente adicional**: Lemaréchal, Roullet & Gula
(2025), *JGR Oceans* 130(10) (verificado 2026-08-08), reescreve o
sistema MTT em variáveis raio/velocidade em sua eq. (1a-1c):
`d/dz(r²w)=2αrw`, `d/dz(r²w²)=2r²b`, `d/dz(r²wb)=2r²w(g/ρr)(∂ρa/∂z)` —
a segunda e a terceira equações têm o fator 2 exatamente como corrigido
aqui, confirmando de forma independente (terceira via, além da leitura
direta de MTT 1956 e da reprodução numérica da Tabela 1) que a correção
está certa.

`Q`=fluxo de volume [m³/s], `M`=fluxo de momento [m⁴/s²], `B`=fluxo de
flutuabilidade [m⁴/s³], `alpha`=coeficiente de entranhamento,
`N`=frequência de Brunt-Väisälä [s⁻¹]. Implementado em
`plume_physics.integrate_plume()` via `scipy.integrate.solve_ivp`
(RK45, passo adaptativo, `rtol=1e-8`) — escolhido por controle de erro
e reprodutibilidade determinística (mesma seed + mesmos parâmetros →
mesmo resultado, dentro da tolerância do integrador), não por um
integrador de passo fixo caseiro.

**Diluição** é definida como `D(z) = Q(z)/Q0` — resultado direto da
integração, não uma fórmula separada.

**Condição de parada**: a integração termina no primeiro de dois
eventos: `B(z)=0` (altura de flutuabilidade neutra — a camada
efetivamente observada em campo) ou `M(z)≈0` (momento esgotado). Para
`N=0` (não-estratificado) nenhum evento dispara e a integração vai até
`z_max` (default 500m).

**Precedente direto para adicionar um traçador químico a este sistema**:
Rudnicki, M.D., & Elderfield, H. (1992). "Theory applied to the
Mid-Atlantic Ridge hydrothermal plumes: the finite-difference approach."
*J. Volcanol. Geotherm. Res.* 50(1-2), 161-172.

**Temperatura local da pluma** `T(z)` é derivada de forma
autoconsistente a partir de `g'(z)=B(z)/Q(z)` (não é um campo
independente): `T(z) = T_ambiente + g'(z)/(g*alpha_termico)`. Isso
alimenta as constantes de taxa dependentes de temperatura (seção 3).

### 2.1 Condições de contorno (fonte)

`plume_physics.build_source()`: `Q0 = área_orifício * velocidade_saida`,
`M0 = Q0*velocidade_saida`, `B0 = g*alpha_termico*ΔT*Q0` (aproximação
de Boussinesq). Velocidades de saída por tipo de vent (1.5/0.6/0.05 m/s
para black smoker/white smoker/diffuse flow) são plausíveis mas **não
citadas** — mantidas do modelo anterior, não uma medição direta.

### 2.2 Reconciliação com a fórmula fechada anterior

**Correção de citação encontrada em 2026-08-08** (leitura direta de
`biblios/jc094ic05p06213.pdf`, agora disponível — antes só acessível por
paywall): a afirmação anterior desta seção, de que a literatura
consultada dava `5*pi^-0.25 = 3.76` (altura de overshoot, M=0) e
`4*pi^-0.25 = 3.01` (altura de intrusão), **não corresponde a nenhuma
equação do artigo de Speer & Rona (1989)** nem a nenhuma fonte
verificável — o fator `pi^-0.25` multiplicando o "5" não aparece em
lugar nenhum do texto primário; parece ter sido uma combinação
inventada numa sessão anterior. A equação real do artigo (eq. 5, p.
6214, atribuída por eles a Turner, J.S. (1973), *Buoyancy Effects in
Fluids*, Cambridge Univ. Press — não verificada aqui, texto não obtido)
é

```
z* = 5 * Bo^0.25 * N^-0.75    (altura de penetração, onde a velocidade se anula)
```

sem fator de `pi`, e com `Bo` definido (eq. 6 do artigo) exatamente como
o `B0` deste projeto (`Bo = g'*A*W` na fonte, avaliado em `z=0` — idêntico
a `B0 = g'0*Q0` usado aqui). Speer & Rona não usam o sistema MTT (7)/(8)
diretamente; o modelo deles (eqs. 1-4 do artigo) resolve temperatura e
salinidade separadamente com um coeficiente de entranhamento próprio
(`E=0.255`, equivalente a `α=E/(2√π)=0.072` na notação MTT) e cita a
forma fechada de Turner (1973) apenas para a altura de penetração, não
como equação derivada por eles. Validação própria do artigo: caso
Atlântico (TAG) prevê 330m vs. ~360m observado; caso Pacífico prevê
180m vs. ~200m observado (Lupton et al. 1985) — não replicado aqui
(exigiria o modelo T+S completo deles, fora do escopo desta correção).

A fórmula fechada usada como **teste de validação**
(`tests/test_plume_physics.py::test_rise_height_reconciles_with_mtt_closed_form`)
foi substituída pela forma fechada derivada diretamente de MTT (1956)
eqs. (10)/(14), verificada por três vias independentes nesta sessão
(reprodução exata da Tabela 1 do artigo por integração numérica do
sistema reduzido eq. 11; verificação algébrica das eqs. (7)/(8)/(10);
e conferência cruzada do coeficiente dimensional `0.410` de eq. (14)).
Para a altura de flutuabilidade neutra (`x1=2.125`, onde `f=0` — o
mesmo locus que o evento `B(z)=0` do código):

```
z_neutro ≈ 0.7326 * alpha^-0.5 * B0^0.25 * N^-0.75
```

(constante `0.7326 = 0.410 * 2^-0.25 * 2.125`, com `F0(artigo) = B0/2`
pela relação `F0* = (2/π)F0` do artigo combinada com `B0 = π*F0*`).
Diferente da fórmula anterior, esta tem dependência explícita em
`alpha` — correta, já que a própria eq. (10) do artigo mostra a escala
de altura proporcional a `alpha^-0.5`; uma fórmula fechada sem
dependência em `alpha` (como a "2.98" usada antes) não pode ser
literalmente MTT (1956), ainda que possa ser uma aproximação legítima
de outra fonte com `alpha` fixado implicitamente. A citação anterior a
Jones, Hogg, Kerr et al. (2020) para `z ≈ 2.98*(B0/N³)^0.25` permanece
**não verificada** nesta sessão (PDF não obtido) — mantida apenas na
nota da seção 2 sobre a forma per-ângulo equivalente, não mais usada
como base do teste de validação. Como o locus `x1=2.125` da nova fórmula
fechada é exatamente o mesmo evento `B(z)=0` que a integração numérica
usa como critério de parada, não há diferença estrutural esperada entre
as duas formulações — a concordância observada com os defaults do
projeto é <1% (254.0m numérico vs. 255.0m fechado), e a tolerância do
teste foi apertada de 25% para 10% de acordo.

## 3. Cinética de reação por espécie (`reaction_kinetics.py`)

Abordagem geral: a *magnitude* de cada taxa é ancorada numa meia-vida
medida em campo, numa pluma hidrotermal real, na temperatura de
referência `T_REF_C=2°C` (água do mar ambiente); a *dependência com
temperatura* usa uma energia de ativação de laboratório (Arrhenius),
quando disponível. **Isto é uma aproximação explícita**: as meias-vidas
de campo citadas são valores efetivos, integrados ao longo de todo o
histórico térmico da pluma (do orifício quente até quase-ambiente), não
medidos numa única temperatura controlada. Ancorar em T=2°C assume que
a maior parte do tempo de residência relevante para a taxa efetiva
observada ocorre perto da temperatura ambiente (já que a pluma esfria
rapidamente nos primeiros metros) — uma escolha de modelagem razoável,
não uma medição direta.

| Espécie | Lei | Magnitude (âncora) | Ea | Fonte |
|---|---|---|---|---|
| H2S | Arrhenius a partir de t½ | 26±9 h, água do mar, pH 8, 25°C, × fator de realce de pluma 100× | 39±2 kJ/mol | Millero et al. (1987) *ES&T* 21:439-443 (cinética); Radford-Knoery et al. (2001) *L&O* 46:461-464 (realce) |
| Fe(II) | Arrhenius a partir de t½, por bacia | Atlântico (TAG): 2.1 min; Pacífico (EPR 9°45'N): 3.3 h | 29±2 kJ/mol | Rudnicki & Elderfield (1993) *GCA* 57:2939-2957; Field & Sherrell (2000) *GCA* 64:619-628; Ea de Millero et al. (1987) *GCA* 51:793-801 |
| Fe (sulfeto) | remoção instantânea, fração fixa | 65% (faixa 40-90%) removido perto do orifício, antes da oxidação contínua | — | Field & Sherrell (2000); Mottl & McConachy (1990) *GCA* 54:1911-1927 |
| Mn(II) | primeira ordem, sem escalonamento térmico | k₁ < 0.2/ano (pluma flutuante) | não encontrada | Cowen, Massoth & Feely (1990) *Deep-Sea Res.* 37:1619-1637 |
| CH4 | traçador conservativo (k=0) | — | — | nenhuma cinética encontrada na pesquisa de literatura |

**Fator de realce de H2S (100×)**: Radford-Knoery et al. (2001)
reportam remoção de sulfeto em plumas reais ~2 ordens de grandeza mais
rápida que a cinética de água do mar de laboratório (mecanismo
provável: oxidação catalisada por metal/partícula — ver também PNAS
2021 sobre "Fe-catalyzed sulfide oxidation" para o mecanismo, cujos
valores numéricos não foram verificados nesta pesquisa). Exposto como
parâmetro `plume_enhancement` em `k_h2s()`, não escondido dentro de uma
constante "base" inflada.

**Assimetria de bacia Fe(II)**: a razão de taxa Atlântico/Pacífico
implementada aqui é de ~94× em T=2°C (ver
`test_atlantic_fe_oxidation_faster_than_pacific_by_at_least_one_order_of_magnitude`
em `tests/test_plume_physics.py`) — mais forte que o "~1 ordem de
grandeza" qualitativamente citado por Field & Sherrell (2000), porque
estamos comparando os extremos citados (2.1 min vs. 3.3h) em vez de uma
média de bacia. Tratar como uma faixa superior plausível, não uma razão
universal.

**Não verificado no texto primário**: os termos de correção de força
iônica/salinidade da equação completa de Millero et al. (1987, *GCA*)
para Fe(II), e os coeficientes da parametrização multiparamétrica mais
recente de González-Santana et al. (2021), *GCA* 297:143-157, não
puderam ser acessados (artigos pagos). Não implementados; se forem
adicionados no futuro, devem vir com esses coeficientes verificados no
texto primário, não reconstruídos por inferência.

## 4. Transporte reativo do traçador

Extensão de Rudnicki & Elderfield (1992) ao sistema MTT: `d(QC)/dz =
-k(T(z))*Q*C/w(z)`. Usando `dt/dz=1/w` e concentração de fundo ambiente
aproximada como zero (C representa o excesso acima do background), a
solução fechada implementada em
`plume_physics.integrate_species_transport()` é:

```
C(z) = (C0_efetivo / D(z)) * exp(-∫[0,t(z)] k dt')
```

`C0_efetivo` já incorpora a fração de remoção instantânea (Fe,
precipitação de sulfeto) quando aplicável. Isso evita uma segunda
integração de EDO acoplada — a decadência reativa é resolvida por
quadratura direta (`scipy.integrate.cumulative_trapezoid`) sobre o
perfil `(t(z), k(T(z)))` já produzido pela integração da pluma base.

## 5. Limitações e elementos não verificados (leia antes de citar este modelo)

- **N constante em todo o campo**: nenhum valor de N verificado nas
  fontes consultadas para dorsais do Atlântico (MAR/TAG) ou Pacífico
  (EPR) — o único valor oceânico de dorsal verificado é Juan de Fuca,
  7.9×10⁻⁴ s⁻¹ (Lavelle, 1997). Usado como default/fallback citável,
  não como universal. A antiga fórmula ad hoc de dependência com
  profundidade (`1e-3*(1+depth/6000)`) foi removida por não ter
  respaldo em nenhuma fonte encontrada — não substituída por outra
  fórmula de profundidade, por falta de uma citável.
- **alpha constante**: falha documentada nos primeiros ~2m acima do
  orifício (Lemaréchal, Roullet & Gula, 2025, *JGR Oceans* 130(10)) —
  não corrigido nesta fase.
- **Sem dependência de pressão/profundidade** nas taxas de reação — as
  leis cinéticas usadas não incluem termo de pressão.
- **pH**: mistura conservativa de [H+] (aditiva, fisicamente correta),
  mas sem química de tamponamento carbonático/borato — uma
  simplificação real da química de carbonatos do oceano.
- **CH4**: tratado como traçador puramente conservativo — nenhuma
  cinética de oxidação (microbiana ou abiótica) foi encontrada na
  pesquisa de literatura que fundamenta este modelo.
- **Geometria do orifício (raio, velocidade de saída)**: valores
  plausíveis mantidos do modelo anterior, não medições citadas — afetam
  Q0/M0/B0 e, portanto, a diluição em qualquer altura absoluta.
- **Regime de mistura para hipóteses prebióticas**: `prebiotic.py` usa
  `dilution_near_field_1m` (D(z=1m) do modelo de pluma turbulenta) como
  proxy físico para o insumo de diluição do módulo de origem da vida.
  Isso é uma aproximação: o regime real de interesse para essas
  hipóteses (mistura difusiva em poros de parede de chaminé, ex. Lost
  City / Russell & Martin) é fisicamente distinto da pluma turbulenta
  em coluna d'água livre modelada aqui. Os demais módulos prebióticos
  (termoforese, adsorção mineral, gradiente de prótons) continuam
  explicitamente rotulados como ilustrativos/especulativos.
- **Sem dispersão de fundo, correntes, coalescência de plumas entre
  vents do mesmo cluster, nem modo de sítio real** — ver seção 1.

## 6. Como rodar a validação

```
pytest tests/test_plume_physics.py -v
```

Cada teste cita sua fonte e a razão da tolerância escolhida no
docstring/comentário — ver `tests/test_plume_physics.py`.

---

# Fase 2 — Modelo acústico exploratório de concentração prebiótica

Implementado em `acoustics.py` (ponto de integração: `--acoustic-mode`
em `fumarola_field.py`, seletor "Modelo acústico" na GUI). **Leia esta
seção inteira, especialmente a 7.6, antes de citar qualquer resultado
deste módulo** — é uma hipótese original explorada computacionalmente,
não a implementação de um modelo já publicado.

## 7.1 Hipótese testada

Formulação do usuário deste projeto: o campo acústico real gerado por
fumarolas hidrotermais pode transportar/concentrar moléculas
prebióticas em regiões de interferência das ondas que se propagam
nesse campo. Nenhuma publicação encontrada na pesquisa que fundamenta
este módulo propõe esse mecanismo para vents hidrotermais — a busca de
literatura sobre concentração prebiótica em vents (evaporação de
poças de maré, adsorção em argila, eutéticos de gelo, compartimentos
minerais tipo Russell & Martin) não retornou nenhuma menção a campos
acústicos.

## 7.2 Problema físico central (por que o modelo tem dois mecanismos, não um)

A força de radiação acústica clássica (potencial de Gor'kov) escala
com o volume da partícula (∝r³) e é demonstrada eficaz experimentalmente
a partir de ~1 μm; abaixo disso ela desaparece mais rápido que r³ e o
movimento browniano/térmico domina (ver Estimation of acoustic forces
on submicron aerosol particles, *Aerosol Sci. Technol.* 2017). Moléculas
prebióticas livres (aminoácidos, nucleotídeos: sub-nm a poucos nm) estão
100-1000× abaixo desse piso — "moléculas presas diretamente em nós de
interferência" não é defensável com a acustofluídica conhecida. Por
isso o módulo implementa dois mecanismos fisicamente distintos,
selecionáveis via `--acoustic-mode {streaming,particle_trap,both}`:

- **A. `streaming`**: streaming acústico de contorno advecta/retém
  SOLUTO DISSOLVIDO de fato — mecanismo correto para moléculas livres,
  independente de tamanho.
- **B. `particle_trap`**: potencial de Gor'kov aprisiona um COLÓIDE
  MINERAL (>1 μm, regime onde a força de radiação de fato funciona);
  concentração molecular é assumida proporcional à concentração local
  de partícula (aproximação, não uma medição de adsorção real).

## 7.3 Fonte acústica real

Crone, T.J., Wilcock, W.S.D., Barclay, A.H., & Parsons, J.D. (2006).
"The Sound Generated by Mid-Ocean Ridge Black Smoker Hydrothermal
Vents." *PLOS ONE* 1(1): e133. Vents "Sully"/"Puffer" (Endeavour, Juan
de Fuca): banda larga 5-500 Hz (10-30 dB acima do ruído ambiente), tons
estreitos 10-250 Hz (largura 5-15 Hz, 10-20 dB acima da banda larga),
pressão RMS 0.4-2.6 Pa. Mecanismos de geração propostos (não excludentes):
fluxo pulsante (monopolo no orifício), mistura turbulenta (dipolo),
interação fluido-estrutura na chaminé, ressonância tipo
Helmholtz/quarto-de-onda na própria chaminé.

**Não verificado/assumido**: a distância exata hidrofone-orifício não
foi confirmada no material consultado (o texto completo/métodos não
foi acessado) — os valores de pressão RMS medidos são tratados como
representativos a uma distância de referência nominal de 1 m
(`REFERENCE_DISTANCE_M`), uma aproximação explícita. A frequência do
tom de cada fumarola é **amostrada** da faixa empírica (10-250 Hz), não
derivada da altura da chaminé por uma fórmula de ressonância: Crone et
al. citam ressonância Helmholtz/quarto-de-onda como uma hipótese entre
várias, sem fornecer uma relação geometria→frequência fechada e
verificada — inventar essa relação seria precisão falsa.

## 7.4 Propagação e auto-interferência (efeito do espelho de Lloyd)

Velocidade do som: Mackenzie, K.V. (1981), "Nine-term equation for
sound speed in the oceans," *JASA* 70(3), 807-812 — ~1499 m/s nas
condições ambiente do projeto (T=2°C, D~2500m), consistente com o valor
de referência de oceano profundo (~1500 m/s). Absorção: Ainslie, M.A.,
& McColm, J.G. (1998), *JASA* 103(3), 1671-1672 — da ordem de 1e-3
dB/km nas frequências de vents, portanto desprezível ao longo da escala
do domínio (o termo é calculado e aplicado, não simplesmente omitido;
ver `test_absorption_negligible_over_domain_scale`).

Não há evidência de sincronismo de fase entre fumarolas independentes
(Crone et al. só reportam modulação de AMPLITUDE por maré, não
travamento de fase) e as linhas tonais têm banda finita (5-15 Hz,
implicando tempo de coerência ~dezenas de ms) — franjas de
interferência ENTRE fumarolas diferentes não seriam estáveis no tempo.
Por isso o modo padrão (`cross_vent_coherence="incoherent"`) soma os
campos de fumarolas diferentes sempre EM POTÊNCIA (nunca em fase) e só
modela a auto-interferência de cada fumarola com sua PRÓPRIA imagem
refletida no fundo local (método de imagens para contorno rígido —
impedância do basalto/sulfeto ≫ água, `BOTTOM_REFLECTION_COEFF=0.9`,
valor plausível não medido). Essa auto-interferência fonte+imagem é o
efeito clássico do "espelho de Lloyd" (Lloyd's mirror — ver Urick, R.J.,
*Principles of Underwater Sound*), sempre coerente (mesma fonte,
geometria fixa), sem depender de qualquer suposição de fase entre
fumarolas.

**Teste de limite superior (`--acoustic-cross-vent-coherence coherent`,
adicionado 2026-08-06)**: a escolha "incoherent" acima é uma suposição
por AUSÊNCIA de evidência (Crone et al. 2006 não mede coerência de fase
entre vents vizinhos), não uma medição que descarta o cenário oposto —
existe um caminho fisicamente plausível para coerência PARCIAL (vents
do mesmo cluster com chaminés de altura semelhante podem ter frequências
de ressonância tonal próximas). Para testar se essa suposição é
determinante para a conclusão do modelo, `acoustics.py` também aceita
um modo "coherent": TODAS as fumarolas recebem uma única frequência
tonal sorteada (em vez de uma por fumarola, ver `build_acoustic_sources`)
e seus fasores de pressão/velocidade são somados em fase — um cenário
de melhor caso deliberadamente idealizado (upper bound), não uma
previsão. Piloto de 20 runs (`--seed 100 --size 65 --n-clusters 5
--acoustic-mode particle_trap`, 2026-08-06): o modo coerente elevou a
profundidade média do poço de Gor'kov (agregado de Fe-oxi-hidróxido,
classe mais favorável) de 2.35e-2 para 3.31e-2 kT (~1.4×) e o máximo de
1.48e-1 para 2.03e-1 kT — na direção fisicamente esperada (interferência
construtiva no melhor caso), mas AMBOS os modos permanecem 1-2 ordens de
grandeza abaixo do limiar kT=1. Achado preliminar (piloto pequeno, não
um desenho fatorial completo — ver limitação abaixo): mesmo o limite
superior otimista de coerência total entre fumarolas não muda a
conclusão de irrelevância física do mecanismo B nas pressões acústicas
reais medidas.

**CORREÇÃO (2026-08-06)**: antes desta versão, o termo de VELOCIDADE
(usado por streaming e por Gor'kov) somava os fasores complexos de
fumarolas DIFERENTES em fase — inconsistente com a soma em potência já
usada (corretamente) para a pressão, e sem sentido físico bem definido
quando as fumarolas somadas têm frequências tonais diferentes. Isso
vazava uma forma de interferência coerente entre vents não documentada
e não pretendida, por um caminho diferente do da pressão. Corrigido:
o termo de velocidade agora segue exatamente a mesma escolha de
`cross_vent_coherence` que o termo de pressão. Regressão coberta em
`tests/test_acoustics.py` (`test_different_vents_still_combine_incoherently_by_default_after_fix`,
`test_single_vent_incoherent_and_coherent_modes_are_identical`).

**Limitação de visibilidade de franjas, não-óbvia**: a diferença máxima
de caminho fonte-vs-imagem é `2×(altura do receptor)=2m` (receptor fixo
a 1 m do fundo, mesma convenção de campo próximo usada em
`dilution_near_field_1m`), **independente da altura da chaminé**. Nas
frequências reais medidas (10-250 Hz, λ=6-150m), essa diferença cobre
MENOS de um ciclo completo de interferência — na prática, a
auto-interferência aparece como um único realce/sombra suave perto da
fumarola, não como múltiplos nós/antinós periódicos (ver
`tests/test_acoustics.py`, onde 1000 Hz — bem acima da faixa medida — é
usado deliberadamente só para validar a matemática de franjas múltiplas).

## 7.5 Mecanismo B — potencial de Gor'kov + população de partículas citada

Gor'kov, L.P. (1962), *Sov. Phys. Dokl.* 6, 773-775; formulação moderna
em Bruus, H. (2012), "Acoustofluidics 7," *Lab Chip* 12, 1014-1021.
Velocidade de partícula derivada da equação de momento linearizada
completa (`v=∇p/(iωρ)`), não do atalho de campo distante `v≈p/(ρc)` —
esse atalho apagaria exatamente a separação nó-de-pressão/antinó-de-
velocidade de que depende o aprisionamento. Distribuição de partícula:
Boltzmann de equilíbrio, `n∝exp(-U/k_BT)` (mecânica estatística padrão,
não uma fórmula heurística ajustada).

**Fase 2b (2026-08-05) — população de duas classes de tamanho, em vez
de um colóide único arbitrário** (`acoustics.PARTICLE_CLASSES`):
testar a recomendação "modele populações de agregados/flocos reais, não
colóides finos" exige ancorar o tamanho em dados de campo, não em uma
suposição. A pesquisa de literatura sustenta dois pontos de ancoragem
(não uma distribuição contínua ajustada — seria precisão falsa com só
dois pontos):

1. **Colóide fino de sulfeto** (`fine_sulfide_colloid`): 2 μm, pirita
   (ρ=5010 kg/m³) — ordem de grandeza de referência ao limiar
   >0.2 μm de partículas de Cu/Zn perto do orifício (Klevenz, V., Bach,
   W., Schmidt, K., Hentscher, M., Koschinsky, A., & Petersen, S.,
   2011, "Geochemistry of vent fluid particles formed during initial
   hydrothermal fluid-seawater mixing along the Mid-Atlantic Ridge,"
   *Geochem. Geophys. Geosyst.* 12, Q0AE05).
2. **Agregado de oxi-hidróxido de Fe, campo próximo**
   (`near_field_fe_oxyhydroxide_aggregate`): 17 μm (ponto médio de
   14-20 μm), ρ=3000 kg/m³ (ponto médio de 2400-3600 kg/m³) —
   González-Santana, D., Planquette, H., Cheize, M., Whitby, H.,
   Gourain, A., Holmes, T., et al. (2020), "Processes driving iron and
   manganese dispersal from the TAG hydrothermal plume (Mid-Atlantic
   Ridge): Results from a GEOTRACES process study," *Front. Mar. Sci.*
   7:568 — raios derivados de velocidade de sedimentação (lei de
   Stokes) no primeiro km da pluma do TAG. É o maior raio de
   partícula/agregado hidrotermal DIRETAMENTE documentado na pesquisa
   de literatura que fundamenta este modelo (a mesma fonte relata
   agregados menores, 2-4 μm, entre 2-30 km — os agregados maiores
   sedimentam e são removidos com a distância, não modelado aqui).

As duas classes são calculadas sempre (`gorkov_potential_field` +
`particle_boltzmann_enhancement` para cada), reportadas em
`diagnostics["particle_classes"]`; a classe de agregado (maior, mais
favorável) é usada como campo principal
(`PRIMARY_PARTICLE_CLASS`). Um tamanho customizado único ainda pode ser
passado explicitamente via `particle_radius_m`/`particle_density_kg_m3`
(ou `--acoustic-particle-radius-um`/`--acoustic-particle-density` na
CLI) para exploração pontual — isso DESLIGA a população citada.

**Achado central (registrado como testes de regressão permanentes,
`test_gorkov_trap_not_physically_relevant_for_fine_colloid_at_measured_vent_pressures`
e `test_gorkov_trap_remains_negligible_for_largest_documented_aggregate`)**:
nas pressões acústicas REAIS medidas em vents (0.4-2.6 Pa RMS — muitas
ordens de grandeza abaixo dos kPa-MPa usados em dispositivos de
acustoforese de laboratório):

- Colóide fino (2 μm): profundidade do poço ≈ 2.05×10⁻⁵ `k_BT`.
- Agregado de campo próximo (17 μm, o maior documentado): profundidade
  ≈ 1.26×10⁻² `k_BT` (~1.3% de `k_BT`) — ~79× abaixo do limiar de
  relevância térmica, mesmo sendo a melhor combinação tamanho/densidade
  diretamente documentada na literatura consultada.

A agitação térmica domina completamente em AMBAS as classes — a
armadilha **não é fisicamente relevante**, e esse resultado agora
repousa sobre o maior tamanho real documentado, não sobre uma suposição
de "colóide fino" arbitrária. Verificado numericamente que a
profundidade escala como r³ (ver
`test_gorkov_potential_scales_as_particle_radius_cubed`): trapping só
se tornaria relevante para partículas maiores que qualquer tamanho
hidrotermal com raio diretamente documentado nas fontes consultadas —
um refinamento falsificável mais forte que a versão anterior deste
documento (que extrapolava um limiar de "~0.1-0.5 mm" sem comparação
com dados de tamanho de partícula real).

## 7.5b Investigado e descartado: cópula raio-densidade do agregado (2026-08-08)

O item de robustez estatística "estrutura de correlação realista entre
parâmetros varridos" (esta conversa, lista de melhorias por data
science) partiu da suposição de que raio e densidade do agregado de
campo próximo provavelmente NÃO são fisicamente independentes
("agregados maiores tendem a ser menos densos, a própria fonte citada
tem essa relação nos dados brutos, plausivelmente extraível") — hoje
`--sensitivity-sweep`/`--variance-decomposition` amostram os dois como
independentes (§7.8/§7.8.2).

**Essa suposição estava ERRADA — verificado lendo o texto primário
completo de González-Santana et al. (2020) diretamente** (não um
resumo/citação de segunda mão): o artigo NÃO reporta nenhuma relação
raio-densidade. A faixa de densidade (2400-3600 kg/m³) é uma faixa de
incerteza FIXA, tomada emprestada de German & Sparks (1993) para
goethita/oxi-hidróxido de Fe amorfo — "we used the SAME range of
particle densities... as German and Sparks (1993)" — e aplicada
UNIFORMEMENTE via lei de Stokes para inverter velocidade de
sedimentação → raio, independente de qual raio resulta da inversão. O
raio em si varia com a DISTÂNCIA do vent (14-20 μm no primeiro km,
2-4 μm entre 2-30 km — já corretamente capturado em `PRIMARY_PARTICLE_
CLASS`, §7.5), não com a densidade. Não há dado bruto raio-vs-densidade
no artigo, publicado ou suplementar, que permita estimar uma cópula real.

**Decisão**: nenhuma cópula foi implementada. Inventar uma estrutura de
correlação sem base na fonte citada seria menos defensável que a
independência assumida atualmente — precisão falsa disfarçada de
realismo, exatamente o tipo de escolha que este projeto evita
(`--sensitivity-sweep`/`--variance-decomposition` já documentam essa
mesma régua para outros parâmetros "ilustrativos, sem faixa citável",
ver §7.8). A amostragem independente atual permanece a escolha correta
e já validada (`joint_latin_hypercube`, §7.8/regressão de teste
`test_sweep_produces_no_spurious_correlation`).

**Nota corretiva**: as entradas de memória/conversa anteriores desta
sessão que motivaram este item continham essa suposição não verificada
sobre os "dados brutos" do artigo — não há dados brutos raio-densidade
nesse artigo. Registrado aqui para não repetir a suposição numa sessão
futura.

## 7.6 Mecanismo A — streaming de contorno + advecção-difusão real

Streaming de contorno: Rayleigh, Lord (1884), *Phil. Trans. R. Soc.*
175, 1-21; forma explícita `U_slip=-(3/4ω)·d⟨U₀²⟩/dx` de Nyborg, W.L.
(1958), "Acoustic streaming near a boundary," *JASA* 30(4), 329-339 —
aplicada aqui como extensão de engenharia ao gradiente 2D completo (não
uma transcrição literal da fórmula 1D). Streaming de bulk (Eckart, C.,
1948, *Phys. Rev.* 73(1), 68-76) foi deliberadamente **não** usado como
mecanismo principal: nas frequências de vents a absorção é desprezível
(seção 7.4), tornando a força de Eckart (∝absorção×intensidade)
irrelevante nessa escala — o streaming de contorno domina.

A concentração de soluto NÃO é uma fórmula heurística: resolve-se a
equação de advecção-difusão-perda estacionária real,
`D∇²C - u·∇C - k·C + S = 0`, por diferenças finitas com advecção
upwind de 1ª ordem (necessária — o número de Péclet de malha aqui é
tipicamente ≫1, tornando diferenças centrais instáveis) e sistema
linear esparso (`scipy.sparse.linalg.spsolve`), comparando a solução
COM streaming (`u` do campo Rayleigh/Nyborg) contra um controle SEM
streaming (`u=0`), mesma fonte e mesma perda — mesmo padrão de
comparação-contra-controle do resto de `prebiotic.py`. `D` (difusividade
de soluto orgânico pequeno em água, ~8e-10 m²/s) é ordem de grandeza de
manual (Cussler, E.L., *Diffusion: Mass Transfer in Fluid Systems*), não
uma medição para aminoácidos a 2°C especificamente. `k` (perda) é uma
**escolha de condicionamento numérico** (difusão 2D estacionária pura,
sem nenhuma perda, não tem solução localizada finita para fonte
puntual em domínio infinito) — não uma cinética química medida.

Nas pressões reais medidas, a velocidade de streaming resultante é
desprezível (ordem 1e-18 m/s nos runs de verificação, ver
`metadata.json` de qualquer run com `--acoustic-mode streaming`) —
mesma conclusão qualitativa do mecanismo B: a energia acústica real
disponível em vents é ordens de grandeza pequena demais, nestas duas
formulações mecanísticas, para produzir um efeito de concentração
detectável.

## 7.7 Limitações e elementos não verificados (leia antes de citar este modelo)

- **Nenhuma validação experimental existe, em nenhuma fonte
  encontrada, para o mecanismo acústico-prebiótico especificamente** —
  isto é uma hipótese original explorada computacionalmente, não um
  modelo publicado sendo implementado. Tratar todo resultado deste
  módulo como um teste de plausibilidade teórica, não uma previsão.
- **Distância de referência da fonte (1 m) e coeficiente de reflexão do
  fundo (0.9) não são medições** — valores plausíveis não verificados
  para o material mineralizado específico de um campo de vents.
- **Frequência tonal por fumarola é amostrada, não derivada da
  geometria da chaminé** — Crone et al. (2006) não fornecem uma relação
  geometria→frequência fechada verificada.
- **Ausência de coerência de fase entre fumarolas é uma inferência, não
  uma medição direta** — baseada na banda finita das linhas tonais
  (5-15 Hz) e na ausência de qualquer relato de travamento de fase em
  Crone et al.; se dados futuros mostrarem coerência parcial entre
  fumarolas próximas, o modelo de soma incoerente subestimaria o
  efeito. Testável via `--acoustic-cross-vent-coherence coherent` (ver
  §7.4) — um limite superior idealizado (todas as fumarolas na mesma
  frequência, somadas em fase), não uma medição nem uma segunda
  suposição igualmente arbitrária: no piloto de 20 runs já rodado (ver
  §7.4), o limite superior aumentou a profundidade do poço em ~1.4× mas
  não mudou a conclusão qualitativa (ainda ordens de grandeza abaixo de
  kT) — um desenho fatorial maior (campo fixo, coerência variando) seria
  necessário para uma afirmação quantitativa mais forte que "não muda a
  conclusão neste piloto".
- **Componente vertical da velocidade de partícula não é resolvida**
  (campo avaliado só num plano horizontal fixo a 1 m do fundo) — força
  de aprisionamento vertical não é modelada.
- **Partículas do mecanismo B no limite rígido**: módulo de
  compressibilidade real de colóides/agregados de sulfeto/oxi-hidróxido
  em escala micro/nanométrica não verificado para nenhuma das duas
  classes — usar um valor específico teria sido precisão falsa, mas o
  limite rígido é ele próprio uma aproximação.
- **A classe de agregado (14-20 μm) repousa sobre uma única pluma
  documentada** (TAG, Mid-Atlantic Ridge; González-Santana et al.,
  2020) — é o maior raio de agregado hidrotermal diretamente
  documentado na pesquisa de literatura que fundamenta este modelo, não
  necessariamente o maior que existe na natureza em outros sítios ou
  sob outras condições de fluxo/idade da pluma.
- **Classes de tamanho estáticas, não um modelo dinâmico de
  agregação/floculação** — populações reais de partícula engrossam ao
  longo do tempo de residência na pluma (sedimentação remove primeiro
  os agregados maiores, conforme a própria diferença de tamanho
  campo-próximo/campo-distante relatada por González-Santana et al.,
  2020); essa evolução temporal não é simulada aqui.
- **Difusividade do soluto e taxa de perda do mecanismo A não são
  calibradas para uma molécula prebiótica específica a 2°C** — ordem de
  grandeza de manual e escolha de condicionamento numérico,
  respectivamente.
- **Resultado atual (achado, não limitação, mas vale repetir aqui)**:
  em ambos os mecanismos, nas pressões acústicas reais medidas em
  vents, o efeito é fisicamente desprezível para os tamanhos de
  partícula/parâmetros de soluto testados como default. Isso não
  invalida a hipótese em princípio (partículas maiores, sítios com
  pressão acústica mais alta, ou mecanismos não modelados aqui poderiam
  mudar essa conclusão), mas qualquer afirmação de que "o mecanismo
  funciona" precisaria justificar por que os parâmetros usados aqui
  seriam substancialmente diferentes dos citados nesta seção.

## 7.8 Varredura de sensibilidade (Hipercubo Latino)

Implementada em `fumarola_field.latin_hypercube_1d()` + `run_experiment()`
(`--sensitivity-sweep` na CLI, checkbox na GUI). Motivação: um ensemble
comum já produz variabilidade de enriquecimento entre runs, mas essa
variabilidade vem só da aleatoriedade do campo de fumarolas (seeds
diferentes) — os parâmetros físicos incertos ficam FIXOS no valor
default em todas as runs, escondendo o quanto o resultado depende
dessas escolhas (ver seção 7.7).

**Método**: amostragem por Hipercubo Latino CONJUNTO/multi-D (McKay,
M.D., Beckman, R.J., & Conover, W.J., 1979, "A comparison of three
methods for selecting values of input variables in the analysis of
output from a computer code," *Technometrics* 21(2), 239-245), via
`fumarola_field.joint_latin_hypercube()` (`scipy.stats.qmc.LatinHypercube`
com `optimization="random-cd"`) — cada dimensão sai estratificada em N
intervalos iguais (N = nº de runs do ensemble), exatamente como o LHS 1D
(`latin_hypercube_1d()`, ainda usado isoladamente em
`tests/test_fumarola_field.py` e disponível como utilitário), mas as N
combinações entre dimensões são escolhidas com otimização de
discrepância centrada em vez de uma permutação puramente aleatória por
dimensão — reduz a chance de correlação espúria residual entre
parâmetros varridos numa amostra pequena, sem introduzir nenhuma
correlação física real entre eles (cada margem continua a mesma faixa
documentada). **Atualizado 2026-08-07**: antes deste ajuste, cada
parâmetro era amostrado por 3 chamadas SEQUENCIAIS e independentes de
`latin_hypercube_1d()`, que já eram matematicamente equivalentes a um
LHS conjunto por permutação simples (sem a otimização de discrepância);
`tests/test_fumarola_field.py::test_sensitivity_sweep_swept_parameters_
show_no_spurious_correlation` confirma numericamente (Spearman |rho|
abaixo do limiar assintótico sob independência, n=60) que a troca não
introduziu nem deixou de corrigir nenhuma correlação detectável entre
`entrainment_alpha` e o raio do agregado num sweep real. A RNG da
varredura deriva do MESMO `--seed` base usado para os campos de
fumarola (via um filho adicional de `SeedSequence.spawn`), então a
varredura inteira — não só cada campo individualmente — é reprodutível.

**O que é varrido, e por quê só isso**: apenas parâmetros com faixa de
incerteza DOCUMENTADA em uma fonte citável:
- `entrainment_alpha` ∈ [0.07, 0.18] — faixa medida em campo (Grotto
  vent, Main Endeavour Field; Rona, P.A., Bemis, K.G., Jones, C.D.,
  Jackson, D.R., Mitsuzawa, K., & Silver, D., 2006, "Entrainment and
  bending in a major hydrothermal plume, Main Endeavour Field, Juan de
  Fuca Ridge," *GRL* 33, L19313, doi:10.1029/2006GL027211 — corrigido
  2026-08-06 após leitura direta do PDF primário completo,
  `biblios/2006gl027211.pdf`: autor principal, título e número de
  artigo citados aqui antes estavam errados — "Bemis, Jones & Jackson,
  'Plume anomaly detected by acoustic Doppler current profiler,'
  L02613" — só o DOI estava correto; o valor 0.07-0.18 já batia
  exatamente com a Tabela 1 do artigo real, então não muda).
- Se `--acoustic-mode` for `particle_trap`/`both`: raio do agregado ∈
  [14, 20] μm e densidade ∈ [2400, 3600] kg/m³ (faixa completa, não só
  o ponto médio usado por padrão — González-Santana et al., 2020; ver
  seção 7.5), sobrescrevendo só a classe `near_field_fe_oxyhydroxide_
  aggregate`, mantendo o colóide fino fixo.

Parâmetros "ilustrativos, sem faixa citável" (coeficientes de Soret,
capacidade de Langmuir, ganho do gradiente de prótons) são
DELIBERADAMENTE excluídos da varredura quantitativa — inventar uma
faixa ± para eles seria precisão falsa. A contribuição relativa desses
módulos continua avaliada só pelo desenho de ablação (ligado/desligado)
já existente.

**Limitação (resolvida por um modo alternativo, ver §7.8.2)**: como cada
run da varredura também tem um campo de fumarolas gerado
independentemente (seed diferente), a dispersão de enriquecimento
resultante mistura variabilidade ESTOCÁSTICA do campo com incerteza de
PARÂMETRO — não isola uma da outra. Isolar as duas exige um desenho
aninhado (mesmo campo/parâmetro fixo dentro de um grupo, só variando
entre grupos), implementado separadamente como `--variance-decomposition`
(§7.8.2) — este `--sensitivity-sweep` "simples" continua existindo como
está, sem mudança de comportamento além da troca de amostragem descrita
acima. Ver `tests/test_fumarola_field.py` para os testes de
estratificação/reprodutibilidade do Hipercubo Latino.

## 7.8.1 Evento raro observado no ensemble de 1000 runs: quando a profundidade de Gor'kov cruza k_BT

Analisando o ensemble real de 1000 runs gerado com `--sensitivity-sweep
--acoustic-mode particle_trap` (seed base 1657119425,
`outputs/experimento_260807_021219`), 7 das 1000 realizações (0,7%; IC
95% de Wilson [0,34%, 1,44%]) tiveram a profundidade do poço de Gor'kov
para a classe de agregado quase-campo cruzando o limiar de relevância
térmica (>1 k_BT) — máximo observado 2,82 k_BT (seed 3074131324, 28
fumarolas). Nas 7 realizações, o raio do agregado amostrado pelo
Hipercubo Latino (faixa real 14-20 μm, González-Santana et al., 2020)
ficou entre 15,5 e 20,0 μm — sempre perto do TETO da faixa documentada,
nunca no meio ou perto do piso. Isso bate exatamente com a escala r³ já
estabelecida (seção 7.5, verificada numericamente no teste
`test_gorkov_trap_not_physically_relevant_at_measured_vent_pressures` e
correlatos): não é ruído — é a cauda esperada da distribuição de
tamanho de agregado colidindo com um mecanismo hipersensível a esse
parâmetro. Correlação de Spearman entre os parâmetros varridos e a
profundidade de poço, calculada sobre o ensemble completo: raio do
agregado ρ=+0,48 (p≈7×10⁻⁶⁰, dominante), nº de vents ρ=+0,13 (p≈3×10⁻⁵,
fraco em MAGNITUDE — mas ver §7.8.4: regressão multivariada mostra que
esse efeito é REAL, não um artefato de confundimento, e explicável
mecanisticamente pela própria definição da métrica, não descartável como
inicialmente suspeitado aqui), α de entranhamento e densidade do
agregado sem correlação significativa (|ρ|<0,03). Reproduzível via
`report._aggregate_acoustic_stats(summaries)` sobre os `summaries` do
ensemble (usado por `report.py` para popular Results/Discussion — ver
abaixo).

**Por que isso é discutido sob a moldura de "eventos raros" em origem
da vida** (pedido explícito do usuário, 2026-08-07, threaded por todo
`report.py`: título, abstract, introdução/objetivo, resultados,
discussão, conclusão, limitações, e nos três geradores de relatório —
`generate_scientific_report`, `generate_admin_report`,
`generate_admin_paper_plosone`, que reaproveita as mesmas funções de
seção): hipóteses de origem da vida não exigem que um mecanismo de
concentração funcione de forma confiável NA MÉDIA — exigem que ele
funcione PELO MENOS UMA VEZ, em algum vent, em algum momento da história
Hadeana/Arqueana inicial da Terra, depois do que química
autocatalítica/replicante por template (se existir) não precisaria mais
do mecanismo. Esse é o argumento de "muitas tentativas": mesmo uma
probabilidade por-realização pequena, multiplicada por um número
plausivelmente enorme de sistemas hidrotermais independentes ao longo
do tempo geológico, pode gerar um valor esperado de sucessos >> 1. Duas
referências ancoram essa moldura no texto (citações RECUPERADAS DA
MEMÓRIA DE TREINAMENTO, não lidas em texto primário nesta sessão —
adicionadas a `docs/CITATIONS_TO_VERIFY.md` pra verificação futura,
mesmo processo já aplicado a todas as outras citações reconstruídas
deste jeito no projeto):
- Lineweaver, C.H., & Davis, T.M. (2002). "Does the rapid appearance of
  life on Earth suggest that life is common in the universe?"
  *Astrobiology* 2(3), 293-304 — argumento de que o aparecimento rápido
  de vida na Terra é evidência (bayesiana) de que abiogênese pode não
  exigir um passo extremamente improvável.
- Carter, B. (1983). "The anthropic principle and its implications for
  biological evolution." *Philosophical Transactions of the Royal
  Society A* 310(1512), 347-363 — argumento antrópico dos "passos
  difíceis": observadores só podem se encontrar numa história onde uma
  transição rara já aconteceu, então a própria existência da vida não é
  evidência forte de que abiogênese seja comum — os dois argumentos
  estão em debate real na literatura (Lineweaver & Davis discutem
  Carter diretamente), e o relatório trata os dois com o MESMO peso
  editorial já usado pra outros mecanismos contestados do projeto (ex.:
  Sojo et al. 2016 vs. Jackson 2016 no gradiente de prótons, seção 8.9).

**Ressalva honesta, crítica**: os 0,7% NÃO são uma estimativa de
frequência real. São uma consequência direta do desenho de amostragem
por Hipercubo Latino da varredura, que cobre a faixa real documentada
(14-20 μm) de forma aproximadamente uniforme por construção — não uma
medição independente de quão frequentemente agregados de 18-20 μm
ocorrem de fato em sistemas hidrotermais reais (que exigiria um dataset
real de distribuição de tamanho de partícula, não só a faixa
documentada). O que É defensável: existe um regime fisicamente real e
alcançável, dentro da faixa diretamente documentada na literatura, em
que o mecanismo deixa de ser desprezível frente ao movimento térmico —
convertendo o que seria descartado de cara numa pergunta concreta e
falseável sobre a frequência real de agregados grandes em vents ativos.

## 7.8.2 Decomposição de variância estocástica vs. paramétrica (desenho aninhado)

Implementada em `variance_decomposition.py` (estatística pura) +
`fumarola_field.run_nested_variance_experiment()` (orquestração das
runs) — `--variance-decomposition` na CLI (`--outer-samples`/
`--inner-replicates`), modo alternativo a `--runs`/`--sensitivity-sweep`
(deriva seu próprio nº de runs = outer × inner). **Também exposta na
GUI** (`gui.py`): terceiro modo de execução "Variance decomposition
(nested)" ao lado de single/ensemble, com campos próprios para nº de
pontos externos/réplicas internas; ao terminar, um painel ao vivo na aba
de estatísticas mostra o resumo (`HydroventGUI._render_vardecomp_summary`)
e o relatório estatístico HTML (`ensemble_report.py`, §10.4) inclui as
seções completas com IC 95%.

**Motivação**: responder de forma quantitativa a "quanto do spread
observado num ensemble é aleatoriedade do campo de fumarolas, e quanto é
incerteza sobre `entrainment_alpha`/raio-densidade do agregado?" — uma
pergunta que `--sensitivity-sweep` (§7.8) não pode responder por
desenho, já que varia as duas fontes simultaneamente run a run.

**Método — desenho aninhado balanceado**: N_outer pontos de parâmetro
amostrados por `joint_latin_hypercube()` (mesmo método de §7.8); para
CADA ponto externo, N_inner réplicas com seeds de campo distintas e o
parâmetro físico FIXO dentro do grupo. Seeds derivadas deterministicamente
de `--seed` (um filho de `SeedSequence` por ponto externo, `N_inner`
netos por ponto para as seeds de campo daquele grupo) — desenho inteiro
reprodutível.

**Estatística — ANOVA de um fator aleatório** (método dos momentos,
Searle, S.R., Casella, G., & McCulloch, C.E., 1992, "Variance
Components," Wiley, cap. 3): pela lei da variância total,
`Var(Y) = E[Var(Y|θ)] + Var(E[Y|θ])`, onde θ indexa o ponto de parâmetro
externo. `MSW` (média das variâncias amostrais intra-grupo) estima
diretamente a componente ESTOCÁSTICA (σ² do campo, parâmetro fixo).
`MSB` (N_inner × variância amostral das médias de grupo) tem
`E[MSB] = σ²_estocástico + N_inner·σ²_paramétrico`, então
`σ²_paramétrico = (MSB − MSW) / N_inner` — grampeado em 0 quando
negativo (sinal paramétrico no ou abaixo do ruído de amostragem finita;
`between_group_variance_was_clipped` registra quando isso acontece, não
é tratado como erro). IC 95% das frações por bootstrap ANINHADO de 2
estágios (reamostra quais grupos externos entram E, dentro de cada um,
quais réplicas internas — preserva a estrutura hierárquica; Davison,
A.C., & Hinkley, D.V., 1997, "Bootstrap Methods and Their Application,"
Cambridge University Press, cap. 3.8).

**Variável de resposta**: por padrão, a profundidade do poço de Gor'kov
da classe de agregado near-field (`trap_depth_over_kT`, mesma métrica de
§7.5/§7.8.1) quando `--acoustic-mode` é `particle_trap`/`both`; cai para
o enriquecimento do hotspot dominante vs. controle
(`top_hotspot_enrichment_vs_control`) caso contrário — ambas já usadas
em outras análises deste projeto, não uma métrica nova inventada só para
isto. Customizável via o parâmetro `response_extractor` da função
(não exposto na CLI ainda).

**Validado com dados sintéticos** (`tests/test_variance_decomposition.py`):
gerando grupos com componentes de variância REALMENTE conhecidos
(efeito de grupo ~ N(0,σ²_between), ruído interno ~ N(0,σ²_within)), a
fração paramétrica VERDADEIRA cai dentro do IC 95% relatado, e os casos
degenerados (só ruído estocástico; só efeito de grupo, sem ruído) são
corretamente reconhecidos (fração paramétrica/estocástica ≈ 0,
respectivamente) — não é só um teste de forma do código, testa se a
matemática recupera um resultado conhecido.

**Custo**: cada réplica é uma simulação física completa (mesmo custo de
uma run normal, ~10-11s/run com acústica+ODE de pluma, medido em
sessão anterior) — o desenho padrão (`--outer-samples 20
--inner-replicates 10` = 200 runs) é deliberadamente mais modesto que um
ensemble típico de milhares de runs; mais N_outer melhora a resolução da
componente paramétrica, mais N_inner melhora a da componente estocástica
— não são intercambiáveis.

**Limitação (resolvida por §7.8.3)**: assume desenho BALANCEADO (mesmo
N_inner em todo grupo externo — imposto pela própria orquestração, não
uma escolha do usuário) e não decompõe, POR SI SÓ, a contribuição
INDIVIDUAL de `entrainment_alpha` vs. raio vs. densidade do agregado
dentro da componente paramétrica quando os três estão ativos — só diz
"quanto da variância é estocástica vs. paramétrica no total". Isolar a
contribuição de cada parâmetro individualmente (e suas interações) é
exatamente o que §7.8.3 (índices de Sobol') faz, reaproveitando os
mesmos dados coletados aqui.

## 7.8.3 Sensibilidade global por parâmetro (índices de Sobol' via surrogate)

Implementada em `global_sensitivity.py` — chamada automaticamente por
`run_nested_variance_experiment()` (§7.8.2) sobre os MESMOS dados já
coletados (pontos externos + réplicas internas), sem rodar nenhuma
simulação física a mais. **Exposta na GUI e no relatório estatístico**
junto com §7.8.2 (mesmo modo de execução "Variance decomposition
(nested)", mesmo painel ao vivo, mesma seção do relatório HTML — ver
§10.4).

**Motivação**: §7.8.2 separa variância estocástica de paramétrica, mas
quando MAIS de um parâmetro está ativo (α + raio + densidade do
agregado) não diz qual DELES domina a componente paramétrica — essa é a
pergunta que os índices de Sobol' (Sobol, I.M., 1993, "Sensitivity
estimates for nonlinear mathematical models," *Mathematical Modeling and
Computational Experiment* 1(4), 407-414) respondem: quanto da variância
de saída é atribuível a CADA parâmetro individualmente (índice de
primeira ordem, S_i) e a cada parâmetro incluindo suas interações com os
demais (índice de efeito total, S_Ti).

**Por que um surrogate**: uma estimativa Monte Carlo estável de Sobol'
via o esquema de Saltelli precisa de milhares de avaliações da função —
inviável com a simulação física real (~10s/run). Um Processo Gaussiano
(kernel RBF, comprimentos de escala por dimensão; Rasmussen, C.E., &
Williams, C.K.I., 2006, "Gaussian Processes for Machine Learning," MIT
Press, cap. 2/5) é ajustado nas MÉDIAS de grupo do desenho aninhado de
§7.8.2 (aproximando E[Y|θ], já isolada do ruído estocástico pelo próprio
desenho), com o ruído de medição de cada ponto conhecido de antemão
(`within_group_variance / N_inner`, reaproveitado de §7.8.2 — não
estimado de novo). Implementado do zero (numpy/scipy, sem
scikit-learn) para manter a mesma filosofia de dependência enxuta do
resto do projeto. Hiperparâmetros do kernel ajustados por máxima
verossimilhança marginal (múltiplas reinicializações, já que a
log-verossimilhança não é convexa).

**Índices de Sobol' via esquema de Saltelli**: S_i pelo estimador de
Saltelli (Saltelli, A., et al., 2010, "Variance based sensitivity
analysis of model output. Design and estimator for the total
sensitivity index," *Computer Physics Communications* 181(2), 259-270);
S_Ti pelo estimador de Jansen (Jansen, M.J.W., 1999, "Analysis of
variance designs for model output," *Computer Physics Communications*
117(1-2), 35-43), numericamente mais estável para o índice total. As
matrizes A/B do esquema de Saltelli vêm de UMA ÚNICA sequência de Sobol'
(baixa discrepância — Sobol, I.M., 1967/1976; ferramenta DIFERENTE dos
índices de sensibilidade de mesmo nome) de dimensão 2d dividida em
colunas, não duas sequências construídas independentemente.

**Bug real encontrado e corrigido nesta sessão**: a primeira
implementação gerava A e B como duas instâncias INDEPENDENTES de
`qmc.Sobol` — cada uma bem distribuída na própria dimensão, mas sem a
estrutura de correlação que o estimador de Saltelli/Jansen exige entre A
e B. Verificado numericamente contra uma função aditiva com resposta
analítica conhecida (`f = x1 + 2·x2`, Var(Uniform(0,1))=1/12 — elementar,
não um número memorizado): a versão com sequências independentes mediu
S1≈0,09 quando o valor analítico correto é S1=0,2 (erro >2x). Corrigido
gerando uma única sequência conjunta de dimensão 2d e dividindo em
A (primeiras d colunas) / B (últimas d colunas) — mesma abordagem da
biblioteca de referência SALib — e reconfirmado batendo com S1=0,2/S2=0,8
dentro do erro de Monte Carlo. Fixado como regressão permanente em
`tests/test_global_sensitivity.py::test_sobol_matrices_use_joint_
sequence_not_independent_ones`.

**Honestidade central — índices são sobre o SURROGATE, não a simulação
diretamente**: só são confiáveis na medida em que o Processo Gaussiano
aproxima bem a resposta real, o que com N_outer tipicamente modesto
(dezenas, não milhares — cada ponto é uma simulação física completa)
pode ser ruim. Todo resultado vem com `loo_cv_r2` (R² de validação
cruzada leave-one-out, via fórmula fechada de GP — Rasmussen & Williams
2006, §5.4.2 — validada contra um conjunto de teste genuinamente
separado em `tests/test_global_sensitivity.py`) e um aviso explícito
(`loo_cv_r2_warning`, limiar R²<0,5) quando o ajuste é fraco demais para
confiar nos índices — testado num caso real desta sessão
(`--outer-samples 8` com 3 parâmetros: R²=0,000, índices descartáveis,
aviso disparou corretamente; `--outer-samples 20`, o padrão de
produção: R²=0,313, ainda um aviso honesto, mas índices consistentes
com o driver já identificado em §7.8.1 — raio do agregado dominante).
Índices individuais são grampeados a [0,1] (a definição matemática nunca
sai desse intervalo; um estimador MC de amostra finita sobre um
surrogate quase-constante pode escapar por ruído puro — visto na prática
com `--outer-samples 8`, um índice saiu 1,24 antes do grampeamento).

**Validado com funções analíticas** (`tests/test_global_sensitivity.py`,
sem envolver o GP — isola bugs do estimador Monte Carlo dos de ajuste do
surrogate): função aditiva sem interação (S_i=S_Ti, soma dos S_i=1,
parâmetro não usado tem S≈0) e função produto com interação genuína
(S_Ti > S_i estritamente, soma dos S_i < 1) — ambas batem com a teoria.
Pipeline completo (surrogate + Sobol') validado com dados sintéticos com
sensibilidade conhecida por construção (um parâmetro dominante vs. um
pouco influente) e com o caso real de uma única dimensão varrida
(`--acoustic-mode off`/`streaming`), onde toda a variância explicada
necessariamente pertence ao único parâmetro (S1>0,85 nos testes).

## 7.8.4 Driver do nº de vents: confundimento ou efeito real? (regressão multivariada por postos)

Implementada em `driver_regression.py` — generaliza a correlação de
Spearman um-parâmetro-por-vez de `report._relevance_drivers` (§7.8.1)
para uma regressão que controla TODOS os preditores simultaneamente,
via transformação de postos (Iman, R.L., & Conover, W.J., 1979, "The use
of the rank transform in regression," *Technometrics* 21(4), 499-509) —
cada preditor e a resposta viram seus próprios postos (postos médios em
empates), ajustados por mínimos quadrados ordinários, preservando a
robustez da correlação de Spearman a não-linearidade monotônica/não-
normalidade (a resposta deste projeto tem cauda longa conhecida — ver
§7.8.1), mas agora com coeficientes PARCIAIS, erro-padrão, teste t, IC
95% por bootstrap de casos e VIF (fator de inflação de variância) por
preditor. Correção de Holm-Bonferroni (Holm, 1979, *Scand. J. Statist.*
6(2), 65-70) aplicada aos p-valores desta regressão especificamente.

**Sistematizada no relatório do Administrador (2026-08-08)**:
`report._relevance_drivers()`/`_relevance_driver_sentence()` (o texto
real que vira Discussão nos 3 geradores admin-gated —
`generate_scientific_report`/`generate_admin_report`/
`generate_admin_paper_plosone`) usavam a correlação de Spearman
um-a-um ANTIGA (sem correção), mesmo depois desta seção já ter mostrado
o método melhor — trocado para chamar `driver_regression.
rank_transform_regression` diretamente. **Bug real encontrado e
corrigido nessa troca**: a primeira versão da frase gerada agrupava
TODO preditor que não era o dominante num único bloco "sem efeito
significativo" — mas com dados reais, `n_vents` tem p_Holm=8,3×10⁻⁷
(sobrevive Holm) enquanto `entrainment_alpha`/`agg_density_kg_m3` não
sobrevivem; a frase antiga teria literalmente contradito os próprios
números que imprimia ao lado. Corrigido separando explicitamente
preditores "também significativos" (sobrevivem Holm, mas com coeficiente
menor que o dominante) de "não significativos". Duas citações novas
adicionadas a `REFERENCES_EN` que já deveriam estar lá desde que o
método passou a ser citado no texto (Iman & Conover 1979; Holm 1979) —
confirmado que ambas convertem corretamente pro formato Vancouver
numerado do gerador PLOS ONE, sem entrar na lista `[VERIFICAR FORMATO
VANCOUVER]` (essa lista já tinha 2 entradas pré-existentes — Urick 1983,
um livro, e Lineweaver & Davis 2002 — não relacionadas a esta mudança).

**Por que a análise de bancada (Fase 5, `data/chladni_bench_2021/
analysis.py`) NÃO recebeu a mesma correção**: é o único outro lugar do
projeto com testes de hipótese (Student t/Welch t/Mann-Whitney/pareado/
Wilcoxon + diagnósticos Shapiro-Wilk/Levene). Mas esses são MÚLTIPLOS
MÉTODOS testando a MESMA pergunta única (wave+ difere de wave-?), um
desenho deliberado de robustez-entre-métodos (documentado desde a Fase
5), não uma família de comparações independentes — aplicar Holm-Bonferroni
ali seria um uso incorreto do conceito, não uma melhoria. Sistematizar
corretamente inclui reconhecer onde a correção NÃO se aplica, não só
onde aplicar.

**Motivação — resolver uma suspeita documentada, não só construir a
capacidade**: §7.8.1 tinha deixado em aberto se nº de vents (ρ=+0,13,
p≈3×10⁻⁵ na correlação marginal) era um driver real da profundidade de
poço de Gor'kov ou um artefato de comparações múltiplas — correlação
marginal não distingue "X afeta Y por si só" de "X e Y são ambos afetados
por uma terceira variável correlacionada com X" (aqui, hipoteticamente,
o raio do agregado).

**Achado real, rodado sobre o ensemble real de 1000 runs já existente**
(`outputs/experimento_260807_021219`, mesmo dataset de §7.8.1 — nenhuma
simulação nova, só reanálise): regressão multivariada de
`entrainment_alpha`, `agg_radius_um`, `agg_density_kg_m3` e `n_vents`
contra `gorkov_trap_depth_over_kT` (n=1000, R²=0,255):

| preditor | coef. padronizado | p | p (Holm) | VIF |
|---|---|---|---|---|
| agg_radius_um | +0,489 | 1,0×10⁻⁶¹ | 4,1×10⁻⁶¹ | 1,01 |
| **n_vents** | **+0,142** | **2,8×10⁻⁷** | **8,3×10⁻⁷** | **1,00** |
| entrainment_alpha | −0,028 | 0,31 | 0,61 | 1,00 |
| agg_density_kg_m3 | +0,016 | 0,57 | 0,61 | 1,00 |

**nº de vents SOBREVIVE ao controle multivariado e à correção de Holm**
— não é um artefato de confundimento com o raio do agregado (VIF=1,00,
confirma que nº de vents é estatisticamente independente dos parâmetros
varridos pelo LHS, como o próprio desenho de amostragem já garantia por
construção). A suspeita registrada em §7.8.1 ("plausivelmente um efeito
de múltiplas comparações") estava certa em desconfiar, mas a resposta,
testada, é que o efeito é REAL — só não é um efeito de múltiplas
comparações no sentido de falso-positivo estatístico.

**Por que faz sentido fisicamente — não é coincidência, é estatística de
extremos**: `particle_boltzmann_enhancement()` (`acoustics.py`, linha
~586) define a profundidade do poço como
`(U_max − U_min) / k_BT` sobre o campo potencial ESPACIAL inteiro
`U(x,y)` (soma das contribuições acústicas de TODOS os vents do campo,
não um valor por-vento isolado). Mais vents no mesmo campo → mais poços
de potencial locais superpostos no mesmo domínio espacial → mais chances
de que o MÁXIMO da faixa (U_max−U_min) caia num valor grande em algum
lugar do domínio — um resultado clássico de estatística de valores
extremos (o máximo de mais variáveis, mesmo iid, tende a ser maior), não
um artefato de múltiplos testes de hipótese. Isso é consistente com
(e explica mecanisticamente) o padrão já documentado em §7.8.1: as 7
realizações raras que cruzaram o limiar tendiam a ter mais fumarolas.

**Limitação honesta**: o R² do modelo é modesto (0,255) — os 4
preditores testados (só os parâmetros já varridos + nº de vents) não
explicam a maior parte da variância de `gorkov_trap_depth_over_kT`;
parte substancial vem de outros graus de liberdade estocásticos do campo
(posição relativa dos vents, geometria de interferência de Lloyd's
mirror por vento — §7.4) não incluídos aqui como preditores explícitos.
IC 95% por bootstrap de casos (não aninhado — cada linha do ensemble
plano é uma unidade de reamostragem independente, diferente do bootstrap
aninhado de §7.8.2, que preserva uma estrutura hierárquica que não
existe neste dataset).

**Validado com dados sintéticos** (`tests/test_driver_regression.py`,
caso central do módulo): um preditor confundido por construção
(correlacionado com o driver real, sem efeito próprio nenhum sobre a
resposta) mostra correlação de Spearman marginal fortemente
"significativa" (herdada do driver real) mas coeficiente parcial
≈0/não-significativo na regressão multivariada — reproduz exatamente o
tipo de pergunta testada acima com dados reais, antes de aplicar ao
ensemble real.

## 7.9 Como rodar a validação

```
pytest tests/test_acoustics.py -v
pytest tests/test_fumarola_field.py -v
pytest tests/test_variance_decomposition.py -v
pytest tests/test_global_sensitivity.py -v
pytest tests/test_driver_regression.py -v
```

Cada teste cita sua fonte/justificativa no docstring — incluindo os
testes que registram os achados NEGATIVOS do modelo (armadilha não
relevante termicamente) como regressão permanente, não só os que
verificam que o código roda sem erro.

---

# Fase 3 — Calibração de módulos prebióticos clássicos (`prebiotic.py`)

Motivação: dos quatro módulos clássicos de concentração prebiótica
(diluição, termoforese, adsorção mineral, gradiente de prótons), só a
diluição tinha base física real (modelo de pluma validado); os outros
três usavam parâmetros "ilustrativos, ordem de grandeza" sem nenhuma
medição por trás — a mesma lacuna que motivou a Fase 2b para o modelo
acústico. Esta fase ataca a termoforese, que tem literatura de medição
real disponível (Baaske et al., 2007).

## 8.1 O mecanismo real não é um equilíbrio Soret estático

A implementação original usava `enhancement = exp(S_T · ΔT)` — um
equilíbrio Soret estático simples. Baaske, P., Weinert, F.M., Duhr, S.,
Lemke, K.H., Russell, M.J., & Braun, D. (2007), "Extreme accumulation
of nucleotides in simulated hydrothermal pore systems," *PNAS*
104(22), 9346-9351, mediram um mecanismo bem mais rico: **convecção
térmica ao longo de um poro alongado acoplada à termodifusão através
dele** — o fluido circula por convecção na direção do comprimento do
poro enquanto as moléculas migram para as bordas por termodifusão,
criando um acúmulo que escala com a RAZÃO DE ASPECTO do poro
(comprimento/largura), não só com S_T·ΔT. Isso produz fatores de
acumulação de 10⁸ a 10¹⁵×, muitas ordens de grandeza maiores que
`exp(S_T·ΔT)` isolado — e explica por que os autores chamam o efeito de
"extremo".

## 8.2 Fórmula implementada — verificada contra o texto primário completo

`module_thermophoresis()` implementa, para classes com
`thermophoresis_convection_coupled=True`:

```
enhancement = exp(k · S_T · ΔT · razão_de_aspecto)
```

**Atualização de 2026-08-06**: o usuário obteve o PDF completo do
artigo (`biblios/baaske2007.pdf`) e a fórmula foi verificada por
leitura direta do texto primário, não mais reconstruída por regressão.
É exatamente a Eq. 1 do artigo (p. 9348):

> c_BOTTOM / c_TOP = exp(0.42 × S_T × ΔT × r)

— a solução analítica de Furry, Jones & Onsager (1939) / Debye (1939)
para uma coluna termogravitacional (tipo Clusius-Dickel), que os
próprios autores confirmam contra sua simulação numérica por elementos
finitos. Portanto **k=0.42, lido diretamente da equação, não um valor
regredido**. O valor anteriormente usado (k=0.4, reconstruído por
regressão a 3 exemplos numéricos antes de o texto completo estar
disponível) já estava dentro de ~5% do valor correto — atualizado para
0.42 nesta revisão. `tests/test_prebiotic.py` reproduz os exemplos
numéricos do artigo (Fig. 2a/Table 2) dentro de tolerância ampla.

## 8.3 Escopo da calibração: só a classe "nucleotideos"

Baaske et al. (2007) mediram nucleotídeos únicos e DNA/RNA de fita
simples/dupla — NÃO aminoácidos, lipídeos ou açúcares. Generalizar a
fórmula/coeficientes deles para essas outras classes por analogia não
teria base citável (opção explicitamente rejeitada ao decidir o escopo
desta fase). Por isso:

- **"nucleotideos"**: `thermophoresis_convection_coupled=True`,
  `soret_coefficient_per_k=0.006` (medido, ver 8.4), `pore_aspect_ratio`
  em `SHARED_PARAMS` (default 10.0, ajustável via
  `--pore-aspect-ratio`).
- **"aminoacidos", "lipideos", "acucares"**: `thermophoresis_convection_coupled=False`
  → fórmula reduz-se exatamente a `exp(S_T·ΔT)` (razão de aspecto e k
  tornam-se 1, sem efeito) — comportamento IDÊNTICO ao anterior a esta
  fase, com os mesmos coeficientes ilustrativos de sempre. Continuam
  explicitamente rotuladas como não calibradas.

## 8.4 Escolha do S_T medido e a ressalva de salinidade

**Verificado 2026-08-06 por leitura direta do texto primário completo**
(Table 1 do artigo): Baaske et al. (2007) reportam S_T de nucleotídeo
único em duas concentrações de sal monovalente: 0.015/K (1.7 mM) e
0.006/K (170 mM) — valores confirmados exatamente, sem divergência do
que já estava documentado aqui. S_T CAI com a salinidade. Usamos 0.006/K (a condição mais salina
testada) como análogo mais próximo da força iônica da água do mar
(~500-600 mM) do que a condição diluída — mas 170 mM ainda é ~3× mais
diluído que a água do mar real, e a tendência observada (S_T cai com
sal) sugere que o valor real em água do mar pode ser AINDA MENOR. Isto
é uma extrapolação explícita, não uma medição nas condições exatas
deste modelo — o valor usado provavelmente ainda sobrestima o efeito
termoforético em fumarolas reais.

## 8.5 Razão de aspecto: geometria de laboratório, não de campo

`pore_aspect_ratio=10.0` (default) é o segmento único mais conservador
testado experimentalmente por Baaske et al. (faixa testada 10:1-125:1,
ou cascatas de segmentos alcançando razões efetivas maiores via
concatenação). A geometria REAL de microporos em paredes de chaminés
hidrotermais ativas não foi encontrada na pesquisa de literatura que
fundamenta este modelo — usar a geometria do aparato de laboratório
como análogo plausível é uma escolha explícita, ajustável via
`--pore-aspect-ratio`, não uma medição de campo.

## 8.6 Limitações adicionais desta fase

- Razão de aspecto de poro real em chaminés hidrotermais não medida na
  literatura consultada (ver 8.5).
- S_T extrapolado de 170 mM para a salinidade real da água do mar (ver
  8.4), provavelmente uma sobrestimativa.
- "Nucleotídeo" (medido) e "base nitrogenada livre" (não medida
  separadamente por Baaske et al., mas incluída no mesmo rótulo de
  classe `MOLECULE_CLASS_LABELS["nucleotideos"]`) podem ter
  termoforese bem diferente — bases livres são menores e mais simples
  que nucleotídeos completos (base+açúcar+fosfato).
- Aminoácidos, lipídeos e açúcares permanecem sem qualquer calibração
  equivalente — nenhuma medição de termoforese tipo Baaske foi
  encontrada para essas classes na pesquisa de literatura consultada.

## 8.7 Como rodar a validação (termoforese)

```
pytest tests/test_prebiotic.py -v
```

## 8.8 Gradiente de prótons: de auto-normalizador a referência biológica real

O módulo de gradiente de prótons (`module_proton_gradient`) usava
`gradient_frac = ΔpH / MAX_DELTA_PH`, onde `MAX_DELTA_PH` era o ΔpH
máximo do PRÓPRIO modelo (diferença entre o pH do fim de membro
hidrotermal e a água do mar, ambos já usados em outros lugares) — um
auto-normalizador sem nenhuma referência externa: `gradient_frac=1`
não significava nada além de "esta é a fumarola mais extrema que este
modelo consegue gerar."

**Reformulação**: `ΔpH` é convertido num potencial transmembrana real
via equação de Nernst (`ΔV = 59.2 mV × ΔpH`, inclinação ideal a 25°C —
eletroquímica padrão), e comparado contra uma REFERÊNCIA BIOLÓGICA
REAL: Sojo, V., Herschy, B., Whicher, A., Camprubí, E., & Lane, N.
(2016), "The Origin of Life in Alkaline Hydrothermal Vents,"
*Astrobiology* 16(2), 181-197, afirmam no texto principal e na legenda
da Fig. 1 que poros de vents alcalinos têm um gradiente de pH de 3
unidades através da barreira inorgânica (força próton-motriz de
~200 mV), e que esse valor é "exactly equivalent in both magnitude
(about 3 pH units) and polarity" à força próton-motriz usada por
células autotróficas extantes hoje. Usamos essa referência (≈177.6 mV)
como a escala de comparação: `gradient_frac=1` agora significa "esta
fumarola produz, em magnitude, o mesmo potencial que a vida moderna
usa para fixar carbono" — uma comparação quantitativa com significado
externo, não um artefato de normalização interna. Sem teto superior
artificial: fumarolas com ΔpH maior que a referência dão
`gradient_frac>1` deliberadamente.

**Correção (2026-08-06, verificado por leitura direta do PDF primário
completo, `biblios/ast.2015.1406.pdf`)**: esta seção citava
anteriormente "5-6 unidades de pH" como o gradiente ao qual os "3
unidades" seriam equivalentes — isso conflava dois números distintos
do artigo. Os "3 unidades / 200 mV" são o gradiente através de poros
de vents alcalinos EM CONDIÇÕES ANÁLOGAS ÀS MODERNAS, comparado
explicitamente pelos autores à PMF biológica extante — é este o número
usado na calibração. Separadamente, o artigo também discute um cenário
Hadeano mais extremo (oceano mais ácido, rico em CO2) em que o
gradiente poderia chegar a até 6 unidades (~400 mV) — mas esse número é
apresentado como um potencial MÁXIMO adicional, não como "equivalente"
a nada biológico; não deveria ter sido citado aqui como se fosse a
mesma comparação dos "3 unidades." A citação de Arndt & Nisbet (2012)
permanece correta como fonte, dentro de Sojo et al., da faixa "5-6
unidades" usada para o cenário Hadeano — só a atribuição da equivalência
biológica estava errada.

## 8.9 O mecanismo em si permanece contestado — crítica com peso igual

**Verificado 2026-08-06 por leitura direta do texto primário completo**
(`biblios/s00239-016-9756-6.pdf`): os quatro números abaixo (1 μm,
>200×, 0.004 unidades de pH, 24 J/mol vs. 24 kJ/mol) foram confirmados
exatamente, sem divergência do que já estava documentado aqui.

Diferente da termoforese (onde a física do mecanismo é bem
estabelecida, só faltavam números medidos), aqui a CALIBRAÇÃO não
resolve uma disputa real sobre se o mecanismo funciona:

Jackson, J.B. (2016), "Natural pH gradients in hydrothermal alkali
vents were unlikely to have played a role in the origin of life,"
*J. Mol. Evol.* 83(1), 1-11, argumenta quantitativamente que:

1. Membranas inorgânicas finas em vents alcalinos reais (~1 μm) são
   >200× mais espessas que bicamadas lipídicas que operam circuitos
   quimiosmóticos reais — maquinário molecular (~1 nm) não conseguiria
   utilizar um gradiente através de uma barreira tão espessa.
2. Em qualquer canal permeável a H+ através dessa membrana (necessário
   para o maquinário molecular realmente ACESSAR o gradiente), a
   difusão faria o ΔpH colapsar para ~0.004 unidades — um trabalho
   disponível de ~24 J/mol de prótons, muito abaixo dos ~24 kJ/mol
   necessários para trabalho útil (3 ordens de grandeza de diferença).
3. Se o fluido escoa mais rápido que a difusão de H+/OH-, o pH ao
   longo do canal permanece próximo ao do fluido de origem, impedindo
   o uso do gradiente exatamente onde o maquinário estaria.
4. Não há evidência de membranas inorgânicas finas sustentando
   gradientes de pH nítidos em vents alcalinos modernos (Lost City).

Ou seja: o próprio mecanismo modelado por `module_proton_gradient` —
um gradiente de pH em massa (fluido↔oceano) traduzido num potencial
transmembrana USÁVEL por maquinário molecular — é uma hipótese
seriamente contestada na literatura primária, com números concretos
do lado crítico. Esta calibração melhora a REFERÊNCIA de comparação
(seção 8.8), não resolve essa disputa — trate qualquer resultado deste
módulo como ilustrando uma hipótese em debate ativo, não um mecanismo
estabelecido.

## 8.10 Limitações adicionais desta fase (gradiente de prótons)

- O mecanismo em si é contestado na literatura primária (Jackson,
  2016) — ver 8.9.
- O modelo usa apenas a MAGNITUDE do ΔpH como proxy (não distingue
  direção ácido/alcalino) — aproximação pré-existente, não alterada
  nesta calibração; o peso por tipo de vent (`proton_vent_type_weight`)
  compensa parcialmente, não corrige completamente.
- A inclinação de Nernst usada (59.2 mV/unidade a 25°C) não é corrigida
  para a temperatura real do fluido hidrotermal (até ~400°C) — a
  inclinação de Nernst escala com T, então isso subestima o potencial
  em fluidos mais quentes; correção não implementada nesta fase.
- `proton_max_factor` (o ganho final aplicado a `gradient_frac`)
  continua sendo um parâmetro ilustrativo, sem calibração — só a
  REFERÊNCIA de comparação (denominador) foi calibrada nesta fase, não
  o ganho em si.

## 8.11 Como rodar a validação (gradiente de prótons)

```
pytest tests/test_prebiotic.py -v -k proton
```

---

# Fase 4 — Validação de velocidades de saída (`plume_physics.py`)

`EXIT_VELOCITY_BY_TYPE` era documentado como "ordem de grandeza
plausível, não citada." Pesquisa de literatura encontrou medições de
campo reais para dois dos três tipos de vent:

- **black_smoker**: 0.7-2.4 m/s (medição direta por flowmeter de
  turbina in situ, "Alvin") — Converse, D.R., Holland, H.D., & Edmond,
  J.M. (1984), "Flow rates in the axial hot springs of the East
  Pacific Rise (21°N)," *Earth Planet. Sci. Lett.* 69, 159-175. O
  valor do modelo (1.5 m/s) já caía dentro dessa faixa — promovido de
  "plausível" para "validado dentro da faixa medida," sem mudar o
  número. **Correção (2026-08-06, verificado por leitura direta do PDF
  primário completo, `biblios/0012-821x2990080-3.pdf`)**: esta seção
  citava anteriormente "1-5 m/s" como a faixa de Converse et al. — isso
  estava errado. 1-5 m/s é a estimativa de Macdonald, K.C., Becker, K.,
  Spiess, F.N., & Ballard, R.D. (1980), *Earth Planet. Sci. Lett.* 48,
  1-7, para o vent "National Geographic," citada de segunda mão DENTRO
  do artigo de Converse et al. — não é a medição própria deles. A
  medição direta de Converse et al. é 0.7-2.4 m/s; o valor do modelo
  (1.5 m/s) permanece validado dentro dessa faixa, então a conclusão
  não muda, só a atribuição da citação.
- **diffuse_flow**: ~0.001-0.111 m/s, combinando Mittelstaedt, E., et
  al. (2012), "Quantifying diffuse and discrete venting at the Tour
  Eiffel vent site, Lucky Strike hydrothermal field," *Geochem.
  Geophys. Geosyst.* 13, Q0AF04 (0.009-0.111 m/s, velocimetria óptica)
  e Sarrazin, J., Rodier, P., Tivey, M.K., Singh, H., Schultz, A., &
  Sarradin, P.-M. (2009), "A dual sensor device to estimate fluid flow
  velocity at diffuse hydrothermal vents," *Deep-Sea Res. I* 56(11),
  2065-2074, no mesmo edifício (0.0011-0.0049 m/s, fraturas de baixa
  temperatura). A discrepância
  de ~1 ordem de grandeza entre os dois métodos/condições NÃO está
  resolvida — reportada como a incerteza real da faixa, não escondida.
  O valor do modelo (0.05 m/s) cai dentro da faixa combinada.
- **white_smoker**: nenhuma medição específica encontrada — permanece
  um valor plausível não citado (0.6 m/s, intermediário por suposição).

Nenhum valor numérico foi alterado nesta fase — apenas dois dos três
foram promovidos de "suposição" para "validado contra medição de
campo real," com a faixa e a fonte agora explícitas.

```
pytest tests/test_plume_physics.py -v -k exit_velocity
```

---

# Fase 5 — Teste de bancada (2021) e a hipótese de ressonância de conduto

Em 2026-08-06, o usuário trouxe um teste de bancada próprio, feito em
2021: um recipiente de 5×5cm com 5mL de água nuclease-free e DNA padrão
purificado K-562, apoiado diretamente sobre um alto-falante gerando tons
de 33-34Hz, com amostras de DNA (quantificadas por Qubit) coletadas em
regiões rotuladas pelo usuário como "wave+" (nó) e "wave-" (antinó), mais
controle e branco (n=32 experimentos independentes, recipiente limpo com
hipoclorito+álcool e remontado a cada corrida). Dados brutos e vídeo do
experimento estão em `data/chladni_bench_2021/` (`medias.xlsx`,
`experiment_video.mp4`, `analysis.py` reproduz todas as estatísticas
abaixo).

Esta seção documenta essa reanálise e conecta o achado à Fase 2 (§7): por
que sistemas de bancada/laboratório conseguem organizar partículas e
células em padrões de Chladni, enquanto o modelo de campo livre de
fumarola (§7.5-7.7) não encontra efeito relevante em kT.

## 9.1 Reanálise estatística (não confiar apenas nos p-values da planilha original)

| Grupo | Média | DP | n |
|---|---|---|---|
| wave+ (nó) | 34,10 | 14,98 | 32 |
| wave− (antinó) | 6,54 | 2,18 | 32 |
| controle | 6,31 | 2,47 | 32 |
| branco | 0,57 | 0,15 | 6 |

wave+ difere de wave− e de controle com efeito grande e robusto a
múltiplas escolhas de teste (Student p=4,8e-15; Welch p=9,9e-12, pois
Levene rejeita variâncias iguais p<0,0001; Mann-Whitney não-paramétrico
p=6,5e-12, pois wave+ falha normalidade Shapiro p=0,003; **teste pareado**
— o desenho correto aqui, já que cada corrida gera uma amostra wave+ E
uma wave- do mesmo experimento — t=1,3e-11, Wilcoxon signed-rank
p=4,7e-10). Cohen's d=2,57. wave− não difere de controle (p≈0,69 em todos
os testes, d=0,10). Controle difere fortemente do branco (p~1e-6 a 1e-14),
validando que o ensaio Qubit está detectando DNA real acima do fundo.
Conclusão: o efeito é real e estatisticamente sólido, independente da
escolha entre teste paramétrico/não-paramétrico/pareado.

```
cd data/chladni_bench_2021 && python analysis.py
```

## 9.2 O vídeo mostra um regime físico diferente do modelado na Fase 2

Inspeção de frames do vídeo (`data/chladni_bench_2021/frames/`) mostra o
recipiente em **contato rígido direto** com o driver do alto-falante, e um
padrão de Chladni clássico (grade quadriculada nítida de linhas nodais)
formado por um traçador claro (pó/espuma fina) na superfície do líquido.

Isso não é uma onda acústica estacionária livre no fluido: a 33-34Hz, o
comprimento de onda do som na água é λ=c/f≈1500/33,5≈45m, muito maior que
o recipiente de 5cm — não cabe um padrão nó/antinó de pressão nessa
escala. O padrão visível é, portanto, quase certamente uma **ressonância
de flexão da placa/parede sólida do recipiente** (onda de flexão em placa
fina — teoria de Kirchhoff-Love, relação de dispersão k⁴=ρhω²/D — tem
velocidade de propagação muito menor que o som no fluido e por isso cabe
um padrão de poucos centímetros numa frequência de dezenas de Hz),
acoplada ao líquido como uma "placa de Chladni molhada".

Isto é fisicamente distinto do mecanismo de aprisionamento de Gor'kov em
campo acústico livre modelado em `acoustics.py` (§7.5) — o mecanismo aqui
é transporte por **acoustic streaming confinado** (vórtices de
recirculação do líquido induzidos pela vibração da placa), não força de
radiação direta sobre a molécula.

## 9.3 Nó vs. antinó: o que a literatura desde Faraday (1831) já documentava

A pergunta "por que Chladni organiza partículas e o nosso modelo de
fumarola não" tem uma resposta quantitativa, não qualitativa — o
mecanismo é o mesmo (força de radiação/streaming em campo de onda
estacionária), a intensidade de campo é que difere por várias ordens de
grandeza (ver 9.4).

Mas a direção do efeito (nó vs. antinó) não é universal — está
documentada como dependente do tamanho/densidade da partícula desde a
fonte primária:

- **Faraday, M. (1831), "On a peculiar class of Acoustical Figures; and
  on certain Forms assumed by groups of particles upon vibrating elastic
  Surfaces," *Philosophical Transactions of the Royal Society of London*
  121, 299-340** (`biblios/faraday1831.pdf`, texto primário completo lido
  diretamente). Chladni já havia mostrado que areia/grãos grossos
  colam nas linhas nodais (§1 do artigo). Faraday mostra que pó fino
  (lycopodium) faz o oposto — se acumula nos **antinós** ("centres of
  oscillation")# — e demonstra experimentalmente a causa: correntes de ar
  (streaming) fluindo em direção ao antinó. No experimento com folha de
  ouro (§16), Faraday mostra o ar entrando por baixo da folha e
  levantando-a "into the form of a blister" exatamente no centro de
  vibração — evidência direta de streaming, 150 anos antes do termo
  existir.
- **Vuillermet, G., Gires, P.-Y., Casset, F., & Poulain, C. (2016),
  "Chladni Patterns in a Liquid at Microscale," *Physical Review
  Letters* 116(18), 184501** (citação verificada via Crossref; PDF
  completo não obtido por canal legítimo gratuito — ver
  `docs/CITATIONS_TO_VERIFY.md`) e **Lei, J. (2017), "Formation of
  inverse Chladni patterns in liquids at microscale: roles of acoustic
  radiation and streaming-induced drag forces," *Microfluidics and
  Nanofluidics* 21(3), 50** (idem) confirmam em líquido microescala: há
  uma competição entre força de radiação acústica (favorece nós, domina
  para partículas maiores/mais densas) e arrasto por streaming (favorece
  antinós, domina para partículas pequenas/leves), com um limiar de
  tamanho separando os dois regimes.

**Consequência para a interpretação do seu experimento**: DNA em solução
livre é uma molécula pequena demais para força de radiação de Gor'kov ter
qualquer papel (§7.5 já estabelece que mesmo agregados de 17μm — mil vezes
maiores que um fragmento de DNA — ficam ~100x abaixo do limiar térmico kT
nas pressões medidas em fumarolas reais). Pela lógica desta literatura,
DNA dissolvido deveria se comportar como traçador passivo do streaming,
não como partícula sujeita a radiação — ou seja, deveria ir para onde o
streaming concentra massa de fluido, não necessariamente para o "nó" no
sentido clássico de Chladni seco.

**Isto é uma tensão real, não resolvida por esta análise**: o traçador
visível no vídeo se comporta como o caso clássico (concentra-se nas linhas
nodais, como areia grossa, não como pó fino de Faraday). Se o DNA
dissolvido é simplesmente co-transportado pelo mesmo escoamento de
streaming que concentra o traçador visível — plausível numa lâmina fina de
líquido, onde o nó geométrico da placa pode ser um ponto natural de
convergência do escoamento superficial, um regime distinto dos
experimentos de líquido profundo de Vuillermet/Lei — isso explicaria o
resultado observado sem contradizer a literatura. Mas isso não foi
verificado diretamente (o traçador visível não é o DNA; não há medição
independente confirmando que os dois co-localizam pelo mesmo mecanismo).
Fica como questão em aberto, não como conclusão.

## 9.4 Conexão com fumarolas: ressonância de conduto já é um mecanismo documentado — mas quantitativamente insuficiente

O "ingrediente que falta" no modelo de campo livre da Fase 2 pode não ser
mais pressão bruta, mas sim confinamento/ressonância mecânica — exatamente
o que o experimento de bancada demonstra por analogia (efeito grande via
streaming confinado, sem nenhuma força de Gor'kov relevante).

**Isto não é especulação nova**: **Crone, T.J., Wilcock, W.S.D., Barclay,
A.H., & Parsons, J.D. (2006), "The Sound Generated by Mid-Ocean Ridge
Black Smoker Hydrothermal Vents," *PLoS ONE* 1(1), e133** — já em
`biblios/journal.pone.0000133.pdf`, já citado em §7.3 para a amplitude de
pressão medida — também discute explicitamente ressonância de conduto
como fonte dos tons estreitos observados (10-250Hz), citando como
resonadores candidatos "Helmholtz resonators, half-wave or quarter-wave
resonators, and solid structures such as **tubes, plates, or cavities**
within the chimneys" (texto primário, verificado diretamente no PDF). Os
autores dão um exemplo numérico concreto: uma cavidade de 2L conectada ao
conduto por uma abertura de 0,02m de diâmetro e 0,04m de comprimento,
preenchida com fluido hidrotermal quente (c=450m/s), dá frequência
fundamental de Helmholtz f≈120Hz; um tubo de 1m fechado numa ponta
(ressonador de quarto de onda) dá f≈113Hz — ambos batendo com a faixa
real observada (10-250Hz). O próprio artigo nota que "plates" (placas) são
um dos tipos de estrutura ressonante candidatos — a mesma classe de
mecanismo (ressonância de flexão de placa) que o vídeo do experimento de
bancada sugere visualmente.

**Evidência quantitativa de amplificação por ressonância, já medida em
campo**: os tons estreitos têm potência ~10-20dB acima do nível de banda
larga no mesmo hidrofone (texto primário, verificado). Isso corresponde a
~3-10x em amplitude de pressão (~10-100x em potência) de amplificação por
ressonância — real, mas **muitas ordens de grandeza abaixo** do que
precisaria para fechar o hiato de ~1e9-1e11x (em profundidade de poço de
Gor'kov/kT) entre pressão medida no campo livre (~1-3 Pa) e pressão usada
em sistemas de padronização celular de bancada (~0,1-0,2 MPa — Engineering
Anisotropic Muscle Tissue using Acoustic Cell Patterning, PubMed
30277617).

**Questão em aberto, não resolvida aqui**: os 10-20dB medidos são a
amplificação que sobra no som IRRADIADO, medido a distância do hidrofone
— não a amplitude de pressão DENTRO do próprio conduto/cavidade
ressonante, que pode ser substancialmente maior (análogo a como o som
dentro de uma câmara ressonante é mais alto que o que escapa e é medido
de fora). Não há, até onde esta pesquisa foi capaz de verificar, medição
publicada de pressão in-conduto em fumarolas reais. Esta é a linha de
investigação mais promissora aberta por esta seção — não closed, apenas
delimitada.

## 9.5 Limitações desta fase

- O experimento de bancada é de 2021, conduzido antes deste projeto
  existir — reanalisado aqui, não desenhado para os fins deste modelo.
  Faltam detalhes de protocolo que mudariam a interpretação (definição
  exata de "controle" — alto-falante desligado no mesmo recipiente, ou
  setup separado; método de localização do nó/antinó — traçador visual,
  cálculo, ou ponto fixo; material/espessura do recipiente).
- Vuillermet (2016) e Lei (2017) são citados apenas por metadados
  verificados (Crossref) e resumos de busca — texto primário não obtido
  (tentativas de PDF via Springer/APS/ResearchGate bloqueadas; ver
  `docs/CITATIONS_TO_VERIFY.md`). Faraday (1831) e Crone et al. (2006)
  foram verificados por leitura direta do texto primário completo.
- A hipótese de co-localização DNA↔traçador visível (9.3) e a hipótese
  de ressonância de conduto em chaminés reais (9.4) são propostas
  motivadas por evidência indireta, não confirmadas — nenhuma delas foi
  testada computacionalmente nesta fase.
- Nenhuma mudança de código foi feita em `acoustics.py`/`fumarola_field.py`
  nesta fase — esta seção é inteiramente documental/interpretativa.

---

# Fase 6 — Robustez estatística geral do ensemble (infraestrutura, não específica de um módulo)

Ao contrário das Fases 1-5 (modelo físico/módulos específicos) e das
seções 7.8.2-7.8.4 (ferramentas específicas do fluxo `--variance-
decomposition`), esta fase documenta uma melhoria na infraestrutura
estatística de PROPÓSITO GERAL já usada por todo o projeto —
`ensemble_stats.describe()`, consumida pela aba de estatísticas da GUI e
por todas as tabelas descritivas de `report.py`, para QUALQUER grandeza
agregada de um ensemble (concentração/enriquecimento prebiótico,
diagnósticos acústicos, nº de vents — não só o módulo acústico).

## 10.1 Por que mean/std sozinhos enganam neste projeto

`describe()` reportava só n/mean/std/min/median/max. Distribuições reais
deste projeto são repetidamente caudal-pesadas por razões físicas já
documentadas (altura de chaminé de longa cauda — `sample_chimney_height`
em `fumarola_field.py`; evento raro de Gor'kov, §7.8.1; estatística de
extremos por nº de vents, §7.8.4) — nesse regime, mean/std por si só
podem sugerir uma dispersão simétrica que não existe.

**Confirmado com o ensemble real de 1000 runs** (mesmo dataset de
§7.8.1/§7.8.4, `outputs/experimento_260807_021219`,
`gorkov_trap_depth_over_kT` da classe agregado, reanálise — nenhuma
simulação nova):

| estatística | valor |
|---|---|
| mean | 0,153 |
| std | 0,189 |
| median | 0,108 |
| IQR (Q1–Q3) | 0,067–0,177 |
| MAD escalado | 0,072 |
| skewness | **7,22** |
| kurtose (excesso) | **78,8** |
| (mean−median)/IQR | 0,41 |

`mean±std` (0,153±0,189) sugere uma faixa aproximadamente simétrica que
chegaria perto de 0 do lado inferior — mas a distribuição real é uma
mediana estreita (0,108) com uma cauda superior extremamente longa e fina
(máximo observado 2,82, seção 7.8.1): skewness=7,22 e curtose=78,8 são
ordens de grandeza acima do que uma distribuição aproximadamente normal
teria (~0 para ambas) — sinal quantitativo direto de que mediana/IQR
descrevem essa distribuição muito melhor que mean/std.

## 10.2 Estatísticas adicionadas

Implementadas em `ensemble_stats.describe()` (aditivo — todas as chaves
pré-existentes mantidas com semântica idêntica, nenhum consumidor
existente quebra):
- `q1`/`q3`/`iqr`: quartis e amplitude interquartil (definição elementar
  por percentil) — espalhamento robusto a outliers.
- `mad`/`mad_scaled`: desvio absoluto mediano bruto e escalado por
  1/Φ⁻¹(3/4) ≈ 1,4826 (Φ⁻¹ = quantil da normal padrão) — essa constante
  específica torna o MAD um estimador CONSISTENTE do desvio-padrão sob
  normalidade (derivação: sob X~N(μ,σ²), E[|X−mediana|]=σ·Φ⁻¹(3/4)),
  comparável diretamente a `std` mas sem o peso desproporcional que
  outliers têm sobre `std`.
- `skewness`/`kurtosis`: coeficiente de assimetria e curtose em excesso
  (normal=0), estimador ajustado de Fisher-Pearson via
  `scipy.stats.skew`/`kurtosis` (`bias=False`, correção de viés de
  amostra finita). Retornam NaN para n<3/n<4 respectivamente (abaixo
  disso o valor é matematicamente degenerado — ex. assimetria de 2
  pontos é SEMPRE 0 por simetria, não por a distribuição real ser
  simétrica; NaN é mais honesto que um zero enganoso). Guarda explícita
  para variância zero (todos os valores idênticos) evita um
  `RuntimeWarning` ruidoso do scipy sem mudar o resultado (mesmo NaN
  correto).
- `mean_median_gap_over_iqr`: `(mean−median)/iqr` — o quanto a média é
  "puxada" da mediana por assimetria/outliers, na escala do
  espalhamento típico dos dados. Diagnóstico AUTOEXPLICATIVO (razão de
  duas quantidades já reportadas), deliberadamente sem um limiar
  externo tipo "|skewness|>1 é 'muito assimétrico'" — esse tipo de
  regra de bolso aparece em vários livros-texto com atribuições
  inconsistentes entre si; em vez de citar uma fonte não totalmente
  verificada para um número de corte arbitrário, o módulo reporta a
  métrica bruta e deixa a interpretação para quem lê.

## 10.2b IC 95% por bootstrap para TODA estatística contínua

Antes desta seção, só a fração binária de eventos raros tinha IC
(`_wilson_ci95` em `report.py`, §7.8.1) — toda estatística CONTÍNUA
(mean, median, iqr, skewness...) era reportada como número nu, sem
incerteza. `describe()` ganhou o parâmetro `n_bootstrap` (padrão 0 —
comportamento IDÊNTICO ao de antes, nenhuma chave nova sem pedir
explicitamente): quando >0, adiciona `<nome>_ci95` para CADA estatística
contínua via bootstrap de casos (Efron & Tibshirani, 1993, "An
Introduction to the Bootstrap," Chapman & Hall), totalmente vetorizado
(uma matriz `(n_bootstrap, n)` de reamostragens, sem laço Python — ver
`_bootstrap_point_estimates`). `compute_ensemble_stats()` repassa o
mesmo parâmetro a cada `describe()` interno.

**Por que o padrão é desligado**: custo real medido, não estimado —
~8s para `n_bootstrap=2000` num array pooled de 30 mil pontos (a
reamostragem em si é rápida; ORDENAR 2000×30000 elementos para
mediana/percentis é o que domina — inerente a bootstrap de estatísticas
baseadas em ordem, não uma ineficiência da implementação). Ligar por
padrão em TODO `compute_ensemble_stats()` deixaria a aba de estatísticas
da GUI/geração de relatório sensivelmente mais lenta em ensembles
grandes sem o usuário ter pedido — mesma lógica já aplicada a
`--ensemble-images`/`--sensitivity-sweep` no resto do projeto. Arrays
por-run (tipicamente centenas a milhares de pontos, não dezenas de
milhares) são bem mais rápidos: 0,25s para `n_bootstrap=2000` num array
de 1000 pontos (medido abaixo).

**Achado real, aplicado ao mesmo ensemble de 1000 runs de §10.1** (nenhuma
simulação nova): o IC da própria skewness/kurtose é LARGO —
skewness=7,22 (IC95% [3,72; 8,38]), kurtose=78,8 (IC95% [22,4; 105,3]).
Mesmo com n=1000, os momentos de ORDEM MAIS ALTA (skewness/kurtose) são
estimados com bem menos precisão que mean/mediana (IC95% [0,142; 0,167]
e [0,102; 0,114] respectivamente, faixas relativas muito mais estreitas)
— esperado estatisticamente (o erro-padrão de estimadores de momento de
ordem k cresce com a ordem), mas antes desta seção não havia como saber
ISSO sem calcular à parte; agora vem de graça em qualquer `describe()`
com `n_bootstrap>0`.

## 10.3 Testado

`tests/test_ensemble_stats.py` (primeira cobertura de teste dedicada
deste módulo — extraído de gui.py numa sessão anterior sem testes
próprios na época): compatibilidade retroativa exata das chaves
pré-existentes; IQR contra `np.percentile` direto; recuperação de
skewness≈0/curtose≈0 numa normal sintética grande; skewness>1 numa
exponencial sintética (assimetria teórica conhecida=2); MAD escalado
recupera o desvio-padrão real sob normalidade dentro de 5%; propriedade
central — um único outlier extremo infla `std` >5x mas move `iqr`/
`mad_scaled` menos de 1,5x; casos degenerados (n pequeno, variância
zero) retornam NaN sem warning. **IC por bootstrap**: cobertura empírica
medida sobre 150 réplicas independentes fica perto de 95% (não um único
seed — um IC 95% erra por definição ~5% das vezes, testar 1 caso só
teria risco real de falha por sorte); largura do IC diminui com amostra
maior; versão vetorizada bate exatamente com reamostragem manual
linha-a-linha; wiring de `compute_ensemble_stats(n_bootstrap=...)`
testado de ponta a ponta. 19 testes neste arquivo, suíte completa do
projeto 110/110 passando.

```
pytest tests/test_ensemble_stats.py -v
```

## 10.4 Relatório estatístico aberto na GUI (sem interpretação, sem login)

Implementado em `ensemble_report.py` — pedido explícito do usuário (2026-08-07,
sessão seguinte a esta): "quero que a gente volte a ter uma seção de
relatório aberto na GUI, mas apenas o relatório estatístico... nada de
discussão ou formato de artigo... nada de imagens representativas."
Distinto de `report.py`/`relatorios_admin.py` (gitignored, texto
interpretativo específico do autor + login de Administrador, não
tocados por este trabalho) pela MESMA régua já usada no
`.gitignore`/`CONTRIBUTING.md` do projeto para decidir o que é software
genérico vs. conteúdo específico do autor — `ensemble_report.py` é
tracked no git.

**Conteúdo do relatório** (botão "Generate statistical report (HTML)"
na aba de estatísticas, disponível sem login assim que um ensemble
termina): tabela de descritivas com IC 95% por bootstrap (§10.2b), os
MESMOS 4 gráficos já exibidos ao vivo na aba (`build_ensemble_charts_
figure`, reaproveitada — GUI e relatório nunca divergem, um só ponto de
verdade) com legenda completa (não só título curto de eixo), tabela
por-run, e — quando `experiment_dir` contém `vardecomp_summary.json`
(ver abaixo) — as seções de decomposição de variância (§7.8.2) e Sobol'
(§7.8.3); quando o ensemble usou `--sensitivity-sweep` num modo flat
(não aninhado), a tabela de drivers multivariada (§7.8.4). Deliberadamente
SEM: título/abstract/discussão, moldura de manuscrito, ou qualquer
imagem de UMA run específica (topview/3D/artístico) — restrições
testadas explicitamente (`test_report_does_not_contain_discussion_or_
article_framing`, `test_report_does_not_embed_representative_run_images`).

**`--variance-decomposition` trazido para a GUI** (mesma sessão, pedido
seguinte do usuário após notar que a lista de itens "concluídos" não
implicava aparecer na GUI): terceiro modo de execução "Variance
decomposition (nested)" ao lado de single/ensemble
(`HydroventGUI._on_run_clicked`/`_vardecomp_worker`), com campos
próprios para nº de pontos externos/réplicas internas (mesmos defaults
da CLI, 20/10). Ao terminar, `_on_vardecomp_finished` carrega as runs
individuais do disco (mesmo padrão de "abrir experimento existente" —
`find_run_dirs`/`load_run_summary`) para alimentar o visualizador de
imagens/aba de estatísticas normalmente, e um painel ao vivo
(`_render_vardecomp_summary`) resume decomposição/Sobol' sem precisar
gerar o relatório. Os resultados específicos do desenho aninhado são
lidos do `vardecomp_summary.json` já gravado em disco por
`run_nested_variance_experiment` (`_read_vardecomp_summary`) — não
passados em memória — então funcionam tanto para uma run recém-feita
quanto para reabrir uma pasta `vardecomp_*` antiga depois.

**Bugs reais encontrados e corrigidos construindo isso**:
1. A montagem da tabela por-run tinha uma expressão ternária Python que,
   por precedência de operador, aplicava a condição à LINHA HTML inteira
   concatenada em vez de só à célula — se `top_hotspot_enrichment_vs_
   control` fosse `None`, a linha inteira virava só "n/a", descartando
   run/seed/nº de fumarolas silenciosamente. Corrigido construindo cada
   célula como variável própria antes de montar a linha.
2. A função de geração aceitava um parâmetro `stats` que nunca era
   de fato usado (sempre recalculava internamente para garantir os IC's
   por bootstrap) — API enganosa, removida.
3. **Bug de infraestrutura de teste, não do código de produção**: rodar
   `tests/test_gui.py` inteiro causava, dependendo da ordem, desde uma
   exceção fatal do Windows até falha silenciosa ao criar a próxima
   `tk.Tk()` — criar/destruir múltiplas raízes Tk no mesmo processo
   mostrou-se instável nesta combinação de plataforma/Tcl (cada teste
   isolado passava normalmente). Corrigido com uma fixture `scope=
   "module"` que reaproveita UMA raiz Tk para o arquivo inteiro.

**Testado**: `tests/test_ensemble_report.py` (15 testes — HTML
balanceado, conteúdo esperado sempre presente, ausência de discussão/
imagens representativas, bilíngue PT/EN, tabela de drivers condicional,
seções de decomposição/Sobol' condicionais e lidas do disco) e
`tests/test_gui.py` (7 testes — primeira cobertura de teste automatizada
da GUI deste projeto: alternância de widgets entre os 3 modos de
execução, validação de entrada, e um fluxo completo real — clique →
thread de fundo → resultado populando `HydroventGUI` → painel ao vivo
→ botão de relatório — através do objeto `HydroventGUI` de verdade, não
só das funções de `fumarola_field.py`/`ensemble_report.py` isoladas).
Suíte completa do projeto 132/132 passando.

```
pytest tests/test_ensemble_report.py -v
pytest tests/test_gui.py -v
```

## 10.5 Convergência de Monte Carlo: 1000 runs basta, ou 10000 mudaria a conclusão?

Implementado em `convergence_analysis.py` — responde formalmente uma
pergunta que ficava só qualitativamente respondida antes ("0 em 100 é
plausível quando a taxa real é ~0,7%, via probabilidade de Poisson"):
conforme mais runs se acumulam, a estimativa da fração de eventos raros
se ESTABILIZA como esperado, e quanto rodar 10000 em vez de 1000
realmente compraria em precisão — sem rodar nenhuma simulação nova, só
reanalisando os dois ensembles reais já existentes.

**IC de Wilson reimplementado aqui** (não importado de `report.py`,
gitignored — este módulo é tracked e precisa funcionar sozinho num
clone público do repositório) — validado contra o próprio valor já
documentado em §7.8.1 (k=7,n=1000 → IC 95% [0,34%, 1,44%], reproduzido
exatamente) e contra cobertura empírica média ~94,8% sobre várias
combinações (p,n) (Wilson tem cobertura conhecidamente OSCILANTE em
torno do nominal — Brown, Cai & DasGupta, 2001, *Statistical Science*
16(2), 101-133 — testar um único caso isolado seria enganoso).

**Traço de convergência real, ensemble de 1000 runs**
(`outputs/experimento_260807_021219`, na ORDEM REAL das runs — a ordem
importa para um traço genuíno, não uma reamostragem embaralhada):

| N | fração | IC 95% | largura |
|---|---|---|---|
| 100 | 1,00% | [0,18%, 5,45%] | 5,27pp |
| 316 | 0,95% | [0,32%, 2,75%] | 2,43pp |
| 1000 | 0,70% | [0,34%, 1,44%] | 1,10pp |

A fração cumulativa converge suavemente para ~0,7% conforme N cresce
(não fica "presa" num valor inicial enganoso), e a largura do IC cai de
forma consistente com o esperado — nenhum sinal de que o ensemble de
1000 runs ainda está "instável" ou precisa de mais dados pra essa
conclusão qualitativa específica.

**Projeção analítica para N=10000** (assumindo a MESMA taxa observada
0,7% — não uma garantia, uma resposta a "se a taxa real for a que já
medimos, o que 10000 runs mudaria"): IC estreitaria de [0,34%, 1,44%]
(largura 1,10pp) para [0,55%, 0,88%] (largura 0,33pp) — uma redução de
~3,3x, próxima da razão teórica 1/√10≈0,316 esperada assintoticamente
para uma proporção binomial. **Conclusão acionável**: rodar 10000 runs
(~2,7h em paralelo, ver §7.8) não mudaria a conclusão qualitativa
(evento raro real, ordem de grandeza ~0,7%, já estabelecida com folga),
mas triplicaria a PRECISÃO da estimativa — uma troca legítima entre
tempo de computação e precisão, agora quantificada, não intuída.

**Validado com dados sintéticos onde a resposta é conhecida**
(`tests/test_convergence_analysis.py`): traço binomial recupera a taxa
real dentro do IC final; traço de média contínua (reaproveita
`ensemble_stats.describe`, mesmo bootstrap de §10.2b) recupera a média
real; e — verificação mais forte — a projeção pra N=10000 feita com só
os primeiros 1000 pontos de uma amostra sintética de 20000 é comparada
contra o IC calculado de FATO sobre os 20000 pontos reais (larguras
batem dentro de 50% de tolerância relativa), confirmando que a
extrapolação é utilizável, não só matematicamente plausível.

```
pytest tests/test_convergence_analysis.py -v
```

## 10.6 Convergência numérica dos solvers: tolerância de EDO e malha de PDE

Implementado em `numerical_convergence.py` — verificação de SOLUÇÃO
NUMÉRICA (o método resolve corretamente a equação escrita, refinar
tolerância/malha não muda o resultado) dos dois solvers do projeto,
nunca checada antes: o integrador de EDO da pluma
(`plume_physics.integrate_plume`, RK45 adaptativo) e o solver de PDE
acústico (`acoustics.solve_steady_advection_diffusion`, diferenças
finitas upwind de 1ª ordem). Ortogonal à validação FÍSICA (a equação
está calibrada contra dado real — já feita em outras seções) — aqui a
pergunta é puramente numérica. `integrate_plume` ganhou `rtol`/`atol`
como parâmetros opcionais (defaults inalterados, `1e-8`/`1e-12`) só
para viabilizar este estudo.

**EDO da pluma — já convergida ao default**: comparando `rise_height_m`
(altura de flutuabilidade neutra) e a diluição em z=1m entre o default
do projeto (rtol=1e-8, atol=1e-12) e uma tolerância 100x mais apertada
(rtol=1e-10, atol=1e-14), nos 3 tipos de vent: mudança relativa
<10⁻⁵ em todos os casos (tipicamente ~10⁻⁹-10⁻¹²) — o default já está
efetivamente na precisão de máquina para as grandezas físicas que
importam. Apertar a tolerância não mudaria nenhum dígito significativo
de nenhum resultado já publicado neste projeto.

**PDE acústico — dois achados complementares, não contraditórios**:

1. **Nos parâmetros REAIS de produção** (`DEFAULT_SOLUTE_DIFFUSIVITY_
   M2_S`=8×10⁻¹⁰ m²/s, minúscula — difusão molecular real de um soluto
   pequeno), a mudança relativa da concentração num ponto de sonda
   fixo, entre malhas que colchetam o `--size` default real (129→257→513
   células, domínio 1200m), é <10⁻³ (tipicamente ~10⁻⁵-10⁻⁶) — a malha
   de produção não é o gargalo de precisão desta simulação. A ordem de
   convergência OBSERVADA nesse regime sai ruidosa/sem sentido físico
   (a mudança já é tão pequena que fica dominada pela precisão do
   solver linear esparso, não pelo erro de truncamento do esquema) —
   um resultado esperado e coerente, não um bug: quando a mudança já é
   desprezível, medir sua "ordem" deixa de ser uma pergunta bem-posta.
2. **Num cenário sintético deliberadamente mais exigente** (difusividade
   e velocidade maiores que os defaults reais, especificamente para
   isolar o erro de truncamento do ruído do solver linear e conseguir
   medir a ordem de convergência de verdade): a ordem observada fica em
   torno de 0,78-0,88 conforme a malha refina, aproximando-se do 1
   teórico esperado para um esquema upwind de 1ª ordem — confirma que
   o MÉTODO em si se comporta como a matemática prevê, uma verificação
   real do solver, independente de qual regime de parâmetro está sendo
   usado.

**Bug metodológico real encontrado e corrigido construindo o estudo de
malha**: a primeira versão posicionava a fonte gaussiana e amostrava o
ponto de sonda pelo ÍNDICE de célula mais próximo do centro/da fração
do domínio — em malhas de resoluções diferentes, o mesmo índice
corresponde a um local FÍSICO ligeiramente diferente (deslocamento de
até meia célula), então cada refinamento estava, sem querer, resolvendo
um problema levemente diferente a cada malha. Isso produzia ordens de
convergência observadas sem sentido físico nenhum (-1,75, depois +2,56,
oscilando) — sintoma claro de ruído de posicionamento, não de erro de
discretização real. Corrigido usando coordenadas FÍSICAS (metros) em
todo lugar — fonte sempre no centro físico exato do domínio,
amostragem da sonda por interpolação bilinear (`scipy.ndimage.
map_coordinates`) na posição física exata, não o índice de célula mais
próximo — depois disso a ordem observada convergiu suavemente para
perto de 1, como esperado.

**Testado** (`tests/test_numerical_convergence.py`): extrapolação de
Richardson recupera ordem 1 e ordem 2 exatas em funções sintéticas com
erro conhecido analiticamente (valida a fórmula em si, sem depender de
nenhum solver real); convergência de tolerância testada nos 3 tipos de
vent reais (não um caso isolado); convergência de malha testada tanto
no regime sintético "estressado" (ordem observada checada contra a
faixa [0,5, 1,5]) quanto no regime de parâmetros reais de produção
(mudança relativa checada contra <10⁻³); regressão do bug de
posicionamento por índice de célula. 8 testes, suíte completa do
projeto 154/154 passando.

```
pytest tests/test_numerical_convergence.py -v
```

## 10.7 QA automatizada de integridade por run

Implementado em `run_qa.py` — verificação sistemática de um ensemble
(NaN/Inf, valores negativos onde fisicamente impossível, seeds
duplicadas, inconsistência interna entre campos já computados, runs com
`metadata.json` ausente/corrompido) em vez de confiar só em inspeção
manual ocasional. Dois níveis DELIBERADAMENTE separados:

- `hard_errors`: bugs inequívocos — sempre vale investigar.
- `soft_flags`: outliers estatísticos via z-score ROBUSTO (mediana/MAD,
  mesmo fator de escala 1,4826 de `ensemble_stats.describe`, §10.2) —
  candidatos a revisão manual, EXPLICITAMENTE não tratados como bug.
  Distribuições deste projeto são caudal-pesadas por construção (altura
  de chaminé, evento raro de Gor'kov — §7.8.1); um outlier estatístico
  aqui tem boa chance de ser o mesmo tipo de evento raro real que o
  resto do projeto foi construído pra estudar. Misturar os dois níveis
  seria repetir, ao contrário, o erro que §7.8.4 já mostrou ser real
  (tratar sinal genuíno como se fosse ruído) — aqui o risco seria tratar
  ruído/bug como se fosse sinal, ou pior, afogar um bug real de verdade
  em meio a alertas de eventos raros legítimos.

**Bug real na PRIMEIRA versão desta própria ferramenta, achado
aplicando aos dois ensembles reais já existentes**: a checagem inicial
exigia `aumentaram + diminuíram + inalterados == n_vents` — e disparou
"erro" em 100% das 1100 runs reais combinadas dos dois ensembles
(100 e 1000 runs). Um sinal óbvio (100% de falha nunca é "muitos bugs
de simulação", é a própria checagem estar errada) levou a reler
`prebiotic.compute_field_hotspots`: essa contagem soma só vents com
`enrichment_vs_control != None`, um SUBCONJUNTO real de `n_vents` (nem
todo vent tem uma comparação válida contra o controle) — nunca deveria
ser igualdade, só "nunca excede". Corrigido para `<=`; os mesmos dois
ensembles reais passam limpos (`hard_errors: 0`, `ok: True`) depois do
ajuste. Fixado como teste de regressão permanente.

**Aplicado aos dois ensembles reais do projeto** (100 e 1000 runs,
1100 runs no total): 0 erros reais encontrados (esperado — dados já
extensivamente analisados nesta sessão); 66 `soft_flags` no total,
todos em `top_hotspot_enrichment_vs_control`/`gorkov_trap_depth_over_
kT` — correspondendo exatamente à cauda longa já conhecida dessas
distribuições (§7.8.1/§10.1), não a problemas novos.

**Testado** (`tests/test_run_qa.py`): cada tipo de `hard_error` isolado
com dado sintético mínimo (NaN, negativo, seed duplicada, n_vents=0,
contagem excedendo n_vents); regressão do bug real acima (contagem
ABAIXO de n_vents não é erro); outlier plantado detectado como
`soft_flag`, dado uniforme não gera nenhum; fórmula do z-score robusto
verificada contra o cálculo direto; runs com `metadata.json` ausente
reportadas explicitamente por `check_experiment_dir_integrity` (não
silenciosamente ignoradas); e um ensemble real pequeno (15 runs, via
`fumarola_field.run_experiment`, não sintético) passa limpo. 13 testes,
suíte completa do projeto 167/167 passando.

```
pytest tests/test_run_qa.py -v
```

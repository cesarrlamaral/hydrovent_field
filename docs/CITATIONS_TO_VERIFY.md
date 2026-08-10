# Citações a verificar no texto primário

Lista compilada em 2026-08-05 a pedido do usuário. Cada item abaixo foi
usado no modelo com base em resumos, trechos secundários ou artigos de
acesso aberto processados por ferramenta de busca/leitura automática —
não em leitura direta e completa do texto primário por um humano. Isto
não significa que estejam errados; significa que precisam de
confirmação antes de qualquer submissão. Organizados por prioridade
(quanto peso aquele número/afirmação tem no resultado final do modelo).

Para cada item: o que baixar, o que especificamente verificar, e onde
isso é usado no código.

**Atualização 2026-08-06 (1ª rodada)**: o usuário baixou e enviou os
PDFs primários completos de 12 dos 16 itens numerados (pasta
`biblios/`) — todos os itens #1-6 e #12-16 foram verificados por
leitura direta. Resultado: 3 erros reais encontrados e corrigidos (#4
conflação de números no Sojo et al.; #6 atribuição errada da faixa de
velocidade de Converse et al.; #16 caracterização errada da linhagem
teórica de Goodman & Lenferink) e 1 erro de metadados de citação real
e sério (#5 — autor principal, título e número de artigo inteiramente
errados, só o DOI estava certo). Todos os outros itens verificados
bateram exatamente com o texto primário.

**Atualização 2026-08-06 (2ª rodada)**: a pedido do usuário, baixei
diretamente (sem intervenção humana) todos os PDFs de acesso aberto
localizáveis para as referências restantes citadas no texto do relatório
(~41 itens no total, além dos 16 numerados
acima). **7 PDFs adicionais obtidos** com sucesso, de fontes legítimas
de acesso aberto (PLOS ONE, PMC/Europe PMC, página institucional do
autor, Internet Archive para material em domínio público):

- Crone, Wilcock, Barclay & Parsons (2006), PLOS ONE 1(1):e133 →
  `journal.pone.0000133.pdf`
- Braun & Libchaber (2002), PRL 89:188103 → `prl.89.188103.pdf`
  (hospedado na página do laboratório do autor)
- Wächtershäuser (1988), Microbiol. Rev. 52(4):452-484 →
  `mr.52.4.452-484.1988.pdf` (PMC373159, via Europe PMC)
- Martin & Russell (2007), Phil. Trans. R. Soc. B 362:1887-1926 →
  `rstb.2006.1881.pdf` (hospedado na página institucional do autor)
- Bruus (2012), Lab Chip 12:1014-1021 → `c2lc21068a.pdf` (hospedado na
  página do laboratório do autor)
- Rayleigh (1884), Phil. Trans. R. Soc. 175:1-21 →
  `rspl.1883.0075.pdf` (domínio público, Internet Archive)
- Radford-Knoery, German, Charlou, Donval & Fouquet (2001), Limnol.
  Oceanogr. 46(2):461-464 → `radford-knoery2001.pdf` (Archimer/IFREMER)
  — **erro real encontrado durante o download**: a lista de
  referências do relatório citava esse artigo como "Radford-Knoery
  & Cutter (2001), 'Kinetics and mechanism of H2S removal...'" —
  autores e título errados (só journal/volume/página coincidiam por
  acaso com o artigo real). O código-fonte (`reaction_kinetics.py`)
  já tinha a citação correta; só a lista de referências do relatório
  estava errada. Corrigido, junto com uma segunda referência de
  Millero et al. (1987) que faltava inteiramente na lista (ver #10).

**Itens que NÃO consegui obter** (paywall sem cópia de acesso aberto
localizável, ou bloqueio anti-bot em todas as fontes tentadas — Wiley/
AGU, Royal Society, APS e HAL retornam páginas de verificação
JavaScript/anti-scraper que as ferramentas de download não conseguem
passar): Ainslie & McColm 1998 (JASA); Baross & Hoffman 1985 (Springer);
Cowen, Massoth & Feely 1990 (Deep-Sea Res., Elsevier); Eckart 1948
(Phys. Rev., APS); Field & Sherrell 2000 (GCA, Elsevier); Gor'kov 1962
(original russo, sem tradução livre localizada); Klevenz et al. 2011
(G3 — é revista de acesso aberto, mas o Wiley bloqueou o download
automatizado); Lavelle 1997 (JGR, AGU); Lemaréchal, Roullet & Gula 2025
(bloqueado no HAL e no Wiley); Mackenzie 1981 (JASA/AIP); McKay,
Beckman & Conover 1979 (Technometrics, Taylor & Francis); Millero,
Sotolongo & Izaguirre 1987 (GCA, Elsevier); Mittelstaedt et al. 2012
(bloqueado no HAL e no Wiley); Morton, Taylor & Turner 1956 (Royal
Society, bloqueio HTTP 403); Mottl & McConachy 1990 (GCA, Elsevier);
Nyborg 1958 (JASA/AIP); Rudnicki & Elderfield 1992 e 1993 (Elsevier).
Não tentei os 4 livros da lista (Cussler; Jensen et al.; Urick; Von
Damm 1995, capítulo de monografia AGU) — livros didáticos/monografias
não costumam ter PDF de acesso aberto legítimo.

**Atualização 2026-08-08**: o usuário baixou manualmente e colocou em
`biblios/` a maioria dos PDFs antes bloqueados. Verificação prioritária
de Morton, Taylor & Turner (1956) — a física central de todo o modelo —
por leitura direta do texto primário completo
(`biblios/rspa.1956.0011.pdf`, 26 páginas):

### Morton, B.R., Taylor, G.I., & Turner, J.S. (1956) — ✅ VERIFICADO E CORRIGIDO (2026-08-08)
**"Turbulent gravitational convection from maintained and instantaneous sources."** *Proc. R. Soc. Lond. A* 234(1196), 1-23. DOI: 10.1098/rspa.1956.0011

- **Erro real encontrado e corrigido — prioridade máxima**: o sistema de
  EDOs implementado em `plume_physics._ode_rhs` estava com `dM/dz = Q*B/M`
  e `dB/dz = -N²*Q`, faltando um fator 2 em ambas as equações (deveria
  ser `dM/dz = 2*Q*B/M`, `dB/dz = -2*N²*Q`). Confirmado por leitura
  direta das eqs. (7,ii-iii) e (8) do artigo (`d/dx(b²u²) =
  2b²g(ρ0-ρ)/ρ1`; `d/dx(b²ug(ρ0-ρ)/ρ1) = 2b²u(g/ρ1)(dρ0/dx)`; forma
  reduzida `dV⁴/dx=4F*W, dF*/dx=-2WG`), e verificado numericamente por
  três vias independentes (reprodução exata da Tabela 1 do artigo,
  verificação algébrica das eqs. 7/8/10, conferência do coeficiente
  dimensional 0.410 de eq. 14 — ver `docs/PHYSICS_MODEL.md` §2 e §2.2
  para a derivação completa). A equação de entranhamento `dQ/dz =
  2√π·α·√M` já estava correta. Corrigido em `plume_physics.py`; impacto
  na altura de ascensão calculada (`rise_height_m`): ~23% menor com os
  defaults do projeto (328m → 254m para um black smoker a 350°C). Todos
  os 167 testes rastreados continuam passando após a correção
  (benchmarks de campo com tolerância de ordem de grandeza/fator ≥3 não
  foram sensíveis o bastante para detectar o erro antes).
- Também corrigida uma citação fabricada em `docs/PHYSICS_MODEL.md` §2.2
  (fórmulas "`5*pi^-0.25=3.76`"/"`4*pi^-0.25=3.01`" atribuídas de forma
  vaga à "literatura consultada" — não correspondem a nenhuma equação
  real; ver entrada #12, Speer & Rona, acima, para o detalhe).

### Lavelle, J.W. (1997) — ✅ VERIFICADO (2026-08-08)
**"Buoyancy-driven plumes in rotating, stratified cross flows: Plume dependence on rotation, turbulent mixing, and cross-flow strength."** *J. Geophys. Res.* 102(C2), 3405-3420.

- **Verificado por leitura direta do PDF primário completo**. Ambas as
  afirmações usadas em `plume_physics.py` batem exatamente: "Buoyancy
  frequency, N over depths 2100-2350 m is 7.9 × 10⁻⁴ s⁻¹" (idêntico ao
  `DEFAULT_N_BRUNT_VAISALA = 7.9e-4` e à faixa de profundidade citada) e
  "the entrainment coefficient [tinha] magnitude comparable to that
  customarily used with integral theory (α ~ 0.1)" (frase quase
  idêntica à do docstring do projeto). Nenhum erro encontrado.

### Lemaréchal, C., Roullet, G., & Gula, J. (2025) — ✅ VERIFICADO E CORRIGIDO (2026-08-08)
**"Hydrothermal Plume Near-Field Dynamics From LES and Observations."** *J. Geophys. Res.: Oceans* 130(10), e2024JC022277. DOI: 10.1029/2024JC022277

- **Erro real encontrado e corrigido**: `report.py` (REFERENCES_EN) tinha
  o primeiro nome do primeiro autor errado ("R." em vez de "C.") e um
  título inteiramente diferente e não-existente ("On the entrainment
  coefficient near hydrothermal vent orifices") — o título real é o
  acima. Corrigido em `report.py`.
- A afirmação usada em `plume_physics.py`/`report.py` (constante-α do
  MTT falha nos primeiros ~2m acima do orifício) bate com o achado real
  do artigo: "a systematic transition region, typically within the
  first two meters, where the entrainment rate varies... above which
  the plume enters the pure plume regime."
- **Achado adicional relevante**: o artigo apresenta, em sua eq.
  (1a-1c), o próprio sistema MTT reescrito em termos de raio/velocidade
  (`d/dz(r²w)=2αrw`, `d/dz(r²w²)=2r²b`, `d/dz(r²wb)=2r²w(g/ρr)(∂ρa/∂z)`)
  — a terceira e a segunda equações têm o fator 2 exatamente como
  corrigido em `plume_physics.py` nesta sessão (ver entrada MTT 1956
  acima), servindo de confirmação independente adicional (terceira via)
  da correção do fator 2.
- **Discrepância observada, não resolvida**: o artigo cita de segunda
  mão (Devenish et al. 2010) os coeficientes `Hnbl=1.04·α^-0.5·B0^0.25·N^-0.75`
  e `Htop=1.36·α^-0.5·B0^0.25·N^-0.75`, usando a mesma definição de B0
  do projeto — diferentes dos coeficientes `0.7326`/`0.9654` derivados
  diretamente de MTT (1956) nesta sessão (ver `docs/PHYSICS_MODEL.md`
  §2.2). Meu coeficiente foi verificado numericamente com <1% de erro
  contra a integração completa do sistema MTT corrigido; a diferença
  com Devenish et al. provavelmente vem de uma convenção de perfil
  diferente (top-hat vs. gaussiano) — não investigado a fundo (Devenish
  et al. 2010 não obtido nesta sessão).

### Mackenzie, K.V. (1981) — ✅ VERIFICADO (2026-08-08)
**"Nine-term equation for sound speed in the oceans."** *J. Acoust. Soc. Am.* 70(3), 807-812.

- **Verificado por leitura direta do PDF primário completo**. Os 9
  coeficientes de `sound_speed_seawater()` batem exatamente com a eq.
  (1) do artigo. Único ajuste: o docstring dizia validade "T=2-30°C"
  quando o artigo diz "-2° a 30°C" — corrigido (a faixa real é ainda
  mais folgada do que o documentado, não um erro de física).

### Ainslie, M.A., & McColm, J.G. (1998) — ✅ VERIFICADO (2026-08-08)
**"A simplified formula for viscous and chemical absorption in sea water."** *J. Acoust. Soc. Am.* 103(3), 1671-1672.

- **Verificado por leitura direta do PDF primário completo**. Todos os
  coeficientes de `seawater_absorption_np_per_m()` (f1, f2, e os três
  termos ácido bórico/MgSO4/água pura) batem exatamente com as eqs.
  (1)-(3) do artigo. Nenhum erro encontrado.

### Nyborg, W.L. (1958) — ✅ VERIFICADO (2026-08-08)
**"Acoustic streaming near a boundary."** *J. Acoust. Soc. Am.* 30(4), 329-339.

- **Verificado por leitura direta do PDF primário completo**. A eq.
  (28c) do artigo dá `u_L = -(3/(8ω))·d(u_a0²)/dx` para o fluxo 1D
  (amplitude de PICO). O código usa `coeff = -0.75/omega` (=`-3/(4ω)`)
  aplicado a `v2_i`, que é construído a partir de `p_ref_tonal_pa`
  (RMS, não pico) — como `⟨U²⟩=u_a0²/2`, as duas formas são
  algebricamente idênticas (`-(3/4ω)·d(u_a0²/2)/dx = -(3/8ω)·d(u_a0²)/dx`).
  Nenhum erro real; documentada a reconciliação em `acoustics.py` para
  evitar alarme falso numa verificação futura.

### Eckart, C. (1948) — ✅ VERIFICADO (2026-08-08)
**"Vortices and streams caused by sound waves."** *Phys. Rev.* 73(1), 68-76.

- **Verificado por leitura direta do PDF primário completo**. O artigo
  não é implementado no código (explicitamente descartado como
  mecanismo desprezível nas frequências de Crone et al., 10-500 Hz) —
  só citado qualitativamente. A afirmação do docstring ("streaming de
  bulk ∝ absorção × intensidade") é uma paráfrase padrão da literatura
  moderna; o artigo original expressa o resultado em termos de
  velocidade ∝ potência irradiada × k² (∝ frequência², um proxy comum
  para absorção viscosa) ÷ ρc³ — consistente em substância, não uma
  citação literal palavra-por-palavra. Sem impacto no código (não
  implementado).

### Klevenz, V., Bach, W., Schmidt, K., Hentscher, M., Koschinsky, A., & Petersen, S. (2011) — ✅ VERIFICADO E CORRIGIDO (2026-08-08)
**"Geochemistry of vent fluid particles formed during initial hydrothermal fluid–seawater mixing along the Mid-Atlantic Ridge."** *Geochem. Geophys. Geosyst.* 12, Q0AE05.

- **Erro real encontrado e corrigido**: `acoustics.py` (comentário sobre
  `PARTICLE_CLASSES["fine_sulfide_colloid"]`) afirmava "Cu/Zn formam
  partículas >0.2 μm perto do orifício" como se fosse um resultado
  medido do artigo. Lido o PDF primário completo (23 páginas): o artigo
  é um estudo de composição geoquímica/mineralógica (chalcopirita,
  esfalerita, pirita, etc. via ICP-OES/ICP-MS) e **não reporta tamanho
  de partícula em nenhum lugar**. O "0.2 μm" vem da seção 2.2
  (metodologia de amostragem): "filtered through 0.2 μM polycarbonate
  membrane filters" — é o poro do filtro usado na coleta, não uma
  medição de granulometria. Corrigido o comentário para refletir
  corretamente que só é citável que as partículas coletadas são ≥0.2 μm
  nominalmente (pelo filtro), não que o artigo mediu ou reportou esse
  valor como tamanho típico de partícula.

### Mittelstaedt, E., Escartín, J., Gracias, N., Olive, J.-A., Barreyre, T., Davaille, A., Cannat, M., & Garcia, R. (2012) — ✅ VERIFICADO (2026-08-08)
**"Quantifying diffuse and discrete venting at the Tour Eiffel vent site, Lucky Strike hydrothermal field."** *Geochem. Geophys. Geosyst.* 13, Q04008.

- **Verificado por leitura direta do PDF primário completo**. "Vertical
  velocities of diffuse effluent between 0.9 cm s⁻¹ and 11.1 cm s⁻¹"
  bate exatamente com a faixa 0.009-0.111 m/s citada em
  `plume_physics.py` (`EXIT_VELOCITY_BY_TYPE["diffuse_flow"]`). Nenhum
  erro encontrado.

### Field, M.P., & Sherrell, R.M. (2000) — ✅ VERIFICADO E CORRIGIDO (2026-08-08)
**"Dissolved and particulate Fe in a hydrothermal plume at 9°45'N, East Pacific Rise: slow Fe(II) oxidation kinetics in Pacific plumes."** *Geochim. Cosmochim. Acta* 64(4), 619-628.

- **Verificado por leitura direta do PDF primário completo**. "Fe(II)
  half-life in ambient deepwater at 9°45'N on the EPR is almost this
  long, at about 3.3 h" — bate exatamente com o `t1/2 = 3.3 h` usado em
  `reaction_kinetics.py` (bacia Pacífico).
- **Erro real encontrado e corrigido**: `reaction_kinetics.py` afirmava
  "fração ~65% medida em EPR 9°45'N (Field & Sherrell 2000)" para
  precipitação imediata de sulfeto de Fe — o artigo calcula "a Fe loss
  of ~68% (~5.6 μM) of the total Fe vented" por balanço de massa, não
  65%. O 65% não aparece em lugar nenhum do texto; possível confusão
  com a citação de Mottl & McConachy (1990) dentro do próprio artigo
  ("40-90% of vent fluid Fe forms sulfides..."). Corrigido
  `FE_PROMPT_SULFIDE_FRACTION_DEFAULT` de 0.65 para 0.68.
- Também confirmado: `report.py` (REFERENCES_EN) tinha um título
  inteiramente diferente e incorreto para este artigo ("Dissolved iron
  in North East Pacific Ocean and Fe hydrothermal plumes") — corrigido
  para o título real.

### Millero, F.J., Sotolongo, S., & Izaguirre, M. (1987) — ✅ VERIFICADO (2026-08-08)
**"The oxidation kinetics of Fe(II) in seawater."** *Geochim. Cosmochim. Acta* 51, 793-801.

- **Verificado por leitura direta do PDF primário completo**. "The
  energy of activation found for the combined data is 29 ± 2 kJ mol⁻¹"
  bate exatamente com `FE_EA_J_MOL = 29_000.0` em `reaction_kinetics.py`.
  Nenhum erro encontrado.

### Cowen, J.P., Massoth, G.J., & Feely, R.A. (1990) — ✅ VERIFICADO (2026-08-08)
**"Scavenging rates of dissolved manganese in a hydrothermal vent plume."** *Deep-Sea Res.* 37(10), 1619-1637.

- **Verificado por leitura direta do PDF primário completo**. "The
  measured scavenging rate constant, k1, was lowest in the buoyant
  plume (<0.2 y⁻¹), increasing to ~2 y⁻¹ in the non-buoyant plume at
  distances of 20 km from the ridge valley axis" bate exatamente com
  `MN_SCAVENGING_RATE_PER_YEAR` em `reaction_kinetics.py`. Nenhum erro
  encontrado.

### Rudnicki, M.D., & Elderfield, H. (1993) — ✅ VERIFICADO (2026-08-08)
**"A chemical model of the buoyant and neutrally buoyant plume above the TAG vent field, 26°N, Mid-Atlantic Ridge."** *Geochim. Cosmochim. Acta* 57, 2939-2957.

- **Verificado por leitura direta do PDF primário completo**. "The
  calculated value for the pseudo-first-order rate constant, k1, is
  0.329 min⁻¹, equivalent to a half-life time... of 2.1 min" bate
  exatamente com `t1/2 = 2.1 min` (bacia Atlântico/TAG) em
  `reaction_kinetics.py`. Nenhum erro encontrado.
- **Achado colateral relevante**: este artigo cita Rudnicki & Elderfield
  (1992a) como "J. Volcanol. Geotherm. Res. 50, 161-172" — confirma
  exatamente a entrada #Rudnicki1992 abaixo.

### Rudnicki, M.D., & Elderfield, H. (1992) — ✅ VERIFICADO (2026-08-08)
**"Theory applied to the Mid-Atlantic Ridge hydrothermal plumes: the finite-difference approach."** *J. Volcanol. Geotherm. Res.* 50(1-2), 161-172.

- **Verificado por leitura direta do PDF primário completo**. Citação
  (título, volume, páginas) confirmada exatamente. Usado em `report.py`
  apenas como citação de contexto (extensão do sistema MTT/Speer&Rona
  para um traçador químico, eq. 5 do artigo) — nenhuma equação
  específica implementada, nenhuma divergência.
- **Confirmação independente adicional (achado colateral)**: a eq. (8)
  deste artigo dá `Zmax = 5·B0^0.25·N^-0.75` (seguindo Turner, 1973),
  **sem nenhum fator de π** — a mesma forma exata de Speer & Rona
  (1989) eq. (5). Isso confirma, por uma SEGUNDA fonte independente,
  que a fórmula "5·π^-0.25=3.76" antes usada em
  `docs/PHYSICS_MODEL.md` §2.2 era mesmo fabricada (ver entrada #12,
  Speer & Rona, e a entrada MTT 1956 acima).

### Mottl, M.J., & McConachy, T.F. (1990) — ✅ VERIFICADO (2026-08-08)
**"Chemical processes in buoyant hydrothermal plumes on the East Pacific Rise near 21°N."** *Geochim. Cosmochim. Acta* 54, 1911-1927.

- **Verificado por leitura direta do PDF primário completo**. A faixa
  "40-90%" citada em `reaction_kinetics.py`/`docs/PHYSICS_MODEL.md` para
  remoção de Fe não aparece literalmente no artigo como um único número,
  mas é derivável exatamente do resultado próprio do artigo: "the plume
  solutions have retained an average of 35 ± 25% of their dissolved Fe"
  → fração removida = 100-35 = 65±25% = faixa 40-90%. Consistente com o
  próprio Field & Sherrell (2000), que parafraseia este resultado como
  "40-90% of vent fluid Fe forms sulfides" (citação de segunda mão já
  usada no projeto). Nenhum erro encontrado.

### McKay, M.D., Beckman, R.J., & Conover, W.J. (1979) — ✅ VERIFICADO (2026-08-08)
**"A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code."** *Technometrics* 21(2), 239-245.

- **Verificado por leitura direta do PDF primário completo**. Título,
  autores e definição do método de Hipercubo Latino ("divide the range
  of each X_k into N strata of equal marginal probability 1/N, and
  sample once from each stratum... matched at random") batem
  exatamente com `latin_hypercube_1d`/`joint_latin_hypercube` em
  `fumarola_field.py`. Nenhum erro encontrado.

### Gor'kov, L.P. (1961) — ✅ VERIFICADO (2026-08-08) — encontrado, não mais "não localizado"
**"О силах, действующих на малую частицу в акустическом поле в идеальной жидкости"** ["On the forces acting on a small particle in an acoustical field in an ideal fluid"]. *Doklady Akademii Nauk SSSR* 140(1), 88-91 (`biblios/dan25473.pdf`, original russo).

- **Achado**: ao contrário do que constava em `docs/CITATIONS_TO_VERIFY.md`
  ("nenhuma cópia... localizada"), o arquivo `biblios/dan25473.pdf` já
  obtido pelo usuário É o artigo original russo de 1961 (confirmado
  pelo cabeçalho "Доклады Академии наук СССР, 1961, Том 140, № 1" e
  pela autoria "Л. П. Горьков").
- **Verificado por leitura direta**: a eq. (12) do artigo,
  `U(r) = 2πR³ρ{⟨p'²⟩/(3ρ²c²)·f1 - ⟨v²⟩/2·f2}`, com
  `f1=1-c²ρ/(c0²ρ0)` e `f2=2(ρ0-ρ)/(2ρ0+ρ)` (ρ0=densidade da
  partícula), é ALGEBRICAMENTE IDÊNTICA, termo a termo, à
  implementação em `acoustics.py`
  (`U = V0*[f1·⟨p²⟩/(2ρc²) - (3/4)·f2·ρ·⟨v²⟩]` com `V0=(4/3)πR³`):
  substituindo V0, os dois primeiros termos coincidem exatamente
  (`(2/3)πR³·f1·⟨p²⟩/(ρc²)`) e os segundos também
  (`πR³·ρ·f2·⟨v²⟩`). O limite rígido `f1=1` usado no projeto também
  bate exatamente com o limite `c0²ρ0→∞` da fórmula geral do artigo.
  Esta é a primeira verificação desta fórmula contra a fonte primária
  ORIGINAL (as verificações anteriores do projeto usavam formulações
  secundárias/modernas equivalentes, ex. Bruus 2012) — nenhuma
  divergência encontrada.

### Baross, J.A., & Hoffman, S.E. (1985) — ✅ VERIFICADO (2026-08-08)
**"Submarine hydrothermal vents and associated gradient environments as sites for the origin and evolution of life."** *Origins Life Evol. Biosph.* 15(4), 327-345.

- **Verificado por leitura direta do PDF primário completo**. Título,
  autores, volume e páginas confirmados exatamente. A citação de
  contexto usada em `report.py` (gradientes físico-químicos íngremes,
  suprimento sustentado de H2S/CH4/Fe(II)/Mn(II), superfícies minerais
  catalíticas) é consistente com a tese central do artigo. Nenhum erro
  encontrado.

### Urick, R.J. (1983) — ⚠️ NÃO VERIFICADO NESTA SESSÃO (arquivo grande demais)
**"Principles of Underwater Sound"** (3rd ed.). McGraw-Hill.

- O PDF (`biblios/pdfcoffee.com_principles-underwater-sound-urick-pdf-free.pdf`)
  está disponível, mas excede o limite de leitura desta sessão (>20MB) —
  não foi possível abrir e conferir diretamente. A citação usada no
  projeto (`acoustics.py`, `report.py`, `tests/test_acoustics.py`) é
  genérica — atribui o efeito clássico e amplamente documentado do
  "espelho de Lloyd" (interferência fonte+imagem perto de um contorno
  refletor) a este livro-texto, sem nenhum número/coeficiente
  específico em jogo. Não verificado diretamente nesta sessão; marcado
  como pendente para uma sessão futura com capacidade de ler PDFs
  maiores (ex. dividir em capítulos).

---

## Prioridade ALTA (sustentam achados centrais do modelo)

### 1. Baaske, P., Weinert, F.M., Duhr, S., Lemke, K.H., Russell, M.J., & Braun, D. (2007) — ✅ VERIFICADO (2026-08-06)
**"Extreme accumulation of nucleotides in simulated hydrothermal pore systems."**
*PNAS* 104(22), 9346–9351. DOI: 10.1073/pnas.0609592104

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/baaske2007.pdf`). A equação primária é a Eq. 1 do artigo
  (p. 9348): `c_BOTTOM/c_TOP = exp(0.42 × S_T × ΔT × r)`. O valor usado
  no modelo (k=0.4, antes reconstruído por regressão) foi atualizado
  para k=0.42, lido diretamente da equação. O valor S_T=0.006/K
  (nucleotídeo único, 170 mM de sal, Tabela 1) foi confirmado
  exatamente. Nenhuma outra divergência encontrada.
- **Onde é usado**: `prebiotic.py`, `module_thermophoresis()`, classe
  "nucleotideos" — é a calibração central do módulo 2.
- Corrigido em: `prebiotic.py`, `docs/PHYSICS_MODEL.md` §8.2/8.4/8.6,
  `tests/test_prebiotic.py`.

### 2. González-Santana, D., Planquette, H., Cheize, M., Whitby, H., Gourain, A., Holmes, T., et al. (2020) — ✅ VERIFICADO (2026-08-06)
**"Processes driving iron and manganese dispersal from the TAG hydrothermal plume (Mid-Atlantic Ridge): Results from a GEOTRACES process study."**
*Frontiers in Marine Science* 7, 568. DOI: 10.3389/fmars.2020.00568

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/fmars.2020.00568.pdf`). Os três números — raio do agregado
  14-20 μm no primeiro km, densidade 2400-3600 kg/m³, velocidade de
  sedimentação 45±6 m/dia perto do vent e 5±2 m/dia mais distante —
  foram confirmados exatamente, sem nenhuma divergência.
- **Onde é usado**: `acoustics.py`, `PARTICLE_CLASSES["near_field_fe_oxyhydroxide_aggregate"]`.
- Corrigido em: `acoustics.py` (nota de verificação adicionada).

### 3. Jackson, J.B. (2016) — ✅ VERIFICADO (2026-08-06)
**"Natural pH gradients in hydrothermal alkali vents were unlikely to have played a role in the origin of life."**
*Journal of Molecular Evolution* 83(1), 1–11. DOI: 10.1007/s00239-016-9756-6

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/s00239-016-9756-6.pdf`). Os quatro números da crítica —
  espessura de membrana ~1 μm, >200× mais espessa que bicamadas
  lipídicas, colapso do gradiente para ~0.004 unidades de pH, ~24 J/mol
  disponíveis vs. ~24 kJ/mol necessários — foram confirmados
  exatamente, sem nenhuma divergência.
- **Onde é usado**: `prebiotic.py` (docstring de `module_proton_gradient`)
  e `docs/PHYSICS_MODEL.md` §8.9 — é a contraposição crítica ao módulo
  de gradiente de prótons, citada com peso igual à hipótese.
- Corrigido em: `prebiotic.py`, `docs/PHYSICS_MODEL.md` §8.9 (notas de
  verificação adicionadas).

### 4. Sojo, V., Herschy, B., Whicher, A., Camprubí, E., & Lane, N. (2016) — ✅ VERIFICADO E CORRIGIDO (2026-08-06)
**"The Origin of Life in Alkaline Hydrothermal Vents."**
*Astrobiology* 16(2), 181–197. DOI: 10.1089/ast.2015.1406

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/ast.2015.1406.pdf`). **Erro real encontrado e corrigido**:
  a documentação conflava dois números distintos do artigo. O artigo
  afirma explicitamente (texto principal + legenda da Fig. 1) que um
  gradiente de **3 unidades de pH** (~200 mV) através de poros de vents
  é "exactly equivalent in both magnitude... and polarity" à força
  próton-motriz biológica extante — este é o número correto usado na
  calibração (`REFERENCE_PROTON_GRADIENT_PH_UNITS=3.0`). Separadamente,
  o artigo discute um cenário Hadeano mais extremo em que o gradiente
  poderia chegar a até **6 unidades** (~400 mV) sob oceano mais ácido —
  mas esse número é apresentado como um potencial máximo adicional,
  NÃO como "equivalente" a nada biológico. A documentação anterior
  atribuía a equivalência biológica aos "5-6 unidades," o que estava
  errado — a equivalência é com os "3 unidades." O valor da constante
  no código (3.0) já estava certo; só a explicação textual precisava de
  correção.
- **Onde é usado**: `prebiotic.py`, constante `REFERENCE_PROTON_GRADIENT_PH_UNITS`
  — é a referência de calibração de todo o módulo 4.
- Corrigido em: `prebiotic.py`, `docs/PHYSICS_MODEL.md` §8.8.

---

## Prioridade MÉDIA (sustentam validações, não achados centrais)

### 5. ~~Bemis, Jones & Jackson (2006)~~ → Rona, P.A., Bemis, K.G., Jones, C.D., Jackson, D.R., Mitsuzawa, K., & Silver, D. (2006) — ✅ VERIFICADO E CORRIGIDO (2026-08-06)
**"Entrainment and bending in a major hydrothermal plume, Main Endeavour Field, Juan de Fuca Ridge."**
*Geophysical Research Letters* 33, L19313. DOI: 10.1029/2006GL027211

- **Verificado por leitura direta do PDF primário completo** (o usuário
  enviou `biblios/2006gl027211.pdf`). **Erro real encontrado e
  corrigido**: a citação anterior tinha autor principal, título e
  número de artigo errados (atribuía o artigo a "Bemis, Jones &
  Jackson," "Plume anomaly detected by acoustic Doppler current
  profiler," L02613) — só o DOI estava correto. O valor numérico usado
  no modelo, faixa 0.07-0.18 do coeficiente de entranhamento alpha
  (Grotto vent, Main Endeavour Field, Tabela 1 do artigo), já estava
  correto e não mudou.
- **Onde é usado**: `plume_physics.py`, `ALPHA_ENTRAINMENT_RANGE`.
- Corrigido em: `plume_physics.py`, `docs/PHYSICS_MODEL.md`,
  `fumarola_field.py`, texto do relatório e lista de
  referências, `tests/test_plume_physics.py`.

### 6. Converse, D.R., Holland, H.D., & Edmond, J.M. (1984) — ✅ VERIFICADO E CORRIGIDO (2026-08-06)
**"Flow rates in the axial hot springs of the East Pacific Rise (21°N): implications for the heat budget and the formation of massive sulfide deposits."**
*Earth and Planetary Science Letters* 69, 159–175.

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/0012-821x2990080-3.pdf`). **Erro real encontrado e
  corrigido**: a faixa citada aqui, "1-5 m/s," NÃO é a medição própria
  de Converse et al. — é a estimativa de Macdonald, K.C., Becker, K.,
  Spiess, F.N., & Ballard, R.D. (1980), *EPSL* 48, 1-7, para o vent
  "National Geographic," citada de segunda mão DENTRO do artigo de
  Converse et al. A medição direta de Converse et al. (flowmeter de
  turbina in situ, "Alvin") é **0.7-2.4 m/s**. O valor do modelo
  (1.5 m/s) permanece dentro dessa faixa correta, então a conclusão de
  validação não muda — só a atribuição da citação e a faixa numérica
  exata mudaram.
- **Onde é usado**: `plume_physics.py`, `EXIT_VELOCITY_BY_TYPE["black_smoker"]`.
- Corrigido em: `plume_physics.py`, `docs/PHYSICS_MODEL.md`,
  `tests/test_plume_physics.py` (faixa do teste ajustada de [1.0, 5.0]
  para [0.7, 2.4]).

### 7. Mittelstaedt, E., et al. (2012)
**"Quantifying diffuse and discrete venting at the Tour Eiffel vent site, Lucky Strike hydrothermal field."**
*Geochemistry, Geophysics, Geosystems* 13, Q0AF04.

- **O que verificar**: a faixa 0.009-0.111 m/s de velocidade de diffuse
  flow.
- **Onde é usado**: `plume_physics.py`, `EXIT_VELOCITY_BY_TYPE["diffuse_flow"]`.

### 8. Sarrazin, J., Rodier, P., Tivey, M.K., Singh, H., Schultz, A., & Sarradin, P.-M. (2009)
**"A dual sensor device to estimate fluid flow velocity at diffuse hydrothermal vents."**
*Deep-Sea Research Part I* 56(11), 2065–2074.

- **O que verificar**: a faixa 0.0011-0.0049 m/s (mais baixa que #7,
  discrepância não resolvida — verificar se as condições diferem o
  suficiente para explicar).
- **Onde é usado**: mesmo local que #7, como segunda fonte da faixa.

### 9. Klevenz, V., Bach, W., Schmidt, K., Hentscher, M., Koschinsky, A., & Petersen, S. (2011)
**"Geochemistry of vent fluid particles formed during initial hydrothermal fluid–seawater mixing along the Mid-Atlantic Ridge."**
*Geochemistry, Geophysics, Geosystems* 12, Q0AE05.

- **O que verificar**: o limiar ">0.2 μm" para partículas de Cu/Zn —
  usado apenas como referência de ordem de grandeza para o colóide
  fino (2 μm), não uma medição precisa.
- **Onde é usado**: `acoustics.py`, `PARTICLE_CLASSES["fine_sulfide_colloid"]`.

---

## Prioridade BAIXA (citadas, mas NÃO implementadas — gaps documentados, não riscos ativos)

### 10. Millero, F.J., Sotolongo, S., & Izaguirre, M. (1987) — ⚠️ ERRO DE ATRIBUIÇÃO ENCONTRADO E CORRIGIDO (2026-08-06)
**"The oxidation kinetics of Fe(II) in seawater."** *GCA* 51, 793-801 (termos de correção de força iônica para Fe(II) — não implementados).

- **Erro real encontrado**: este item da lista atribuía o artigo GCA
  51:793-801 aos autores "Hubinger, Fernandez & Garnett" — na verdade
  esses são os coautores de um artigo DIFERENTE de Millero de 1987
  (H2S em água do mar, *Environ. Sci. Technol.* 21:439-443, item já
  citado corretamente no texto do relatório e em `reaction_kinetics.py`).
  O artigo GCA 51:793-801 sobre oxidação de Fe(II) tem coautores
  Sotolongo & Izaguirre. O código (`reaction_kinetics.py`) já usava a
  atribuição correta — só esta lista e a lista de referências do
  relatório estavam erradas/incompletas (a entrada de Fe(II) estava
  totalmente ausente da lista de referências do relatório).
- Corrigido no texto do relatório (adicionada a referência que faltava),
  `docs/CITATIONS_TO_VERIFY.md` (este item).

### 11. González-Santana, D., et al. (2021)
*GCA* 297:143-157 (parametrização multiparamétrica de Fe(II) — **atenção**: é um artigo DIFERENTE do item #2, mesmo grupo/nome de autor principal, não confundir; não implementado).

### 12. Speer, K.G., & Rona, P.A. (1989) — ✅ VERIFICADO (2026-08-06)
**"A model of an Atlantic and Pacific hydrothermal plume."** *JGR* 94(C5), 6213-6220.

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/jc094ic05p06213.pdf`). Citação (autores, título, volume,
  páginas) confirmada exatamente. A assimetria qualitativa Atlântico
  (perfil de salinidade instável → pluma sobe mais, anomalia final fria
  e doce) vs. Pacífico (perfil estável → anomalia final quente e salgada,
  ascensão menor) é exatamente o que o artigo reporta. Só o resultado
  qualitativo é usado com confiança; as equações específicas não foram
  implementadas (o modelo usa MTT 1956 direto) — nenhuma divergência
  encontrada.
- **Atualização 2026-08-08 — achado real adicional**: a verificação
  acima checou apenas a citação qualitativa usada em `report.py`. Uma
  segunda checagem (nesta data, motivada pela correção do sistema de
  EDOs MTT em `plume_physics.py`) achou que `docs/PHYSICS_MODEL.md` §2.2
  citava formas fechadas "`5*pi^-0.25=3.76`" e "`4*pi^-0.25=3.01`" como
  vindas "da literatura consultada" (implicitamente Speer & Rona) — isso
  **não corresponde a nenhuma equação do artigo real**. A eq. (5) do
  artigo é `z* = 5*Bo^0.25*N^-0.75` (sem fator de `pi`), atribuída por
  eles a Turner (1973), não derivada no próprio artigo. Corrigido em
  `docs/PHYSICS_MODEL.md` §2.2 e no teste de validação correspondente
  (`tests/test_plume_physics.py`), que agora usa uma forma fechada
  derivada diretamente de MTT (1956) eqs. (10)/(14) em vez desses
  números fabricados.

### 13. Arndt, N., & Nisbet, E. (2012) — ✅ VERIFICADO (2026-08-06)
**"Processes on the young earth and the habitats of early life."** *Annu. Rev. Earth Planet. Sci.* 40, 521-549.

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/annurev-earth-042711-105316.pdf`). O artigo NÃO afirma
  diretamente um número "5-6 unidades de pH" — é um dos três artigos
  (junto com Pinti 2005 e Zahnle et al. 2007) que Sojo et al. (2016)
  cita como apoio geral à ideia de que concentrações mais altas de CO2
  tornaram o oceano Hadeano mais ácido. O número "6 unidades" é
  aritmética do próprio Sojo et al. (pH do vent, 11, menos pH do oceano
  Hadeano, ~5, de Pinti 2005) — Sojo representa Arndt & Nisbet
  corretamente como apoio geral ao tema (CO2 elevado, química do oceano
  Hadeano/Arqueano), não como fonte do número específico. Nenhum erro
  encontrado. Usado apenas como citação de contexto no texto do
  relatório, não em nenhum cálculo direto do código.

---

## Adicionadas para o recorte da Astrobiology (verificar antes de usar no texto final)

### 14. Hsu, H.-W., et al. (2015) — ✅ VERIFICADO (2026-08-06)
**"Ongoing hydrothermal activities within Enceladus."** *Nature* 519, 207–210.

- **Correção de metadados**: esta lista tinha o primeiro autor errado
  ("Hsu, S.-M."); o nome correto é Hsiang-Wen Hsu ("H.-W."), já citado
  corretamente no texto do relatório.
- **Verificado por leitura direta do PDF primário completo**
  (`biblios/nature14262.pdf`). A afirmação usada no texto do relatório
  ("≳90°C within Enceladus, inferred from silica nanoparticles
  detected in its plume") corresponde exatamente ao resultado central
  do artigo: partículas de nanosílica no penacho indicam reações
  hidrotermais de alta temperatura (>90°C) na interface núcleo-oceano.
  Nenhuma divergência encontrada.

### 15. Waite, J.H., et al. (2017) — ✅ VERIFICADO (2026-08-06)
**"Cassini finds molecular hydrogen in the Enceladus plume: Evidence for hydrothermal processes."** *Science* 356(6334), 155–159.

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/science.aai8703.pdf`). Confirma exatamente o uso no texto
  do relatório: a detecção de H2 molecular no penacho é interpretada
  pelos autores como evidência de reações hidrotermais água-rocha,
  comparada explicitamente pelo próprio artigo aos sistemas
  hidrotermais terrestres como Lost City. Nenhuma divergência
  encontrada.

### 16. Goodman, J.C., & Lenferink, E. (2012) — ⚠️ ERRO REAL ENCONTRADO E CORRIGIDO (2026-08-06)
**"Numerical simulations of marine hydrothermal plumes for Europa and other icy worlds."** *Icarus* 221(2), 970–983.

- **Verificado por leitura direta do PDF primário completo**
  (`biblios/j.icarus.2012.08.027.pdf`). **Erro real encontrado**: o
  texto do relatório afirmava que este artigo usa "modelos de pluma
  integral na mesma linhagem de Morton-Taylor-Turner (1956)" usada
  neste projeto — isso está ERRADO. Goodman & Lenferink (2012) usam o
  MIT GCM, um modelo de circulação oceânica rotativo e não-hidrostático,
  com a física escalonada por um "número de Rossby natural"
  (Ro* = (Bf³)^0.25/H), apropriado para o oceano profundo e dominado
  por Coriolis de Europa — uma linhagem teórica distinta (Fernando
  et al. 1998; Goodman et al. 2004), que nem cita Morton, Taylor &
  Turner (1956) nas referências. A teoria MTT de entranhamento
  turbulento (usada neste projeto, via `plume_physics.py`) é apropriada
  perto do orifício, antes da rotação planetária dominar a dinâmica —
  as duas abordagens não são intercambiáveis.
- Corrigido no texto do relatório (introdução reescrita para
  descrever corretamente a linhagem teórica do artigo).

---

## Adicionadas na Fase 5 (teste de bancada Chladni/DNA, 2026-08-06)

### 17. Faraday, M. (1831) — ✅ VERIFICADO (2026-08-06)
**"On a peculiar class of Acoustical Figures; and on certain Forms assumed by groups of particles upon vibrating elastic Surfaces."**
*Philosophical Transactions of the Royal Society of London* 121, 299–340. DOI: 10.1098/rstl.1831.0018

- **Verificado por leitura direta do PDF primário completo** (domínio
  público, baixado do Internet Archive, `biblios/faraday1831.pdf`). O
  texto confirma exatamente o uso em `docs/PHYSICS_MODEL.md` §9.3: pó
  grosso (areia) se acumula nas linhas nodais (§1-2 do artigo); pó fino
  (lycopodium) faz o oposto, se acumula nos antinós/"centres of
  oscillation" (§2); Faraday demonstra a causa (correntes de ar/streaming)
  no experimento da folha de ouro (§16), que mostra ar entrando sob a
  folha e levantando-a no centro de vibração.
- **Onde é usado**: `docs/PHYSICS_MODEL.md` §9.3, como base histórica
  para interpretar a convenção nó/antinó do experimento de bancada do
  usuário. Não implementado em código.

### 18. Crone, T.J., Wilcock, W.S.D., Barclay, A.H., & Parsons, J.D. (2006) — passagem adicional verificada (2026-08-06)
**"The Sound Generated by Mid-Ocean Ridge Black Smoker Hydrothermal Vents."**
*PLoS ONE* 1(1), e133. (já em `biblios/journal.pone.0000133.pdf`, já citado em §7.3 para a amplitude de pressão medida)

- **Passagem nova verificada por leitura direta do PDF**: a discussão de
  ressonância de conduto (Helmholtz, quarto-de-onda, "tubes, plates, or
  cavities within the chimneys") e o exemplo numérico (cavidade de 2L,
  abertura 0,02m×0,04m, c=450m/s → f≈120Hz; tubo de 1m → f≈113Hz), além
  do valor "~10-20dB acima do nível de banda larga" para os tons
  estreitos — todos confirmados exatamente no texto primário (página 5
  do PDF).
- **Onde é usado**: `docs/PHYSICS_MODEL.md` §9.4, nova seção conectando
  o experimento de bancada à hipótese de ressonância de conduto em
  chaminés reais. Não implementado em código (seção documental).

### 19. Vuillermet, G., Gires, P.-Y., Casset, F., & Poulain, C. (2016) — ⚠️ NÃO VERIFICADO, PDF não obtido
**"Chladni Patterns in a Liquid at Microscale."** *Physical Review Letters* 116(18), 184501. DOI: 10.1103/PhysRevLett.116.184501

- **Citação verificada via Crossref** (autores/título/volume/página
  confirmados por metadados oficiais), mas o **texto primário não foi
  lido** — tentativas de acesso via APS (physics.aps.org, 403) e
  ResearchGate (403, exige login) bloqueadas. Conteúdo usado
  (competição entre força de radiação acústica e arrasto por streaming,
  determinando migração para nó vs. antinó conforme tamanho/densidade da
  partícula) vem de resumos de busca, não de leitura direta.
- **Onde é usado**: `docs/PHYSICS_MODEL.md` §9.3, como apoio à discussão
  de nó-vs-antinó em líquido microescala. Não implementado em código.

### 20. Lei, J. (2017) — ⚠️ NÃO VERIFICADO, PDF não obtido
**"Formation of inverse Chladni patterns in liquids at microscale: roles of acoustic radiation and streaming-induced drag forces."** *Microfluidics and Nanofluidics* 21(3), 50. DOI: 10.1007/s10404-017-1888-5

- **Citação verificada via Crossref**, texto primário não lido (Springer
  bloqueou acesso automatizado — redireciona para página de login).
  Mesma ressalva do item #19.
- **Onde é usado**: mesmo local que #19.

## Adicionadas na discussão de eventos raros / origem da vida (2026-08-07)

Pedido direto do usuário: discutir o achado real do ensemble de 1000
runs (7/1000 realizações cruzando o limiar de relevância térmica de
Gor'kov, ver `docs/PHYSICS_MODEL.md` §7.8.1) sob a moldura de "eventos
raros" em teoria de origem da vida. As duas citações abaixo foram
**recuperadas da memória de treinamento**, não lidas em texto primário
nesta sessão — mesmo tratamento de risco dos itens #19/#20.

### 21. Lineweaver, C.H., & Davis, T.M. (2002) — ⚠️ NÃO VERIFICADO, citado de memória
**"Does the rapid appearance of life on Earth suggest that life is common in the universe?"** *Astrobiology* 2(3), 293-304.

- **Não verificado nesta sessão** — metadados (autores/ano/revista/
  volume/páginas) e o argumento central (leitura bayesiana do
  aparecimento rápido de vida na Terra como evidência de que abiogênese
  pode não exigir um passo extremamente improvável) vêm da memória de
  treinamento do modelo, não de leitura direta do PDF. Prioridade ALTA
  pra verificação, já que sustenta um argumento central da nova
  discussão, não uma validação periférica.
- **Onde é usado**: `report.py` (`_discussion_section`,
  `_abstract_section`, `_introduction_section`, `_conclusion_section`,
  `generate_admin_report`) e `docs/PHYSICS_MODEL.md` §7.8.1.

### 22. Carter, B. (1983) — ⚠️ NÃO VERIFICADO, citado de memória
**"The anthropic principle and its implications for biological evolution."** *Philosophical Transactions of the Royal Society A* 310(1512), 347-363.

- **Não verificado nesta sessão** — mesma ressalva do item #21. O
  argumento citado (raciocínio antrópico dos "passos difíceis": a
  existência de observadores não é evidência estatística forte de que
  uma transição rara seja comum, por seleção de observador) é um
  argumento clássico e amplamente citado na literatura de astrobiologia/
  seleção antrópica, mas o número exato de volume/páginas e a formulação
  precisa não foram conferidos contra o texto primário. Prioridade ALTA
  pelo mesmo motivo do item #21 — usado deliberadamente como contraponto
  de peso igual ao argumento de Lineweaver & Davis, não como ilustração
  descartável.
- **Onde é usado**: mesmo local que #21.

---

## Como usar esta lista

1. Baixe os PDFs dos itens de prioridade ALTA primeiro (#1-4, #21-22) —
   são os que sustentam os achados centrais reportados no relatório
   científico.
2. Envie aqui para eu conferir cada número/equação citado contra o
   texto primário real.
3. Se algum número estiver errado ou eu tiver interpretado mal, eu
   corrijo o código, os testes e a documentação imediatamente.
4. Os itens de prioridade BAIXA (#10-13) não afetam nenhum resultado
   atual — podem esperar.

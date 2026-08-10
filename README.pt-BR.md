# hydrovent_field

**Português (Brasil) | [English](README.md)**

[![tests](https://github.com/cesarrlamaral/hydrovent_field/actions/workflows/tests.yml/badge.svg)](https://github.com/cesarrlamaral/hydrovent_field/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Um gerador procedural de campos de fumarolas hidrotermais de dorsais
meso-oceânicas e ferramenta de simulação, construído para testar hipóteses
sobre concentração de moléculas prebióticas perto de fumarolas hidrotermais —
com base física real e citada, não em suposições ilustrativas. Inclui
diluição de pluma/cinética de espécies reativas, quatro mecanismos clássicos
de concentração (termoforese em poros, adsorção mineral, compartimentalização
por gradiente de prótons, diluição de pluma), e uma hipótese original de
concentração acústica avaliada contra pressões sonoras reais medidas em
fumarolas.

Esta é uma ferramenta de simulação real, não um demo/protótipo educacional:
cada modelo físico é implementado a partir de uma fonte primária citada,
validado contra medições reais de campo onde há dado disponível, e toda
escolha não-verificada ou ilustrativa é sinalizada explicitamente em vez de
escondida. Veja [`docs/PHYSICS_MODEL.md`](docs/PHYSICS_MODEL.md) para a
documentação completa equação-por-equação, com citações, benchmarks e uma
seção explícita de "limitações e elementos não-verificados".

![Campo de fumarolas gerado proceduralmente](assets/splash_vent_field.png)

## Funcionalidades

- **Terreno e campo de fumarolas procedurais**: heightmap por diamond-square,
  vale axial de rifte, fumarolas em clusters classificadas como black smoker /
  white smoker / diffuse flow.
- **Física de pluma validada**: modelo integral de pluma estratificada de
  Morton–Taylor–Turner (1956) (`plume_physics.py`, resolvido via
  `scipy.integrate.solve_ivp`) e cinética de reação citada para oxidação de
  H₂S/Fe(II)/Mn(II) (`reaction_kinetics.py`), calibrada contra valores reais
  medidos em campo (Mottl & McConachy 1990, Lupton et al. 1985, Rudnicki &
  Elderfield 1993, Field & Sherrell 2000).
- **Quatro módulos clássicos de concentração prebiótica** (`prebiotic.py`):
  diluição, termoforese em poros (calibrada contra Baaske et al. 2007 para
  nucleotídeos), adsorção mineral, e compartimentalização por gradiente de
  prótons (calibrada contra uma referência biológica real, Sojo et al. 2016).
- **Hipótese original de concentração acústica** (`acoustics.py`): testa se o
  campo acústico real, medido, de fumarolas hidrotermais (Crone et al. 2006)
  consegue concentrar moléculas prebióticas, via dois mecanismos fisicamente
  distintos — streaming acústico de contorno (uma PDE real de
  advecção-difusão em regime estacionário) e aprisionamento de partículas por
  força de radiação de Gor'kov — cada um selecionável independentemente.
- **Análise de sensibilidade**: amostragem por Hipercubo Latino sobre
  parâmetros com faixa de incerteza documentada na literatura
  (`--sensitivity-sweep`).
- **Ensembles de até 10.000 runs**, sequenciais ou paralelizados entre
  núcleos de CPU (`--parallel`), com seeding reprodutível por run, detecção
  de interrupção (crash) e suporte a retomada.
- **GUI desktop** (`gui.py`, Tkinter) e uma **CLI** roteirizável
  (`fumarola_field.py`), ambas rodando o mesmo código de simulação por baixo.
- Um **dataset independente de teste de bancada de padronização acústica**
  (coletado pelo autor, 2021, `data/chladni_bench_2021/`), usado para
  cross-checar a hipótese acústica contra medições reais de laboratório.

## Instalação

Requer **Python 3.10+**.

```bash
git clone https://github.com/cesarrlamaral/hydrovent_field.git
cd hydrovent_field
pip install -r requirements.txt
```

A GUI usa Tkinter, parte da biblioteca padrão do Python. Já vem incluído nos
instaladores oficiais Windows/macOS do python.org; no Linux costuma ser um
pacote de sistema separado:

```bash
# Debian/Ubuntu
sudo apt install python3-tk
```

## Uso

### GUI

```bash
python gui.py
```

Configure os parâmetros de terreno/campo de fumarolas/módulos prebióticos,
escolha simulação única ou ensemble (até 10.000 runs, opcionalmente em
paralelo entre núcleos de CPU), e veja os resultados num visualizador de
imagens com zoom/pan e uma aba de estatísticas do ensemble ao vivo.

### CLI

```bash
# Simulação única
python fumarola_field.py --seed 42 --size 257 --n-clusters 6 --spreading-rate 60

# Ensemble de 1000 runs, hipótese acústica + varredura de sensibilidade, em paralelo
python fumarola_field.py --seed 42 --runs 1000 --acoustic-mode both \
    --sensitivity-sweep --parallel

# Referência completa de flags
python fumarola_field.py --help
```

Sem `--runs`, um menu interativo guia pela escolha entre simulação única ou
ensemble, geração de imagens e (para ensembles) execução paralela.

### Testes

```bash
pytest tests/
```

## Estrutura do repositório

```
fumarola_field.py       Geração de terreno/campo de fumarolas, CLI, orquestração de ensembles
plume_physics.py        Modelo de ascensão/diluição de pluma de Morton-Taylor-Turner
reaction_kinetics.py    Cinética de decaimento de espécies reativas, citada
prebiotic.py            Módulos clássicos de concentração + análise de hotspots
acoustics.py            Hipótese de concentração acústica (streaming + trap de Gor'kov)
ensemble_stats.py       Estatísticas de ensemble compartilhadas (usadas por gui.py) —
                        descritivas, estatísticas robustas (IQR/MAD/skewness/kurtose),
                        IC por bootstrap opcional
variance_decomposition.py  Decomposição de variância estocástica vs. paramétrica (desenho aninhado)
global_sensitivity.py   Índices de sensibilidade global de Sobol' via surrogate GP próprio
driver_regression.py    Regressão multivariada por transformação de postos (controla todos
                        os preditores de uma vez, em vez de correlação um-a-um)
ensemble_report.py      Relatório estatístico HTML aberto na GUI, sem login (só tabelas +
                        gráficos agregados do ensemble — sem discussão/interpretação)
convergence_analysis.py Traço de convergência de Monte Carlo + projeção analítica de IC
                        (vale a pena rodar um ensemble maior?)
numerical_convergence.py  Verificação de solução numérica: tolerância de EDO / malha de PDE
run_qa.py               QA automatizada de integridade por run (erros inequívocos vs.
                        outliers estatísticos, deliberadamente separados)
gui.py / i18n.py        GUI desktop e suas tabelas de strings PT/EN
tests/                  Suíte pytest
docs/PHYSICS_MODEL.md   Documentação física completa: cada equação, citação, benchmark
data/chladni_bench_2021/      Dataset de teste de bancada acústico + reanálise
```

## Como citar

Se você usar este software, cite o repositório (um arquivo `CITATION.cff`
está incluído para o botão "Cite this repository" do GitHub).

## Licença

[MIT](LICENSE) — veja o arquivo LICENSE. Copyright (c) 2026 Cesar Amaral.

## Autor

Dr. Cesar Amaral — Núcleo de Genética Molecular Ambiental e Astrobiologia
(NGA), Depto. de Biofísica e Biometria, IBRAG, UERJ.
[www.ngauerj.org](http://www.ngauerj.org)

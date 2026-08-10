"""
Traduções da interface gráfica (gui.py). Inglês é o idioma padrão; o
usuário escolhe o idioma na tela de apresentação, antes da janela
principal ser construída. `set_language`/`get_language` guardam o
idioma escolhido em estado de módulo (processo tem uma única GUI por
vez, então não há necessidade de passar o idioma explicitamente por
toda a árvore de widgets) — chamado uma vez, não muda durante a sessão.

Apenas texto visível ao usuário (labels, botões, mensagens) é traduzido
aqui. Comentários/docstrings do código-fonte continuam em português,
como o resto do projeto.
"""

from __future__ import annotations

_LANG = "en"


def set_language(lang: str) -> None:
    global _LANG
    if lang not in ("en", "pt"):
        raise ValueError(f"idioma desconhecido: {lang!r}")
    _LANG = lang


def get_language() -> str:
    return _LANG


STRINGS: dict[str, dict[str, str]] = {
    # --- Splash screen ---
    "splash_version": {"en": "Version {v}", "pt": "Versão {v}"},
    "splash_desc": {
        "en": ("Procedural generator of hydrothermal vent fields and simulator of the "
               "prebiotic-concentration model for organic molecules at hydrothermal "
               "hotspots, with plume physics modules (Morton-Taylor-Turner), reactive "
               "kinetics, dilution, thermophoresis, mineral adsorption, proton gradient, "
               "and an exploratory acoustic model."),
        "pt": ("Gerador procedural de campos de fumarolas hidrotermais e simulador do "
               "modelo de concentração prebiótica de moléculas orgânicas em hotspots "
               "hidrotermais, com módulos físicos de pluma (Morton-Taylor-Turner), "
               "cinética reativa, diluição, termoforese, adsorção mineral, gradiente de "
               "prótons e um modelo acústico exploratório."),
    },
    "splash_credits": {
        "en": ("Dr. Cesar Amaral — Environmental Molecular Genetics and Astrobiology "
               "Group (NGA)\nDept. of Biophysics and Biometry, IBRAG, UERJ"),
        "pt": ("Dr. Cesar Amaral — Núcleo de Genética Molecular Ambiental e "
               "Astrobiologia (NGA)\nDepto. de Biofísica e Biometria, IBRAG, UERJ"),
    },
    "splash_continue": {"en": "Continue", "pt": "Continuar"},

    # --- Menu ---
    "menu_file": {"en": "File", "pt": "Arquivo"},
    "menu_open_experiment": {"en": "Open existing experiment...", "pt": "Abrir experimento existente..."},
    "menu_resume_experiment": {"en": "Resume aborted experiment...", "pt": "Continuar experimento abortado..."},
    "menu_exit": {"en": "Exit", "pt": "Sair"},
    "menu_help": {"en": "Help", "pt": "Ajuda"},
    "menu_about": {"en": "About", "pt": "Sobre"},

    # --- Form: experiment mode ---
    "frame_experiment_mode": {"en": "Experiment mode", "pt": "Modo do experimento"},
    "experiment_mode_desc": {
        "en": ("\"Experiment run\" locks every parameter of the real protocol used for the "
               "author's reference study (molecule class, acoustic mode, terrain/plume-"
               "physics parameters, prebiotic modules, sensitivity sweep) — so anyone can "
               "reproduce the same ensemble configuration. Only seed, number of runs, image "
               "generation, parallelism, output location and 3D-visualization options remain "
               "free (they don't affect the scientific results). \"Exploratory\" unlocks all "
               "options below."),
        "pt": ("\"Experiment run\" trava todo parâmetro do protocolo real usado no estudo "
               "de referência do autor (classe de molécula, modo acústico, parâmetros de "
               "terreno/física da pluma, módulos prebióticos, varredura de sensibilidade) — "
               "assim qualquer pessoa reproduz a mesma configuração de ensemble. Só seed, nº "
               "de runs, geração de imagem, paralelismo, local de saída e opções de "
               "visualização 3D continuam livres (não afetam o resultado científico). "
               "\"Exploratório\" libera todas as opções abaixo."),
    },
    "radio_exploratory": {"en": "Exploratory (all options)", "pt": "Exploratório (todas as opções)"},
    "radio_experiment_run": {
        "en": "Experiment run (nucleotides, acoustic A+B fixed)",
        "pt": "Experiment run (nucleotídeos, acústico A+B fixos)",
    },

    # --- Form: execution ---
    "frame_execution": {"en": "Execution", "pt": "Execução"},
    "radio_single_run": {"en": "Single run", "pt": "Single run"},
    "radio_ensemble": {"en": "Ensemble", "pt": "Ensemble"},
    "label_n_runs": {"en": "No. of runs (1-10000):", "pt": "Nº de runs (1-10000):"},
    "chk_ensemble_images": {
        "en": "Generate images for ensembles (slower)",
        "pt": "Gerar imagens em ensembles (mais lento)",
    },
    "chk_sensitivity_sweep": {
        "en": "Sensitivity sweep (Latin Hypercube over alpha, and acoustic aggregate if active)",
        "pt": "Varredura de sensibilidade (Hipercubo Latino sobre alpha, e agregado acústico se ativo)",
    },
    "radio_vardecomp": {
        "en": "Variance decomposition (nested)", "pt": "Decomposição de variância (aninhada)",
    },
    "label_outer_samples": {
        "en": "Outer parameter points (>=2):", "pt": "Pontos de parâmetro externos (>=2):",
    },
    "label_inner_replicates": {
        "en": "Inner field replicates (>=2):", "pt": "Réplicas de campo internas (>=2):",
    },
    "chk_parallel": {
        "en": "Parallel execution ({n} processes) — faster, log order may vary",
        "pt": "Execução paralela ({n} processos) — mais rápido, ordem do log pode variar",
    },
    "label_workers": {
        "en": "Worker processes (blank = auto):",
        "pt": "Processos paralelos (vazio = automático):",
    },
    "label_seed": {"en": "Base seed (empty = random):", "pt": "Seed base (vazio = aleatória):"},

    # --- Form: terrain ---
    "frame_terrain": {"en": "Terrain and vent field", "pt": "Terreno e campo de fumarolas"},
    "label_grid_size": {"en": "Grid size (2^n+1):", "pt": "Tamanho da grade (2^n+1):"},
    "label_roughness": {"en": "Roughness (0-1):", "pt": "Rugosidade (0-1):"},
    "label_n_clusters": {"en": "No. of clusters:", "pt": "Nº de clusters:"},
    "label_vents_min": {"en": "Vents/cluster (min):", "pt": "Fumarolas/cluster (mín):"},
    "label_vents_max": {"en": "Vents/cluster (max):", "pt": "Fumarolas/cluster (máx):"},
    "label_spreading_rate": {"en": "Spreading rate (mm/yr):", "pt": "Taxa de espalhamento (mm/ano):"},
    "label_local_relief": {"en": "Real local relief (m):", "pt": "Relevo local real (m):"},
    "label_ocean_depth": {"en": "Baseline ocean depth (m):", "pt": "Profundidade oceânica base (m):"},

    # --- Form: plume physics ---
    "frame_plume_physics": {
        "en": "Plume physics (see docs/PHYSICS_MODEL.md)",
        "pt": "Física da pluma (ver docs/PHYSICS_MODEL.md)",
    },
    "label_entrainment_alpha": {"en": "Entrainment coefficient alpha:", "pt": "Coef. de entranhamento alpha:"},
    "label_stratification_n": {
        "en": "Brunt-Väisälä frequency N (s⁻¹):",
        "pt": "Freq. de Brunt-Väisälä N (s⁻¹):",
    },
    "label_basin": {"en": "Basin (Fe kinetics):", "pt": "Bacia (cinética Fe):"},
    "chk_export_plume_profiles": {
        "en": "Export full plume profiles (slower)",
        "pt": "Exportar perfis completos de pluma (mais lento)",
    },

    # --- Form: 3D visualization ---
    "frame_3d": {"en": "3D visualization", "pt": "Visualização 3D"},
    "chk_gen_3d": {"en": "Generate stylized 3D scene", "pt": "Gerar cena 3D estilizada"},
    "chk_true_scale": {
        "en": "Also generate true-scale scene",
        "pt": "Também gerar cena em escala verdadeira",
    },
    "chk_artistic_render": {
        "en": "Also generate artistic view (non-scientific, requires pyvista)",
        "pt": "Também gerar visualização artística (não-científica, requer pyvista)",
    },
    "label_z_exag": {"en": "Vertical exaggeration (z_exag):", "pt": "Exagero vertical (z_exag):"},
    "label_chimney_scale": {"en": "Chimney scale:", "pt": "Escala das chaminés:"},
    "label_view_elev": {"en": "Camera - elevation:", "pt": "Câmera - elevação:"},
    "label_view_azim": {"en": "Camera - azimuth:", "pt": "Câmera - azimute:"},
    "label_domain_size": {"en": "Real domain extent (m):", "pt": "Extensão real do domínio (m):"},

    # --- Form: prebiotic modules ---
    "frame_prebiotic_modules": {
        "en": "Prebiotic concentration modules",
        "pt": "Módulos de concentração prebiótica",
    },
    "label_molecule_class": {"en": "Molecule class:", "pt": "Classe de moléculas:"},
    "label_pore_aspect_ratio": {
        "en": "Pore aspect ratio (blank = 10:1):",
        "pt": "Razão de aspecto do poro (vazio = 10:1):",
    },
    "chk_dilution": {"en": "Dilution / plume advection", "pt": "Diluição / advecção da pluma"},
    "chk_thermophoresis": {"en": "Thermophoresis in mineral pores", "pt": "Termoforese em poros minerais"},
    "chk_mineral_adsorption": {
        "en": "Adsorption on mineral surfaces",
        "pt": "Adsorção em superfícies minerais",
    },
    "chk_proton_gradient": {
        "en": "Proton gradient (alkaline compartments)",
        "pt": "Gradiente de prótons (compart. alcalinos)",
    },

    # --- Form: acoustic model ---
    "frame_acoustic": {
        "en": "Acoustic model (exploratory hypothesis)",
        "pt": "Modelo acústico (hipótese exploratória)",
    },
    "acoustic_desc": {
        "en": ("Tests whether the real sound field of hydrothermal vents (Crone et al. "
               "2006) could concentrate prebiotic molecules. No experimental validation "
               "— see docs/PHYSICS_MODEL.md."),
        "pt": ("Testa se o campo sonoro real das fumarolas (Crone et al. 2006) pode "
               "concentrar moléculas prebióticas. Sem validação experimental — ver "
               "docs/PHYSICS_MODEL.md."),
    },
    "acoustic_mode_off": {"en": "Off", "pt": "Desligado"},
    "acoustic_mode_streaming": {
        "en": "A — Acoustic streaming (dissolved solute)",
        "pt": "A — Streaming acústico (soluto dissolvido)",
    },
    "acoustic_mode_particle_trap": {
        "en": "B — Particle trapping (Gor'kov)",
        "pt": "B — Aprisionamento de partícula (Gor'kov)",
    },
    "acoustic_mode_both": {"en": "A + B — Both combined", "pt": "A + B — Ambos combinados"},
    "label_acoustic_particle_radius": {
        "en": "Custom particle radius, μm (blank = cited population, B/A+B only):",
        "pt": "Raio de partícula customizado, μm (vazio = população citada, só B/A+B):",
    },
    "label_acoustic_particle_density": {
        "en": "Custom particle density, kg/m³ (blank = cited population, B/A+B only):",
        "pt": "Densidade de partícula customizada, kg/m³ (vazio = população citada, só B/A+B):",
    },
    "acoustic_coherence_desc1": {
        "en": ("Combination between different vents (does not affect each vent's own "
               "self-interference with its seafloor image, which is always coherent):"),
        "pt": ("Combinação entre fumarolas diferentes (não afeta a auto-interferência de "
               "cada fumarola com sua própria imagem, que é sempre coerente):"),
    },
    "chk_coherent_bound": {
        "en": "Test idealized upper bound (all vents in phase, same frequency)",
        "pt": "Testar limite superior idealizado (todas as fumarolas em fase, mesma frequência)",
    },
    "acoustic_coherence_desc2": {
        "en": ("Off (default): incoherent summation between vents — no evidence of "
               "phase-locking between independent vents. On: optimistic best-case "
               "scenario, used only to test whether this assumption affects the "
               "conclusion."),
        "pt": ("Desligado (padrão): soma incoerente entre fumarolas — não há evidência "
               "de travamento de fase entre fumarolas independentes. Ligado: cenário de "
               "melhor caso otimista, usado só para testar se essa suposição afeta a "
               "conclusão."),
    },

    # --- Form: output ---
    "frame_output": {"en": "Output", "pt": "Saída"},
    "label_basename": {"en": "Base filename:", "pt": "Nome base dos arquivos:"},
    "btn_run": {"en": "Run", "pt": "Executar"},
    "status_ready": {"en": "Ready.", "pt": "Pronto."},

    # --- Output panel ---
    "btn_open_outputs": {"en": "Open output folder", "pt": "Abrir pasta de saída"},
    "tab_images": {"en": "Images", "pt": "Imagens"},
    "tab_stats": {"en": "Ensemble analysis", "pt": "Análise do ensemble"},
    "tab_about": {"en": "About", "pt": "Sobre"},
    "label_log": {"en": "Log:", "pt": "Log:"},

    # --- Images tab ---
    "label_image": {"en": "Image:", "pt": "Imagem:"},
    "btn_prev_run": {"en": "< Previous run", "pt": "< Run anterior"},
    "btn_next_run": {"en": "Next run >", "pt": "Run seguinte >"},
    "no_image_placeholder": {
        "en": "No image yet — run a simulation.",
        "pt": "Nenhuma imagem ainda — execute uma run.",
    },
    "image_unavailable": {
        "en": "(image unavailable for this run — were images disabled for this ensemble?)",
        "pt": "(imagem não disponível para esta run — imagens desligadas neste ensemble?)",
    },

    # --- IMAGE_KEYS labels ---
    "img_2d": {"en": "2D map", "pt": "Mapa 2D"},
    "img_3d": {"en": "3D scene (stylized)", "pt": "Cena 3D (estilizada)"},
    "img_truescale": {"en": "3D scene (true scale)", "pt": "Cena 3D (escala verdadeira)"},
    "img_artistic": {"en": "Artistic view (non-scientific)", "pt": "Visualização artística (não-científica)"},
    "img_hotspots": {"en": "Prebiotic hotspots", "pt": "Hotspots prebióticos"},
    "img_acoustic": {
        "en": "Acoustic field (exploratory hypothesis)",
        "pt": "Campo acústico (hipótese exploratória)",
    },
    "img_module_dilution": {"en": "Gradient map — Dilution", "pt": "Mapa de gradiente — Diluição"},
    "img_module_thermophoresis": {
        "en": "Gradient map — Thermophoresis",
        "pt": "Mapa de gradiente — Termoforese",
    },
    "img_module_mineral_adsorption": {
        "en": "Gradient map — Mineral adsorption",
        "pt": "Mapa de gradiente — Adsorção mineral",
    },
    "img_module_proton_gradient": {
        "en": "Gradient map — Proton gradient",
        "pt": "Mapa de gradiente — Gradiente de prótons",
    },

    # --- ZoomPanCanvas ---
    "btn_fit_window": {"en": "Fit to window", "pt": "Ajustar à janela"},
    "zoom_hint": {
        "en": "(mouse wheel = zoom, drag = pan, double-click = fit)",
        "pt": "(roda do mouse = zoom, arrastar = mover, duplo clique = ajustar)",
    },

    # --- Stats tab ---
    "stats_placeholder": {
        "en": "Run an ensemble (>1 run) to see statistical analysis here.",
        "pt": "Rode um ensemble (>1 run) para ver a análise estatística aqui.",
    },
    "stats_loading": {
        "en": "Computing ensemble statistics...",
        "pt": "Calculando estatísticas do ensemble...",
    },
    "frame_desc_stats": {"en": "Descriptive statistics", "pt": "Estatísticas descritivas"},
    "frame_charts": {"en": "Charts", "pt": "Gráficos"},
    "frame_runs_table": {"en": "Experiment runs", "pt": "Runs do experimento"},

    "stat_top_hotspot": {
        "en": "Leading hotspot per run (x control)",
        "pt": "Hotspot líder por run (x controle)",
    },
    "stat_pooled_enrich": {
        "en": "Pooled enrichment across all vents (x control)",
        "pt": "Enriquecimento de todas as fumarolas agrupadas (x controle)",
    },
    "stat_up_down_label": {
        "en": "Vents that increased / decreased / unchanged vs. control",
        "pt": "Fumarolas que aumentaram / diminuíram / não mudaram vs. controle",
    },
    "stat_up_down_value": {
        "en": "{n_up} increased | {n_down} decreased | {n_eq} unchanged (of {n_total})",
        "pt": "{n_up} aumentaram | {n_down} diminuíram | {n_eq} inalteradas (de {n_total})",
    },
    "stat_max_conc": {
        "en": "Maximum hotspot per run (µM, absolute)",
        "pt": "Hotspot máximo por run (µM, absoluto)",
    },
    "stat_mean_conc": {
        "en": "Mean concentration per run (µM, absolute)",
        "pt": "Concentração média por run (µM, absoluto)",
    },
    "stat_n_vents": {"en": "No. of vents per run", "pt": "Nº de fumarolas por run"},
    "stat_top_type": {
        "en": "Leading hotspot type, per run:",
        "pt": "Tipo do hotspot líder, por run:",
    },
    "stat_fmt": {
        "en": "n={n} | mean={mean:.3f} | std={std:.3f} | min={min:.3f} | median={median:.3f} | max={max:.3f}",
        "pt": "n={n} | média={mean:.3f} | desvio={std:.3f} | mín={min:.3f} | mediana={median:.3f} | máx={max:.3f}",
    },

    # --- Ensemble charts (matplotlib) ---
    "chart_enrichment_hist_title": {
        "en": "Enrichment vs. control —\nall vents (log2, 0=1x)",
        "pt": "Enriquecimento vs. controle —\ntodas as fumarolas (log2, 0=1x)",
    },
    "chart_log2_xlabel": {"en": "log2(x control)", "pt": "log2(x controle)"},
    "chart_top_hist_title": {
        "en": "Leading hotspot,\nper run (x control)",
        "pt": "Hotspot líder,\npor run (x controle)",
    },
    "chart_xcontrol_xlabel": {"en": "x control", "pt": "x controle"},
    "chart_type_bar_title": {
        "en": "Leading hotspot type\n(run count)",
        "pt": "Tipo do hotspot líder\n(contagem de runs)",
    },
    "chart_scatter_title": {
        "en": "No. of vents vs.\nleading hotspot (x control)",
        "pt": "Nº de fumarolas vs.\nhotspot líder (x controle)",
    },
    "chart_scatter_xlabel": {"en": "no. of vents", "pt": "nº de fumarolas"},

    # --- Run table columns ---
    "col_run": {"en": "Run", "pt": "Run"},
    "col_seed": {"en": "Seed", "pt": "Seed"},
    "col_n_vents": {"en": "No. vents", "pt": "Nº fumarolas"},
    "col_top_enrich": {"en": "Leading hotspot (x control)", "pt": "Hotspot líder (x controle)"},
    "col_mean_enrich": {"en": "Mean enrich. (x)", "pt": "Enriq. médio (x)"},
    "col_n_up": {"en": "Increased", "pt": "Aumentaram"},
    "col_n_down": {"en": "Decreased", "pt": "Diminuíram"},
    "col_top_type": {"en": "Leading hotspot type", "pt": "Tipo do hotspot líder"},

    # --- Ensemble statistical report (open to all users, no interpretation/
    # discussion — see ensemble_report.py) ---
    "btn_generate_ensemble_report": {
        "en": "Generate statistical report (HTML)",
        "pt": "Gerar relatório estatístico (HTML)",
    },
    "ensemble_report_generating": {
        "en": "Generating statistical report...",
        "pt": "Gerando relatório estatístico...",
    },
    "ensemble_report_done": {
        "en": "Statistical report saved: {path}",
        "pt": "Relatório estatístico salvo: {path}",
    },
    "ensemble_report_error": {
        "en": "Could not generate the statistical report: {error}",
        "pt": "Não foi possível gerar o relatório estatístico: {error}",
    },
    "ensemble_report_title": {
        "en": "Ensemble Statistical Report",
        "pt": "Relatório Estatístico do Ensemble",
    },
    "ensemble_report_intro": {
        "en": "Statistical summary of the ensemble only — no interpretation, "
              "hypothesis discussion, or manuscript framing. Descriptive "
              "statistics, charts, and per-run results below reflect exactly "
              "what this ensemble produced.",
        "pt": "Resumo estatístico do ensemble apenas — sem interpretação, "
              "discussão de hipóteses ou moldura de manuscrito. As estatísticas "
              "descritivas, gráficos e resultados por run abaixo refletem "
              "exatamente o que este ensemble produziu.",
    },
    "ensemble_report_meta_experiment": {"en": "Experiment folder", "pt": "Pasta do experimento"},
    "ensemble_report_meta_generated": {"en": "Report generated", "pt": "Relatório gerado em"},
    "ensemble_report_meta_base_seed": {"en": "Base seed", "pt": "Seed base"},
    "ensemble_report_meta_n_runs": {"en": "No. of runs", "pt": "Nº de runs"},
    "ensemble_report_meta_sweep": {"en": "Sensitivity sweep", "pt": "Varredura de sensibilidade"},
    "ensemble_report_meta_acoustic": {"en": "Acoustic mode", "pt": "Modo acústico"},
    "ensemble_report_meta_molecule": {"en": "Molecule class", "pt": "Classe de molécula"},
    "ensemble_report_meta_basin": {"en": "Basin", "pt": "Bacia"},
    "ensemble_report_meta_modules": {"en": "Active modules", "pt": "Módulos ativos"},
    "ensemble_report_yes": {"en": "yes", "pt": "sim"},
    "ensemble_report_no": {"en": "no", "pt": "não"},

    "ensemble_report_table1_title": {"en": "Descriptive statistics", "pt": "Estatísticas descritivas"},
    "ensemble_report_table1_th_metric": {"en": "Metric", "pt": "Métrica"},
    "ensemble_report_table1_th_n": {"en": "n", "pt": "n"},
    "ensemble_report_table1_th_mean": {"en": "Mean [95% CI]", "pt": "Média [IC 95%]"},
    "ensemble_report_table1_th_median": {"en": "Median [95% CI]", "pt": "Mediana [IC 95%]"},
    "ensemble_report_table1_th_std": {"en": "Std. dev.", "pt": "Desvio-padrão"},
    "ensemble_report_table1_th_iqr": {"en": "IQR (Q1–Q3)", "pt": "IQR (Q1–Q3)"},
    "ensemble_report_table1_th_skew": {"en": "Skewness", "pt": "Assimetria"},
    "ensemble_report_table1_note": {
        "en": "95% CIs computed by case bootstrap ({n_bootstrap} resamples; see "
              "docs/PHYSICS_MODEL.md §10.2b). Skewness far from 0 means the "
              "mean/median describe the distribution differently — prefer the "
              "median for a skewed metric.",
        "pt": "IC 95% via bootstrap de casos ({n_bootstrap} reamostragens; ver "
              "docs/PHYSICS_MODEL.md §10.2b). Assimetria longe de 0 significa "
              "que média/mediana descrevem a distribuição de formas diferentes — "
              "prefira a mediana para uma métrica assimétrica.",
    },

    "ensemble_report_figure_title": {"en": "Figure 1", "pt": "Figura 1"},
    "ensemble_report_figure_caption": {
        "en": "Figure 1. Ensemble-level summary of prebiotic enrichment vs. "
              "control, pooled across all runs. (A) Histogram of log2(enrichment) "
              "for every vent in every run (0 = no change vs. control, dashed "
              "line); right-shifted bars indicate net enrichment. (B) Histogram of "
              "the leading (highest-enrichment) hotspot per run, in linear x-control "
              "units (dashed line at 1x = no change). (C) Count of runs by the vent "
              "type of the leading hotspot. (D) Leading hotspot enrichment vs. "
              "number of vents in that run's field, one point per run (dashed line "
              "at 1x) — a visual, univariate look at whether larger fields trend "
              "toward higher enrichment (see Table 3 below for a multivariate test "
              "of this, when applicable).",
        "pt": "Figura 1. Resumo do ensemble de enriquecimento prebiótico vs. "
              "controle, agrupado entre todas as runs. (A) Histograma de "
              "log2(enriquecimento) de cada fumarola de cada run (0 = sem mudança "
              "vs. controle, linha tracejada); barras deslocadas à direita indicam "
              "enriquecimento líquido. (B) Histograma do hotspot líder (maior "
              "enriquecimento) de cada run, em unidades lineares x-controle (linha "
              "tracejada em 1x = sem mudança). (C) Contagem de runs pelo tipo de "
              "fumarola do hotspot líder. (D) Enriquecimento do hotspot líder vs. "
              "nº de fumarolas do campo daquela run, um ponto por run (linha "
              "tracejada em 1x) — um olhar visual e univariado sobre se campos "
              "maiores tendem a maior enriquecimento (ver Tabela 3 abaixo para um "
              "teste multivariado disso, quando aplicável).",
    },

    "ensemble_report_table2_title": {"en": "Per-run results", "pt": "Resultados por run"},

    "ensemble_report_table3_title": {
        "en": "Sensitivity sweep — driver analysis (multivariate)",
        "pt": "Varredura de sensibilidade — análise de drivers (multivariada)",
    },
    "ensemble_report_table3_note": {
        "en": "Rank-transform multiple regression (Iman & Conover, 1979) against "
              "{response_name}, controlling all predictors simultaneously — a "
              "coefficient near zero with a high p-value after Holm correction "
              "means that predictor's apparent marginal effect (if any) does not "
              "survive controlling for the others. VIF > ~5 flags collinearity "
              "between predictors. See docs/PHYSICS_MODEL.md §7.8.4.",
        "pt": "Regressão múltipla por transformação de postos (Iman & Conover, "
              "1979) contra {response_name}, controlando todos os preditores "
              "simultaneamente — um coeficiente perto de zero com p-valor alto "
              "após correção de Holm significa que o efeito marginal aparente "
              "daquele preditor (se houver) não sobrevive ao controle dos "
              "demais. VIF > ~5 sinaliza colinearidade entre preditores. Ver "
              "docs/PHYSICS_MODEL.md §7.8.4.",
    },
    "ensemble_report_table3_th_predictor": {"en": "Predictor", "pt": "Preditor"},
    "ensemble_report_table3_th_coef": {"en": "Coefficient", "pt": "Coeficiente"},
    "ensemble_report_table3_th_p": {"en": "p", "pt": "p"},
    "ensemble_report_table3_th_p_holm": {"en": "p (Holm)", "pt": "p (Holm)"},
    "ensemble_report_table3_th_vif": {"en": "VIF", "pt": "VIF"},

    "ensemble_report_meta_vardecomp_value": {
        "en": "nested design ({outer} outer x {inner} inner)",
        "pt": "desenho aninhado ({outer} externos x {inner} internos)",
    },
    "ensemble_report_vardecomp_title": {
        "en": "Variance decomposition (stochastic vs. parametric)",
        "pt": "Decomposição de variância (estocástica vs. paramétrica)",
    },
    "ensemble_report_vardecomp_note": {
        "en": "Nested design: {outer} parameter points ({params}) x {inner} field-seed "
              "replicates each, decomposed via one-way random-effects ANOVA (Searle, "
              "Casella & McCulloch, 1992). See docs/PHYSICS_MODEL.md §7.8.2.",
        "pt": "Desenho aninhado: {outer} pontos de parâmetro ({params}) x {inner} réplicas "
              "de seed de campo cada, decomposto via ANOVA de um fator aleatório (Searle, "
              "Casella & McCulloch, 1992). Ver docs/PHYSICS_MODEL.md §7.8.2.",
    },
    "ensemble_report_vardecomp_th_component": {"en": "Component", "pt": "Componente"},
    "ensemble_report_vardecomp_th_fraction": {"en": "Fraction of variance [95% CI]", "pt": "Fração da variância [IC 95%]"},
    "ensemble_report_vardecomp_row_stochastic": {
        "en": "Stochastic (vent field, seed)", "pt": "Estocástica (campo de fumarolas, seed)",
    },
    "ensemble_report_vardecomp_row_parametric": {
        "en": "Parametric (swept parameters)", "pt": "Paramétrica (parâmetros varridos)",
    },
    "ensemble_report_vardecomp_clipped_note": {
        "en": "The parametric component's raw estimate was negative and was clipped to "
              "zero — the parametric signal is at or below the sampling noise floor in "
              "this run.",
        "pt": "A estimativa bruta da componente paramétrica saiu negativa e foi grampeada "
              "em zero — o sinal paramétrico está no ou abaixo do ruído de amostragem "
              "nesta run.",
    },
    "ensemble_report_sobol_title": {
        "en": "Global sensitivity (Sobol' indices, GP surrogate)",
        "pt": "Sensibilidade global (índices de Sobol', surrogate GP)",
    },
    "ensemble_report_sobol_note": {
        "en": "First-order (S1) and total-effect (ST) Sobol' indices from a Gaussian "
              "Process surrogate fit to the nested design above (Saltelli/Jansen "
              "estimators). Surrogate leave-one-out CV R² = {loo_r2:.3f}. See "
              "docs/PHYSICS_MODEL.md §7.8.3.",
        "pt": "Índices de Sobol' de primeira ordem (S1) e efeito total (ST) de um "
              "surrogate de Processo Gaussiano ajustado ao desenho aninhado acima "
              "(estimadores de Saltelli/Jansen). R² de validação cruzada "
              "leave-one-out do surrogate = {loo_r2:.3f}. Ver docs/PHYSICS_MODEL.md §7.8.3.",
    },
    "ensemble_report_sobol_warning": {
        "en": "(weak fit — R² < 0.5, treat these indices as indicative, not conclusive)",
        "pt": "(ajuste fraco — R² < 0,5, trate estes índices como indicativos, não conclusivos)",
    },
    "ensemble_report_sobol_th_param": {"en": "Parameter", "pt": "Parâmetro"},
    "ensemble_report_sobol_th_s1": {"en": "S1 [95% CI]", "pt": "S1 [IC 95%]"},
    "ensemble_report_sobol_th_st": {"en": "ST [95% CI]", "pt": "ST [IC 95%]"},

    # --- About tab ---
    "about_site": {"en": "Lab website: www.ngauerj.org", "pt": "Site do laboratório: www.ngauerj.org"},
    "about_instagram": {
        "en": "Instagram: @chuck_nga_uerj  |  @ngamediauerj",
        "pt": "Instagram: @chuck_nga_uerj  |  @ngamediauerj",
    },

    # --- Dialogs / messageboxes ---
    "msg_running_title": {"en": "Running", "pt": "Em execução"},
    "msg_running_body": {
        "en": "Wait for the current run to finish.",
        "pt": "Aguarde a execução atual terminar.",
    },
    "dialog_select_experiment_title": {
        "en": "Select the experiment folder (experimento_YYMMDD_HHMMSS)",
        "pt": "Selecione a pasta do experimento (experimento_AAMMDD_HHMMSS)",
    },
    "msg_invalid_folder_title": {"en": "Invalid folder", "pt": "Pasta inválida"},
    "msg_invalid_folder_body": {
        "en": ("This folder doesn't contain an experiment_metadata.json — select an "
               "\"experimento_...\" folder generated by this program."),
        "pt": ("Essa pasta não contém um experiment_metadata.json — selecione uma pasta "
               "\"experimento_...\" gerada por este programa."),
    },
    "msg_empty_experiment_title": {"en": "Empty experiment", "pt": "Experimento vazio"},
    "msg_empty_experiment_body": {
        "en": "No completed run was found in this folder.",
        "pt": "Nenhuma run completa foi encontrada nesta pasta.",
    },
    "msg_no_aborted_title": {"en": "No aborted experiment", "pt": "Nenhum experimento abortado"},
    "msg_no_aborted_body": {
        "en": "No incomplete experiment was found in:\n{path}",
        "pt": "Nenhum experimento incompleto foi encontrado em:\n{path}",
    },
    "msg_cannot_resume_title": {"en": "Cannot resume", "pt": "Não é possível retomar"},
    "msg_cannot_resume_body": {
        "en": ("This experiment was created by an earlier version of the program, which "
               "didn't save the full parameters needed to safely resume execution "
               "(terrain, physics, modules). Run a new, complete experiment."),
        "pt": ("Este experimento foi criado por uma versão anterior do programa, que não "
               "salvava os parâmetros completos necessários para retomar a execução com "
               "segurança (terreno, física, módulos). Rode um novo experimento completo."),
    },
    "status_resuming": {
        "en": "Resuming experiment ({n_completed}/{n_runs} runs already completed)...",
        "pt": "Retomando experimento ({n_completed}/{n_runs} runs já completas)...",
    },

    "dialog_resume_title": {"en": "Resume aborted experiment", "pt": "Continuar experimento abortado"},
    "dialog_resume_label": {
        "en": "Incomplete experiments found:",
        "pt": "Experimentos incompletos encontrados:",
    },
    "col_dir": {"en": "Folder", "pt": "Pasta"},
    "col_progress": {"en": "Completed runs", "pt": "Runs completas"},
    "col_seed_base": {"en": "Base seed", "pt": "Seed base"},
    "col_resumable": {"en": "Resumable", "pt": "Retomável"},
    "val_yes": {"en": "Yes", "pt": "Sim"},
    "val_no_old_format": {"en": "No (old format)", "pt": "Não (formato antigo)"},
    "btn_cancel": {"en": "Cancel", "pt": "Cancelar"},
    "btn_resume_this": {"en": "Resume this experiment", "pt": "Continuar este experimento"},

    "msg_invalid_param_title": {"en": "Invalid parameter", "pt": "Parâmetro inválido"},
    "msg_invalid_param_body": {
        "en": "Check that all numeric fields are filled in correctly.",
        "pt": "Verifique se todos os campos numéricos estão preenchidos corretamente.",
    },
    "msg_invalid_value_title": {"en": "Invalid value", "pt": "Valor inválido"},
    "msg_invalid_value_body": {
        "en": "The number of runs must be between 1 and 10000.",
        "pt": "O nº de runs deve estar entre 1 e 10000.",
    },
    "msg_invalid_vardecomp_body": {
        "en": "Outer parameter points and inner field replicates must each be >= 2.",
        "pt": "Pontos de parâmetro externos e réplicas de campo internas precisam ser >= 2 cada.",
    },
    "status_running": {"en": "Running {n} run(s)...", "pt": "Executando {n} run(s)..."},
    "status_running_vardecomp": {
        "en": "Running nested variance decomposition ({outer} outer x {inner} inner)...",
        "pt": "Rodando decomposição de variância aninhada ({outer} externos x {inner} internos)...",
    },
    "status_done_vardecomp": {
        "en": "Done: {n} run(s) (nested design) in {dir}",
        "pt": "Concluído: {n} run(s) (desenho aninhado) em {dir}",
    },
    "frame_vardecomp": {
        "en": "Variance decomposition (this run)", "pt": "Decomposição de variância (esta run)",
    },
    "status_error": {"en": "Execution error.", "pt": "Erro na execução."},
    "msg_error_title": {"en": "Error", "pt": "Erro"},
    "status_done": {"en": "Done: {n} run(s) in {dir}", "pt": "Concluído: {n} run(s) em {dir}"},
    "run_indicator": {"en": "Run {i}/{n}: {name}", "pt": "Run {i}/{n}: {name}"},
}


def t(key: str, **kwargs) -> str:
    entry = STRINGS[key]
    text = entry[_LANG]
    return text.format(**kwargs) if kwargs else text

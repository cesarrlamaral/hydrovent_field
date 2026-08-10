"""
GUI (Tkinter) para o gerador de campos de fumarolas hidrotermais e o
modelo de hotspots prebióticos.

Formulário com os parâmetros de geração, escala real e módulos
prebióticos ligáveis/desligáveis; roda a simulação em uma thread separada
(para a janela não travar); mostra o log de progresso, um visualizador
de imagens com zoom/pan, e (para ensembles) uma aba de análise
estatística completa do experimento.

Idioma da interface (inglês por padrão, português como alternativa) é
escolhido pelo usuário na tela de apresentação, antes da janela
principal ser construída — ver i18n.py.

Uso:
    python gui.py
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import fumarola_field as ff
import plume_physics as pp
import reaction_kinetics as rk
import acoustics as ac
import ensemble_stats as es
import ensemble_report as er
import i18n
from i18n import t

_SPLASH_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "assets", "splash_vent_field.png")
_DEVELOPMENT_YEAR = "2026"


def _show_splash():
    """Tela de apresentação exibida antes da janela principal — nome/versão
    do software, a mesma descrição e créditos da aba 'Sobre', uma imagem de
    campo de fumarolas, um seletor de idioma (Inglês/Português, Inglês
    selecionado por padrão) e um botão "Continue"/"Continuar" (canto
    inferior direito) que fecha a splash e libera a abertura da janela
    principal já no idioma escolhido (ver i18n.py)."""
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    bg = "#0d1b2a"
    splash.configure(bg=bg)

    width, height = 640, 830
    sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
    splash.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")

    border = tk.Frame(splash, bg="#3a5468", bd=0)
    border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    body = tk.Frame(border, bg=bg)
    body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    tk.Label(body, text="HYDROVENT", font=("Segoe UI", 28, "bold"),
             fg="#e8f1f2", bg=bg).pack(pady=(24, 0))
    lbl_version = tk.Label(body, font=("Segoe UI", 11), fg="#8fb3c7", bg=bg)
    lbl_version.pack(pady=(0, 12))

    lang_bar = tk.Frame(body, bg=bg)
    lang_bar.pack(pady=(0, 14))

    photo = None
    if os.path.exists(_SPLASH_IMAGE_PATH):
        img = Image.open(_SPLASH_IMAGE_PATH)
        img.thumbnail((520, 380), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        tk.Label(body, image=photo, bg=bg).pack(pady=(0, 16))
    splash._splash_photo_ref = photo  # evita coleta de lixo do PhotoImage

    lbl_desc = tk.Label(body, font=("Segoe UI", 9), fg="#c7d8e0", bg=bg,
                         wraplength=560, justify="center")
    lbl_desc.pack(padx=24, pady=(0, 14))

    lbl_credits = tk.Label(body, font=("Segoe UI", 9), fg="#c7d8e0", bg=bg, justify="center")
    lbl_credits.pack(pady=(0, 6))

    tk.Label(body, text="www.ngauerj.org   |   @chuck_nga_uerj   |   @ngamediauerj",
             font=("Segoe UI", 8), fg="#7fa0b3", bg=bg).pack(pady=(0, 18))

    bottom_bar = tk.Frame(body, bg=bg)
    bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=(0, 16))
    tk.Label(bottom_bar, text=f"© {_DEVELOPMENT_YEAR}", font=("Segoe UI", 8),
              fg="#5f7d8f", bg=bg).pack(side=tk.LEFT)
    btn_continue = tk.Button(bottom_bar, command=splash.destroy,
                              bg="#3a5468", fg="#e8f1f2", activebackground="#4d6f88",
                              activeforeground="#ffffff", relief=tk.FLAT, padx=16, pady=4,
                              cursor="hand2", font=("Segoe UI", 9, "bold"))
    btn_continue.pack(side=tk.RIGHT)

    def _lang_btn_style(btn: tk.Button, selected: bool):
        btn.configure(bg="#4d6f88" if selected else "#25384a",
                      fg="#ffffff" if selected else "#9fb8c8")

    def apply_lang(lang: str):
        i18n.set_language(lang)
        lbl_version.configure(text=t("splash_version", v=ff.__version__))
        lbl_desc.configure(text=t("splash_desc"))
        lbl_credits.configure(text=t("splash_credits"))
        btn_continue.configure(text=t("splash_continue"))
        _lang_btn_style(btn_en, lang == "en")
        _lang_btn_style(btn_pt, lang == "pt")

    btn_en = tk.Button(lang_bar, text="English", relief=tk.FLAT, padx=12, pady=3,
                        cursor="hand2", font=("Segoe UI", 8, "bold"),
                        command=lambda: apply_lang("en"))
    btn_pt = tk.Button(lang_bar, text="Português", relief=tk.FLAT, padx=12, pady=3,
                        cursor="hand2", font=("Segoe UI", 8, "bold"),
                        command=lambda: apply_lang("pt"))
    btn_en.pack(side=tk.LEFT, padx=3)
    btn_pt.pack(side=tk.LEFT, padx=3)

    apply_lang("en")  # inglês é o padrão

    splash.mainloop()


class TextRedirector:
    """Redireciona stdout (os prints de fumarola_field) para uma fila lida pela GUI."""

    def __init__(self, msg_queue: queue.Queue):
        self.queue = msg_queue

    def write(self, text):
        if text:
            self.queue.put(("log", text))

    def flush(self):
        pass


class ZoomPanCanvas(ttk.Frame):
    """
    Canvas de imagem com zoom (roda do mouse, centrado no cursor) e
    panorâmica (arrastar com o botão esquerdo). Mantém a imagem PIL
    original em memória e re-amostra a cada zoom, para não perder
    qualidade ao ampliar.
    """

    MIN_ZOOM = 0.05
    MAX_ZOOM = 4.0

    def __init__(self, parent):
        super().__init__(parent)
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="−", width=3, command=lambda: self._zoom_by(1 / 1.2)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1.2)).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(toolbar, text=t("btn_fit_window"), command=self.fit_to_window).pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_zoom = ttk.Label(toolbar, text="100%")
        self.lbl_zoom.pack(side=tk.LEFT, padx=8)
        ttk.Label(toolbar, text=t("zoom_hint"),
                  foreground="#777").pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg="#1a1a19", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.pil_image: Image.Image | None = None
        self.photo = None
        self.zoom = 1.0

        self.canvas.bind("<MouseWheel>", self._on_wheel)          # Windows
        self.canvas.bind("<Button-4>", lambda e: self._zoom_by(1.2, e))   # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self._zoom_by(1 / 1.2, e))  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Double-Button-1>", lambda e: self.fit_to_window())

    def load(self, path: str):
        self.pil_image = Image.open(path)
        self.canvas.update_idletasks()
        self.fit_to_window()

    def clear(self, message: str = ""):
        self.pil_image = None
        self.photo = None
        self.canvas.delete("all")
        if message:
            cw = self.canvas.winfo_width() or 400
            ch = self.canvas.winfo_height() or 300
            self.canvas.create_text(cw / 2, ch / 2, text=message, fill="#888", font=("Segoe UI", 10))
        self.lbl_zoom.configure(text="—")

    def fit_to_window(self):
        if self.pil_image is None:
            return
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        iw, ih = self.pil_image.size
        self.zoom = max(self.MIN_ZOOM, min(cw / iw, ch / ih))
        self._render()

    def _render(self):
        if self.pil_image is None:
            return
        iw, ih = self.pil_image.size
        new_w, new_h = max(1, int(iw * self.zoom)), max(1, int(ih * self.zoom))
        resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))
        self.lbl_zoom.configure(text=f"{self.zoom * 100:.0f}%")

    def _zoom_by(self, factor: float, event=None):
        if self.pil_image is None:
            return
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return

        ex = event.x if event is not None else self.canvas.winfo_width() / 2
        ey = event.y if event is not None else self.canvas.winfo_height() / 2
        img_x = self.canvas.canvasx(ex) / self.zoom
        img_y = self.canvas.canvasy(ey) / self.zoom

        self.zoom = new_zoom
        self._render()

        iw, ih = self.pil_image.size
        new_cx, new_cy = img_x * self.zoom, img_y * self.zoom
        frac_x = max(0.0, (new_cx - ex) / max(1, iw * self.zoom))
        frac_y = max(0.0, (new_cy - ey) / max(1, ih * self.zoom))
        self.canvas.xview_moveto(frac_x)
        self.canvas.yview_moveto(frac_y)

    def _on_wheel(self, event):
        self._zoom_by(1.15 if event.delta > 0 else 1 / 1.15, event)

    def _on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


# --------------------------------------------------------------------------
# Protocolo "Experiment Run" — pedido do usuário 2026-08-09: travar TODO
# parâmetro que definiu os ensembles reais usados na análise submetida à
# Astrobiology (outputs/experimento_260807_021219 e
# experimento_260808_111609, ambos molécula=nucleotideos/acústica=both/
# sensitivity-sweep, bacia atlantic), não só molécula+modo acústico como
# antes — pra qualquer pessoa que clique "Experiment Run" reproduzir
# exatamente o mesmo protocolo científico, mesmo que os defaults do
# código mudem no futuro (por isso os valores são travados aqui de forma
# explícita, não apenas herdados do valor-padrão de cada widget).
#
# Deliberadamente DE FORA deste protocolo (continuam livres mesmo em modo
# "Experiment Run"): seed (cada ensemble novo precisa de uma seed própria,
# nunca reaproveitar a mesma), nº de runs, geração de imagens, paralelismo/
# nº de processos, pasta de saída/nome base, e todos os parâmetros de
# visualização 3D (afetam só a renderização, nunca os dados científicos).
# nº de runs (pedido do usuário 2026-08-09: já relatamos os tamanhos de
# ensemble usados no próprio paper, então esse número pode continuar
# livre) só ganha aqui um valor SUGERIDO pra pré-preencher o campo (o
# maior dos dois ensembles reais, experimento_260807_021219/
# experimento_260808_111609) — não uma trava; o campo continua editável.
_EXPERIMENT_RUN_SUGGESTED_N_RUNS = 1000

_EXPERIMENT_RUN_PROTOCOL = {
    "size": 257,
    "roughness": 0.55,
    "n_clusters": 6,
    "vents_min": 2,
    "vents_max": 9,
    "spreading_rate": 60.0,
    "local_relief_m": 150.0,
    "ocean_depth_baseline_m": 2500.0,
    "entrainment_alpha": pp.DEFAULT_ALPHA_ENTRAINMENT,
    "stratification_n": pp.DEFAULT_N_BRUNT_VAISALA,
    "basin": "atlantic",
}


class HydroventGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("hydrovent_field")
        self.root.geometry("1280x820")

        # Rótulos que dependem do idioma escolhido na splash (i18n.get_language()
        # já foi setado antes desta janela ser construída) — montados uma vez
        # aqui em vez de constantes de classe, já que dependem do idioma.
        self.IMAGE_KEYS = [
            ("png_2d_path", t("img_2d")),
            ("png_3d_path", t("img_3d")),
            ("png_truescale_path", t("img_truescale")),
            ("png_artistic_path", t("img_artistic")),
            ("png_hotspots_path", t("img_hotspots")),
            ("png_acoustic_path", t("img_acoustic")),
            ("png_module_dilution_path", t("img_module_dilution")),
            ("png_module_thermophoresis_path", t("img_module_thermophoresis")),
            ("png_module_mineral_adsorption_path", t("img_module_mineral_adsorption")),
            ("png_module_proton_gradient_path", t("img_module_proton_gradient")),
        ]
        self.ACOUSTIC_MODE_LABELS = {
            "off": t("acoustic_mode_off"),
            "streaming": t("acoustic_mode_streaming"),
            "particle_trap": t("acoustic_mode_particle_trap"),
            "both": t("acoustic_mode_both"),
        }
        self.RUN_TABLE_COLUMNS = [
            ("run", t("col_run"), 130),
            ("seed", t("col_seed"), 90),
            ("n_vents", t("col_n_vents"), 90),
            ("top_enrich", t("col_top_enrich"), 150),
            ("mean_enrich", t("col_mean_enrich"), 110),
            ("n_up", t("col_n_up"), 90),
            ("n_down", t("col_n_down"), 90),
            ("top_type", t("col_top_type"), 140),
        ]
        # Rótulos de classe de molécula (fumarola_field.py/prebiotic.py já
        # mantêm as duas versões, PT/EN, reaproveitadas aqui — ver
        # MOLECULE_CLASS_LABELS_EN, originalmente só para as figuras).
        self.molecule_labels = (ff.MOLECULE_CLASS_LABELS_EN if i18n.get_language() == "en"
                                 else ff.MOLECULE_CLASS_LABELS)

        self.msg_queue: queue.Queue = queue.Queue()
        self.run_thread: threading.Thread | None = None
        self.stats_thread: threading.Thread | None = None
        self.last_summaries: list[dict] = []
        self.last_ensemble_stats: dict | None = None
        self.last_pooled_hotspots: list[dict] | None = None
        self.last_experiment_dir: str | None = None
        self.last_vardecomp_result: dict | None = None
        self.current_run_idx = 0
        self._sort_state: dict[str, bool] = {}

        self._build_vars()
        self._build_menu()
        self._build_layout()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Variáveis / valores padrão (espelham os defaults do CLI)
    # ------------------------------------------------------------------
    def _build_vars(self):
        self.v_experiment_mode = tk.StringVar(value="exploratory")
        self.v_run_mode = tk.StringVar(value="single")
        self.v_n_runs = tk.IntVar(value=1)
        self.v_ensemble_images = tk.BooleanVar(value=False)
        self.v_sensitivity_sweep = tk.BooleanVar(value=False)
        self.v_parallel = tk.BooleanVar(value=False)
        # --variance-decomposition (ver fumarola_field.run_nested_variance_experiment):
        # modo de execução alternativo a single/ensemble, mesmos defaults da CLI.
        self.v_outer_samples = tk.IntVar(value=20)
        self.v_inner_replicates = tk.IntVar(value=10)
        self.v_workers = tk.StringVar(value="")

        self.v_seed = tk.StringVar(value="")
        self.v_size = tk.IntVar(value=257)
        self.v_roughness = tk.DoubleVar(value=0.55)
        self.v_n_clusters = tk.IntVar(value=6)
        self.v_vents_min = tk.IntVar(value=2)
        self.v_vents_max = tk.IntVar(value=9)
        self.v_spreading_rate = tk.DoubleVar(value=60.0)
        self.v_local_relief_m = tk.DoubleVar(value=150.0)
        self.v_ocean_depth_baseline_m = tk.DoubleVar(value=2500.0)
        self.v_entrainment_alpha = tk.DoubleVar(value=pp.DEFAULT_ALPHA_ENTRAINMENT)
        self.v_stratification_n = tk.DoubleVar(value=pp.DEFAULT_N_BRUNT_VAISALA)
        self.v_basin = tk.StringVar(value="atlantic")
        self.v_export_plume_profiles = tk.BooleanVar(value=False)

        self.v_gen_3d = tk.BooleanVar(value=True)
        self.v_true_scale = tk.BooleanVar(value=False)
        self.v_artistic_render = tk.BooleanVar(value=False)
        self.v_z_exag = tk.DoubleVar(value=25.0)
        self.v_chimney_scale = tk.DoubleVar(value=1.0)
        self.v_view_elev = tk.DoubleVar(value=55.0)
        self.v_view_azim = tk.DoubleVar(value=-50.0)
        self.v_domain_size_m = tk.DoubleVar(value=1200.0)

        self.v_molecule_class = tk.StringVar(value=self.molecule_labels[ff.DEFAULT_MOLECULE_CLASS])
        self.v_pore_aspect_ratio = tk.StringVar(value="")

        self.v_dilution = tk.BooleanVar(value=True)
        self.v_thermophoresis = tk.BooleanVar(value=True)
        self.v_mineral_adsorption = tk.BooleanVar(value=True)
        self.v_proton_gradient = tk.BooleanVar(value=True)
        self.v_acoustic_mode_label = tk.StringVar(value=self.ACOUSTIC_MODE_LABELS["off"])
        self.v_acoustic_coherent_bound = tk.BooleanVar(value=False)
        self.v_acoustic_particle_radius_um = tk.StringVar(value="")
        self.v_acoustic_particle_density = tk.StringVar(value="")

        self.v_outputs_dir = tk.StringVar(value=ff.DEFAULT_OUTPUTS_DIR)
        self.v_basename = tk.StringVar(value="fumarola_field")

        self.v_status = tk.StringVar(value=t("status_ready"))
        self.v_image_choice = tk.StringVar(value=self.IMAGE_KEYS[0][1])

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label=t("menu_open_experiment"), command=self._on_open_experiment_clicked)
        file_menu.add_command(label=t("menu_resume_experiment"), command=self._on_resume_experiment_clicked)
        file_menu.add_separator()
        file_menu.add_command(label=t("menu_exit"), command=self.root.quit)
        menubar.add_cascade(label=t("menu_file"), menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label=t("menu_about"), command=lambda: self.notebook.select(self.about_tab))
        menubar.add_cascade(label=t("menu_help"), menu=help_menu)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=10)
        right = ttk.Frame(paned, padding=10)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_form(left)
        self._build_output_panel(right)

    def _build_form(self, parent):
        canvas = tk.Canvas(parent, width=380, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Modo do experimento ---
        box = self._labeled_frame(form, t("frame_experiment_mode"))
        ttk.Label(box, foreground="#777", wraplength=340, justify="left",
                  text=t("experiment_mode_desc")).pack(anchor="w", pady=(0, 6))
        ttk.Radiobutton(box, text=t("radio_exploratory"), variable=self.v_experiment_mode,
                        value="exploratory", command=self._on_experiment_mode_change).pack(anchor="w", pady=1)
        ttk.Radiobutton(box, text=t("radio_experiment_run"),
                        variable=self.v_experiment_mode, value="experiment_run",
                        command=self._on_experiment_mode_change).pack(anchor="w", pady=1)

        # --- Execução ---
        box = self._labeled_frame(form, t("frame_execution"))
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(row, text=t("radio_single_run"), variable=self.v_run_mode, value="single",
                        command=self._on_run_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text=t("radio_ensemble"), variable=self.v_run_mode, value="ensemble",
                        command=self._on_run_mode_change).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(row, text=t("radio_vardecomp"), variable=self.v_run_mode, value="vardecomp",
                        command=self._on_run_mode_change).pack(side=tk.LEFT, padx=(10, 0))

        self.entry_n_runs = self._labeled_entry(box, t("label_n_runs"), self.v_n_runs)
        self.entry_n_runs.configure(state="disabled")
        self.chk_ensemble_images = ttk.Checkbutton(box, text=t("chk_ensemble_images"),
                                                     variable=self.v_ensemble_images)
        self.chk_ensemble_images.pack(anchor="w", pady=2)
        self.chk_ensemble_images.configure(state="disabled")

        self.chk_sensitivity_sweep = ttk.Checkbutton(
            box, text=t("chk_sensitivity_sweep"),
            variable=self.v_sensitivity_sweep)
        self.chk_sensitivity_sweep.pack(anchor="w", pady=2)
        self.chk_sensitivity_sweep.configure(state="disabled")

        self.entry_outer_samples = self._labeled_entry(box, t("label_outer_samples"), self.v_outer_samples)
        self.entry_outer_samples.configure(state="disabled")
        self.entry_inner_replicates = self._labeled_entry(box, t("label_inner_replicates"), self.v_inner_replicates)
        self.entry_inner_replicates.configure(state="disabled")

        self.chk_parallel = ttk.Checkbutton(
            box, text=t("chk_parallel", n=ff._default_parallel_workers()),
            variable=self.v_parallel)
        self.chk_parallel.pack(anchor="w", pady=2)
        self.chk_parallel.configure(state="disabled")
        self.entry_workers = self._labeled_entry(box, t("label_workers"), self.v_workers)
        self.entry_workers.configure(state="disabled")

        self._labeled_entry(box, t("label_seed"), self.v_seed)

        # --- Terreno ---
        box = self._labeled_frame(form, t("frame_terrain"))
        self.entry_size = self._labeled_entry(box, t("label_grid_size"), self.v_size)
        self.entry_roughness = self._labeled_entry(box, t("label_roughness"), self.v_roughness)
        self.entry_n_clusters = self._labeled_entry(box, t("label_n_clusters"), self.v_n_clusters)
        self.entry_vents_min = self._labeled_entry(box, t("label_vents_min"), self.v_vents_min)
        self.entry_vents_max = self._labeled_entry(box, t("label_vents_max"), self.v_vents_max)
        self.entry_spreading_rate = self._labeled_entry(box, t("label_spreading_rate"), self.v_spreading_rate)
        self.entry_local_relief = self._labeled_entry(box, t("label_local_relief"), self.v_local_relief_m)
        self.entry_ocean_depth = self._labeled_entry(box, t("label_ocean_depth"), self.v_ocean_depth_baseline_m)

        # --- Física da pluma (Morton-Taylor-Turner + cinética reativa) ---
        box = self._labeled_frame(form, t("frame_plume_physics"))
        self.entry_entrainment_alpha = self._labeled_entry(box, t("label_entrainment_alpha"), self.v_entrainment_alpha)
        self.entry_stratification_n = self._labeled_entry(box, t("label_stratification_n"), self.v_stratification_n)
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text=t("label_basin"), width=26).pack(side=tk.LEFT)
        self.basin_menu = ttk.Combobox(row, textvariable=self.v_basin, state="readonly",
                                        values=list(rk.BASIN_PARAMS.keys()))
        self.basin_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chk_export_plume_profiles = ttk.Checkbutton(
            box, text=t("chk_export_plume_profiles"), variable=self.v_export_plume_profiles)
        self.chk_export_plume_profiles.pack(anchor="w", pady=2)

        # --- Visualização 3D ---
        box = self._labeled_frame(form, t("frame_3d"))
        ttk.Checkbutton(box, text=t("chk_gen_3d"), variable=self.v_gen_3d).pack(anchor="w", pady=2)
        ttk.Checkbutton(box, text=t("chk_true_scale"),
                       variable=self.v_true_scale).pack(anchor="w", pady=2)
        ttk.Checkbutton(box, text=t("chk_artistic_render"),
                       variable=self.v_artistic_render).pack(anchor="w", pady=2)
        self._labeled_entry(box, t("label_z_exag"), self.v_z_exag)
        self._labeled_entry(box, t("label_chimney_scale"), self.v_chimney_scale)
        self._labeled_entry(box, t("label_view_elev"), self.v_view_elev)
        self._labeled_entry(box, t("label_view_azim"), self.v_view_azim)
        self._labeled_entry(box, t("label_domain_size"), self.v_domain_size_m)

        # --- Módulos prebióticos ---
        box = self._labeled_frame(form, t("frame_prebiotic_modules"))
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text=t("label_molecule_class"), width=26).pack(side=tk.LEFT)
        self.molecule_menu = ttk.Combobox(row, textvariable=self.v_molecule_class, state="readonly",
                                           values=list(self.molecule_labels.values()))
        self.molecule_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_pore_aspect_ratio = self._labeled_entry(
            box, t("label_pore_aspect_ratio"), self.v_pore_aspect_ratio)
        self.chk_dilution = ttk.Checkbutton(box, text=t("chk_dilution"), variable=self.v_dilution)
        self.chk_dilution.pack(anchor="w", pady=2)
        self.chk_thermophoresis = ttk.Checkbutton(
            box, text=t("chk_thermophoresis"), variable=self.v_thermophoresis)
        self.chk_thermophoresis.pack(anchor="w", pady=2)
        self.chk_mineral_adsorption = ttk.Checkbutton(
            box, text=t("chk_mineral_adsorption"), variable=self.v_mineral_adsorption)
        self.chk_mineral_adsorption.pack(anchor="w", pady=2)
        self.chk_proton_gradient = ttk.Checkbutton(
            box, text=t("chk_proton_gradient"), variable=self.v_proton_gradient)
        self.chk_proton_gradient.pack(anchor="w", pady=2)

        # --- Modelo acústico (hipótese exploratória) ---
        box = self._labeled_frame(form, t("frame_acoustic"))
        ttk.Label(box, foreground="#777", wraplength=340, justify="left",
                  text=t("acoustic_desc")).pack(anchor="w", pady=(0, 6))
        self.acoustic_radiobuttons = []
        for value in ["off", "streaming", "particle_trap", "both"]:
            rb = ttk.Radiobutton(box, text=self.ACOUSTIC_MODE_LABELS[value],
                                  variable=self.v_acoustic_mode_label,
                                  value=self.ACOUSTIC_MODE_LABELS[value],
                                  command=self._on_acoustic_mode_change)
            rb.pack(anchor="w", pady=1)
            self.acoustic_radiobuttons.append(rb)

        self.entry_acoustic_particle_radius = self._labeled_entry(
            box, t("label_acoustic_particle_radius"), self.v_acoustic_particle_radius_um)
        self.entry_acoustic_particle_density = self._labeled_entry(
            box, t("label_acoustic_particle_density"), self.v_acoustic_particle_density)
        self.entry_acoustic_particle_radius.configure(state="disabled")
        self.entry_acoustic_particle_density.configure(state="disabled")

        ttk.Separator(box).pack(fill=tk.X, pady=6)
        ttk.Label(box, foreground="#777", wraplength=340, justify="left",
                  text=t("acoustic_coherence_desc1")).pack(anchor="w")
        self.chk_coherent_bound = ttk.Checkbutton(
            box, text=t("chk_coherent_bound"),
            variable=self.v_acoustic_coherent_bound)
        self.chk_coherent_bound.pack(anchor="w", pady=(2, 0))
        ttk.Label(box, foreground="#777", wraplength=340, justify="left",
                  text=t("acoustic_coherence_desc2"),
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 2))

        # --- Saída ---
        box = self._labeled_frame(form, t("frame_output"))
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.v_outputs_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3, command=self._browse_outputs_dir).pack(side=tk.LEFT, padx=(4, 0))
        self._labeled_entry(box, t("label_basename"), self.v_basename)

        # --- Executar ---
        self.btn_run = ttk.Button(form, text=t("btn_run"), command=self._on_run_clicked)
        self.btn_run.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(form, textvariable=self.v_status, foreground="#555").pack(anchor="w", pady=(0, 10))

    def _build_output_panel(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text=t("btn_open_outputs"), command=self._open_outputs_folder).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(8, 8))

        images_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(images_tab, text=t("tab_images"))
        self.notebook.add(self.stats_tab, text=t("tab_stats"))
        self.notebook.add(self.about_tab, text=t("tab_about"))

        self._build_images_tab(images_tab)
        self._build_stats_tab_placeholder()
        self._build_about_tab(self.about_tab)

        log_frame = ttk.Frame(parent)
        log_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(log_frame, text=t("label_log")).pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=9, bg="#0d0d0d", fg="#d6d6d6", font=("Consolas", 9))
        self.log_text.pack(fill=tk.X)

    def _build_images_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(bar, text=t("label_image")).pack(side=tk.LEFT)
        self.image_menu = ttk.Combobox(bar, textvariable=self.v_image_choice, state="readonly",
                                        values=[label for _, label in self.IMAGE_KEYS], width=28)
        self.image_menu.pack(side=tk.LEFT, padx=6)
        self.image_menu.bind("<<ComboboxSelected>>", lambda e: self._show_current_image())

        ttk.Button(bar, text=t("btn_prev_run"), command=lambda: self._change_run(-1)).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(bar, text=t("btn_next_run"), command=lambda: self._change_run(1)).pack(side=tk.LEFT, padx=2)
        self.lbl_run_indicator = ttk.Label(bar, text="—")
        self.lbl_run_indicator.pack(side=tk.LEFT, padx=8)

        self.image_canvas = ZoomPanCanvas(parent)
        self.image_canvas.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.image_canvas.clear(t("no_image_placeholder"))

    def _build_stats_tab_placeholder(self):
        for child in self.stats_tab.winfo_children():
            child.destroy()
        ttk.Label(self.stats_tab, text=t("stats_placeholder"),
                  foreground="#777", padding=20).pack(anchor="center", expand=True)

    def _build_about_tab(self, parent):
        frame = ttk.Frame(parent, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="hydrovent_field", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text=t("splash_version", v=ff.__version__), foreground="#777").pack(anchor="w", pady=(0, 16))

        ttk.Label(frame, text=t("splash_desc"), wraplength=560, justify="left").pack(anchor="w", pady=(0, 16))

        ttk.Separator(frame).pack(fill=tk.X, pady=(0, 16))

        ttk.Label(frame, text=t("splash_credits"), wraplength=560, justify="left").pack(anchor="w", pady=(0, 12))

        ttk.Label(frame, text=t("about_site")).pack(anchor="w", pady=1)
        ttk.Label(frame, text=t("about_instagram")).pack(anchor="w", pady=1)

    # ------------------------------------------------------------------
    # Helpers de layout
    # ------------------------------------------------------------------
    @staticmethod
    def _labeled_frame(parent, title):
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill=tk.X, padx=4, pady=6)
        return box

    @staticmethod
    def _labeled_entry(parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=26).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return entry

    def _on_run_mode_change(self):
        mode = self.v_run_mode.get()
        exp_locked = self.v_experiment_mode.get() == "experiment_run"
        if mode == "single":
            self.v_n_runs.set(1)
            if not exp_locked:
                self.v_sensitivity_sweep.set(False)
            self.v_parallel.set(False)
        elif mode == "vardecomp" and not exp_locked:
            self.v_sensitivity_sweep.set(False)  # não usado pelo desenho aninhado
        self._refresh_lock_states()

    def _on_experiment_mode_change(self):
        locked = self.v_experiment_mode.get() == "experiment_run"
        if locked:
            # Trava TODO parâmetro do protocolo científico real usado nos
            # ensembles submetidos — ver _EXPERIMENT_RUN_PROTOCOL — não só
            # molécula/modo acústico como antes. Seed, nº de runs, imagens,
            # paralelismo e visualização 3D continuam livres (ver comentário
            # da constante).
            # nº de runs continua LIVRE (não travado) — só troca pra modo
            # "ensemble" automaticamente (nossos ensembles reais nunca
            # foram "single run") e sugere 1000 como ponto de partida
            # (o maior dos dois ensembles do paper), editável livremente.
            if self.v_run_mode.get() != "ensemble":
                self.v_run_mode.set("ensemble")
                self.v_n_runs.set(_EXPERIMENT_RUN_SUGGESTED_N_RUNS)
            self.v_molecule_class.set(self.molecule_labels["nucleotideos"])
            self.v_acoustic_mode_label.set(self.ACOUSTIC_MODE_LABELS["both"])
            p = _EXPERIMENT_RUN_PROTOCOL
            self.v_size.set(p["size"])
            self.v_roughness.set(p["roughness"])
            self.v_n_clusters.set(p["n_clusters"])
            self.v_vents_min.set(p["vents_min"])
            self.v_vents_max.set(p["vents_max"])
            self.v_spreading_rate.set(p["spreading_rate"])
            self.v_local_relief_m.set(p["local_relief_m"])
            self.v_ocean_depth_baseline_m.set(p["ocean_depth_baseline_m"])
            self.v_entrainment_alpha.set(p["entrainment_alpha"])
            self.v_stratification_n.set(p["stratification_n"])
            self.v_basin.set(p["basin"])
            self.v_export_plume_profiles.set(True)
            self.v_sensitivity_sweep.set(True)
            self.v_dilution.set(True)
            self.v_thermophoresis.set(True)
            self.v_mineral_adsorption.set(True)
            self.v_proton_gradient.set(True)
            self.v_pore_aspect_ratio.set("")
            self.v_acoustic_coherent_bound.set(False)
            self.v_acoustic_particle_radius_um.set("")
            self.v_acoustic_particle_density.set("")
        self.molecule_menu.configure(state="disabled" if locked else "readonly")
        for rb in self.acoustic_radiobuttons:
            rb.configure(state="disabled" if locked else "normal")
        self._refresh_lock_states()

    def _on_acoustic_mode_change(self):
        label_to_acoustic_mode = {v: k for k, v in self.ACOUSTIC_MODE_LABELS.items()}
        uses_particles = label_to_acoustic_mode[self.v_acoustic_mode_label.get()] in ("particle_trap", "both")
        exp_locked = self.v_experiment_mode.get() == "experiment_run"
        state = "normal" if (uses_particles and not exp_locked) else "disabled"
        self.entry_acoustic_particle_radius.configure(state=state)
        self.entry_acoustic_particle_density.configure(state=state)

    def _refresh_lock_states(self):
        """Recalcula o estado (normal/disabled) de todo widget cujo
        habilitar/desabilitar depende de mais de um controle (modo de
        execução single/ensemble/vardecomp E modo Experiment Run/
        exploratório) — chamado pelos dois handlers em vez de cada um
        mexer nos widgets compartilhados isoladamente, pra eles nunca se
        sobrescreverem um ao outro."""
        mode = self.v_run_mode.get()
        ensemble = mode == "ensemble"
        vardecomp = mode == "vardecomp"
        exp_locked = self.v_experiment_mode.get() == "experiment_run"

        self.entry_n_runs.configure(state="normal" if ensemble else "disabled")
        self.chk_ensemble_images.configure(state="normal" if (ensemble or vardecomp) else "disabled")
        self.chk_sensitivity_sweep.configure(state="normal" if (ensemble and not exp_locked) else "disabled")
        self.chk_parallel.configure(state="normal" if (ensemble or vardecomp) else "disabled")
        self.entry_workers.configure(state="normal" if (ensemble or vardecomp) else "disabled")
        self.entry_outer_samples.configure(state="normal" if vardecomp else "disabled")
        self.entry_inner_replicates.configure(state="normal" if vardecomp else "disabled")

        protocol_state = "disabled" if exp_locked else "normal"
        for entry in (self.entry_size, self.entry_roughness, self.entry_n_clusters,
                      self.entry_vents_min, self.entry_vents_max, self.entry_spreading_rate,
                      self.entry_local_relief, self.entry_ocean_depth,
                      self.entry_entrainment_alpha, self.entry_stratification_n,
                      self.entry_pore_aspect_ratio):
            entry.configure(state=protocol_state)
        self.basin_menu.configure(state="disabled" if exp_locked else "readonly")
        for chk in (self.chk_export_plume_profiles, self.chk_dilution, self.chk_thermophoresis,
                    self.chk_mineral_adsorption, self.chk_proton_gradient, self.chk_coherent_bound):
            chk.configure(state=protocol_state)
        self._on_acoustic_mode_change()

    def _browse_outputs_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.v_outputs_dir.get() or ".")
        if chosen:
            self.v_outputs_dir.set(chosen)

    def _open_outputs_folder(self):
        path = self.v_outputs_dir.get() or ff.DEFAULT_OUTPUTS_DIR
        os.makedirs(path, exist_ok=True)
        os.startfile(path)  # Windows

    # ------------------------------------------------------------------
    # Abrir experimento existente / continuar experimento abortado
    # ------------------------------------------------------------------
    def _on_open_experiment_clicked(self):
        if self.run_thread and self.run_thread.is_alive():
            messagebox.showinfo(t("msg_running_title"), t("msg_running_body"))
            return
        start_dir = self.v_outputs_dir.get() or ff.DEFAULT_OUTPUTS_DIR
        chosen = filedialog.askdirectory(
            initialdir=start_dir,
            title=t("dialog_select_experiment_title"))
        if not chosen:
            return

        meta_path = os.path.join(chosen, "experiment_metadata.json")
        if not os.path.exists(meta_path):
            messagebox.showerror(
                t("msg_invalid_folder_title"),
                t("msg_invalid_folder_body"))
            return

        summaries = [s for rd in ff.find_run_dirs(chosen) if (s := ff.load_run_summary(rd)) is not None]
        if not summaries:
            messagebox.showwarning(t("msg_empty_experiment_title"), t("msg_empty_experiment_body"))
            return

        self.log_text.delete("1.0", tk.END)
        self._on_run_finished({"experiment_dir": chosen, "summaries": summaries})

    def _on_resume_experiment_clicked(self):
        if self.run_thread and self.run_thread.is_alive():
            messagebox.showinfo(t("msg_running_title"), t("msg_running_body"))
            return

        outputs_dir = self.v_outputs_dir.get() or ff.DEFAULT_OUTPUTS_DIR
        aborted = ff.find_aborted_experiments(outputs_dir)
        if not aborted:
            messagebox.showinfo(
                t("msg_no_aborted_title"),
                t("msg_no_aborted_body", path=os.path.abspath(outputs_dir)))
            return

        chosen = self._ask_pick_aborted_experiment(aborted)
        if chosen is None:
            return

        if not chosen["resumable"]:
            messagebox.showerror(
                t("msg_cannot_resume_title"),
                t("msg_cannot_resume_body"))
            return

        workers_raw = self.v_workers.get().strip()
        try:
            n_workers = int(workers_raw) if workers_raw else None
        except ValueError:
            messagebox.showerror(t("msg_invalid_value_title"), t("msg_invalid_value_body"))
            return

        self.log_text.delete("1.0", tk.END)
        self.btn_run.configure(state="disabled")
        self.v_status.set(
            t("status_resuming", n_completed=chosen["n_completed"], n_runs=chosen["n_runs"]))
        self.run_thread = threading.Thread(
            target=self._resume_worker, args=(chosen["experiment_dir"], n_workers), daemon=True)
        self.run_thread.start()

    def _resume_worker(self, experiment_dir: str, n_workers=None):
        import sys
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.msg_queue)
        try:
            result = ff.resume_experiment(experiment_dir, parallel=self.v_parallel.get(), n_workers=n_workers)
            self.msg_queue.put(("done", result))
        except Exception as exc:  # noqa: BLE001 - reportar qualquer erro na GUI
            self.msg_queue.put(("error", str(exc)))
        finally:
            sys.stdout = old_stdout

    def _ask_pick_aborted_experiment(self, aborted: list[dict]) -> dict | None:
        """Diálogo modal simples para escolher, entre os experimentos incompletos
        encontrados, qual retomar."""
        dialog = tk.Toplevel(self.root)
        dialog.title(t("dialog_resume_title"))
        dialog.geometry("640x280")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=t("dialog_resume_label"),
                  padding=(10, 10, 10, 0)).pack(anchor="w")

        columns = ("dir", "progress", "seed", "resumable")
        tree = ttk.Treeview(dialog, columns=columns, show="headings", height=8)
        tree.heading("dir", text=t("col_dir"))
        tree.column("dir", width=280)
        tree.heading("progress", text=t("col_progress"))
        tree.column("progress", width=110, anchor="center")
        tree.heading("seed", text=t("col_seed_base"))
        tree.column("seed", width=90, anchor="center")
        tree.heading("resumable", text=t("col_resumable"))
        tree.column("resumable", width=140, anchor="center")
        for item in aborted:
            tree.insert("", tk.END, values=(
                os.path.basename(item["experiment_dir"]),
                f"{item['n_completed']}/{item['n_runs']}",
                item["base_seed"],
                t("val_yes") if item["resumable"] else t("val_no_old_format"),
            ))
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        if aborted:
            tree.selection_set(tree.get_children()[0])

        result: dict = {"value": None}

        def on_ok():
            sel = tree.selection()
            if sel:
                result["value"] = aborted[tree.index(sel[0])]
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btns = ttk.Frame(dialog)
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btns, text=t("btn_cancel"), command=on_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text=t("btn_resume_this"), command=on_ok).pack(side=tk.RIGHT, padx=(0, 8))
        tree.bind("<Double-1>", lambda e: on_ok())

        dialog.wait_window()
        return result["value"]

    # ------------------------------------------------------------------
    # Construção dos args + execução
    # ------------------------------------------------------------------
    def _build_args(self) -> argparse.Namespace | None:
        try:
            seed_raw = self.v_seed.get().strip()
            label_to_class = {v: k for k, v in self.molecule_labels.items()}
            label_to_acoustic_mode = {v: k for k, v in self.ACOUSTIC_MODE_LABELS.items()}
            experiment_mode = self.v_experiment_mode.get()
            exp_locked = experiment_mode == "experiment_run"
            # Reforço defensivo: em experiment_run, TODO parâmetro do
            # protocolo científico real (ver _EXPERIMENT_RUN_PROTOCOL) é
            # SEMPRE o valor travado, independente do estado dos widgets
            # (que já ficam desabilitados/forçados por
            # _on_experiment_mode_change/_refresh_lock_states) — não confia
            # só na camada de UI, porque um widget desabilitado ainda pode
            # carregar um valor obsoleto na sua Variable caso o código mude
            # no futuro.
            p = _EXPERIMENT_RUN_PROTOCOL
            if exp_locked:
                molecule_class = "nucleotideos"
                acoustic_mode = "both"
            else:
                molecule_class = label_to_class[self.v_molecule_class.get()]
                acoustic_mode = label_to_acoustic_mode[self.v_acoustic_mode_label.get()]
            return argparse.Namespace(
                seed=int(seed_raw) if seed_raw else None,
                size=p["size"] if exp_locked else self.v_size.get(),
                roughness=p["roughness"] if exp_locked else self.v_roughness.get(),
                n_clusters=p["n_clusters"] if exp_locked else self.v_n_clusters.get(),
                vents_min=p["vents_min"] if exp_locked else self.v_vents_min.get(),
                vents_max=p["vents_max"] if exp_locked else self.v_vents_max.get(),
                spreading_rate=p["spreading_rate"] if exp_locked else self.v_spreading_rate.get(),
                local_relief_m=p["local_relief_m"] if exp_locked else self.v_local_relief_m.get(),
                ocean_depth_baseline_m=(p["ocean_depth_baseline_m"] if exp_locked
                                         else self.v_ocean_depth_baseline_m.get()),
                entrainment_alpha=p["entrainment_alpha"] if exp_locked else self.v_entrainment_alpha.get(),
                stratification_n=p["stratification_n"] if exp_locked else self.v_stratification_n.get(),
                basin=p["basin"] if exp_locked else self.v_basin.get(),
                export_plume_profiles=True if exp_locked else self.v_export_plume_profiles.get(),
                outputs_dir=self.v_outputs_dir.get() or ff.DEFAULT_OUTPUTS_DIR,
                basename=self.v_basename.get() or "fumarola_field",
                no_3d=not self.v_gen_3d.get(),
                z_exag=self.v_z_exag.get(),
                view_elev=self.v_view_elev.get(),
                view_azim=self.v_view_azim.get(),
                chimney_scale=self.v_chimney_scale.get(),
                true_scale=self.v_true_scale.get(),
                artistic_render=self.v_artistic_render.get(),
                domain_size_m=self.v_domain_size_m.get(),
                runs=None,
                ensemble_images=self.v_ensemble_images.get(),
                sensitivity_sweep=True if exp_locked else self.v_sensitivity_sweep.get(),
                no_dilution=False if exp_locked else not self.v_dilution.get(),
                no_thermophoresis=False if exp_locked else not self.v_thermophoresis.get(),
                no_mineral_adsorption=False if exp_locked else not self.v_mineral_adsorption.get(),
                no_proton_gradient=False if exp_locked else not self.v_proton_gradient.get(),
                molecule_class=molecule_class,
                pore_aspect_ratio=(None if exp_locked else
                                    (float(self.v_pore_aspect_ratio.get())
                                     if self.v_pore_aspect_ratio.get().strip() else None)),
                acoustic_mode=acoustic_mode,
                acoustic_cross_vent_coherence=(
                    "incoherent" if exp_locked else
                    ("coherent" if self.v_acoustic_coherent_bound.get() else "incoherent")),
                experiment_mode=experiment_mode,
                # None => usa a população de duas classes citadas (ver acoustics.py
                # PARTICLE_CLASSES) em vez de um único tamanho customizado
                acoustic_particle_radius_um=(None if exp_locked else
                                              (float(self.v_acoustic_particle_radius_um.get())
                                               if self.v_acoustic_particle_radius_um.get().strip() else None)),
                acoustic_particle_density=(None if exp_locked else
                                            (float(self.v_acoustic_particle_density.get())
                                             if self.v_acoustic_particle_density.get().strip() else None)),
            )
        except (tk.TclError, ValueError):
            messagebox.showerror(t("msg_invalid_param_title"), t("msg_invalid_param_body"))
            return None

    def _on_run_clicked(self):
        if self.run_thread and self.run_thread.is_alive():
            return

        args = self._build_args()
        if args is None:
            return

        workers_raw = self.v_workers.get().strip()
        try:
            n_workers = int(workers_raw) if workers_raw else None
        except ValueError:
            messagebox.showerror(t("msg_invalid_value_title"), t("msg_invalid_value_body"))
            return

        mode = self.v_run_mode.get()
        if mode == "ensemble":
            n_runs = self.v_n_runs.get()
            if not (1 <= n_runs <= 10000):
                messagebox.showerror(t("msg_invalid_value_title"), t("msg_invalid_value_body"))
                return
            make_images = self.v_ensemble_images.get()
            parallel = self.v_parallel.get()
        elif mode == "vardecomp":
            outer_n = self.v_outer_samples.get()
            inner_n = self.v_inner_replicates.get()
            if outer_n < 2 or inner_n < 2:
                messagebox.showerror(t("msg_invalid_value_title"), t("msg_invalid_vardecomp_body"))
                return
            make_images = self.v_ensemble_images.get()
            parallel = self.v_parallel.get()
        else:
            n_runs = 1
            make_images = True
            parallel = False

        self.log_text.delete("1.0", tk.END)
        self.btn_run.configure(state="disabled")

        if mode == "vardecomp":
            self.v_status.set(t("status_running_vardecomp", outer=outer_n, inner=inner_n))
            self.run_thread = threading.Thread(
                target=self._vardecomp_worker,
                args=(args, outer_n, inner_n, make_images, parallel, n_workers), daemon=True)
        else:
            self.v_status.set(t("status_running", n=n_runs))
            self.run_thread = threading.Thread(
                target=self._run_worker, args=(args, n_runs, make_images, parallel, n_workers), daemon=True)
        self.run_thread.start()

    def _run_worker(self, args, n_runs, make_images, parallel, n_workers=None):
        import sys
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.msg_queue)
        try:
            result = ff.run_experiment(args, n_runs, make_images, parallel=parallel, n_workers=n_workers)
            self.msg_queue.put(("done", result))
        except Exception as exc:  # noqa: BLE001 - reportar qualquer erro na GUI
            self.msg_queue.put(("error", str(exc)))
        finally:
            sys.stdout = old_stdout

    def _vardecomp_worker(self, args, outer_n, inner_n, make_images, parallel, n_workers=None):
        import sys
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.msg_queue)
        try:
            result = ff.run_nested_variance_experiment(
                args, outer_n, inner_n, make_images=make_images, parallel=parallel, n_workers=n_workers)
            self.msg_queue.put(("vardecomp_done", result))
        except Exception as exc:  # noqa: BLE001 - reportar qualquer erro na GUI
            self.msg_queue.put(("error", str(exc)))
        finally:
            sys.stdout = old_stdout

    # ------------------------------------------------------------------
    # Fila de mensagens (thread de execução -> thread da GUI)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_text.insert(tk.END, payload)
                    self.log_text.see(tk.END)
                elif kind == "done":
                    self._on_run_finished(payload)
                elif kind == "vardecomp_done":
                    self._on_vardecomp_finished(payload)
                elif kind == "stats_ready":
                    self._render_stats_tab(*payload)
                elif kind == "error":
                    self.btn_run.configure(state="normal")
                    self.v_status.set(t("status_error"))
                    messagebox.showerror(t("msg_error_title"), payload)
                elif kind == "ensemble_report_ready":
                    self.btn_ensemble_report.configure(state="normal")
                    self.v_ensemble_report_status.set(t("ensemble_report_done", path=payload))
                    try:
                        os.startfile(payload)  # noqa: S606 - abrir o HTML gerado localmente
                    except OSError:
                        pass
                elif kind == "ensemble_report_error":
                    self.btn_ensemble_report.configure(state="normal")
                    self.v_ensemble_report_status.set("")
                    messagebox.showerror(t("msg_error_title"), t("ensemble_report_error", error=payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_run_finished(self, result: dict):
        self.btn_run.configure(state="normal")
        self.last_summaries = result["summaries"]
        self.last_experiment_dir = result["experiment_dir"]
        self.last_vardecomp_result = None  # não é um experimento --variance-decomposition
        self.current_run_idx = 0
        n = len(self.last_summaries)
        self.v_status.set(t("status_done", n=n, dir=os.path.basename(result["experiment_dir"])))
        self._show_current_image()

        if n > 1:
            self._build_stats_tab_loading()
            self.stats_thread = threading.Thread(target=self._stats_worker, args=(self.last_summaries,), daemon=True)
            self.stats_thread.start()
        else:
            self._build_stats_tab_placeholder()

    def _on_vardecomp_finished(self, result: dict):
        """`ff.run_nested_variance_experiment` devolve um dict de formato
        DIFERENTE de `run_experiment` (decomposição/Sobol', não uma lista
        de summaries) — carrega as runs individuais do disco (mesmo
        padrão de "abrir experimento existente") para alimentar o
        visualizador de imagens/aba de estatísticas normalmente; os
        resultados ESPECÍFICOS do desenho aninhado (decomposição/Sobol')
        são lidos de `vardecomp_summary.json` pelo relatório estatístico
        (ensemble_report.py) quando o usuário pedir, e resumidos ao vivo
        aqui embaixo."""
        self.btn_run.configure(state="normal")
        experiment_dir = result["experiment_dir"]
        run_dirs = ff.find_run_dirs(experiment_dir)
        summaries = [s for s in (ff.load_run_summary(rd) for rd in run_dirs) if s is not None]
        self.last_summaries = summaries
        self.last_experiment_dir = experiment_dir
        self.last_vardecomp_result = result
        self.current_run_idx = 0
        self.v_status.set(t("status_done_vardecomp", n=len(summaries), dir=os.path.basename(experiment_dir)))
        self._show_current_image()

        if len(summaries) > 1:
            self._build_stats_tab_loading()
            self.stats_thread = threading.Thread(target=self._stats_worker, args=(summaries,), daemon=True)
            self.stats_thread.start()
        else:
            self._build_stats_tab_placeholder()

    # ------------------------------------------------------------------
    # Visualizador de imagens
    # ------------------------------------------------------------------
    def _change_run(self, delta):
        if not self.last_summaries:
            return
        self.current_run_idx = max(0, min(len(self.last_summaries) - 1, self.current_run_idx + delta))
        self._show_current_image()

    def _show_current_image(self):
        if not self.last_summaries:
            self.lbl_run_indicator.configure(text="—")
            return

        summary = self.last_summaries[self.current_run_idx]
        self.lbl_run_indicator.configure(
            text=t("run_indicator", i=self.current_run_idx + 1, n=len(self.last_summaries),
                   name=os.path.basename(summary["run_dir"]))
        )

        key_map = dict(self.IMAGE_KEYS)
        label_to_key = {v: k for k, v in key_map.items()}
        path = summary.get(label_to_key[self.v_image_choice.get()])

        if not path or not os.path.exists(path):
            self.image_canvas.clear(t("image_unavailable"))
            return

        self.image_canvas.load(path)

    # ------------------------------------------------------------------
    # Análise estatística do ensemble
    # ------------------------------------------------------------------
    def _build_stats_tab_loading(self):
        for child in self.stats_tab.winfo_children():
            child.destroy()
        ttk.Label(self.stats_tab, text=t("stats_loading"),
                  foreground="#777", padding=20).pack(anchor="center", expand=True)

    def _stats_worker(self, summaries: list[dict]):
        pooled = es.load_pooled_hotspots(summaries)
        stats = es.compute_ensemble_stats(summaries, pooled)
        self.msg_queue.put(("stats_ready", (summaries, stats, pooled)))

    def _render_stats_tab(self, summaries: list[dict], stats: dict, pooled: list[dict]):
        self.last_ensemble_stats = stats
        self.last_pooled_hotspots = pooled

        for child in self.stats_tab.winfo_children():
            child.destroy()

        self.stats_tab.columnconfigure(0, weight=1)
        row_idx = 0

        # --- relatório estatístico (HTML, aberto a todos — sem discussão/
        # interpretação, sem imagem representativa de run específica; ver
        # ensemble_report.py. Distinto do relatório do Administrador. ) ---
        report_frame = ttk.Frame(self.stats_tab, padding=(4, 4))
        report_frame.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=(4, 0))
        row_idx += 1
        self.btn_ensemble_report = ttk.Button(
            report_frame, text=t("btn_generate_ensemble_report"),
            command=self._on_generate_ensemble_report_clicked)
        self.btn_ensemble_report.pack(side=tk.LEFT)
        self.v_ensemble_report_status = tk.StringVar(value="")
        ttk.Label(report_frame, textvariable=self.v_ensemble_report_status,
                  foreground="#555").pack(side=tk.LEFT, padx=8)

        # --- decomposição de variância / Sobol' (só quando a run atual veio
        # de --variance-decomposition; ver _on_vardecomp_finished) ---
        if self.last_vardecomp_result is not None:
            vd_frame = ttk.LabelFrame(self.stats_tab, text=t("frame_vardecomp"), padding=8)
            vd_frame.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=4)
            row_idx += 1
            self._render_vardecomp_summary(vd_frame, self.last_vardecomp_result)

        # --- estatísticas descritivas ---
        desc_frame = ttk.LabelFrame(self.stats_tab, text=t("frame_desc_stats"), padding=8)
        desc_frame.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=4)
        row_idx += 1

        def fmt(d):
            return t("stat_fmt", n=d["n"], mean=d["mean"], std=d["std"],
                      min=d["min"], median=d["median"], max=d["max"])

        n_up = stats["n_vents_increased_vs_control"]
        n_down = stats["n_vents_decreased_vs_control"]
        n_eq = stats["n_vents_unchanged_vs_control"]
        n_total = n_up + n_down + n_eq

        rows = [
            (t("stat_top_hotspot"), fmt(stats["top_hotspot_enrichment"])),
            (t("stat_pooled_enrich"), fmt(stats["pooled_enrichment"])),
            (t("stat_up_down_label"),
             t("stat_up_down_value", n_up=n_up, n_down=n_down, n_eq=n_eq, n_total=n_total)),
            (t("stat_max_conc"), fmt(stats["max_concentration"])),
            (t("stat_mean_conc"), fmt(stats["mean_concentration"])),
            (t("stat_n_vents"), fmt(stats["n_vents"])),
        ]
        for i, (label, value) in enumerate(rows):
            ttk.Label(desc_frame, text=label + ":", width=48).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Label(desc_frame, text=value, font=("Consolas", 9)).grid(row=i, column=1, sticky="w", pady=1)

        type_counts = stats["top_hotspot_type_counts"]
        type_txt = " | ".join(f"{k}: {v}" for k, v in type_counts.items())
        ttk.Label(desc_frame, text=t("stat_top_type"), width=48).grid(
            row=len(rows), column=0, sticky="w", pady=1)
        ttk.Label(desc_frame, text=type_txt, font=("Consolas", 9)).grid(
            row=len(rows), column=1, sticky="w", pady=1)

        # --- gráficos ---
        chart_frame = ttk.LabelFrame(self.stats_tab, text=t("frame_charts"), padding=4)
        chart_frame.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=4)
        row_idx += 1
        self._render_ensemble_charts(chart_frame, stats)

        # --- tabela por run ---
        table_frame = ttk.LabelFrame(self.stats_tab, text=t("frame_runs_table"), padding=4)
        table_frame.grid(row=row_idx, column=0, sticky="nsew", padx=4, pady=4)
        self.stats_tab.rowconfigure(row_idx, weight=1)
        self._render_runs_table(table_frame, summaries)

    def _render_vardecomp_summary(self, parent, result: dict):
        """Resumo AO VIVO da decomposição de variância/Sobol' de uma run
        --variance-decomposition recém-concluída — versão compacta do que
        o relatório estatístico (ensemble_report.py, lido do disco) mostra
        em detalhe, pra não depender de clicar em "gerar relatório" só
        para ver os números principais."""
        d = result["decomposition"]
        gs = result.get("global_sensitivity")

        rows = [
            (t("ensemble_report_meta_vardecomp_value", outer=result["outer_n"], inner=result["inner_n"]), ""),
            (t("ensemble_report_vardecomp_row_stochastic"),
             f"{d['stochastic_fraction']:.3f} [{d['stochastic_fraction_ci95'][0]:.3f}, "
             f"{d['stochastic_fraction_ci95'][1]:.3f}]"),
            (t("ensemble_report_vardecomp_row_parametric"),
             f"{d['parametric_fraction']:.3f} [{d['parametric_fraction_ci95'][0]:.3f}, "
             f"{d['parametric_fraction_ci95'][1]:.3f}]"),
        ]
        for i, (label, value) in enumerate(rows):
            ttk.Label(parent, text=label + (":" if value else ""), width=48).grid(
                row=i, column=0, sticky="w", pady=1)
            ttk.Label(parent, text=value, font=("Consolas", 9)).grid(row=i, column=1, sticky="w", pady=1)

        if gs is not None:
            r = len(rows)
            warn = f"  {t('ensemble_report_sobol_warning')}" if gs["loo_cv_r2_warning"] else ""
            ttk.Label(parent, text=t("ensemble_report_sobol_th_param") + " (S1 / ST):",
                      width=48).grid(row=r, column=0, sticky="w", pady=(6, 1))
            ttk.Label(parent, text=f"LOO CV R²={gs['loo_cv_r2']:.3f}{warn}",
                      font=("Consolas", 9)).grid(row=r, column=1, sticky="w", pady=(6, 1))
            for j, name in enumerate(gs["param_names"], start=r + 1):
                s1, st = gs["first_order"][name], gs["total_order"][name]
                ttk.Label(parent, text=name, width=48).grid(row=j, column=0, sticky="w", pady=1)
                ttk.Label(parent, text=f"S1={s1:.3f} | ST={st:.3f}",
                          font=("Consolas", 9)).grid(row=j, column=1, sticky="w", pady=1)

    def _render_ensemble_charts(self, parent, stats: dict):
        # Figura construída em ensemble_report.py (não aqui) — reaproveitada
        # tanto pela GUI (embutida via FigureCanvasTkAgg abaixo) quanto pelo
        # relatório estatístico HTML (embutida como PNG), para as duas nunca
        # divergirem.
        fig = er.build_ensemble_charts_figure(stats)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_generate_ensemble_report_clicked(self):
        if not self.last_summaries or self.last_ensemble_stats is None or self.last_experiment_dir is None:
            return
        self.btn_ensemble_report.configure(state="disabled")
        self.v_ensemble_report_status.set(t("ensemble_report_generating"))
        thread = threading.Thread(
            target=self._ensemble_report_worker,
            args=(self.last_summaries, self.last_pooled_hotspots or [], self.last_experiment_dir),
            daemon=True)
        thread.start()

    def _ensemble_report_worker(self, summaries: list[dict], pooled: list[dict], experiment_dir: str):
        try:
            path = er.generate_ensemble_statistics_report(summaries, pooled, experiment_dir)
            self.msg_queue.put(("ensemble_report_ready", path))
        except Exception as exc:  # noqa: BLE001 - reportar qualquer erro na GUI
            self.msg_queue.put(("ensemble_report_error", str(exc)))

    def _render_runs_table(self, parent, summaries: list[dict]):
        columns = [c[0] for c in self.RUN_TABLE_COLUMNS]
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        for col_id, label, width in self.RUN_TABLE_COLUMNS:
            tree.heading(col_id, text=label, command=lambda c=col_id: self._sort_table(tree, c))
            tree.column(col_id, width=width, anchor="center")

        for s in summaries:
            ps = s["prebiotic_summary"]
            top_enrich = ps["top_hotspot_enrichment_vs_control"]
            mean_enrich = ps["mean_enrichment_vs_control"]
            tree.insert("", tk.END, values=(
                os.path.basename(s["run_dir"]), s["seed"], s["n_vents"],
                f"{top_enrich:.3f}" if top_enrich is not None else "n/a",
                f"{mean_enrich:.3f}" if mean_enrich is not None else "n/a",
                ps["n_vents_increased_vs_control"], ps["n_vents_decreased_vs_control"],
                ps["top_hotspot_vent_type"],
            ))

        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _sort_table(self, tree: ttk.Treeview, col: str):
        items = [(tree.set(k, col), k) for k in tree.get_children("")]
        try:
            items.sort(key=lambda pair: float(pair[0]))
        except ValueError:
            items.sort(key=lambda pair: pair[0])
        reverse = self._sort_state.get(col, False)
        items.sort(reverse=reverse)
        self._sort_state[col] = not reverse
        for index, (_, k) in enumerate(items):
            tree.move(k, "", index)


def main():
    _show_splash()
    root = tk.Tk()
    HydroventGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

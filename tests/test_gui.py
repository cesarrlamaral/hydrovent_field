"""
Testes de integração da GUI (Tkinter) — primeira cobertura de teste
automatizada deste módulo (antes só verificado manualmente via "python -c
'import gui'"/smoke tests ad-hoc, ver histórico do projeto). Foco no
modo de execução "vardecomp" (--variance-decomposition trazido para a
GUI): alternância de widgets, o clique do botão
disparando o worker/thread real, e o resultado populando a aba de
estatísticas — não só chamando as funções de `fumarola_field.py`/
`ensemble_report.py` isoladamente (essas já têm cobertura própria),
mas o objeto REAL `HydroventGUI` e seus handlers.

`root.withdraw()` roda a janela sem exibi-la — funciona no Windows sem
servidor de display (testado). `root.update()`
em loop processa a fila de eventos do Tk (equivalente a deixar o
mainloop rodar) enquanto uma thread de fundo (run_thread/stats_thread)
termina — mesmo padrão usado pela própria aplicação via `root.after`.

Rodar com: pytest tests/test_gui.py -v
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tk = pytest.importorskip("tkinter")

import i18n
import gui


def _wait_until(condition, timeout_s=30.0, root=None):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if root is not None:
            root.update()
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture(scope="module")
def root():
    """UMA raiz Tk para o módulo inteiro — criar/destruir várias
    instâncias de tk.Tk() no mesmo processo mostrou-se instável nesta
    combinação de plataforma/Tcl (achado real:
    rodar os testes deste arquivo em sequência causava desde exceções
    fatais do Windows até falha silenciosa ao criar a próxima Tk(),
    sempre que cada teste criava/destruía sua própria raiz — cada teste
    isolado passava normalmente). Reaproveitar uma raiz e limpar seus
    widgets entre testes evita o padrão de churn que disparava o bug."""
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def app(root, tmp_path):
    i18n.set_language("en")
    for child in root.winfo_children():
        child.destroy()
    instance = gui.HydroventGUI(root)
    instance.v_size.set(33)
    instance.v_n_clusters.set(2)
    instance.v_vents_min.set(2)
    instance.v_vents_max.set(3)
    instance.v_outputs_dir.set(str(tmp_path))
    yield instance


# --------------------------------------------------------------------------
# 1. Alternância de modo de execução — widgets certos habilitados/
#    desabilitados para single / ensemble / vardecomp
# --------------------------------------------------------------------------

def test_run_mode_single_disables_ensemble_and_vardecomp_widgets(app):
    app.v_run_mode.set("single")
    app._on_run_mode_change()
    assert str(app.entry_n_runs["state"]) == "disabled"
    assert str(app.entry_outer_samples["state"]) == "disabled"
    assert str(app.entry_inner_replicates["state"]) == "disabled"
    assert str(app.chk_parallel["state"]) == "disabled"


def test_run_mode_ensemble_enables_only_ensemble_widgets(app):
    app.v_run_mode.set("ensemble")
    app._on_run_mode_change()
    assert str(app.entry_n_runs["state"]) == "normal"
    assert str(app.chk_sensitivity_sweep["state"]) == "normal"
    assert str(app.entry_outer_samples["state"]) == "disabled"
    assert str(app.entry_inner_replicates["state"]) == "disabled"


def test_run_mode_vardecomp_enables_only_vardecomp_widgets(app):
    app.v_run_mode.set("vardecomp")
    app._on_run_mode_change()
    assert str(app.entry_outer_samples["state"]) == "normal"
    assert str(app.entry_inner_replicates["state"]) == "normal"
    assert str(app.chk_parallel["state"]) == "normal"
    assert str(app.entry_n_runs["state"]) == "disabled"
    assert str(app.chk_sensitivity_sweep["state"]) == "disabled"


def test_switching_to_vardecomp_clears_sensitivity_sweep_flag(app):
    app.v_run_mode.set("ensemble")
    app._on_run_mode_change()
    app.v_sensitivity_sweep.set(True)
    app.v_run_mode.set("vardecomp")
    app._on_run_mode_change()
    assert app.v_sensitivity_sweep.get() is False


# --------------------------------------------------------------------------
# 2. Validação de entrada antes de disparar a run
# --------------------------------------------------------------------------

def test_vardecomp_run_rejects_outer_samples_below_two(app, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, body: shown.append((title, body)))
    app.v_run_mode.set("vardecomp")
    app._on_run_mode_change()
    app.v_outer_samples.set(1)
    app.v_inner_replicates.set(5)
    app._on_run_clicked()
    assert len(shown) == 1
    assert app.run_thread is None or not app.run_thread.is_alive()


# --------------------------------------------------------------------------
# 3. Fluxo completo: clique real -> thread de fundo -> resultado na GUI
# --------------------------------------------------------------------------

def test_vardecomp_full_run_populates_result_and_stats(app):
    app.v_run_mode.set("vardecomp")
    app._on_run_mode_change()
    app.v_acoustic_mode_label.set(app.ACOUSTIC_MODE_LABELS["particle_trap"])
    app.v_outer_samples.set(3)
    app.v_inner_replicates.set(3)
    app.v_seed.set("123")

    app._on_run_clicked()
    assert _wait_until(lambda: app.last_vardecomp_result is not None, root=app.root)

    result = app.last_vardecomp_result
    assert result["outer_n"] == 3
    assert result["inner_n"] == 3
    assert "decomposition" in result
    assert "global_sensitivity" in result
    assert len(app.last_summaries) == 9  # 3 x 3

    assert _wait_until(lambda: app.last_ensemble_stats is not None, root=app.root)

    # o painel ao vivo de decomposição/Sobol' deveria estar na aba de
    # estatísticas (LabelFrame com o título esperado).
    titles = []
    for child in app.stats_tab.winfo_children():
        try:
            titles.append(child.cget("text"))
        except tk.TclError:
            pass
    assert any("Variance decomposition" in title for title in titles)


def test_normal_ensemble_run_does_not_set_vardecomp_result(app):
    """Regressão: `last_vardecomp_result` precisa ser resetado numa run
    de ensemble normal (não deixar dado de uma vardecomp anterior
    vazando pro painel ao vivo de uma run que não é aninhada)."""
    app.v_run_mode.set("ensemble")
    app._on_run_mode_change()
    app.v_n_runs.set(3)
    app.v_seed.set("7")

    app._on_run_clicked()
    assert _wait_until(lambda: len(app.last_summaries) == 3, root=app.root)
    assert app.last_vardecomp_result is None

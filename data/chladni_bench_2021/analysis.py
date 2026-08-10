"""Reanalise estatistica do teste de bancada Chladni/DNA (abril de 2021).

Le medias.xlsx (aba "Plan1") e recomputa, do zero, as comparacoes entre
wave+ (no), wave- (antino), controle e branco -- sem confiar nos p-values
ja calculados dentro da planilha original. Ver docs/PHYSICS_MODEL.md
Fase 5 para a interpretacao completa.

Convencao herdada do usuario que desenhou o experimento: wave+ = regiao de
NO; wave- = regiao de ANTINO. Ver Fase 5.1 sobre a ambiguidade desta
convencao frente a literatura classica de Chladni (Faraday, 1831).

Uso:
    python analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats


def load_data(path="medias.xlsx"):
    xl = pd.ExcelFile(path)
    df = xl.parse("Plan1", header=1).iloc[:, 1:5]
    df.columns = ["wave_minus", "wave_plus", "control", "blank"]
    df = df.iloc[0:32].apply(pd.to_numeric, errors="coerce")
    return df


def compare(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    t_eq = stats.ttest_ind(a, b, equal_var=True)
    t_w = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b)
    sh_a, sh_b = stats.shapiro(a), stats.shapiro(b)
    lev = stats.levene(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    print(f"--- {label} ---")
    print(f"mean a={a.mean():.3f} sd={a.std(ddof=1):.3f} n={len(a)} | "
          f"mean b={b.mean():.3f} sd={b.std(ddof=1):.3f} n={len(b)}")
    print(f"Student t (var. iguais): p={t_eq.pvalue:.3e}")
    print(f"Welch t (var. desiguais): p={t_w.pvalue:.3e}")
    print(f"Mann-Whitney U (nao-parametrico): p={mw.pvalue:.3e}")
    print(f"Shapiro-Wilk normalidade: a p={sh_a.pvalue:.4f}, b p={sh_b.pvalue:.4f}")
    print(f"Levene var. homogenea: p={lev.pvalue:.4f}")
    print(f"Cohen d = {d:.3f}")
    print()


def main():
    data = load_data()
    wm, wp = data["wave_minus"].dropna(), data["wave_plus"].dropna()
    ct, bl = data["control"].dropna(), data["blank"].dropna()

    compare(wp, wm, "wave+ (no) vs wave- (antino)")
    compare(wm, ct, "wave- (antino) vs controle")
    compare(wp, ct, "wave+ (no) vs controle")
    compare(ct, bl, "controle vs branco")

    paired = data.dropna(subset=["wave_minus", "wave_plus"])
    tt = stats.ttest_rel(paired["wave_plus"], paired["wave_minus"])
    wsr = stats.wilcoxon(paired["wave_plus"], paired["wave_minus"])
    print(f"--- pareado (mesma corrida = wave+ e wave- do mesmo experimento) ---")
    print(f"n pares = {len(paired)}")
    print(f"t pareado: p={tt.pvalue:.3e}")
    print(f"Wilcoxon signed-rank: p={wsr.pvalue:.3e}")


if __name__ == "__main__":
    main()

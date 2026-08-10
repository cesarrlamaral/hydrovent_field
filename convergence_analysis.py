"""
Convergência de Monte Carlo do próprio ensemble: acompanha como uma
estatística (fração binomial de evento raro, ou média de uma grandeza
contínua) se estabiliza conforme mais runs são acumuladas, e projeta
(extrapola analiticamente, sem rodar mais nada) a largura do IC 95%
esperada num N maior — respondendo formalmente "vale a pena rodar mais
runs?" antes de gastar o tempo de computação.

Motivação concreta deste projeto: o ensemble de 1000 runs
(`outputs/experimento_260807_021219`) achou 7/1000 (0,7%) de eventos
raros cruzando o limiar de relevância térmica de Gor'kov; um ensemble de
100 runs separado (`outputs/experimento_260807_092429`) achou 0/100 —
já sabíamos que isso é ESTATISTICAMENTE CONSISTENTE (Poisson/Wilson),
mas não havia uma resposta formal e visual pra "1000 já é o suficiente,
ou 10000 mudaria a conclusão?" (custo real medido: ~2,7h pra 10000 runs
em paralelo — ver docs/PHYSICS_MODEL.md §7.8).

Módulo puro (sem Tkinter/relatório), reaproveita `ensemble_stats.describe`
(bootstrap já construído no projeto) para a componente contínua — não
reimplementa CI de novo.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 1. IC de Wilson — reimplementado aqui (não importado de report.py, que é
#    gitignored e não existe num clone público do repositório; este
#    módulo é tracked e precisa funcionar sozinho).
# --------------------------------------------------------------------------

_Z95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975), constante — ver teste de validação


def wilson_ci95_from_rate(phat: float, n: float) -> Tuple[float, float]:
    """IC de Wilson (1927, J. Am. Stat. Assoc. 22(158), 209-212) para uma
    proporção binomial, parametrizado por taxa (`phat`) e `n` diretamente
    — `n` não precisa ser inteiro nem `phat*n` precisa ser um nº de
    sucessos real, o que permite reutilizar a MESMA fórmula tanto para
    ICs reais (`phat=k/n`) quanto para PROJEÇÕES hipotéticas ("se a taxa
    observada se mantivesse, qual seria o IC com n_alvo tentativas?").
    """
    if n <= 0:
        return (0.0, 0.0)
    z2 = _Z95 ** 2
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    adj = _Z95 * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n ** 2))
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return (float(max(0.0, lo)), float(min(1.0, hi)))


def wilson_ci95(k: int, n: int) -> Tuple[float, float]:
    """IC de Wilson para `k` sucessos em `n` tentativas reais (inteiros)."""
    if n <= 0:
        return (0.0, 0.0)
    return wilson_ci95_from_rate(k / n, n)


# --------------------------------------------------------------------------
# 2. Traço de convergência — evento raro binário (fração acumulada + IC)
# --------------------------------------------------------------------------

def _log_spaced_sample_points(n_total: int, n_points: int) -> np.ndarray:
    """N's espaçados logaritmicamente entre 1 e n_total (inclusive),
    inteiros únicos e ordenados — cobre a faixa toda sem gastar tempo
    computando em TODO N quando n_total é grande (relevante para o
    traço de média contínua, que reamostra por bootstrap em cada ponto;
    o traço binomial, muito mais barato, pode usar todo N sem custo real,
    mas usa a mesma função por consistência)."""
    n_points = min(n_points, n_total)
    raw = np.unique(np.round(np.logspace(0, np.log10(n_total), n_points)).astype(int))
    raw = raw[(raw >= 1) & (raw <= n_total)]
    if raw[-1] != n_total:
        raw = np.append(raw, n_total)
    return raw


def running_binomial_fraction_trace(successes: np.ndarray, n_points: int = 200) -> dict:
    """`successes`: array 0/1 (ou bool) NA ORDEM REAL das runs (a ordem
    importa para um traço de convergência genuíno — não embaralhar).
    Devolve a fração cumulativa e o IC de Wilson em N's espaçados
    logaritmicamente de 1 até `len(successes)`."""
    successes = np.asarray(successes, dtype=float)
    n_total = successes.size
    if n_total == 0:
        raise ValueError("successes não pode ser vazio")
    cum = np.cumsum(successes)
    ns = _log_spaced_sample_points(n_total, n_points)
    fractions = cum[ns - 1] / ns
    ci = np.array([wilson_ci95(int(cum[n - 1]), int(n)) for n in ns])
    return {
        "n": ns, "fraction": fractions,
        "ci_lo": ci[:, 0], "ci_hi": ci[:, 1],
        "ci_width": ci[:, 1] - ci[:, 0],
        "n_total": n_total, "k_total": int(cum[-1]),
    }


# --------------------------------------------------------------------------
# 3. Traço de convergência — estatística contínua (média + IC bootstrap,
#    reaproveitando ensemble_stats.describe)
# --------------------------------------------------------------------------

def running_mean_trace(values: np.ndarray, n_points: int = 25, n_bootstrap: int = 1000,
                        rng: Optional[np.random.Generator] = None) -> dict:
    """`values`: array NA ORDEM REAL das runs. Em cada N amostrado
    (espaçamento log), calcula a média e o IC 95% por bootstrap (via
    `ensemble_stats.describe`, mesmo mecanismo já usado no resto do
    projeto — ver docs/PHYSICS_MODEL.md §10.2b) sobre os PRIMEIROS N
    valores (não uma reamostragem aleatória do array inteiro — é
    literalmente "o que eu saberia se tivesse parado em N runs")."""
    import ensemble_stats as es

    values = np.asarray(values, dtype=float)
    n_total = values.size
    if n_total == 0:
        raise ValueError("values não pode ser vazio")
    if rng is None:
        rng = np.random.default_rng()
    ns = _log_spaced_sample_points(n_total, n_points)
    ns = ns[ns >= 2]  # bootstrap/CI não fazem sentido com N=1

    means, ci_lo, ci_hi = [], [], []
    for n in ns:
        d = es.describe(values[:n], n_bootstrap=n_bootstrap, rng=rng)
        means.append(d["mean"])
        lo, hi = d["mean_ci95"]
        ci_lo.append(lo)
        ci_hi.append(hi)
    means, ci_lo, ci_hi = np.array(means), np.array(ci_lo), np.array(ci_hi)
    return {
        "n": ns, "mean": means, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "ci_width": ci_hi - ci_lo, "n_total": n_total,
    }


# --------------------------------------------------------------------------
# 4. Projeção analítica — "valeria a pena rodar N_alvo runs?"
# --------------------------------------------------------------------------

def predict_binomial_ci_at_n(k_current: int, n_current: int, n_target: int) -> dict:
    """Assumindo que a taxa observada (`k_current/n_current`) se mantém,
    projeta o IC de Wilson (e sua largura) num `n_target` maior — SEM
    rodar nenhuma simulação nova. Não é uma garantia (a taxa real pode
    diferir da observada, especialmente com `k_current` pequeno), é uma
    resposta a "SE a taxa real for a que já medimos, o que N_target
    runs mudaria".
    """
    phat = k_current / n_current if n_current > 0 else 0.0
    ci_current = wilson_ci95(k_current, n_current)
    ci_target = wilson_ci95_from_rate(phat, n_target)
    return {
        "rate": phat,
        "n_current": n_current, "ci_current": ci_current,
        "ci_width_current": ci_current[1] - ci_current[0],
        "n_target": n_target, "ci_target": ci_target,
        "ci_width_target": ci_target[1] - ci_target[0],
        "width_ratio": (ci_target[1] - ci_target[0]) / (ci_current[1] - ci_current[0])
        if ci_current[1] > ci_current[0] else float("nan"),
    }


def predict_mean_ci_half_width_at_n(current_half_width: float, n_current: int, n_target: int) -> float:
    """Escala analítica clássica: o erro-padrão de uma média cai com
    1/sqrt(n) (Teorema Central do Limite — assintótico, mais preciso
    quanto maior `n_current` já for), então a largura do IC também cai
    nessa proporção. `current_half_width` = (ci_hi-ci_lo)/2 já observado
    em `n_current`."""
    if n_current <= 0 or n_target <= 0:
        raise ValueError("n_current e n_target precisam ser positivos")
    return float(current_half_width * np.sqrt(n_current / n_target))

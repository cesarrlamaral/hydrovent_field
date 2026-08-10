"""
Índices de sensibilidade global (Sobol', variance-based) sobre um
surrogate (Processo Gaussiano) treinado nos pontos externos do desenho
aninhado de `variance_decomposition.py`/`fumarola_field.run_nested_
variance_experiment` — reaproveita os MESMOS dados (médias de grupo +
variância intra-grupo já estimada) sem rodar nenhuma simulação física a
mais.

Por que um surrogate: os índices de Sobol' (Sobol, I.M., 1993, "Sensitivity
estimates for nonlinear mathematical models," MMCE 1(4):407-414; ver
também Sobol, I.M., 2001, Math. Comput. Simul. 55:271-280) precisam de
milhares de avaliações da função para uma estimativa Monte Carlo estável
(esquema de Saltelli, ver abaixo) — inviável com a simulação física real
(~10s/run). Um Processo Gaussiano (Rasmussen, C.E., & Williams, C.K.I.,
2006, "Gaussian Processes for Machine Learning," MIT Press, cap. 2/5)
ajustado nos poucos pontos externos reais aproxima a resposta MÉDIA
(E[Y|theta], já isolada da variância estocástica pelo próprio desenho
aninhado — ver variance_decomposition.py) como função suave dos
parâmetros varridos, e pode ser avaliado milhões de vezes por segundo.

Implementado do zero (não via scikit-learn) para manter a mesma filosofia
de dependência enxuta do resto do projeto (só numpy/scipy) — mesmo
padrão de "método citado, implementação própria, testes próprios" já
usado em plume_physics.py/acoustics.py.

**Honestidade central**: os índices de Sobol' aqui são sobre o
SURROGATE, não sobre a simulação física diretamente — só são confiáveis
na medida em que o surrogate aproxima bem a resposta real. Por isso todo
resultado vem acompanhado de `loo_cv_r2` (R² de validação cruzada
leave-one-out) e de IC 95% por bootstrap; um R² baixo é um sinal
explícito para não confiar nos índices, não um detalhe escondido.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


# --------------------------------------------------------------------------
# 1. Processo Gaussiano (kernel RBF, ajuste próprio, sem scikit-learn)
# --------------------------------------------------------------------------

class GaussianProcessSurrogate:
    """
    Regressão por Processo Gaussiano com kernel RBF (exponencial
    quadrático) de comprimento de escala por dimensão, ruído de medição
    CONHECIDO (não estimado — vem da própria variância intra-grupo já
    calculada em variance_decomposition.py, ver `fit`). Hiperparâmetros
    (variância de sinal + comprimentos de escala) ajustados por máxima
    verossimilhança marginal (Rasmussen & Williams 2006, eq. 2.30), várias
    reinicializações aleatórias (log-verossimilhança não é convexa).

    Entradas normalizadas para [0,1]^d via `bounds` antes de qualquer
    cálculo — necessário porque os parâmetros varridos têm escalas MUITO
    diferentes (ex. alpha ~0.07-0.18 vs. densidade ~2400-3600 kg/m3).
    """

    def __init__(self, bounds: List[tuple]):
        self.bounds = bounds
        self.d = len(bounds)
        self._lows = np.array([b[0] for b in bounds])
        self._highs = np.array([b[1] for b in bounds])
        self._fitted = False

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._lows) / (self._highs - self._lows)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, sigma_f2: float, lengthscales: np.ndarray) -> np.ndarray:
        diff = (X1[:, None, :] - X2[None, :, :]) / lengthscales
        sqdist = np.sum(diff ** 2, axis=-1)
        return sigma_f2 * np.exp(-0.5 * sqdist)

    def fit(self, X: np.ndarray, y: np.ndarray, noise_variance, n_restarts: int = 8,
            rng: Optional[np.random.Generator] = None) -> "GaussianProcessSurrogate":
        """
        `noise_variance`: escalar ou array (n,) — variância de medição
        CONHECIDA de cada observação (não ajustada, ao contrário do sigma_f/
        lengthscales). Tipicamente `within_group_variance / n_inner`
        (variância da MÉDIA de n_inner réplicas, não da réplica individual).
        """
        if rng is None:
            rng = np.random.default_rng()
        Xn = self._normalize(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        n = Xn.shape[0]
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        yn = (y - self._y_mean) / self._y_std

        noise = np.broadcast_to(np.asarray(noise_variance, dtype=float) / (self._y_std ** 2), (n,)).copy()
        jitter = 1e-8

        def neg_log_marginal_likelihood(log_theta):
            # Clampeado a [-20,20]: exp(20)~5e8 já é um comprimento de escala
            # efetivamente "infinito" num domínio normalizado [0,1] (kernel
            # praticamente constante) — evita overflow/RuntimeWarning do
            # otimizador explorando regiões numericamente inúteis sem mudar
            # qual hiperparâmetro é o ótimo real (que nunca está lá fora).
            log_theta = np.clip(log_theta, -20.0, 20.0)
            log_sigma_f2, log_lengthscales = log_theta[0], log_theta[1:]
            sigma_f2 = np.exp(log_sigma_f2)
            lengthscales = np.exp(log_lengthscales)
            K = self._kernel(Xn, Xn, sigma_f2, lengthscales) + np.diag(noise + jitter)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            nll = 0.5 * yn @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            if not np.isfinite(nll):
                return 1e10
            return float(nll)

        best = None
        for _ in range(n_restarts):
            x0 = rng.uniform(-2.0, 2.0, size=1 + self.d)
            res = minimize(neg_log_marginal_likelihood, x0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

        log_sigma_f2, log_lengthscales = best.x[0], best.x[1:]
        self.sigma_f2 = float(np.exp(log_sigma_f2))
        self.lengthscales = np.exp(log_lengthscales)
        self._X = Xn
        self._y = yn
        self._noise = noise
        K = self._kernel(Xn, Xn, self.sigma_f2, self.lengthscales) + np.diag(noise + jitter)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, yn))
        self._K_inv = np.linalg.solve(self._L.T, np.linalg.solve(self._L, np.eye(n)))
        self._fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Média posterior (des-normalizada) em `X_new` (n_new, d) — em
        unidades reais dos parâmetros e da resposta."""
        if not self._fitted:
            raise RuntimeError("chame fit() antes de predict()")
        Xn_new = self._normalize(np.asarray(X_new, dtype=float))
        K_star = self._kernel(Xn_new, self._X, self.sigma_f2, self.lengthscales)
        mean_n = K_star @ self._alpha
        return mean_n * self._y_std + self._y_mean

    def loo_cv_r2(self) -> float:
        """
        R² de validação cruzada leave-one-out, via fórmula fechada de GP
        (Rasmussen & Williams 2006, §5.4.2, eq. 5.12) — evita reajustar o
        modelo n vezes: mu_i_loo = y_i - alpha_i / K_inv_ii,
        sigma2_i_loo = 1 / K_inv_ii, usando o MESMO ajuste já feito em
        `fit()` (hiperparâmetros não são re-otimizados por fold — LOO
        aproximado, padrão para GPs, não LOO exato de um pipeline
        completo de seleção de hiperparâmetros).
        """
        if not self._fitted:
            raise RuntimeError("chame fit() antes de loo_cv_r2()")
        diag_kinv = np.diag(self._K_inv)
        mu_loo_n = self._y - self._alpha / diag_kinv
        mu_loo = mu_loo_n * self._y_std + self._y_mean
        y_real = self._y * self._y_std + self._y_mean
        ss_res = np.sum((y_real - mu_loo) ** 2)
        ss_tot = np.sum((y_real - y_real.mean()) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------
# 2. Índices de Sobol' via esquema de Saltelli, MC sobre o surrogate (ou
#    qualquer função vetorizável — testado diretamente contra funções
#    analíticas em tests/test_global_sensitivity.py, sem envolver o GP)
# --------------------------------------------------------------------------

def _saltelli_matrices(bounds: List[tuple], n: int, rng: np.random.Generator):
    """A, B: (n, d) via UMA ÚNICA sequência de Sobol' conjunta de dimensão
    2d, dividida em A (primeiras d colunas) e B (últimas d colunas) — não
    duas sequências de Sobol' construídas independentemente. Isso importa:
    duas instâncias independentes de `qmc.Sobol` são cada uma bem
    distribuída na SUA própria dimensão, mas não têm a estrutura de
    correlação (praticamente zero, testável) que o estimador de
    Saltelli/Jansen precisa entre A e B — usar duas instâncias
    independentes mediu, na prática (ver commit desta mudança), índices
    ~2-3x errados numa função aditiva com resposta analítica conhecida.
    Split de uma sequência conjunta é o procedimento padrão (mesma
    abordagem da biblioteca de referência SALib,
    `sample.saltelli.sample`). (Sobol, I.M., 1967/1976, a sequência de
    baixa discrepância — DIFERENTE dos índices de sensibilidade de
    Sobol', mesmo nome, ferramentas relacionadas mas distintas.) `n` é
    arredondado para a próxima potência de 2 (exigido por
    `Sobol.random_base2`)."""
    d = len(bounds)
    m = int(np.ceil(np.log2(max(n, 2))))
    sampler = qmc.Sobol(d=2 * d, seed=rng)
    unit = sampler.random_base2(m=m)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    A = lows + unit[:, :d] * (highs - lows)
    B = lows + unit[:, d:] * (highs - lows)
    return A, B


def sobol_indices_from_function(f: Callable[[np.ndarray], np.ndarray], bounds: List[tuple],
                                 n: int = 4096, rng: Optional[np.random.Generator] = None) -> dict:
    """
    Índices de Sobol' de primeira ordem (S_i) e de efeito total (S_Ti) de
    `f` sobre `bounds` (lista de (low,high), entradas assumidas
    independentes e uniformes na faixa — mesma suposição de
    `--sensitivity-sweep`/`--variance-decomposition`), via esquema de
    Saltelli. `f` deve aceitar um array (n, d) e devolver (n,).

    S_i: estimador de Saltelli (Saltelli, A., et al., 2010, "Variance
    based sensitivity analysis of model output. Design and estimator for
    the total sensitivity index," Computer Physics Communications,
    181(2), 259-270 — eq. 4.16 nesse artigo/prática padrão).
    S_Ti: estimador de Jansen (Jansen, M.J.W., 1999, "Analysis of
    variance designs for model output," Computer Physics Communications,
    117(1-2), 35-43 — eq. 27 nesse artigo), numericamente mais estável
    que o estimador original de Sobol' para o índice total.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(bounds)
    A, B = _saltelli_matrices(bounds, n, rng)
    n_eff = A.shape[0]

    f_A = np.asarray(f(A), dtype=float)
    f_B = np.asarray(f(B), dtype=float)
    var_y = np.var(np.concatenate([f_A, f_B]), ddof=1)

    s_i, s_ti = {}, {}
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        f_ABi = np.asarray(f(A_Bi), dtype=float)

        v_i = np.mean(f_B * (f_ABi - f_A))
        v_ti = 0.5 * np.mean((f_A - f_ABi) ** 2)

        # Índices de Sobol' verdadeiros são, por definição, limitados a
        # [0,1] — um estimador MC de amostra finita pode escapar um pouco
        # dessa faixa por ruído (visto na prática com `f` quase constante,
        # ex. um surrogate mal ajustado por poucos dados: var_y fica perto
        # de zero e a razão v_i/var_y explode). Grampear comunica "ruído
        # do estimador", não esconde a instabilidade — o diagnóstico real
        # dessa situação é `loo_cv_r2`/`loo_cv_r2_warning` em
        # `fit_surrogate_and_compute_sobol`, não este grampeamento.
        s_i[i] = float(np.clip(v_i / var_y, 0.0, 1.0)) if var_y > 0 else 0.0
        s_ti[i] = float(np.clip(v_ti / var_y, 0.0, 1.0)) if var_y > 0 else 0.0

    return {"first_order": s_i, "total_order": s_ti, "variance": float(var_y), "n_mc": n_eff}


# --------------------------------------------------------------------------
# 3. Orquestração: ajusta o surrogate nos dados do desenho aninhado e
#    devolve os índices de Sobol' + diagnóstico de qualidade do ajuste
# --------------------------------------------------------------------------

def fit_surrogate_and_compute_sobol(outer_params: np.ndarray, outer_groups: List[np.ndarray],
                                     within_group_variance: float, bounds: List[tuple],
                                     param_names: List[str], n_mc: int = 4096, n_bootstrap: int = 200,
                                     rng: Optional[np.random.Generator] = None) -> dict:
    """
    `outer_params`: (n_outer, d) — os mesmos pontos externos do desenho
    aninhado. `outer_groups`: lista de n_outer arrays de réplicas
    internas (mesmo formato de `variance_decomposition.
    nested_variance_decomposition`). `within_group_variance`: componente
    ESTOCÁSTICA já estimada por `nested_variance_decomposition` — usada
    como ruído de medição conhecido da média de cada grupo
    (within_group_variance / n_inner).
    """
    if rng is None:
        rng = np.random.default_rng()
    outer_params = np.asarray(outer_params, dtype=float)
    n_outer = outer_params.shape[0]
    n_inner = len(outer_groups[0])
    group_means = np.array([g.mean() for g in outer_groups])
    noise_var_of_mean = within_group_variance / n_inner

    def _fit_and_sobol(params, means, seed_rng):
        gp = GaussianProcessSurrogate(bounds).fit(params, means, noise_var_of_mean, rng=seed_rng)
        sobol = sobol_indices_from_function(gp.predict, bounds, n=n_mc, rng=seed_rng)
        return gp, sobol

    gp, sobol = _fit_and_sobol(outer_params, group_means, rng)
    loo_r2 = gp.loo_cv_r2()

    boot_s = {i: [] for i in range(len(bounds))}
    boot_st = {i: [] for i in range(len(bounds))}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_outer, size=n_outer)
        try:
            _, boot_sobol = _fit_and_sobol(outer_params[idx], group_means[idx], rng)
        except np.linalg.LinAlgError:
            continue
        for i in range(len(bounds)):
            boot_s[i].append(boot_sobol["first_order"][i])
            boot_st[i].append(boot_sobol["total_order"][i])

    result = {
        "param_names": param_names,
        "loo_cv_r2": loo_r2,
        "loo_cv_r2_warning": loo_r2 < 0.5,
        "gp_sigma_f2": gp.sigma_f2,
        "gp_lengthscales": gp.lengthscales.tolist(),
        "first_order": {param_names[i]: sobol["first_order"][i] for i in range(len(bounds))},
        "total_order": {param_names[i]: sobol["total_order"][i] for i in range(len(bounds))},
        "first_order_ci95": {
            param_names[i]: (float(np.percentile(boot_s[i], 2.5)), float(np.percentile(boot_s[i], 97.5)))
            if boot_s[i] else (float("nan"), float("nan"))
            for i in range(len(bounds))
        },
        "total_order_ci95": {
            param_names[i]: (float(np.percentile(boot_st[i], 2.5)), float(np.percentile(boot_st[i], 97.5)))
            if boot_st[i] else (float("nan"), float("nan"))
            for i in range(len(bounds))
        },
        "n_bootstrap_valid": len(boot_s[0]),
    }
    return result

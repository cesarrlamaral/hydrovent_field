"""
QA automatizada de integridade de um ensemble — verificação sistemática
em vez de confiar só em inspeção manual ocasional. Dois níveis DELIBERADAMENTE separados:

- `hard_errors`: problemas inequívocos (NaN/Inf, valor negativo onde
  fisicamente impossível, seeds duplicadas, inconsistência interna
  entre campos já computados) — sempre vale investigar.
- `soft_flags`: outliers estatísticos (z-score robusto, mediana/MAD —
  ver ensemble_stats.py) — candidatos a revisão manual, EXPLICITAMENTE
  não tratados como bug por padrão. Distribuições deste projeto são
  conhecidamente caudal-pesadas (altura de chaminé, evento raro de
  Gor'kov — docs/PHYSICS_MODEL.md §7.8.1/§10.1); um outlier estatístico
  aqui é, com boa probabilidade, o mesmo tipo de evento raro real que
  o resto do projeto foi construído pra estudar, não um bug — misturar
  os dois níveis seria exatamente o erro que §7.8.4 (docs/PHYSICS_MODEL.md)
  já mostrou ser real: tratar sinal real como ruído.

Módulo puro (sem Tkinter/relatório), opera sobre a MESMA lista de
`summaries` (dicts) já usada pelo resto do projeto — não um parser de
`runs_index.csv` à parte, para nunca divergir do dado real.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _is_bad_number(x) -> bool:
    return x is None or (isinstance(x, (int, float)) and not np.isfinite(x))


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Z-score robusto (mediana/MAD escalado — mesmo fator 1/Φ⁻¹(3/4)
    ≈1,4826 de `ensemble_stats.describe`, ver docs/PHYSICS_MODEL.md
    §10.2) — resistente a outliers de verdade, ao contrário do z-score
    clássico (mean/std), que o próprio outlier infla."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    mad_scaled = mad * 1.4826
    return (values - median) / mad_scaled


def check_run_integrity(summaries: List[dict], z_threshold: float = 5.0,
                         n_runs_expected: Optional[int] = None) -> dict:
    """Verifica `summaries` (lista de dicts, mesmo formato de
    `fumarola_field.load_run_summary`) por problemas de integridade.

    `n_runs_expected`: se fornecido (ex. de `experiment_metadata.json`),
    compara contra `len(summaries)` para detectar runs faltando/com
    metadata.json corrompido (que `load_run_summary` já filtra como
    None antes de chegar aqui — o CHAMADOR precisa contar isso à parte,
    ver `check_experiment_dir_integrity` abaixo, que faz isso).
    """
    hard_errors: List[str] = []
    seeds_seen: dict = {}

    numeric_nonneg_fields = [
        ("n_vents", lambda s: s.get("n_vents")),
        ("max_concentration_uM", lambda s: (s.get("prebiotic_summary") or {}).get("max_concentration_uM")),
        ("mean_concentration_uM", lambda s: (s.get("prebiotic_summary") or {}).get("mean_concentration_uM")),
    ]

    for i, s in enumerate(summaries):
        run_label = s.get("run_dir", f"run[{i}]")

        seed = s.get("seed")
        if seed is not None:
            if seed in seeds_seen:
                hard_errors.append(
                    f"{run_label}: seed {seed} duplicada (já usada em {seeds_seen[seed]}) — "
                    "possível bug de derivação de seed")
            else:
                seeds_seen[seed] = run_label

        for field_name, getter in numeric_nonneg_fields:
            val = getter(s)
            if _is_bad_number(val):
                hard_errors.append(f"{run_label}: {field_name} é NaN/Inf/None ({val!r})")
            elif val < 0:
                hard_errors.append(f"{run_label}: {field_name} negativo ({val}) — fisicamente impossível")

        n_vents = s.get("n_vents")
        if isinstance(n_vents, (int, float)) and n_vents == 0:
            hard_errors.append(f"{run_label}: n_vents=0 — falha de geração do campo")

        # invariante REAL (verificado em prebiotic.compute_field_hotspots):
        # aumentaram+diminuíram+inalterados conta só vents com
        # enrichment_vs_control != None — um SUBCONJUNTO de n_vents (nem
        # todo vent tem uma comparação válida contra o controle), não
        # necessariamente igual. Só o caso de EXCEDER n_vents é um bug
        # real (contagem duplicada ou vazamento de outro campo).
        ps = s.get("prebiotic_summary") or {}
        n_up = ps.get("n_vents_increased_vs_control")
        n_down = ps.get("n_vents_decreased_vs_control")
        n_eq = ps.get("n_vents_unchanged_vs_control")
        if all(isinstance(v, (int, float)) for v in (n_up, n_down, n_eq, n_vents)) and n_vents:
            if n_up + n_down + n_eq > n_vents:
                hard_errors.append(
                    f"{run_label}: aumentaram+diminuíram+inalterados ({n_up}+{n_down}+{n_eq}"
                    f"={n_up + n_down + n_eq}) EXCEDE n_vents ({n_vents}) — inconsistência interna")

        diag = s.get("acoustic_diagnostics")
        if diag:
            for key in ("gorkov_trap_depth_over_kT", "streaming_speed_max_m_s"):
                val = diag.get(key)
                if val is not None:
                    if _is_bad_number(val):
                        hard_errors.append(f"{run_label}: acoustic_diagnostics.{key} é NaN/Inf ({val!r})")
                    elif val < 0:
                        hard_errors.append(f"{run_label}: acoustic_diagnostics.{key} negativo ({val})")

    soft_flags: List[str] = []
    outlier_fields = [
        ("prebiotic_summary.top_hotspot_enrichment_vs_control",
         [(s.get("prebiotic_summary") or {}).get("top_hotspot_enrichment_vs_control") for s in summaries]),
        ("acoustic_diagnostics.gorkov_trap_depth_over_kT",
         [(s.get("acoustic_diagnostics") or {}).get("gorkov_trap_depth_over_kT") for s in summaries]),
    ]
    for field_name, raw_values in outlier_fields:
        valid_idx = [i for i, v in enumerate(raw_values) if v is not None and np.isfinite(v)]
        if len(valid_idx) < 5:
            continue
        values = np.array([raw_values[i] for i in valid_idx])
        z = _robust_z_scores(values)
        for local_i, zi in zip(valid_idx, z):
            if abs(zi) > z_threshold:
                run_label = summaries[local_i].get("run_dir", f"run[{local_i}]")
                soft_flags.append(
                    f"{run_label}: {field_name}={raw_values[local_i]:.4g} é um outlier estatístico "
                    f"(z robusto={zi:+.1f}, limiar={z_threshold}) — candidato a revisão manual, "
                    "NÃO necessariamente um bug (pode ser um evento raro real, ver docs/PHYSICS_MODEL.md §7.8.1)")

    return {
        "n_runs_checked": len(summaries),
        "n_runs_expected": n_runs_expected,
        "hard_errors": hard_errors,
        "soft_flags": soft_flags,
        "ok": len(hard_errors) == 0,
    }


def check_experiment_dir_integrity(experiment_dir: str, z_threshold: float = 5.0) -> dict:
    """Conveniência: carrega um experimento do disco (mesmo padrão de
    `fumarola_field.find_run_dirs`/`load_run_summary`, reaproveitado sem
    duplicar) e roda `check_run_integrity`, incluindo runs com
    `metadata.json` faltando/corrompido (que `load_run_summary` filtra
    como None) como um hard_error explícito — não silenciosamente
    ignoradas."""
    import fumarola_field as ff

    run_dirs = ff.find_run_dirs(experiment_dir)
    summaries = []
    missing = []
    for rd in run_dirs:
        s = ff.load_run_summary(rd)
        if s is None:
            missing.append(rd)
        else:
            summaries.append(s)

    result = check_run_integrity(summaries, z_threshold=z_threshold, n_runs_expected=len(run_dirs))
    for rd in missing:
        result["hard_errors"].insert(
            0, f"{rd}: metadata.json ausente ou corrompido — run incompleta/crashada")
    result["ok"] = len(result["hard_errors"]) == 0
    return result

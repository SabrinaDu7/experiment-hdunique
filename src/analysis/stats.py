"""The statistical tests this project's conclusions rest on.

Until now every one of these lived only inside a bash heredoc pasted into a results document. That
made the headline findings — the exit-state null, the three-level variance split, the correlations
ruling out timing and duration effects — impossible to re-run, impossible to test, and impossible
to check for drift. They are library code here.

Each function returns a plain dict so a result can go straight into a `Values` record without a
formatting step in between.
"""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def correlation(*, x: pd.Series, y: pd.Series, method: str = "spearman") -> dict[str, float]:
    """Rank (or Pearson) correlation with its p-value."""
    fn = {"spearman": stats.spearmanr, "pearson": stats.pearsonr}[method]
    rho, p = fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"rho": float(rho), "p": float(p), "n": len(x)}


def compare_groups(*, values: pd.Series, groups: pd.Series) -> dict[str, Any]:
    """Kruskal-Wallis across however many groups `groups` defines.

    Used to ask "do these mice differ at all?" without the variance-partitioning ambition of an
    ICC — a question that stays answerable when the ICC's interval does not.
    """
    arrays = [np.asarray(g, dtype=float) for _, g in values.groupby(groups)]
    stat, p = stats.kruskal(*arrays)
    return {"statistic": float(stat), "p": float(p), "n_groups": len(arrays), "n": len(values)}


def paired_difference(*, a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Wilcoxon signed-rank on paired samples, with the median difference."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    stat, p = stats.wilcoxon(diff)
    return {
        "median_diff": float(np.median(diff)),
        "n_positive": int((diff > 0).sum()),
        "n": len(diff),
        "statistic": float(stat),
        "p": float(p),
    }


def rank_sum(*, a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Mann-Whitney U for two independent samples."""
    stat, p = stats.mannwhitneyu(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return {"statistic": float(stat), "p": float(p), "n_a": len(a), "n_b": len(b)}


def demean_within(*, frame: pd.DataFrame, columns: list[str], group: str) -> pd.DataFrame:
    """Subtract each group's mean from `columns`, adding `<col>_c`.

    The cheapest way to ask whether an effect lives *within* groups or merely *between* them — the
    distinction that turned the apparent bout-context effect into a session confound.
    """
    out = frame.copy()
    for column in columns:
        out[f"{column}_c"] = out[column] - out.groupby(group)[column].transform("mean")
    return out


def effect_within_and_between(
    *, frame: pd.DataFrame, outcome: str, predictor: str, group: str
) -> pd.DataFrame:
    """The same effect estimated three ways: ignoring groups, with group as a random effect, and
    with group as fixed effects.

    A predictor that looks strong raw and vanishes under the other two is a between-group confound,
    not an effect of the predictor. Reporting all three side by side makes that visible instead of
    leaving it to be discovered.
    """
    rows = []
    fits = {
        "raw": smf.ols(f"{outcome} ~ {predictor}", frame).fit(),
        "group_random": smf.mixedlm(f"{outcome} ~ {predictor}", frame, groups=frame[group]).fit(),
        "group_fixed": smf.ols(f"{outcome} ~ {predictor} + C({group})", frame).fit(),
    }
    for name, fit in fits.items():
        rows.append(
            {
                "model": name,
                "beta": float(fit.params[predictor]),
                "p": float(fit.pvalues[predictor]),
            }
        )
    return pd.DataFrame(rows)


def nested_variance(
    *, frame: pd.DataFrame, outcome: str, outer: str, inner: str
) -> dict[str, float]:
    """Three-level variance components for a nested design, by Searle's unbalanced ANOVA.

    Splits the variance in `outcome` into between-`outer` (e.g. mouse), between-`inner`-within-outer
    (session), and residual (bout). Method of moments rather than REML because `statsmodels`'
    crossed variance-component optimiser does not converge on this design, whereas this estimator is
    exact and needs no optimiser.
    """
    n_total, n_outer = len(frame), frame[outer].nunique()
    counts_inner = frame.groupby([outer, inner]).size()
    counts_outer = frame.groupby(outer).size()
    n_inner = len(counts_inner)

    grand = frame[outcome].mean()
    mean_outer = frame.groupby(outer)[outcome].mean()
    mean_inner = frame.groupby([outer, inner])[outcome].mean()

    ss_outer = sum(counts_outer[i] * (mean_outer[i] - grand) ** 2 for i in counts_outer.index)
    ss_inner = sum(
        counts_inner[k] * (mean_inner[k] - mean_outer[k[0]]) ** 2 for k in counts_inner.index
    )
    ss_resid = sum(
        ((g[outcome] - g[outcome].mean()) ** 2).sum() for _, g in frame.groupby([outer, inner])
    )

    ms_outer = ss_outer / (n_outer - 1)
    ms_inner = ss_inner / (n_inner - n_outer)
    ms_resid = ss_resid / (n_total - n_inner)

    weighted = sum(counts_inner[k] ** 2 / counts_outer[k[0]] for k in counts_inner.index)
    k2 = (n_total - weighted) / (n_inner - n_outer)
    k1 = (weighted - sum(counts_inner**2) / n_total) / (n_outer - 1)
    k0 = (n_total - sum(counts_outer**2) / n_total) / (n_outer - 1)

    var_resid = ms_resid
    var_inner = max((ms_inner - ms_resid) / k2, 0.0)
    var_outer = max((ms_outer - ms_resid - k1 * var_inner) / k0, 0.0)
    total = var_outer + var_inner + var_resid

    return {
        "var_outer": float(var_outer),
        "var_inner": float(var_inner),
        "var_resid": float(var_resid),
        "total": float(total),
        "frac_outer": float(var_outer / total),
        "frac_inner": float(var_inner / total),
        "frac_resid": float(var_resid / total),
        "icc_outer": float(var_outer / total),
        "icc_outer_inner": float((var_outer + var_inner) / total),
        "n": n_total,
        "n_outer": n_outer,
        "n_inner": n_inner,
    }


def residualise(*, frame: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.Series:
    """Residuals of `outcome` after regressing out `predictors`.

    Note this is only interpretable when the predictors are *upstream* of the outcome. Regressing
    out a variable that is partly a consequence of the outcome removes real signal — the trap that
    made decode-quality correction look as though it abolished the between-mouse effect.
    """
    formula = f"{outcome} ~ " + " + ".join(predictors)
    return smf.ols(formula, frame).fit().resid


def percentile_ci(*, samples: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile confidence interval from bootstrap replicates."""
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))

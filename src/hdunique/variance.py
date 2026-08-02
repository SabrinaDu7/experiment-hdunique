"""Between-mouse vs between-session variance of the REM diffusion constant.

Each mouse contributes several sessions, each with one D. Two obstacles rule out a naive variance
comparison: sessions from one mouse are not independent, and mice have unequal session counts. So
the spread is split with a random-intercept linear mixed model fitted by REML:

    log(D) ~ 1 + (1 | mouse)

    tau^2   (random-intercept variance) = between-mouse variance
    sigma^2 (residual variance)         = between-session, within-mouse variance
    ICC = tau^2 / (tau^2 + sigma^2)     = fraction of variability due to mouse identity

Three modelling choices, all load-bearing:

1. Sessions are **gated on ADn cell count**. D inflates when the ring is undersampled, and cell
   yield is confounded with mouse identity (Mouse20 is thin in every session), so ungated the
   artefact loads onto tau^2 and masquerades as a between-mouse difference.
2. D is **log-transformed**. It is positive, spans ~0.4-7 and has mean-dependent spread; on the log
   scale the components read as fractional variability and normality is defensible.
3. CIs come from a **mouse-level parametric bootstrap**, not Wald. Variance components are bounded
   at zero with skewed sampling distributions and ~6 mice carry very little information about
   tau^2, so Wald intervals would be actively misleading.

sigma^2 absorbs everything the mouse intercept does not, which includes real session-to-session
biology *plus* D-estimation noise, so it is an upper bound on within-mouse biological variance. It
is also a single pooled parameter under an assumed-common variance, not an average of per-mouse
variances; the per-mouse variances are in fact wildly heterogeneous.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

from hdunique.config import HEADLINE_WINDOW_MS
from hdunique.diffusion import window_column


def d_column(*, window_ms: int) -> str:
    """Parquet column holding D for a fit window."""
    return window_column(stat="D", window_ms=window_ms, headline_ms=HEADLINE_WINDOW_MS)


def gate_sessions(
    *,
    df: pd.DataFrame,
    cell_set: str,
    min_cells: int,
    window_ms: int = HEADLINE_WINDOW_MS,
    count_col: str = "n_adn",
) -> pd.DataFrame:
    """Keep one cell set's well-sampled sessions and add `log_D`.

    `count_col` is the cell count the gate reads; "n_adn" suits the ADn set, while a PoS or union
    panel must pass "n_cells" since PoS-only sessions have n_adn = 0.
    """
    out = df[(df["cell_set"] == cell_set) & (df[count_col] >= min_cells)].copy()
    out["log_D"] = np.log(out[d_column(window_ms=window_ms)])
    return out.reset_index(drop=True)


def fit_lmm(*, df: pd.DataFrame) -> MixedLMResults:
    """Fit log(D) ~ 1 + (1 | mouse) by REML."""
    return smf.mixedlm("log_D ~ 1", df, groups=df["mouse"]).fit(reml=True)


def variance_components(*, result: MixedLMResults) -> dict[str, float]:
    """Extract tau^2, sigma^2, their total, the ICC and the grand mean from a fit."""
    tau2 = float(np.asarray(result.cov_re)[0, 0])
    sigma2 = float(result.scale)
    total = tau2 + sigma2
    return {
        "tau2_between_mouse": tau2,
        "sigma2_within_mouse": sigma2,
        "total": total,
        "icc": tau2 / total if total > 0 else float("nan"),
        "grand_mean_log_D": float(result.fe_params.iloc[0]),
    }


def mouse_effects(*, result: MixedLMResults, df: pd.DataFrame) -> pd.DataFrame:
    """Per-mouse shrunken random-effect estimates (BLUPs) with conditional uncertainty.

    For a random-intercept model the posterior variance of a mouse's intercept is
    1 / (n_i/sigma^2 + 1/tau^2), so a mouse with few sessions is both shrunk harder toward the
    grand mean and less certain. Sorted ascending by effect.
    """
    comp = variance_components(result=result)
    tau2, sigma2 = comp["tau2_between_mouse"], comp["sigma2_within_mouse"]
    n_by_mouse = df.groupby("mouse")["log_D"].size()
    raw_mean = df.groupby("mouse")["log_D"].mean()

    rows = []
    for mouse, effect in result.random_effects.items():
        n = int(n_by_mouse[mouse])
        blup = float(np.asarray(effect)[0])
        cond_var = 1.0 / (n / sigma2 + 1.0 / tau2) if tau2 > 0 else 0.0
        rows.append(
            {
                "mouse": mouse,
                "n_sessions": n,
                "blup": blup,
                "cond_sd": float(np.sqrt(cond_var)),
                "mean_log_D": comp["grand_mean_log_D"] + blup,
                "raw_mean_log_D": float(raw_mean[mouse]),
            }
        )
    return pd.DataFrame(rows).sort_values("blup").reset_index(drop=True)


def bootstrap_components(*, df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Parametric bootstrap of the variance components.

    Simulates mouse intercepts ~ N(0, tau^2) and residuals ~ N(0, sigma^2) on the **observed**
    design, so the unbalanced session counts are preserved, then refits. Replicates that fail to
    converge are dropped; replicates landing on the tau^2 = 0 boundary are kept, and are exactly
    what produces the CI's lower edge near zero.
    """
    comp = variance_components(result=fit_lmm(df=df))
    rng = np.random.default_rng(seed)
    tau = np.sqrt(comp["tau2_between_mouse"])
    sigma = np.sqrt(comp["sigma2_within_mouse"])
    mu = comp["grand_mean_log_D"]

    mice = df["mouse"].to_numpy()
    unique_mice = np.unique(mice)
    sim = df[["mouse"]].copy()

    rows: list[dict[str, float]] = []
    for _ in range(n_boot):
        intercepts = dict(zip(unique_mice, rng.normal(0.0, tau, len(unique_mice))))
        sim["log_D"] = (
            mu + np.array([intercepts[m] for m in mice]) + rng.normal(0.0, sigma, len(mice))
        )
        try:
            rows.append(variance_components(result=fit_lmm(df=sim)))
        except Exception:  # noqa: BLE001, S112 -- statsmodels raises several types on a
            continue        # non-converging replicate; dropping it is the documented behaviour.
    return pd.DataFrame(rows)


def component_cis(
    *,
    boot: pd.DataFrame,
    keys: tuple[str, ...] = ("tau2_between_mouse", "sigma2_within_mouse", "icc"),
) -> dict[str, tuple[float, float]]:
    """2.5 / 97.5 percentile CIs for each variance component from the bootstrap replicates."""
    return {
        k: (float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))) for k in keys
    }


def anova_icc(*, df: pd.DataFrame) -> float:
    """One-way ANOVA intraclass correlation on log_D, as an independent cross-check on the LMM.

    Uses the standard unbalanced-design correction n0 for the group-size mismatch. Agreement with
    the LMM ICC indicates the decomposition itself is sound and the uncertainty is sampling.
    """
    groups = [g["log_D"].to_numpy() for _, g in df.groupby("mouse")]
    k, n = len(groups), np.array([len(g) for g in groups])
    grand = np.concatenate(groups).mean()
    ms_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n.sum() - k)
    n0 = (n.sum() - (n**2).sum() / n.sum()) / (k - 1)
    tau2 = max((ms_between - ms_within) / n0, 0.0)
    return tau2 / (tau2 + ms_within) if tau2 + ms_within > 0 else float("nan")

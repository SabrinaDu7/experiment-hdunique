"""Pipeline configuration.

Every default here is the setting that produced the published results, so a bare
`uv run hd-diffusion --scope all` reproduces the sweep with no flags. (In the source repo the
sweep settings differed from the config defaults, which made the tables unreproducible from the
code alone — see docs/2026-08-02-port-rem-diffusion-and-variance.md, problem P4.)
"""

import dataclasses
from typing import Literal

#: Paper-reported REM diffusion constants (rad^2/s), for reference in printouts and plots.
#: NOTE: the paper used CRCNS REM scoring; this repo uses DANDI's. The two disagree substantially
#: on Mouse28-140313, so these are context, not a target. See docs/rem-diffusion.md.
PAPER_REM_TARGETS: dict[str, float] = {"Mouse28-140313": 1.1, "Mouse25-140130": 0.52}

#: Mice in DANDI 000056 that have at least one usable session.
DANDI_MICE: tuple[int, ...] = (12, 17, 20, 24, 25, 28)

#: Bin separations used for the diffusion curve, in units of `dt` -> 100..500 ms at dt = 0.1 s.
DIFFUSION_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Origin-anchored fit windows (ms). 200 ms is the paper's window and stays unsuffixed in the
#: parquet ("D"); the wider ones ("D_300"...) are a saturation diagnostic, NOT alternative
#: estimates of D.
FIT_WINDOWS_MS: tuple[int, ...] = (200, 300, 400, 500)

#: The paper's fit window; its columns are unsuffixed in the parquet.
HEADLINE_WINDOW_MS: int = 200

# --- long-timescale analysis (hd-timescale) ---

#: Fit windows for the timescale comparison: the paper's 200 ms, the widest window the original
#: sweep already reported, and 5 s — where circular wrapping dominates and must be circumvented.
TIMESCALE_WINDOWS_MS: tuple[int, ...] = (200, 500, 5000)

#: Longest lag the timescale curves are evaluated at, in bins (50 x 100 ms = 5 s).
TIMESCALE_MAX_LAG: int = 50


@dataclasses.dataclass(frozen=True)
class DiffusionConfig:
    """REM diffusion-constant sweep.

    REM epochs come from the DANDI `states` interval set, so the pipeline is self-contained (no
    CRCNS files). D is invariant to a constant shift and to a flip of the decoded angle, so REM's
    absent head direction is passed as zeros.
    """

    # --- scope ---
    #: "session" = one (mouse, session); "mouse" = every session of one mouse; "all" = everything.
    scope: Literal["session", "mouse", "all"] = "session"
    mouse: int = 28
    session: int = 140313

    # --- cells ---
    #: Brain areas to take, by DANDI per-unit `location`. ADn carries the ring; PoS alone is
    #: near-noise and adding it to ADn only inflates variance, so ADn is the analysis set.
    #: PoS / ADn+PoS remain reachable for the diagnostic documented in docs/rem-diffusion.md.
    cell_areas: tuple[str, ...] = ("ADn",)

    # --- rates (paper: ~100 ms bins, Gaussian kernel of s.d. 100 ms) ---
    dt: float = 0.1
    sigma: float = 0.1
    #: First-N cap on the points fed to Isomap, matching the original pipeline's `counts[:15000]`.
    n_samples: int = 15000

    # --- embedding (paper: Isomap, 5 neighbours, 3-D before decoding) ---
    n_neighbors: int = 5
    fit_dim: int = 3

    # --- ring fit (paper: K = 12 knots, k-means init, length-multiplier penalty) ---
    n_knots: int = 12
    knot_order: str = "wt_per_len"
    penalty: str = "mult_len"
    dalpha: float = 0.005
    #: Best-of-N ring per refit, selected on `fit_err` (the unsupervised spline cost, so this
    #: needs no ground-truth angle and is valid in REM).
    n_restarts: int = 10
    #: Independent refits -> D mean +/- std. NOT a confidence interval: it shrinks by construction
    #: as `n_restarts` rises, so read it as residual ring instability.
    n_refits: int = 5
    #: Fraction of REM points the ring is FIT on; decoding always uses 100%. Nothing is held out
    #: for scoring (REM has no ground-truth HD), so 1.0 simply gives a better-constrained ring.
    fit_frac: float = 1.0

    #: Bootstrap replicates for the CI on D (the paper's: resample 200 ms epochs with replacement,
    #: 1000x). Set to 0 to skip. Unlike `D_std` this IS a confidence interval.
    n_bootstrap: int = 1000

    # --- reproducibility ---
    seed: int = 0

    # --- caching ---
    #: Persist the Isomap embedding + per-refit decoded angles, so changing a diffusion metric is
    #: a seconds-long recompute rather than a refit.
    use_cache: bool = True
    #: Skip all NWB loading / Isomap / ring fitting and rebuild the parquets from cached decoded
    #: angles. For trying a new lag window or metric.
    recompute_only: bool = False

    #: Save a per-session diffusion-curve PNG (single-session scope).
    make_plot: bool = False
    #: DANDI `states` label to decode within.
    rem_label: str = "REM"

    @property
    def session_id(self) -> str:
        """Session identifier, e.g. 'Mouse28-140313'."""
        return f"Mouse{self.mouse}-{self.session}"

    @property
    def cell_set(self) -> str:
        """Cell-set label used in the parquet and cache filenames, e.g. 'ADn' or 'ADn+PoS'."""
        return "+".join(self.cell_areas)


@dataclasses.dataclass(frozen=True)
class VarianceGateConfig:
    """Which sessions enter a variance decomposition, and how its CIs are drawn.

    Shared by both decompositions — the headline one and the by-fit-window robustness check — so
    that the robustness check is guaranteed to gate on exactly what the headline gates on. (In the
    source repo the two gates were written separately and had drifted apart; see the port record,
    problem P3.)
    """

    #: Sessions below this ADn cell count have an undersampled ring and an inflated D. Cell yield
    #: is confounded with mouse identity (Mouse20 is thin in every session), so ungated the
    #: artefact masquerades as a between-mouse difference.
    min_adn_cells: int = 15
    cell_set: str = "ADn"
    mice: tuple[int, ...] = DANDI_MICE

    #: Which of the two co-headline estimates to decompose: "D" pools every pair in the
    #: concatenated trace, "D_bout_aware" excludes pairs straddling a REM-bout boundary.
    estimator: Literal["D", "D_bout_aware"] = "D"

    #: Parametric-bootstrap replicates -> percentile CIs. Variance components are bounded at zero
    #: with skewed sampling distributions, so Wald intervals would mislead here.
    n_bootstrap: int = 2000
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class VarianceConfig(VarianceGateConfig):
    """Between-mouse vs between-session variance decomposition of D at one fit window."""

    #: Which fit window's D to decompose. The wider windows are a saturation diagnostic; the ICC
    #: is invariant to this choice, which is itself a reported result.
    window_ms: int = HEADLINE_WINDOW_MS

    #: Also fit the ungated data, so the gate's effect is visible rather than hidden.
    sensitivity_ungated: bool = True
    make_plot: bool = True

# Methods — the paper's specification

Verbatim extracts from **Chaudhuri, Gercek, Pandey, Peyrache & Fiete (2019), *The intrinsic
population dynamics of a canonical cognitive circuit*, Nature Neuroscience**
([paper](https://www.nature.com/articles/s41593-019-0460-x)), covering only the parts this repo
implements. This is the authoritative spec; everything in `src/` is measured against it.

Where this repo deliberately departs from the paper, the departure is listed in §5 and argued in
[`2026-08-02-port-rem-diffusion-and-variance.md`](./2026-08-02-port-rem-diffusion-and-variance.md).

---

## 1. Data

> Recordings were made in the ADn of mice during wake, REM and nREM periods, along with measured
> head angles. For some of the mice, the data also contain recordings from the postsubiculum.
> Including data from the postsubiculum allows for slightly better manifold decoding of waking HD
> in some animals.

## 2. Preprocessing

> We first converted spike times into time-varying rates. We included all recorded thalamic cells,
> **without subselection** based on tuning or other criteria, using binned spike counts throughout
> (**~100-ms resolution**). For analyses except persistent homology, we estimated firing rates by
> convolving the spike times with a Gaussian kernel of standard deviation **100 ms**. [...] In all
> cases, we then replaced the rates by their **square root to stabilize the variance**.

> **Isomap:** Before fitting the manifold or applying topological methods, we used Isomap to reduce
> the large (N-dimensional) ambient dimension by re-embedding the data into a smaller, but still
> relatively high-dimensional, embedding space. We set the number of neighbors to be 5 (higher
> values also work well) and embedded into 3–20 dimensions (3 for visualization and before
> decoding; 10 before applying the topological methods like persistent homology).

## 3. Spline fit, parameterization and decoding

> We fit the manifolds using piecewise linear curves. A curve L(y) is specified by K knots, with
> locations {y₁ ⋯ y_K}. The knots are ordered, and the ith segment of the curve is a straight line
> between the ith and i+1th knot. Given data points xᵢ and a number of knots K, we first used
> k-means to identify K clusters in the data and set the centers of these clusters to be the
> initial knot locations. We then iteratively updated these knot locations to minimize
> (Σᵢ‖xᵢ − L(y)‖)·|L(y)|, where ‖xᵢ − L(y)‖ is the Euclidean distance between the ith data point and
> the nearest point on the curve L(y), and |L(y)| is the length of the curve. The multiplication by
> |L(y)| acts as a regularizer that penalizes excessively long or convoluted curves.

> **Decoding:** We parameterized points on the manifold by distance along the curve (in embedding
> space) from some arbitrary origin, with distances rescaled between 0 and 2π for comparison to the
> actual head angle. We primarily used **K = 12** and embedding dimension **D = 3**. Points were
> decoded by mapping them to the nearest point on the manifold, based on the Euclidean norm in the
> embedding space, and reading off the parameter value there.

The paper also notes it did *not* force an interpretation by regression onto a previously
characterized library of states — which rules out fitting the ring on wake and projecting REM
through it. The ring must be fit and decoded **within** the state being analysed.

## 4. Diffusion curves

> The diffusion curve at time shift τ is D(τ) = ⟨(α(t + τ) − α(t))²⟩_t, where the average value is
> taken over time (that is, all pairs of time points separated by τ). To compute diffusion
> constants, we fitted a straight line to the **first 200 ms** of the squared change in decoded
> angle against time. To obtain a bootstrapped estimate of error, we resampled 200-ms epochs from
> the data with replacement (number of samples chosen to match the length of data) and recomputed
> the diffusion constant. We repeated this resampling procedure **1,000 times**.

> *(Note: the squaring inside the average is stated in the paper's formula for the diffusion curve
> and is what the released code computes; some transcriptions of this sentence drop the square.)*

**Reported REM values:**

| Animal / session | REM diffusion constant (rad²/s) |
|---|---|
| Mouse28-140313 | 1.1 ± 0.04 |
| Mouse25-140130 | 0.52 ± 0.03 |
| Mouse12 (session unspecified) | 1.3 ± 0.06 |

D is reported as the **slope** of the fitted line. Note this is twice the diffusion coefficient of
the physics convention, where ⟨Δx²⟩ = 2Dτ.

## 5. Where this repo departs from the paper

Each departure is deliberate and argued in the port doc; none is an oversight.

| Paper | This repo | Why |
|---|---|---|
| "all recorded thalamic cells, without subselection" | **ADn only** | The paper's own reported numbers were established to be ADn-only in the predecessor repo's CRCNS work. Adding postsubiculum cells does not lower D and PoS alone is near-noise. |
| Sleep-state epochs as scored for the paper (CRCNS `.states.REM`) | **DANDI `states` REM** | Keeps the pipeline self-contained on one public dataset. The two scorings differ substantially on some sessions, so absolute D here is **not** directly comparable to the paper's numbers. |
| Bootstrap over resampled 200 ms epochs, ×1000 | **not implemented** | The source implementation's "bootstrap" resampled per-bin squared changes, which is a different quantity; rather than port a mislabelled statistic, it was dropped. `D_std` (spread over refits) is reported instead and is explicitly a ring-stability diagnostic, not a CI. |
| Straight-line fit to the first 200 ms | **line forced through the origin** | At zero lag the squared change is exactly zero, so the intercept is not a free parameter. |

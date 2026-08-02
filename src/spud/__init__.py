"""Vendored analysis code from Chaudhuri et al. (2019), *The intrinsic population dynamics of a
canonical cognitive circuit* (Nature Neuroscience).

Upstream: https://github.com/FieteLab/SPUD (repo `shared_scripts/`, `read_in_data/`).

These modules are kept **as the authors wrote them** — untyped, `from __future__ import division`,
no keyword-only arguments — so that this port can be diffed against the original line by line. They
are deliberately exempt from this repo's coding style; everything in `hdunique/` follows it.

The only edits made:
- `manifold_fit_and_decode_fns.py`: `import shared_scripts.X` -> `import spud.X` (package rename).
- `kernel_rates.py`: extracted the two functions actually used from `read_in_data/rate_functions.py`,
  dropping that module's `sys.path` manipulation and its CRCNS-only loaders.
"""

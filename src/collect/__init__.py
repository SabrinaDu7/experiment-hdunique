"""Data-collection sweeps: the steps that turn the decode cache into per-session tables.

These are library modules, not CLIs. An experiment that needs one declares `collect()` and calls
into here, so the expensive step is shared rather than re-derived per question. The one genuinely
expensive collector — the NWB -> cache sweep — stays a console script (`hd-diffusion`), because it
is the prerequisite for everything and is run on its own.
"""

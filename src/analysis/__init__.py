"""Reusable analysis primitives.

Experiment modules should be thin: load through `io`, compute through `curves`/`stats`, emit
through `values`. Anything that would be written twice belongs here instead.
"""

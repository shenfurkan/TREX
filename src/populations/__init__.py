"""Scenario/companion priors, background tables and imaging constraints.

Use owning submodules directly. ``priors`` retains the public sampler API and
companion/mass-ratio re-exports; kernels live in ``companion_priors`` and
``mass_ratio_priors``. Shared parameters have no sampler dependencies.
``spatial_prior`` retains diagnostics, not a source-location likelihood.
No eager imports or package-level facade are needed.
"""

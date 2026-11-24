"""Native time axes, Gaussian baselines, phase bins and photometric conversions.

Import the owning modules directly; no eager imports or package-level facades.
``baseline`` owns the explicit proper-prior aperture likelihood, ``binning``
inverse-variance phase bins, ``covariance`` descriptive OOT covariance (not a
noise likelihood), and ``photometric_relations`` colour/magnitude conversions.
``time_axis`` owns precise already-barycentric TDB reference/boxcar-anchor
arithmetic. ``time_conversion`` separately qualifies observer clock/geometry
conversion; ``exposure`` owns declared temporal quadrature, not cadence inference.
"""

"""Portable standards, numerical primitives and shared errors.

Import each owner module directly; the package re-exports nothing, so that
``_numerics`` (NumPy/SciPy only) can be used without loading Astropy.

* ``physical_constants``: canonical owner of the IAU/CODATA standards and
  unit conversions (loads Astropy).
* ``constants``: short CGS aliases (``Msun``, ``Rsun``, ``G`` ...) over those
  standards, plus the Gaia DR2 close-pair resolution assumption.
* ``_numerics``: log-space evidence reductions, ESS and box quadrature.
* ``errors``: unresolved-input and numerical-likelihood exceptions.
"""

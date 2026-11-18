"""Pin positive-weight sparse-corner refusal, not rectangular-grid invention."""

import numpy as np
import pytest

from trex.stellar_light.interpolation import (
    PhotonAtmosphereGrid, UnsupportedAtmosphereError, interpolate_photons,
)


def _quadratic_table(axis):
    value = (axis/100.)**2
    return np.stack((value, np.ones_like(value), np.zeros_like(value), np.zeros_like(value)),
                    axis=-1)[:, None, None, None]


def test_uniform_axis_bracket_reproduces_linear_interpolation_at_knots_and_midpoints(monkeypatch):
    """A quadratic table exposes a one-cell bracket error that a linear table hides."""
    axis = np.array([100., 200., 300., 400.])
    table = _quadratic_table(axis)
    axes = (axis, np.array([0.]), np.array([0.]), np.array([0.]))
    points = np.r_[axis, np.nextafter(axis[1:-1], np.inf), np.nextafter(axis[1:-1], -np.inf),
                   [150., 250., 350.]]
    expected = np.interp(points, axis, (axis/100.)**2)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a uniform axis must not run searchsorted")

    monkeypatch.setattr(np, "searchsorted", refuse)
    values, valid = interpolate_photons(axes, table, (points, 0., 0., 0.))
    assert np.all(valid)
    np.testing.assert_allclose(values[:, 0], expected, rtol=1e-13, atol=0.)


def test_nonuniform_axis_keeps_the_search_bracket(monkeypatch):
    axis = np.array([100., 175., 300., 400.])
    table = _quadratic_table(axis)
    axes = (axis, np.array([0.]), np.array([0.]), np.array([0.]))
    points = np.array([100., 140., 175., 240., 300., 350., 400.])
    expected = np.interp(points, axis, (axis/100.)**2)
    calls = []
    original = np.searchsorted

    def counting(*args, **kwargs):
        calls.append(np.asarray(args[0]).shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", counting)
    values, valid = interpolate_photons(axes, table, (points, 0., 0., 0.))
    assert calls and np.all(valid)
    np.testing.assert_allclose(values[:, 0], expected, rtol=1e-13, atol=0.)


@pytest.mark.parametrize("compiled", [False, True])
def test_native_affine_bracket_matches_numpy_at_knots_and_nextafter(compiled):
    from trex.accelerators import _jax_models as jm

    if not jm.HAS_JAX_TREX:
        pytest.skip("JAX float64 backend is unavailable")
    axis = np.array([100., 200., 300., 400.])
    table = _quadratic_table(axis)
    axes = (axis, np.array([0.]), np.array([0.]), np.array([0.]))
    points = np.r_[axis, np.nextafter(axis[1:-1], np.inf), np.nextafter(axis[1:-1], -np.inf),
                   [150., 250., 350.]]
    def evaluate(coordinates):
        return interpolate_photons(
            tuple(jm.jnp.asarray(a) for a in axes), jm.jnp.asarray(table),
            (coordinates, 0., 0., 0.), xp=jm.jnp)

    operation = jm.jax.jit(evaluate) if compiled else evaluate
    values, valid = operation(jm.jnp.asarray(points))
    assert np.all(np.asarray(valid))
    np.testing.assert_allclose(np.asarray(values[:, 0]), np.interp(points, axis, (axis/100.)**2),
                               rtol=1e-13, atol=0.)


def test_actual_nodes_only_and_flux_identity_through_interpolation():
    axes = (np.array([3000., 3100.]), np.array([4.5, 5.]), np.array([0.]), np.array([0.]))
    table = np.ones((2, 2, 1, 1, 4))
    table[..., 1:] = [1., -.3, -.1]
    table[..., 0] = np.pi*(1-.3/3-.1/6)
    table[1, 1] = np.nan
    grid = PhotonAtmosphereGrid(axes, table, {"source": "synthetic"})
    value = grid.evaluate(3000., 4.5, 0., 0.)
    assert value[0] == pytest.approx(np.pi*(value[1]+value[2]/3+value[3]/6))
    for point in [(3050., 4.75), (3100., 5.), (2999., 4.5)]:
        with pytest.raises(UnsupportedAtmosphereError):
            grid.evaluate(*point, 0., 0.)

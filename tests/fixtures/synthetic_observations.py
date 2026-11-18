"""Deterministic, target-neutral synthetic TREX inputs for tests only.

Copied from EXONYM's fixture module when the TREX-only tests moved here; the
TREX package never imports EXONYM, so these fixtures stay exonym-free.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .independent_transit_model import exposure_averaged_flux
from trex.foundation.physical_constants import SECONDS_PER_DAY


SYNTHETIC_SOLAR_METALLICITY_DEX = float(Fraction(0, 1))


def synthetic_portable_trex_inputs(directory):
    """The packaged low-information example inputs, so test and example cannot diverge."""
    from trex.app.example import synthetic_inputs

    return synthetic_inputs(directory)


def synthetic_trex_tp_integrand_case():
    """Target-neutral box photometry for scalar/native-integrand comparisons.

    The observation is analytic, not an implementation-generated reference.
    Expected physical transforms/likelihoods are evaluated independently in
    the test through the retained SciPy priors and Batman production path.
    Times/exposure are days from conjunction in TDB; flux/errors are fractional.
    """
    time = np.linspace(-float(Fraction(1, 8)), float(Fraction(1, 8)), 33)
    depth = float(Fraction(1, 1000))
    flux = 1 - (np.abs(time) < float(Fraction(1, 24))) * depth
    sigma = np.linspace(float(Fraction(1, 2000)), float(Fraction(1, 1000)), time.size)
    return {
        "time": time, "flux": flux, "sigma": sigma,
        "period_days": float(Fraction(2, 1)),
        "exptime_days": float(Fraction(1, 1000)),
        "u1": float(Fraction(1, 4)), "u2": float(Fraction(1, 8)),
    }


def synthetic_trex_dense_tp_integrand_case():
    """Same analytic box observation on a denser grid for real block scheduling.

    This computational fixture creates several immutable evaluation blocks
    without injecting a different forward model or an external observation.
    Time/exposure: TDB days from conjunction; flux/errors: fractional.
    """
    scene = synthetic_trex_tp_integrand_case()
    time = np.linspace(-float(Fraction(1, 8)), float(Fraction(1, 8)), 513)
    scene.update(time=time,
                 flux=1 - (np.abs(time) < float(Fraction(1, 24))) * float(Fraction(1, 1000)),
                 sigma=np.linspace(float(Fraction(1, 2000)), float(Fraction(1, 1000)), time.size))
    return scene


def synthetic_trex_eb_integrand_case(period_days):
    """Analytic primary/secondary box photometry for full native EB parity.

    TEST_DESIGN: two time windows expose both independently limb-darkened
    eclipsed bodies; depths are rational fixtures, not a generated truth model.
    Time/exposure are TDB days from primary conjunction, flux/errors fractional.
    """
    scene = synthetic_trex_tp_integrand_case()
    half_period = float(period_days) * float(Fraction(1, 2))
    secondary = scene["time"] + half_period
    scene["time"] = np.concatenate((scene["time"], secondary))
    scene["flux"] = np.concatenate((scene["flux"],
        1.0 - (np.abs(secondary - half_period) < float(Fraction(1, 24))) * float(Fraction(1, 2000))))
    scene["sigma"] = np.tile(scene["sigma"], 2)
    scene.pop("period_days")
    scene["effective_period"] = float(period_days)
    return scene


def synthetic_trex_background_integrand_population():
    """TEST_DESIGN: three unrelated stars spanning the retained radius-prior branches.

    These are explicit target-neutral table inputs, not catalog observations or
    a stellar calibration. Ordering is deliberately non-monotone so the real
    background-population constructor must retain its stable magnitude sort.
    """
    return {
        "tess_mag": np.asarray([13.0, 12.0, 14.0]),
        "mass_solar": np.asarray([0.6, 1.0, 0.3]),
        "radius_solar": np.asarray([0.6, 1.0, 0.3]),
        "teff_k": np.asarray([4000.0, 5750.0, 3500.0]),
        "mh_dex": np.asarray([-1.0, 0.5, 0.0]),
    }


def synthetic_batch_likelihood_case(
    *,
    period_days=5.0,
    impact_parameter=0.5,
    a_over_rs=10.0,
    secondary_radius_solar=0.1,
    eb_fluxratio=0.01,
    exptime_days=1.0 / 1440.0,
    nsamples=1,
):
    """A valid target-neutral EB/TP geometry for batch execution tests.

    Execution tests only need finite physical arrays and the common per-draw
    parameters; this is not an observation or a calibration input. Tests that
    assert a forward-model value carry their own analytic oracle.
    """
    from trex.foundation.constants import Rsun

    time = np.linspace(-0.1, 0.1, 41)
    flux = np.ones_like(time)
    return time, flux, {
        "R_EB_solar": secondary_radius_solar,
        "R_s_solar": 1.0,
        "EB_fluxratio": eb_fluxratio,
        "P_orb": period_days,
        "inc_deg": math.degrees(math.acos(impact_parameter / a_over_rs)),
        "a_cm": a_over_rs * Rsun,
        "u1": 0.3,
        "u2": 0.2,
        "u1_EB": 0.25,
        "u2_EB": 0.2,
        "exptime_days": exptime_days,
        "nsamples": nsamples,
    }


def synthetic_neighbor_preflight_case(root, *, neighbor_fraction, characterized=False,
                                      prepend_characterized=False):
    """Tiny real-engine input; the neighbor's flux bounds any possible eclipse.

    TEST_DESIGN: the injected fractional depth is exactly 1/512; the retained
    population and solar host are explicit synthetic inputs, not fitted truth.
    """
    import hashlib

    from trex.app.scene import TargetScene
    from tests.fixtures.trex_scene import caller_provenance

    population = root / "preflight_population.csv"
    population.write_text("TESS,logTe,logg,Mact,[M/H]\n13.0,3.70,4.5,0.8,0.0\n", encoding="utf-8")
    neighbor = {
        "source_id": "synthetic-uncharacterized", "M_s": 0.8 if characterized else None,
        "R_s": 0.8 if characterized else None, "delta_mag": 2.0,
        "separation_arcsec": 1.0,
        "stellar_parameter_unavailable_reason": None if characterized else "synthetic missing stellar data",
        "aperture_flux_fractions": [{"sector": 1, "fraction": neighbor_fraction}],
    }
    neighbors = [neighbor]
    if prepend_characterized:
        neighbors.insert(0, {**neighbor, "source_id": "synthetic-characterized", "M_s": 0.8, "R_s": 0.8})
    scene = TargetScene(
        tic_id=123, ra_deg=10.0, dec_deg=20.0, M_s_Msun=1.0, R_s_Rsun=1.0,
        Teff_K=5750.0, Tmag=11.5, plx_mas=10.0, feh_dex=0.0, sectors=[1],
        metallicity_provenance=caller_provenance(0.0),
        contrast_separations=np.array([0.1, 1.0]), contrast_values=np.array([2.0, 5.0]),
        target_aperture_flux_fractions=[{"sector": 1, "fraction": 1.0 - len(neighbors) * neighbor_fraction}],
        resolved_neighbors=neighbors, N_background=1, trilegal_cache=population,
        background_sha256=hashlib.sha256(population.read_bytes()).hexdigest(), background_area_sq_deg=0.1,
    )
    time = np.linspace(-0.2, 0.2, 21)
    flux = 1.0 - (np.abs(time) < 1 / 16) / 512
    return scene, {
        "time": time, "flux": flux, "sigma": 1 / 4096,
        "period_days": 2.0, "depth_ppm": 1e6 / 512, "exptime_days": 1 / 1024,
        "sector_weights": {1: 1.0}, "n_draws": 8, "random_seed": 7, "nsamples": 1,
        "escalation_checkpoint_directory": str(root / "escalation"),
    }


def synthetic_secondary_ldc_case(*, companion_is_host=False, radial_nodes=128):
    """Grazing circular EB, independently integrated with each body's grid LDC.

    TEST_DESIGN: exact rational radii, orbit, and flux fractions deliberately
    expose the companion's intensity profile during a partial occultation.
    The expected flux uses the independent polar-disc quadrature, not TREX.
    """
    from trex.foundation.constants import G, Msun, Rsun
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    host_radius = float(Fraction(1, 1))
    secondary_radius = float(Fraction(1, 2))
    host_mass = float(Fraction(1, 1))
    secondary_mass = float(Fraction(1, 2))
    grid = load_triceratops_tess_grid()
    host = grid.select(5750, math.log10(G * host_mass * Msun / (host_radius * Rsun)**2), 0)
    secondary = grid.select(3500, math.log10(G * secondary_mass * Msun / (secondary_radius * Rsun)**2), 0)
    period = float(Fraction(5, 1))
    a_rs = float(Fraction(10, 1))
    impact = float(Fraction(5, 4))
    exposure = float(Fraction(1, 1000))
    offsets = np.linspace(-float(Fraction(1, 10)), float(Fraction(1, 10)), 21)
    time = np.concatenate((offsets, period / 2 + offsets))
    k = secondary_radius / host_radius

    def disc_flux(local_time, radius_ratio, axes, b, node):
        return exposure_averaged_flux(
            tuple(local_time), period, radius_ratio, axes, b,
            (node.u1 + node.u2)**2, node.u1 / (2 * (node.u1 + node.u2)),
            exposure * SECONDS_PER_DAY, subsamples=1, radial_nodes=radial_nodes,
        )

    primary = disc_flux(time, k, a_rs, impact, host)
    occulted_secondary = disc_flux(time - period / 2, 1 / k, a_rs / k, impact / k, secondary)
    eb_fraction = float(Fraction(1, 4))
    third_fraction = float(Fraction(1, 3))
    f_eb = eb_fraction / (1 - eb_fraction)
    f_third = third_fraction / (1 - third_fraction)
    w_host, w_other = (f_third, 1) if companion_is_host else (1, f_third)
    expected = (w_host * primary + f_eb * occulted_secondary + w_other) / (w_host + f_eb + w_other)
    return time, expected, {
        "R_EB_solar": secondary_radius, "EB_fluxratio": eb_fraction,
        "P_orb": period, "inc_deg": math.degrees(math.acos(impact / a_rs)),
        "a_cm": a_rs * host_radius * Rsun, "R_s_solar": host_radius,
        "u1": host.u1, "u2": host.u2, "u1_EB": secondary.u1, "u2_EB": secondary.u2,
        "exptime_days": exposure, "nsamples": 1,
        "companion_fluxratio": third_fraction, "companion_is_host": companion_is_host,
    }

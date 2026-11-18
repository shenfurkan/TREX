"""Shared isolated, headless TREX test settings."""

import os
from pathlib import Path

import pytest

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_configure(config):
    """Retain per-test durations in remote runner logs for runtime ranking.

    The Kaggle runner exports ``EXONYM_TEST_KIT_SHA256`` for every target; a
    local run keeps its normal short output.
    """
    if os.environ.get("EXONYM_TEST_KIT_SHA256") and getattr(config.option, "durations", None) is None:
        config.option.durations = 60


def pytest_collection_modifyitems(items):
    root = Path(__file__).parent
    for item in items:
        relative = item.path.relative_to(root).as_posix()
        if (item.path.stem.startswith("test_trex")
                or relative.startswith(("trex/accelerators/", "trex/app/", "trex/evidence/"))
                or relative in {
                    "trex/foundation/test_engine_numerics.py",
                    "trex/models/test_engine_primitives.py",
                    "trex/models/test_per_point_sigma.py",
                    "trex/photometry/test_baseline_consumers.py",
                    "trex/populations/test_companion_population.py",
                    "trex/populations/test_engine_priors.py",
                    "trex/stellar_light/test_evidence_adapters.py",
                    "trex/stellar_light/test_host_limb_darkening.py",
                }):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Keep retry attempts while avoiding real network backoff in tests."""
    from urllib3.util.retry import Retry

    monkeypatch.setattr(Retry, "sleep", lambda self, response=None: None)


@pytest.fixture(autouse=True)
def _trex_recovery_directory(monkeypatch, tmp_path_factory):
    monkeypatch.setenv(
        "TREX_ESCALATION_CHECKPOINT_DIRECTORY",
        str(tmp_path_factory.mktemp("trex-escalation")),
    )


@pytest.fixture
def field_extinction():
    """The synthetic dust-map record a TRILEGAL test submits as extinction_infty."""
    return {"status": "available", "map": "sfd98-ebv", "map_file": "trilegal_field_extinction.fits",
            "retrieved_utc": "2026-11-10T00:00:00Z",
            "ebv_sfd": {"max": 0.18235, "centre": 0.18235, "mean": 0.18235, "min": 0.18235, "pixels": 1},
            "av_infinity_mag": 0.5}


@pytest.fixture(autouse=True)
def _offline_field_extinction(monkeypatch, field_extinction):
    """No TRILEGAL test contacts the dust-map service; the field value is stated."""
    from trex.acquisition import trilegal

    def synthetic(*_args, **_kwargs):
        return field_extinction

    monkeypatch.setattr(trilegal, "field_extinction_from_map", synthetic)


@pytest.fixture(autouse=True)
def _offline_lsst_sim_fallback(monkeypatch):
    """The background-population fallback never reaches the Astro Data Lab in tests."""
    from trex.acquisition import background_population
    from trex.acquisition.trilegal_errors import TrilegalUnavailableError

    def unreachable(*_args, **_kwargs):
        raise TrilegalUnavailableError("offline tests do not contact the Astro Data Lab")

    monkeypatch.setattr(background_population, "acquire_lsst_sim_population", unreachable)


@pytest.fixture(autouse=True)
def _offline_gaia_notifications(monkeypatch):
    from astroquery.utils.tap.conn.tapconn import TapConn

    original = TapConn.execute_tapget

    def without_bulletin(self, context, *args, **kwargs):
        if context == "notification?action=GetNotifications":
            raise OSError("offline tests do not retrieve Gaia service bulletins")
        return original(self, context, *args, **kwargs)

    monkeypatch.setattr(TapConn, "execute_tapget", without_bulletin)

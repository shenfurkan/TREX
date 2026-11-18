"""Background population for the unresolved scenarios: TRILEGAL first, a recorded fallback second.

The TRILEGAL 1.6 web service is the primary source, with the field submitted
in Galactic degrees (``trex.acquisition.trilegal``). When it cannot supply a
population, the same field is taken from the TRILEGAL LSST simulation
(``trex.acquisition.lsst_sim``) inside that simulation's footprint; outside it
the TRILEGAL failure is raised unchanged in meaning. A fallback population
says so in its record and is never treated as current: the next acquisition
tries TRILEGAL again.
"""

import copy
import json
import math
from pathlib import Path

import requests

from trex.runtime.logging import host_logger

from .lsst_sim import FIELD_SUBMISSION as LSST_SIM_FIELD_SUBMISSION
from .lsst_sim import MODEL as LSST_SIM_MODEL
from .lsst_sim import RECORD_FILE_NAME as QUERY_RECORD_FILE_NAME
from .lsst_sim import acquire_lsst_sim_population
from .storage import write_bytes
from .trilegal import STAGE_RECORD_FILE_NAME, acquire_trilegal, is_galactic_field_record
from .transport import AcquisitionError
from .trilegal_errors import TrilegalUnavailableError

# The same-cone count deficit is measured; its effect on a particular FPP is not.
# The scenarios with an unresolved background star -- DTP/DEB/DEBx2P (eclipse on
# the target, diluted by the background star) and BTP/BEB/BEBx2P (the background
# star is the host) -- carry log(star_count / area * pi * radius^2) in their
# evidence (``background_population_expectation``). The FPP counts DTP as a
# planet, like the target-hosted TP and companion-hosted PTP, while the bound-
# companion S* scenarios do not use the population at all, so scaling the
# background evidence can move the normalized FPP either way.
FALLBACK_LIMITATION = (
    "background population from the TRILEGAL LSST simulation fallback; its same-cone star counts can "
    "fall below Gaia DR3 counts, but the effect of that deficit on a particular FPP is not measured: "
    "the background-hosted scenarios draw from it and are weighted by its star count, while the "
    "normalized FPP also contains scenarios that do not, so a fallback-derived FPP is not "
    "interchangeable with one from the TRILEGAL web service"
)


def record_simulates_target_field(record):
    """Return whether a population record was simulated at its target's field.

    The fallback model and the web model use different field records; a record
    that claims the fallback flag without the fallback model is inconsistent and
    is not admitted as either.
    """
    if not isinstance(record, dict):
        return False
    if record.get("model") == LSST_SIM_MODEL:
        return record.get("field_submission") == LSST_SIM_FIELD_SUBMISSION
    if record.get("fallback") is True:
        return False
    return is_galactic_field_record(record)


def field_attenuation_envelope(record):
    """Return the upper end (V-band mag) of the field attenuation envelope a population record states.

    A TRILEGAL web record carries the field extinction its submitted form was
    normalized with (``field_extinction``, the SFD98 map value at the field); a
    fallback record carries the largest per-star extinction of its field. A
    record stating neither raises ``ValueError``.
    """
    if not isinstance(record, dict):
        raise ValueError("background population record is not an object")
    if record.get("model") == LSST_SIM_MODEL:
        value = record.get("field_av_max")
    else:
        extinction = record.get("field_extinction")
        if not isinstance(extinction, dict) or extinction.get("status") != "available":
            raise ValueError("TRILEGAL record does not state the field's map-derived extinction")
        value = extinction.get("av_infinity_mag")
    if isinstance(value, bool):
        raise ValueError("field extinction must be real, not boolean")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("field extinction is not numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("field extinction must be finite and nonnegative")
    return value


def is_fallback_record(record):
    """Return whether a population record identifies the fallback provider.

    The fallback acquisition names ``model: trilegal-lsst-sim-dr2`` and writes
    ``fallback: true``. Either marker is enough to keep a fallback population
    from being published as a web result; ``fallback_provenance`` refuses a
    record that claims the flag without the fallback model.
    """
    if not isinstance(record, dict):
        return False
    return record.get("model") == LSST_SIM_MODEL or record.get("fallback") is True


def fallback_provenance(record):
    """Return the provenance a result record carries for a fallback population, or None.

    A TRILEGAL web population returns None so its inference record stays
    unchanged. The projection carries the population digest, so a portable scene
    can be checked against the file it was built from. The limitation states the
    measured same-cone count deficit and that its effect on a particular FPP is
    not measured; it does not assert a bias direction.
    """
    if not is_fallback_record(record):
        return None
    model = record.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("fallback population record does not state its model")
    if model != LSST_SIM_MODEL:
        raise ValueError("fallback population record does not name the fallback model")
    # Also accepts its own projection (a portable scene's input), which has no
    # field record: the field is checked where the full record is read
    # (``retained_fallback_record``); the caller compares this digest with its file.
    digest = record.get("population_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("fallback population record does not name its population digest")
    return {"model": model, "fallback": True,
            "population_sha256": record.get("population_sha256"),
            "fallback_trigger": copy.deepcopy(record.get("fallback_trigger")),
            "limitation": FALLBACK_LIMITATION}


def retained_fallback_record(population_path, population_sha256):
    """Return the fallback record retained beside a population file, or raise.

    The acquisition writes ``trilegal_query.json`` beside ``trilegal_cache.csv``.
    A missing or web record returns None, so a TRILEGAL web scene stays exactly
    as it was. A fallback record that does not simulate the target field or does
    not name this population is refused rather than silently treated as a web
    population; so is an unreadable or malformed record, which cannot show
    which provider produced the population.
    """
    path = Path(population_path).parent / QUERY_RECORD_FILE_NAME
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"background population record {path.name} is unreadable: {exc}") from exc
    if not is_fallback_record(record):
        return None
    if not record_simulates_target_field(record):
        raise ValueError("fallback population record does not simulate the target field")
    if record.get("population_sha256") != population_sha256:
        raise ValueError("fallback population record does not name the retained population")
    return record


def _trigger(output_dir, exc):
    """The stage record ``acquire_trilegal`` wrote for this failure, else a minimal one.

    A stage record left by an earlier attempt describes a different failure, so
    it is used only when it names this exception's class and message.
    """
    try:
        record = json.loads((Path(output_dir) / STAGE_RECORD_FILE_NAME).read_text())
    except (OSError, ValueError):
        record = None
    error = record.get("error") if isinstance(record, dict) else None
    if (isinstance(error, dict) and record.get("status") == "unavailable"
            and error.get("exception_class") == type(exc).__name__
            and error.get("message") == str(exc).strip()):
        return record
    return {"status": "unavailable", "error": {"exception_class": type(exc).__name__, "message": str(exc)}}


def acquire_background_population(ra_deg, dec_deg, output_dir, policy, *, query_runner=None, candidate_id=None,
                                  record_prefix=Path(), writer=write_bytes):
    """Return ``(population_csv, record_json)`` from TRILEGAL, or from the fallback when TRILEGAL fails.

    ``query_runner`` is TRILEGAL's query seam and receives
    ``(form_overrides, field_extinction)``: the field's dust-map record is
    acquired (or resumed) by ``acquire_trilegal`` before the runner is called,
    so the submitted form and the retained record always agree.

    Transport failures reach this point after the session's own bounded
    retries, so they are final for this attempt like a typed outage. When the
    fallback cannot serve the field either, the original TRILEGAL exception
    is raised unchanged (callers keep their retry contract), with the fallback
    failure attached as ``fallback_failure`` and written to the stage record.
    """
    try:
        return acquire_trilegal(ra_deg, dec_deg, output_dir, policy, query_runner=query_runner,
                                candidate_id=candidate_id, record_prefix=record_prefix, writer=writer)
    except (TrilegalUnavailableError, requests.RequestException) as exc:
        trilegal_failure = exc
    host_logger(__name__).warning("TRILEGAL unavailable (%s); trying the LSST-simulation fallback",
                                  type(trilegal_failure).__name__)
    trigger = _trigger(output_dir, trilegal_failure)
    try:
        result = acquire_lsst_sim_population(
            ra_deg, dec_deg, output_dir, policy, trigger=trigger,
            candidate_id=candidate_id, record_prefix=record_prefix, writer=writer)
    except AcquisitionError as exc:
        fallback_failure = str(exc)
        host_logger(__name__).warning("LSST-simulation fallback unavailable too; TRILEGAL error is raised")
    else:
        host_logger(__name__).warning("background population is the recorded LSST-simulation fallback")
        return result
    trilegal_failure.fallback_failure = fallback_failure
    stage_path = Path(output_dir) / STAGE_RECORD_FILE_NAME
    if trigger.get("stage") is not None and stage_path.is_file():
        stage = dict(trigger, fallback={"model": LSST_SIM_MODEL, "status": "unavailable",
                                        "message": fallback_failure})
        writer(stage_path, (json.dumps(stage, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    # Raised outside the handler so its own cause chain stays exactly as TRILEGAL produced it.
    raise trilegal_failure

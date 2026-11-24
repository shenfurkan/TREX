"""Host-owned logging hooks and an explicit standalone CLI lifecycle.

Portable callers use the ``trex`` namespace. Embedded callers explicitly supply
their root namespace so the host's handlers, context and redaction stay active.
Importing this module configures nothing. Logs are not scientific evidence.
Unknown messages/fields are visibly omitted. Reviewed message templates and
warning summaries below must be updated with their producers; source modules,
function names and exception types are validated structurally (the ``trex``
namespace, identifier syntax, and live exception classes), so new code needs no
allowlist edit. Paths and arbitrary payloads never become safe merely because
they came from a package module.
"""

import argparse
import errno
import importlib
import itertools
import io
import json
import logging
import math
import multiprocessing.util
import numbers
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import warnings
from datetime import datetime, timezone
from multiprocessing.pool import RemoteTraceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

#: Host-supplied ``() -> (initializer, args)``; TREX worker pools run it in each worker.
_worker_log_setup: Optional[Callable[[], Tuple[Any, Tuple[Any, ...]]]] = None

#: Host-supplied ``(drain, adopt)`` for warnings captured inside pool workers.
_worker_warning_hooks: Optional[
    Tuple[Callable[[], List[Dict[str, Any]]], Callable[[List[Dict[str, Any]]], None]]
] = None

#: The namespace an embedded TREX shares with its host; standalone TREX keeps ``trex``.
_host_namespace: str = "trex"

#: One worker's transport payload is bounded by the host's own capture cap.
WORKER_WARNING_PAYLOAD_LIMIT = 50
_EXCEPTION_FRAME_LIMIT = 64
_EXCEPTION_CHAIN_LIMIT = 8
_EXCEPTION_NOTE_LIMIT = 16
_WORKER_WARNING_BYTES_LIMIT = 1_000_000

# Only operational summaries cross this boundary, never arbitrary ``extra`` payloads.
_METRIC_FIELDS = frozenset({
    "stage", "phase", "status", "category", "error_type", "next_action", "scenario", "stop_reason", "reason",
    "n_draws", "draws", "nsamples", "n_jobs", "random_seed", "budget", "max_likelihood_calls",
    "n_likelihood_calls", "ess", "ESS", "ln_evidence", "FPP", "NFPP", "fpp", "nfpp", "wall_seconds",
    "elapsed_seconds", "remaining_seconds", "draw_seconds", "wait_seconds", "evaluate_seconds",
    "cpu_draws_per_second", "accelerator_draws_per_second",
    "cpu_reference_overlap", "cpu_timing_condition", "accelerator_timing_condition",
    "kernel", "pilot", "chosen", "device", "warning_module", "warning_category", "warning_location",
    "warning_summary", "count",
    "exit_code", "command",
})
_PILOT_FIELDS = frozenset({
    "device", "draws", "seconds", "draws_per_second", "reference", "eligible", "compile_seconds", "parity",
})
_PARITY_FIELDS = frozenset({"same_nonfinite_pattern", "max_abs_lnL_difference", "tolerance", "passed"})
_EVENTS = frozenset({
    "trex.cli.start", "trex.cli.stopped", "trex.cli.finish", "trex.worker.start", "trex.worker.stop",
    "trex.doctor.inputs", "trex.doctor.mast", "trex.engine.start", "trex.engine.stopped", "trex.engine.finish",
    "trex.engine.admission", "trex.engine.phase", "trex.target.refused", "trex.run.refused", "trex.run.deferred",
    "trex.batch.retry", "trex.batch_plan", "trex.device_pilot", "trex.scenario", "trex.nested", "trex.nested.failed",
    "warning.captured", "warning.worker", "warning.setup.degraded", "warning.transport.unavailable",
    "warning.transport.marker_unavailable", "warning.transport.report_failed", "warning.transport.cleanup_failed",
    "warning.transport.lost",
    "trex.hardware.admission", "trex.hardware.compile_failed", "trex.hardware.fallback",
    "trex.hardware.lease_cleanup",
    "stellar_light.download", "stellar_light.preparation",
})
_DEVICE_ID = re.compile(r"(?:(?:cpu|gpu|cuda|rocm|tpu)(?::[0-9]{1,10})?|TFRT_CPU_[0-9]{1,10})", re.ASCII)
_REMOTE_FRAME = re.compile(r'^  File "([^"\r\n]+)", line ([0-9]{1,10}), in ([^\r\n]+)$')
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
# A TREX module identifier is a namespace name, never a filesystem path.
_TREX_SOURCE_FILE = re.compile(r"trex(?:/[A-Za-z_][A-Za-z0-9_]*)+\.py", re.ASCII)
# Function names are code identifiers with this documented safe syntax.
_FUNCTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}", re.ASCII)
# Exception names are code identities: a live ``BaseException`` subclass or the
# conventional suffix of a class from a module this process has not imported.
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}(?:Error|Exception)", re.ASCII)
_LOGGERS = frozenset({
    "trex", "trex.cli", "trex.worker", "trex.warnings", "trex.engine", "trex.target", "trex.doctor",
    "trex.run", "trex.hardware", "trex.acquisition.aperture", "trex.acquisition.gspspec", "trex.acquisition.transport",
    "trex.populations", "trex.populations.background", "trex.populations.companion_imaging",
    "trex.populations.companion_priors", "trex.populations.mass_ratio_priors",
    "trex.populations.priors", "trex.populations.spatial_prior",
    "trex.runtime", "trex.acquisition.tic", "trex.acquisition.tpf",
    "trex.models", "trex.photometry",
    "trex.acquisition.trilegal", "trex.acquisition.background_population",
    "trex.stellar", "trex.stellar_light", "trex.stellar_light.release", "trex.stellar_light.preparation",
})
_WARNING_TYPES = frozenset({
    "Warning", "UserWarning", "RuntimeWarning", "DeprecationWarning", "PendingDeprecationWarning", "FutureWarning",
    "ImportWarning", "ResourceWarning", "BytesWarning", "EncodingWarning", "SyntaxWarning", "UnicodeWarning",
    "AstropyWarning", "AstropyDeprecationWarning", "AstropyUserWarning", "UnitsWarning", "VerifyWarning",
    "ErfaWarning", "IERSWarning", "OptimizeWarning", "WorkerWarningLoss", "WorkerWarningCaptureLimit",
    "WorkerWarningTransportUnavailable", "WarningCaptureLimit",
})
_TEXT_VALUES = {
    "stage": frozenset({"full", "resume", "input-admission", "input-loading", "input-binding", "target-api",
                        "target-binding", "checkpoint", "evidence", "execution"}),
    "phase": frozenset({"full", "resume", "nested", "input-admission", "monte-carlo", "mc-importance",
                        "nested-escalation", "mc_importance", "nested_escalation"}),
    "status": frozenset({"ok", "ready", "finished", "restored", "paused", "failed", "refused", "deferred",
                         "not-run", "anomaly", "all_neginf", "degenerate_evidence"}),
    "category": frozenset({"paused", "numerical-failure", "convergence", "missing-evidence", "invalid-input",
                           "execution-error", "interrupted", "WorkerWarningLoss", "WorkerWarningTransportUnavailable",
                           "provider_error", "rate_limit", "http_status", "tls_verification", "dns", "tls_transport",
                           "connect_timeout", "incomplete_transfer", "timeout", "probe-error", "application-error"}),
    "next_action": frozenset({
        "Resume the retained checkpoint with the same inputs and budget.",
        "Repair the failed supported likelihood; retain every scenario.",
        "Inspect the recorded scenario failures, repair the supported likelihood and retain every scenario.",
        "Inspect retained scenario convergence; do not lower the scientific budget.",
        "Supply the recorded required evidence or supported model configuration.",
        "Correct the supplied inputs or unsupported option before retrying.",
        "Inspect the original traceback and repair the failing production path.",
        "Inputs are admissible; run TREX to assess evidence and convergence.",
    }),
    "stop_reason": frozenset({"evidence-tolerance", "likelihood-plateau", "truncated", "unresolved", "unknown",
                              "nested-importance-sampling", "mc-pilot-defensive-importance"}),
    "cpu_timing_condition": frozenset({"previous-scenario-in-flight", "serial"}),
    "accelerator_timing_condition": frozenset({"serial-after-previous-scenario"}),
    "warning_module": frozenset({"astropy", "astroquery"}),
    "command": frozenset({"run", "doctor"}),
}
# Third-party frames are reviewed library identifiers, not filesystem paths.
_LIBRARY_FILES = frozenset({
    "dynesty/sampler.py", "dynesty/dynamicsampler.py", "dynesty/internal_samplers.py", "dynesty/bounding.py",
    "dynesty/utils.py", "multiprocessing/pool.py", "multiprocessing/process.py", "multiprocessing/spawn.py",
    "multiprocessing/util.py", "concurrent/futures/thread.py", "concurrent/futures/process.py", "contextlib.py",
    "json/__init__.py", "json/decoder.py", "threading.py", "warnings.py", "pathlib.py", "runpy.py",
    "ctypes/__init__.py", "ctypes/util.py", "jax/_src/xla_bridge.py", "jax/_src/compiler.py",
    "jax/_src/dispatch.py", "jax/_src/api.py", "numpy/linalg/_linalg.py", "numpy/linalg/linalg.py",
    "astropy/io/votable/table.py", "astropy/io/votable/tree.py", "astropy/utils/exceptions.py",
})
_SOURCE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
_LIBRARY_ROOTS = (os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix))
_SCENARIOS = frozenset({
    "TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P",
    "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P", "NTP", "NEB", "NEBx2P", "host",
})
_DEVICE_REASONS = frozenset({
    "eligible accelerators beat the CPU at float64 parity",
    "no accelerator passed float64 parity",
    "CPU throughput >= summed eligible accelerator throughput",
    "accelerator admission incomplete", "allocator memory probe failed",
    "available device memory is unknown", "driver memory probe unavailable",
    "driver memory counter unavailable", "driver identity unavailable",
    "previous driver device could not be restored",
    "allocation exhausted memory", "compiler exhausted memory", "pipeline exhausted memory",
    "capacity counter unavailable; single block only",
    "scheduler budget below immutable granule; existing single-block floor",
    "live device/compiler/host capacity",
})
# These are audited operational reasons, not strings inferred safe from source
# constants: even a literal can contain observations or credentials.
_SAFE_EXCEPTION_REASONS = frozenset({
    "stellar upper bounds must match masses and be finite positive numbers",
    "stellar masses must be a finite real numeric 1-D array",
    "stellar masses outside retained spline-node domain [0.1, 40.0] Msun; no extrapolation",
    "stellar-relation radii must be finite and positive",
    "stellar-relation effective temperatures must be finite and positive",
    "proposal radii must be finite nonnegative real numbers",
    "main-sequence inversion requires a finite positive Teff",
    "main-sequence inversion failed its forward temperature check",
    "retained stellar relations must disclose their uncalibrated status",
    "atmosphere preparation requires a complete demand record",
    "atmosphere demand nodes and rows must be sequences",
    "atmosphere demand nodes must have three official coordinates",
    "atmosphere demand has an incomplete population support row",
    "atmosphere demand row nodes must be sequences",
    "atmosphere demand has an invalid sampling declaration",
    "requires one supported release and ordered nonnegative A_V knots",
    "stellar radius and temperature pieces must share mass knots",
    "secondary relation has nonpositive radius/temperature in its support",
    "atmosphere demand requires temperature, mass, radius, metallicity and extinction rows",
    "atmosphere demand needs every finite, aligned population row",
    "population temperatures/masses/radii must be positive and A_V nonnegative",
    "bound host indices must be an iterable of retained row indices",
    "bound host indices must refer to retained population rows",
    "atmosphere interpolation requires four axes and coordinates",
    "atmosphere interpolation axes must be finite and strictly increasing",
    "atmosphere coordinates must be real numeric values, not booleans",
    "atmosphere grids require real numeric arrays, not booleans",
    "angular ray tables require real numeric mu and intensity arrays",
    "invalid atmosphere grid or absent release provenance",
    "v3 photon atmospheres require their original angular profiles",
    "atmosphere grid has no nodes or a corrupt partial/nonfinite node",
    "nonpositive atmosphere photon light", "negative quadratic angular intensity",
    "quadratic intensity and surface photon flux are inconsistent",
    "angular ray tables do not match the atmosphere grid",
    "incomplete/corrupt angular photon profile",
    "angular intensity and surface photon flux are inconsistent",
    "this compiled atmosphere has no authentic angular profiles",
    "compiled atmosphere lacks required axes, table or source",
    "compiled atmosphere has an incomplete angular representation",
    "compiled atmosphere source must be scalar JSON text",
    "compiled atmosphere has an unsupported physical release",
    "compiled atmosphere does not match the explicitly selected stellar-light model",
    "requires actual nodes from one supported PHOENIX release",
    "different PHOENIX releases cannot fill one another's missing nodes",
    "spectrum is not an official SPECINT node", "duplicate atmosphere node",
    "one atmosphere node must share its original rays across extinction knots",
    "atmosphere nodes have incompatible ray counts",
    "angular quadrature order must be a positive integer",
    "angular occultation inputs must be real numeric values, not booleans",
    "occultor radius ratio must be a scalar",
    "occultor radius ratio must be finite and positive",
    "occultation separation must be nonnegative",
    "invalid complete angular photon profile",
    "requires an official scaled-solar PHOENIX specific-intensity release",
    "PHOENIX angular spectrum lacks valid required FITS fields",
    "unsupported PHOENIX radiation-field metadata",
    "SPECINT-2011 adapter supports only the scaled-solar release",
    "v3 specific intensities require GRIDVER=3, not a renamed older spectrum",
    "invalid PHOENIX angular spectrum",
    "unsupported PHOENIX release or published atmosphere coordinate",
    "explicit full-resolution acquisition is supported for the selected v3 release only",
    "retained atmosphere has no acquisition URL/date receipt",
    "retained atmosphere acquisition receipt is unreadable",
    "retained atmosphere acquisition receipt differs from the selected release/node",
    "downloaded atmosphere coordinates do not match the requested node",
    "atmosphere cache changed during acquisition; refusing replacement",
    "retained atmosphere coordinates do not match the requested node",
    "physical stellar light requires target A_V [mag]",
    "physical atmosphere requires its own composition and extinction",
    "physical atmosphere coordinates must be finite real numbers",
    "physical atmosphere needs finite positive host temperature, mass and radius",
    "atmosphere has no nonnegative quadratic reduction; consume its prepared original rays",
    "photon flux, radius and distance must be finite real numbers",
    "surface photon flux must be nonnegative; radius and distance must be positive",
    "physical background atmosphere requires retained TRILEGAL A_V (av_mag)",
    "physical background atmosphere requires retained tess_mag rows",
    "background av_mag must be finite, nonnegative and row aligned",
    "bound population and evidence integrand require the identical stellar_light input",
    "native stellar light requires JAX/jaxoplanet float64",
    "physical atmosphere requires stellar coordinates, composition and extinction",
    "physical atmosphere coordinates must be real numbers",
    "physical background atmosphere requires retained A_V",
    "invalid photon passband/extinction tabulation or missing source",
    "quadratic reduction requires finite ordered rays and aligned nonnegative intensities",
    "quadratic reduction integrates the whole disc: rays must span mu=0..1",
    "atmosphere has nonpositive photon intensity/flux",
    "flux-conserving quadratic law has negative intensity",
    "A_V must be finite and nonnegative",
    "photon integration requires a finite ordered angular spectrum",
    "spectrum does not cover the complete passband",
    "TESS photon response rows must contain wavelength and response",
    "TESS photon response must be in [0,1]",
    "binary light selection must be a 1-D boolean mask",
    "physical binary light requires both stellar coordinates and source composition/extinction",
    "limb-darkening supplement must not replace an existing node",
    "TRICERATOPS TESS grid lacks an exact host coefficient node",
    "teff_k must be positive",
    "Claret extension metallicity planes differ from the reference grid",
    "host atmosphere arrays must have matching per-draw shapes",
    "limb-darkening supplement must use a reference metallicity plane",
    "host atmosphere coordinates must be real numeric values, not booleans",
    "metallicity_dex must be scalar or match the host atmosphere shape",
    "metallicity_dex must be finite", "host atmosphere arrays must be finite",
    "vendored TRICERATOPS TESS limb-darkening grid is unavailable",
    "vendored TRICERATOPS TESS grid has unexpected columns",
    "vendored TRICERATOPS TESS grid has an invalid row",
    "vendored TRICERATOPS TESS grid is unreadable",
    "vendored TRICERATOPS TESS grid has no coefficient rows",
    "TESS limb-darkening supplement is unavailable",
    "TESS limb-darkening supplement has unexpected columns",
    "TESS limb-darkening supplement is unreadable",
    "TESS limb-darkening supplement has no nodes or repeats a node",
    "packaged PHOENIX-COND table has invalid source coordinates",
    "TESS limb-darkening supplement has negative quadratic intensity",
    "packaged Claret (2017) TESS limb-darkening table is unavailable",
    "packaged Claret (2017) grid has unexpected columns",
    "packaged Claret (2017) grid is unreadable",
    "packaged Claret (2017) grid has no coefficient rows",
    "packaged Claret (2017) grid has an unexpected model parameter",
    "packaged Claret (2017) grid repeats an atmosphere node",
    "Claret (2017) extension disagrees with a packaged reference node",
    "Claret (2017) extension does not match the packaged reference coverage",
    "limb-darkening provenance entries must be mappings",
    "importance worker initializer did not receive its block callable",
    "scalar importance likelihood returned NaN or +inf",
    "an adaptive importance pass returned NaN or +inf",
    "the importance-refinement likelihood returned NaN or +inf",
    "no importance draw landed on the likelihood support",
    "importance refinement needs at least one nested posterior",
    "no accelerator lease was admitted for MC-pilot importance",
    "no accelerator passed MC-pilot importance float64 parity",
    "planet integrand received non-finite parameters",
    "planet parameters do not match the scenario dimensions",
    "planet likelihood returned a non-finite value on supported geometry",
    "binary integrand received non-finite parameters",
    "binary parameters do not match the scenario dimensions",
    "binary likelihood returned a non-finite value on supported geometry",
    "nested wall deadline reached; sampler checkpoint retained",
    "nested wall deadline reached before sampling",
    "nested wall deadline reached before refinement",
    "escalation wall deadline reached; retained runs are resumable",
    "nested evidence ensemble is non-finite",
    "nested sampler checkpoint is corrupt or incomplete",
    "nested checkpoint inputs do not match this integration",
    "importance checkpoint inputs do not match this refinement",
    "importance checkpoint progress is incomplete or invalid",
    "nested wall deadline reached during importance adaptation",
    "importance checkpoint is missing its completed adaptation",
    "importance adaptation checkpoint has too many evaluated likelihoods",
    "adaptive importance likelihood must return one value per point",
    "importance adaptation checkpoint is corrupt or incomplete",
    "importance checkpoint is corrupt or incomplete",
    "nested result checkpoint is corrupt or incomplete",
    "nested sampler checkpoint has no valid monitored likelihood count",
    "invalid adaptive posterior",
    "invalid adaptive passes",
    "invalid adaptive likelihood count",
    "invalid adaptive exponent",
    "invalid adaptive ESS",
    "invalid adaptive completion state",
    "inconsistent adaptive completion state",
    "invalid adaptive likelihood prefix",
    "nested posterior requires finite unit-cube samples and normalized non-negative weights",
    "log weights must contain real numbers, not boolean or text",
    "scenario log-evidences must contain real numbers, not boolean or text",
    "photometry must contain real numbers, not boolean or text",
    "time must contain finite real numbers, not boolean or text",
    "time must be finite",
    "MAST transport returned no response and no failure record",
    "host escalation deadline must be a finite monotonic time",
    "prepared inputs must be regular files in a non-symlink directory",
    "no retained scenarios are available to plot",
    "atmosphere cache files must not be symlinks",
    "resumable escalation requires a candidate-owned checkpoint directory",
    "native JAX baseline marginalization is unsupported; explicitly select dynesty",
    "BlackJAX resumable escalation is unsupported; select dynesty explicitly",
    "TREX source files are unavailable for checkpoint identity",
    "companion flux fractions must be real numbers in [0, 1)",
    "companion flux fractions must be finite and in [0, 1)",
    "flux must be a non-empty finite real one-dimensional array",
    "flux uncertainty must be real, not boolean or text",
    "flux uncertainty must be finite and positive",
    "per-point flux uncertainty must match the flux array shape",
    "host aperture flux fraction must be a real number",
    "host aperture flux fraction must be finite and within (0, 1]",
    "companion_is_host must be boolean",
    "companion flux fraction must be scalar",
    "binary flux fractions must be scalar",
    "orbital period and total mass must be finite positive real numbers",
    "semi-major axis cannot be represented finitely and positively",
    "period, stellar mass and radius must be finite positive real scalars",
    "planet mass must be a finite nonnegative real scalar",
    "inclination must be a finite real angle in [0, 180] degrees",
    "scaled semi-major axis must be finite and positive",
    "eccentricity must be finite and in [0, 1)",
    "argument of periastron must be finite",
    "period and scaled radii must be finite positive real scalars",
    "limb-darkening coefficients and epoch must be finite real scalars",
    "nsamples must be a positive integer",
    "time must be a non-empty finite real one-dimensional array",
    "forward model produced nonfinite flux",
    "exptime_days must be a real number of days",
    "exptime_days must be finite and positive",
    "physical radii and semi-major axis must be finite positive real scalars",
    "a tabulated binary requires both bodies' original angular profiles",
    "secondary limb-darkening coefficients must be finite real scalars",
    "time and flux must be aligned one-dimensional arrays",
    "model likelihood is nonfinite",
    "TREX noise calibration requires aligned native photometry",
    "TREX noise calibration requires real, nonboolean native photometry",
    "TREX noise calibration requires finite native photometry",
    "TREX noise calibration requires positive integer sectors",
    "TREX noise calibration requires finite real ephemeris values",
    "TREX noise calibration requires T14 shorter than half the period",
    "TREX noise calibration deferred; inspect per-sector diagnostics",
    "time_btjd must be a non-empty finite real one-dimensional array",
    "ephemeris and exclusion factor must be finite real scalars",
    "duration must be positive and shorter than P/2; exclusion factor must be positive",
    "duration_days and cadence_days must be finite real scalars",
    "duration_days and cadence_days must be positive",
    "duration-to-cadence ratio must be finite",
    "residuals and time_btjd must be aligned finite real one-dimensional arrays",
    "cadence and gap tolerance must be finite positive real scalars",
    "gap tolerance in days must be finite",
    "bin point counts must be nonnegative integers, not booleans",
    "maximum bin points must not be below minimum bin points",
    "noise-model and prior-inflation switches must be boolean",
    "time_btjd and residuals must be aligned one-dimensional arrays",
    "baseline time_days must be real, not boolean or text",
    "baseline design must be real, not boolean or text",
    "baseline prior_mean must be real, not boolean or text",
    "baseline prior_covariance must be real, not boolean or text",
    "baseline time_days must be finite",
    "baseline design must be finite",
    "baseline prior_mean must be finite",
    "baseline prior_covariance must be finite",
    "baseline requires strictly increasing native TDB days, not folded phase",
    "baseline design must align with every retained native cadence",
    "baseline prior dimensions do not match its design",
    "baseline prior covariance must be symmetric",
    "baseline prior covariance must be positive definite",
    "baseline prior covariance Cholesky must be finite",
    "baseline needs an explicit mode and independent prior/basis source",
    "Gaussian baseline requires its exact native-time observations without phase bins",
    "baseline likelihood needs aligned finite aperture flux and positive reported errors",
    "baseline prior prediction cannot be represented finitely in float64",
    "baseline host aperture fraction must lie in (0,1]",
    "baseline covariance evaluation failed",
    "baseline covariance cannot be represented finitely in float64",
    "baseline requires one finite physical prediction per native observation",
    "baseline physical prediction is nonfinite",
    "baseline likelihood evaluation overflowed",
    "baseline likelihood is nonfinite; no zero-support replacement",
    "baseline device density requires JAX float64 mode",
    "phase bins require finite real one-dimensional photometry and edges",
    "phase-bin inputs must be representable finite float64 values",
    "phase bins require aligned flux and strictly positive reported errors",
    "phase-bin edges must contain at least two strictly increasing values",
    "phase-bin widths must be representable finite days",
    "phase-bin mean or uncertainty cannot be represented finitely",
    "known covariance-exclusion event requires a finite BTJD_TDB ephemeris",
    "known covariance-exclusion events must be an iterable of ephemeris mappings",
    "folded covariance requires real, nonboolean photometry",
    "folded covariance requires aligned finite photometry and positive errors",
    "invalid folded covariance ephemeris, bin count or shared error factor",
    "half-phase covariance-exclusion epoch cannot be represented finitely",
    "folded covariance summary is nonfinite",
    "photometric conversions require non-empty real inputs, not boolean or text",
    "photometric conversion inputs must be finite",
    "J_Ks_to_Tmag: J - Ks outside the Stassun et al. (2018) calibration domain [-0.1, 1.0] mag; no extrapolation is performed",
    "magnitude difference has no finite positive flux ratio in float64",
    "flux ratios must be finite and strictly positive",
    "installed nvJitLink wheel is missing its linker library",
    "cannot preload installed CUDA linker dependency",
    "runtime library unavailable or missing required symbols",
    "CUDA driver identity API unavailable", "lease metadata lock cleanup failed",
    "TRILEGAL population cache has empty or duplicate column names",
    "TRILEGAL population cache has no simulated stars",
    "TRILEGAL population cache has a non-physical mass or temperature",
    "TRILEGAL population has no simulated star fainter than the target",
    "TRILEGAL population has an unrepresentable stellar radius",
    "background population requires retained TRILEGAL [M/H] (mh_dex)",
    "background population requires retained TESS magnitudes (tess_mag)",
    "background mh_dex must be finite and match the retained stars",
    "imaging requires positive companion and nonnegative secondary flux",
    "image-source flux ratio is unrepresentable",
    "contrast curve requires matching separation and contrast arrays",
    "contrast curve separations must be positive and strictly increasing",
    "projected separations must be nonnegative",
    "imaging priors and fluxes must be real, not boolean or text",
    "imaging priors and separations must be real, not boolean or text",
    "imaging limits must be real finite values or NaN off-curve",
    "imaging log priors must be finite or negative infinity",
    "contrast separations and values must be supplied together",
    "deferred imaging constraints must be a list",
    "hierarchical stability requires a finite positive observed period, 0 <= outer eccentricity < 1 and 0 <= mutual inclination <= pi",
    "hierarchical stability requires finite periods and positive masses",
    "Winters et al. (2019) projected separation requires a primary mass of at least 0.075 and at most 0.60 solar masses",
    "Winters et al. (2019) M-dwarf prior requires 0.075 <= primary mass <= 0.60 solar masses",
    "Winters et al. (2019) M-dwarf prior requires a primary mass of at least 0.075 solar masses",
    "Winters projected-separation variance must exceed the orbital projection variance",
    "Moe & Di Stefano companion frequency is calibrated for primaries up to 40 solar masses; the TREX Moe branch requires mass above 0.60 solar masses",
    "Moe & Di Stefano companion fraction is outside its physical domain",
    "companion orbital period must lie inside the measured 0.2 <= log10(P/day) <= 8.0 domain",
    "Moe twin fractions require 0.60 < primary mass <= 40 solar masses",
    "total pair masses must be positive",
    "bound-companion model has no supported stellar mass-ratio domain",
    "Moe & Di Stefano companion period sampling requires a primary mass above the 0.60 solar-mass Winters boundary",
    "Moe & Di Stefano companion frequency has no positive normalization",
    "Moe mass-ratio support must start in [0.1, 1)",
    "Moe & Di Stefano period cumulative must be finite and monotone",
    "Moe & Di Stefano mass-ratio law requires one orbital period per draw",
    "companion mass-ratio quantiles need one per-draw orbital period",
    "primary masses must be positive",
    "mass-ratio floor cannot represent the stellar secondary-mass boundary",
    "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass",
    "twin fractions must lie in [0, 1)",
    "Moe & Di Stefano mass-ratio normalization is not positive",
    "mass-ratio quantiles must match the prepared host shape",
    "mass-ratio inverse is unrepresentable on its support",
    "mass ratios must match the prepared host shape",
    "EB mass ratios must lie in (0, 1]",
    "host masses must be positive",
    "inclination bounds must satisfy 0 <= lower < upper <= 180 degrees",
    "inclination bounds are unresolved in float64",
    "Moe & Di Stefano Eq. 17 has no finite eta at log10(P/day) <= 0.5",
    "eccentricity draws must be finite values in [0, 1)",
    "eclipsing-pair quantiles need one per-draw host mass",
    "background population requires a positive simulated star count",
    "background population expectation is unrepresentable",
    "neighbor separations must be finite nonnegative real arcseconds",
}) | frozenset(
    name + suffix
    for name in (
        "outer log periods", "inner binary masses", "outer companion masses",
        "observed period", "outer eccentricity", "mutual inclination", "log projected separations",
        "companion separation quantiles", "primary mass", "companion orbital periods",
        "companion log periods", "total pair masses", "log semimajor axes",
        "companion period quantiles", "companion mass-ratio quantiles", "_uncapped",
        "flatpriors", "radius quantiles", "host masses", "radii", "inclination quantiles",
        "lower inclination", "upper inclination", "binary period", "planet", "orbital period",
        "eccentricity quantiles", "periastron quantiles", "scenario period", "use_2xP",
        "mass-ratio quantiles", "simulated sky area", "field radius", "companion_is_host",
        "companion flux ratios", "secondary flux ratios", "contrast separations", "contrast values",
        "projected separations", "require_fainter", "primary masses", "mass-ratio floors",
        "mass-ratio slopes", "twin fractions", "mass ratios", "target TESS magnitude",
        "background mh_dex", "background TESS magnitudes",
    )
    for suffix in (
        " must contain finite real numbers, not boolean or text", " must be a finite real scalar",
        " must be a finite real scalar greater than zero", " must lie in [0, 1]",
        " must lie in (0, 1)", " must be a boolean",
    )
)
_KNOWN_WARNINGS = frozenset({
    ("The sampling was stopped short due to maxiter/maxcall limit the delta(log(z)) criterion is not achieved; "
     "posterior may be poorly sampled"),
    "We have reached the plateau in the likelihood we are stopping sampling",
    ("All the initial likelihood values are the same. You likely have a plateau in the likelihood. "
     "Nested sampling may not be the best sampler in this case."),
    "Weights do not sum to 1 and have been renormalized.",
})
_WARNING_OMITTED = "Warning text omitted; inspect the source and owning record"
_JAX_WARNING_TYPES = frozenset({
    "RuntimeError", "ValueError", "MemoryError", "TypeError", "NumericalLikelihoodError", "XlaRuntimeError",
})
_SAFE_WARNING_SUMMARIES = _KNOWN_WARNINGS | frozenset({
    _WARNING_OMITTED,
    "Limb darkening grid bounds exceeded; nearest nodes used",
    "Progress callback failed; details omitted",
    "JAX batch failed; NumPy reference likelihood used",
    "Pool worker warning reports lost",
    "pool warning transport unavailable; worker warnings cannot be adopted",
    "Additional distinct warnings omitted",
    "Additional distinct worker warnings were omitted",
}) | frozenset(
    "JAX batched %s likelihood failed (%s); NumPy reference likelihood used" % (kind, error_type)
    for kind in ("TP", "EB") for error_type in _JAX_WARNING_TYPES
)
_SAFE_MESSAGES = frozenset({
    "PHOENIX atmosphere download started", "PHOENIX atmosphere download completed",
    "PHOENIX atmosphere preparation started", "PHOENIX atmosphere preparation completed",
    "PHOENIX atmosphere node unavailable",
    "TREX CLI started", "TREX CLI finished", "TREX CLI stopped",
    "TREX worker started", "TREX worker stopped", "Python warning captured",
    "Worker warning received", "TREX calculation started", "TREX input refused",
    "No idle identifiable accelerator; CPU execution selected",
    "no accelerator meets the memory policy floor; CPU execution selected",
    "pool warning transport unavailable; worker warnings cannot be adopted",
    "Worker warning start marker unavailable", "Worker warning report failed",
    "Worker warning spool cleanup failed", "Warning capture library setup failed",
    "Accelerator admission incomplete", "Accelerator probe failed; CPU execution selected",
    "Accelerator lease cleanup incomplete", "Lease metadata unlock failed; closing descriptor",
    "JAX setup failed; CPU reference execution selected", "JAX placement probe failed; CPU execution selected",
    "TRILEGAL job submitted; polling its result",
    "LSST-simulation fallback unavailable too; TRILEGAL error is raised",
    "background population is the recorded LSST-simulation fallback",
})
_BATCH_WORKING_SET = "TREX %s: batch=%d, compiled working set=%d bytes, memory budget=%d bytes"
_BATCH_COMPILED_BLOCK = "TREX %s: batch=%d, compiled block=%d draws/%d bytes, memory budget=%d bytes"
_SAFE_FORMATS = {
    "TREX calculation stopped: %s": "TREX calculation stopped",
    "TREX calculation %s": "TREX calculation finished or paused",
    "TREX input refused: %s": "TREX input refused",
    "TREX input check: %s": "TREX input check finished",
    "MAST application check: %s": "MAST application check finished",
    "TREX inputs admitted: %d cadences, %d neighbors, %d draws per scenario": "TREX inputs admitted",
    "TREX Monte Carlo phase %s: %d draws per scenario, %.2f s": "TREX Monte Carlo phase finished or restored",
    "TREX MC-pilot importance phase finished: %.2f s": "TREX MC-pilot importance phase finished",
    "TREX nested escalation phase finished: %.2f s": "TREX nested escalation phase finished",
    "TREX scenario %s: draws=%d, ESS=%s, lnZ=%r, draw=%.2fs, wait=%.2fs, evaluate=%.2fs": "TREX scenario finished",
    "TREX nested %s: seed=%d, calls=%d, lnZ=%r, stop=%s, %.0f s": "TREX nested run finished",
    "TREX device pilot: chose %s (%s)": "TREX device pilot finished",
    "nested ensemble for %s failed": "TREX nested ensemble failed",
    "TREX nested escalation of %s failed: %s: %s": "TREX nested escalation failed",
    "TREX importance refinement of %s failed: %s: %s": "TREX importance refinement failed",
    "JAX batch failed, retrying with halved chunks: %s: %s": "JAX batch failed; retrying with halved chunks",
    "TREX %s: overlapped setup failed; retrying through the batch path (%s)":
        "Overlapped accelerator setup failed; batch path retried",
    _BATCH_WORKING_SET: "TREX accelerator batch planned",
    _BATCH_COMPILED_BLOCK: "TREX accelerator batch planned",
    "%s: pilot compilation exhausted memory at batch=%d; halving (%s)": "Pilot compilation exhausted memory; halving batch",
    "%s: proposed compilation exhausted memory at batch=%d; halving (%s)": "Proposed compilation exhausted memory; halving batch",
    "%s: compiler memory analysis unavailable; using conservative batch=%d": "Compiler memory analysis unavailable; conservative batch used",
    "%s: final compiler memory counter unavailable; keeping batch=%d": "Final compiler memory counter unavailable; previous batch kept",
    "%s: compiler memory analysis unavailable; using one immutable block (%d draws)": "Compiler memory analysis unavailable; one block used",
    "warnings from %d pool worker process(es) were lost (%d of %d reported)": "Pool worker warning reports lost",
    "%s: device memory unavailable; probing with one draw, no guessed RAM/VRAM budget":
        "Device memory unavailable; one-draw probe used",
    "%s: allocation exhausted at batch=%d; retrying the same draws with a smaller working set (%s)":
        "Allocation exhausted; same draws retried with smaller working set",
    "%s: device compile failed; skipping this device (%s)": "Device compile failed; device skipped",
    "Accelerator admission: %s": "Accelerator admission incomplete",
    "MAST %s failed after %d attempt(s): %s": "MAST request attempts exhausted",
    "MAST provider request failed: %s": "MAST provider request failed",
    "archive request failed: %s": "Archive request failed",
    "downloading SPOC TPF for TIC %s sector %s": "SPOC TPF download started",
    "TPF download failed for TIC %s sector %s (retryable=%s)": "SPOC TPF download failed",
    "TRILEGAL stage %s failed: %s (retryable=%s)": "TRILEGAL acquisition stage failed",
    "TRILEGAL unavailable (%s); trying the LSST-simulation fallback": "TRILEGAL unavailable; LSST fallback attempted",
    "GSP-Spec response was not a complete VOTable: %s": "GSP-Spec VOTable response incomplete",
    "target phase Powell exhausted (status=%s, nfev=%s): %s; refining same image":
        "Target-phase optimizer exhausted; same image refined",
}
_BATCH_FIELDS = {
    _BATCH_WORKING_SET: ("batch_size", "compiled_bytes", "memory_budget_bytes"),
    _BATCH_COMPILED_BLOCK: ("batch_size", "block_draws", "compiled_bytes", "memory_budget_bytes"),
}
_standalone_state: Optional[Dict[str, Any]] = None
_standalone_warnings: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_standalone_warning_lock = threading.Lock()


def get_logger(name: str, *, namespace: str = "trex") -> logging.Logger:
    """Return a logger under the supplied host namespace without side effects."""
    normalized = str(name)
    if normalized == namespace or normalized.startswith(namespace + "."):
        return logging.getLogger(normalized)
    return logging.getLogger(namespace + "." + normalized.lstrip("."))


def host_logger(name: str) -> logging.Logger:
    """The logger for ``name`` under the host namespace set by the embedding package."""
    return get_logger(name, namespace=_host_namespace)


def set_host_namespace(namespace: Optional[str]) -> None:
    """Name the package an embedded TREX shares; an empty value restores standalone ``trex``."""
    global _host_namespace
    _host_namespace = str(namespace) if namespace else "trex"


def set_worker_log_setup(setup: Optional[Callable[[], Tuple[Any, Tuple[Any, ...]]]]) -> None:
    """Let the host say how a spawned worker joins its logging, without TREX importing it."""
    global _worker_log_setup
    _worker_log_setup = setup


def worker_log_setup() -> Optional[Tuple[Any, Tuple[Any, ...]]]:
    """The picklable ``(initializer, args)`` for a pool started now, if a host set one."""
    return None if _worker_log_setup is None else _worker_log_setup()


def set_worker_warning_hooks(
    hooks: Optional[
        Tuple[Callable[[], List[Dict[str, Any]]], Callable[[List[Dict[str, Any]]], None]]
    ],
) -> None:
    """Let the host drain and adopt pool-worker warnings without TREX importing it."""
    global _worker_warning_hooks
    _worker_warning_hooks = hooks


def _metric(value: Any) -> Any:
    """Keep numeric metrics; text, arbitrary objects and scientific arrays are not metrics."""
    if value is None:
        return value
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    return "<omitted>"


def _allowed(value: Any, choices: Any) -> Any:
    if value is None:
        return None
    return value if isinstance(value, str) and value in choices else "<omitted>"


def _invocation(value: Any) -> str:
    return value if isinstance(value, str) and _INVOCATION_ID.fullmatch(value) else "<omitted>"


def _source_file(value: Any) -> str:
    """Keep a module identifier, never an absolute or unreviewed filesystem path.

    A bare ``trex/...`` name is already a namespace identifier. An absolute path
    is accepted only when its resolved location is inside the TREX source root or
    the interpreter's library roots, and only its relative module name is emitted.
    """
    if not isinstance(value, str):
        return "<external>"
    if value in _LIBRARY_FILES or _TREX_SOURCE_FILE.fullmatch(value):
        return value
    if not os.path.isabs(value):
        return "<external>"
    try:
        path = os.path.realpath(value)
        if os.path.commonpath((path, _SOURCE_ROOT)) == _SOURCE_ROOT:
            identifier = "trex/" + os.path.relpath(path, _SOURCE_ROOT).replace(os.sep, "/")
            if _TREX_SOURCE_FILE.fullmatch(identifier):
                return identifier
        if any(os.path.commonpath((path, root)) == root for root in _LIBRARY_ROOTS):
            for identifier in _LIBRARY_FILES:
                if path.endswith(os.sep + identifier.replace("/", os.sep)):
                    return identifier
    except (OSError, ValueError):
        pass
    return "<external>"


def _name(value: Any, pattern: re.Pattern[str]) -> Any:
    if value is None:
        return None
    return value if isinstance(value, str) and pattern.fullmatch(value) else "<omitted>"


def _live_exception_types() -> frozenset[str]:
    """Names of every exception class this process has loaded, including new ones."""
    names: set[str] = set()
    seen: set[type[BaseException]] = set()
    pending = [BaseException]
    while pending:
        exception = pending.pop()
        if exception in seen:
            continue
        seen.add(exception)
        names.add(exception.__name__)
        pending.extend(exception.__subclasses__())
    return frozenset(names)


def _exception_type(value: Any) -> Any:
    """Exception type names are code identities; payload text is not one."""
    if value is None:
        return None
    if not isinstance(value, str):
        return "<omitted>"
    return value if _EXCEPTION_NAME.fullmatch(value) or value in _live_exception_types() else "<omitted>"


def _line(value: Any) -> Optional[int]:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool) and 0 <= value <= sys.maxsize:
        return int(value)
    return None


def _source(value: Any, line: Any, function: Any) -> Dict[str, Any]:
    filename = _source_file(value)
    return {"file": filename, "line": _line(line),
            "function": _name(function, _FUNCTION_NAME)}


def _warning_location(value: Any) -> str:
    if not isinstance(value, str):
        return "<omitted>"
    if value in ("trex.runtime.logging", "trex.runtime.logging.collect_worker_warnings",
                 "trex.logging.collect_worker_warnings"):
        return "trex/runtime/logging.py"
    if ":" not in value:
        return "<omitted>"
    filename, line = value.rsplit(":", 1)
    if not re.fullmatch(r"[0-9]{1,10}", line, re.ASCII):
        return "<omitted>"
    return "%s:%d" % (_source_file(filename), int(line))


def _scenario(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and (value in _SCENARIOS or re.fullmatch(
            r"(?:N[0-9]{1,10}|Neighbor [0-9]{1,10}(?: N(?:TP|EB|EBx2P))?)", value)):
        return value
    return "<omitted>"


def _kernel(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        for name in ("_batch_lnL_TP_jit", "_batch_lnL_EB_jit"):
            if value == name or re.fullmatch(r"<PjitFunction of <function " + name + r" at 0x[0-9a-f]+>>", value):
                return name
    return "<omitted>"


def _device(value: Any) -> str:
    """Keep only recognized device identifiers, not arbitrary device labels."""
    return value if isinstance(value, str) and _DEVICE_ID.fullmatch(value) else "<omitted>"


def _pilot_metric(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in ("reference", "eligible", "same_nonfinite_pattern", "passed"):
        return value if isinstance(value, bool) else "<omitted>"
    if key == "draws":
        return int(value) if isinstance(value, numbers.Integral) and not isinstance(value, bool) else "<omitted>"
    return _metric(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else "<omitted>"


def _exception_reason(text: Any) -> Optional[str]:
    """Match the same reviewed reasons in local and pickled worker exceptions."""
    if not isinstance(text, str):
        return None
    if text in _SAFE_EXCEPTION_REASONS:
        return text
    if re.match(r"phoenix-[a-z0-9-]+: [0-9]+ unsupported atmosphere request\(s\) ", text):
        return "unsupported atmosphere requests; no star removed or fallback used"
    if re.fullmatch(r"Teff [0-9.eE+-]+ K lies outside the retained main-sequence relation", text):
        return "Teff lies outside the retained main-sequence relation"
    if re.fullmatch(r"Teff [0-9.eE+-]+ K is reached by more than one mass on the retained relation", text):
        return "Teff is reached by more than one mass on the retained relation"
    if re.fullmatch(r"effective temperatures must be finite and inside the blackbody reference table \[[0-9.eE+, -]+\] K", text):
        return "effective temperatures outside the finite blackbody reference table"
    # Retain the failure category/column/row, never the caller's cache path or cells.
    match = re.fullmatch(
        r"(TRILEGAL population cache has (?:data after its footer|a truncated row)"
        r"|TRILEGAL population has (?:a non-numeric|invalid) (?:TESS|Mact|logg|logTe|\[M/H\]|Av|m-M0))"
        r" at .+:([0-9]+)", text,
    )
    if match:
        return f"{match[1]} at row {match[2]}; path omitted"
    if text.startswith("TRILEGAL population cache is unreadable: "):
        return "TRILEGAL population cache is unreadable; path omitted"
    prefix = "TRILEGAL population cache lacks required columns: "
    if text.startswith(prefix) and all(
        name in {"TESS", "Mact", "logg", "logTe", "[M/H]"} for name in text[len(prefix):].split(", ")
    ):
        return text
    if text.startswith("native unit numerical/execution integrity failure: "):
        return "native unit numerical/execution integrity failure; details omitted"
    if re.fullmatch(r"monitored nested likelihood budget of \d+ evaluations exhausted", text):
        return "monitored nested likelihood budget exhausted"
    if re.fullmatch(
        r"nested wall deadline reached during importance refinement; \d+ of \d+ likelihoods retained", text,
    ):
        return "nested wall deadline reached during importance refinement; likelihood progress retained"
    if re.fullmatch(
        r"the .+ nested run exhausted its registered \d+-evaluation likelihood budget after \d+ "
        r"monitored evaluations; the partial history is unresolved and must defer", text,
    ):
        return "nested likelihood budget exhausted; partial history is unresolved and must defer"
    if re.fullmatch(r".+: \d+ supported likelihood evaluations failed", text):
        return "supported likelihood evaluations failed"
    return None


def _failure_record_note(note: str) -> Optional[str]:
    """Keep the app's record-write status without copying paths or exception payloads."""
    if note.startswith("TREX failure record: "):
        return "TREX failure record written; location omitted"
    prefix = "TREX failure record not written: "
    if not note.startswith(prefix):
        return None
    problem = note[len(prefix):]
    known = (
        "the target failure record could not be written",
        "failure record could not be written",
        "output directory could not be created",
        "output directory already exists",
        "the empty output is retained for resume",
    )
    reason = next((item for item in known if problem == item or problem.startswith(item + ": ")), None)
    if reason is None:
        return prefix + "details omitted"
    detail = problem[len(reason):].removeprefix(": ")
    match = re.match(
        r"(?:\[Errno (\d{1,6})\]|(PermissionError|FileNotFoundError|FileExistsError|OSError|RuntimeError)\((\d{1,6})?)",
        detail,
    )
    if match:
        number = match.group(1) or match.group(3)
        if number and int(number) in errno.errorcode:
            reason += " (" + os.strerror(int(number)) + ")"
        elif match.group(2):
            reason += " (" + match.group(2) + "; details omitted)"
    return prefix + reason


def _field(key: str, value: Any) -> Any:
    if key == "warning_location":
        return _warning_location(value)
    if key == "warning_category":
        return _allowed(value, _WARNING_TYPES)
    if key == "error_type":
        return _exception_type(value)
    if key == "device":
        return _device(value)
    if key in _TEXT_VALUES:
        return _allowed(value, _TEXT_VALUES[key])
    if key == "reason":
        return _allowed(value, _DEVICE_REASONS)
    if key == "warning_summary":
        return _allowed(value, _SAFE_WARNING_SUMMARIES)
    if key == "kernel":
        return _kernel(value)
    if key == "chosen":
        return [_device(item) for item in value[:64]] if isinstance(value, (list, tuple)) else "<omitted>"
    if key == "pilot":
        if not isinstance(value, (list, tuple)):
            return "<omitted>"
        rows = []
        for row in value[:64]:
            if isinstance(row, dict):
                clean = {}
                for name, item in row.items():
                    if name == "device":
                        clean[name] = _device(item)
                    elif name == "parity":
                        clean[name] = ({part: _pilot_metric(part, number) for part, number in item.items()
                                        if part in _PARITY_FIELDS} if isinstance(item, dict) else "<omitted>")
                    elif name in _PILOT_FIELDS:
                        clean[name] = _pilot_metric(name, item)
                rows.append(clean)
        return rows
    if key == "parity":
        return ({part: _pilot_metric(part, number) for part, number in value.items()
                 if part in _PARITY_FIELDS} if isinstance(value, dict) else "<omitted>")
    if key in ("scenario", "cpu_reference_overlap"):
        if key == "cpu_reference_overlap" and isinstance(value, bool):
            return value
        return _scenario(value)
    return _metric(value)


def _message(record: logging.LogRecord) -> str:
    """Describe known templates without interpolating untrusted arguments."""
    if not isinstance(record.msg, str):
        return "Unstructured message omitted"
    if not record.args:
        return record.msg if record.msg in _SAFE_MESSAGES else "Unstructured message omitted"
    return _SAFE_FORMATS.get(record.msg, "Unstructured message omitted")


def _exception_frames(exc: BaseException) -> Dict[str, Any]:
    """Keep raise sites and safe reasons without serializing exception arguments or locals."""
    seen = set()

    def visit(error: BaseException) -> Dict[str, Any]:
        seen.add(id(error))
        traceback_frames = []
        frames_omitted = 0
        for frame, line in traceback.walk_tb(error.__traceback__):
            traceback_frames.append(_source(frame.f_code.co_filename, line, frame.f_code.co_name))
            if len(traceback_frames) > _EXCEPTION_FRAME_LIMIT:
                traceback_frames.pop(0)
                frames_omitted += 1
        text = error.args[0] if len(error.args) == 1 else None
        message = _exception_reason(text)
        result = {
            "type": _exception_type(type(error).__name__),
            "message_omitted": message is None,
            "frames": traceback_frames,
        }
        if message is not None:
            result["message"] = message
            if message != text:
                result["message_redacted"] = True
        if frames_omitted:
            result["frames_omitted"] = frames_omitted
        notes = []
        for note in itertools.islice(getattr(error, "__notes__", ()), _EXCEPTION_NOTE_LIMIT):
            safe = (_failure_record_note(note) if isinstance(note, str) else None)
            safe = safe or _exception_reason(note)
            notes.append(safe or "<note omitted>")
        if notes:
            result["notes"] = notes
        note_count = len(getattr(error, "__notes__", ()))
        if note_count > _EXCEPTION_NOTE_LIMIT:
            result["notes_omitted"] = note_count - _EXCEPTION_NOTE_LIMIT
        if isinstance(error, OSError) and isinstance(error.errno, int):
            try:
                result["reason"] = os.strerror(error.errno)
            except (OverflowError, ValueError):
                pass
        if isinstance(error, RemoteTraceback):
            # Pool exceptions carry the original worker traceback as text in their
            # cause. Extract only frame locations, never source lines or error text.
            in_stack = False
            for line in io.StringIO(error.tb):
                line = line.rstrip("\r\n")
                if line == "Traceback (most recent call last):":
                    in_stack = True
                elif in_stack:
                    frame = _REMOTE_FRAME.fullmatch(line)
                    if frame:
                        result["frames"].append(_source(frame[1], int(frame[2]), frame[3]))
                        if len(result["frames"]) > _EXCEPTION_FRAME_LIMIT:
                            result["frames"].pop(0)
                            result["frames_omitted"] = result.get("frames_omitted", 0) + 1
                    elif line and not line.startswith(" "):
                        in_stack = False
        cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
        if cause is not None and id(cause) not in seen:
            if len(seen) < _EXCEPTION_CHAIN_LIMIT:
                result["cause"] = visit(cause)
            else:
                result["cause_omitted"] = True
        return result

    return visit(exc)


class _JsonFormatter(logging.Formatter):
    def __init__(self, invocation_id: str):
        super().__init__()
        self.invocation_id = invocation_id

    def format(self, record: logging.LogRecord) -> str:
        raw_event = getattr(record, "event", None)
        event = raw_event if raw_event is None or isinstance(raw_event, str) and raw_event in _EVENTS else "<omitted>"
        try:
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        except (OSError, OverflowError, TypeError, ValueError):
            timestamp = None
        levels = {logging.DEBUG: "DEBUG", logging.INFO: "INFO", logging.WARNING: "WARNING",
                  logging.ERROR: "ERROR", logging.CRITICAL: "CRITICAL"}
        canonical_name = record.name
        if _host_namespace != "trex":
            if record.name == _host_namespace:
                canonical_name = "trex"
            elif record.name.startswith(_host_namespace + "."):
                suffix = record.name[len(_host_namespace) + 1:]
                canonical_name = suffix if suffix == "trex" or suffix.startswith("trex.") else "trex." + suffix
        payload = {
            "timestamp_utc": timestamp,
            "invocation_id": _invocation(self.invocation_id),
            "process": _line(record.process),
            "level": levels.get(record.levelno, "OTHER") if isinstance(record.levelno, int) else "OTHER",
            "logger": record.name if canonical_name in _LOGGERS else "<omitted>",
            "event": event,
            "message": _message(record),
            "source": _source(record.pathname, record.lineno, record.funcName),
        }
        fields = {key: _field(key, value) for key, value in record.__dict__.items() if key in _METRIC_FIELDS}
        if (canonical_name == "trex" and event == "trex.batch_plan" and isinstance(record.msg, str)
                and isinstance(record.args, tuple) and record.msg in _BATCH_FIELDS):
            names = _BATCH_FIELDS[record.msg]
            values = record.args[1:]
            if len(values) == len(names) and all(
                isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in values
            ):
                fields.update(zip(names, map(int, values), strict=True))
        if fields:
            payload["fields"] = fields
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = _exception_frames(record.exc_info[1])
        return json.dumps(payload, allow_nan=False, sort_keys=True)


def _diagnostic(text: str) -> None:
    """Last-resort stderr path must not echo a failed record or its arguments."""
    identity = " process=%d" % os.getpid()
    if _standalone_state is not None:
        identity += " invocation=%s" % _invocation(_standalone_state["invocation_id"])
    try:
        os.write(2, (text + identity + "\n").encode("utf-8"))
    except OSError:
        pass


class _StreamHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        _diagnostic("TREX logging: stderr output failed; a diagnostic was lost")


class _FileHandler(logging.StreamHandler):
    def __init__(self, path: Path):
        # O_EXCL guarantees that even a reused PID cannot share an existing sink.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        super().__init__(stream)
        self.failed_writes = 0

    def handleError(self, record: logging.LogRecord) -> None:
        self.failed_writes += 1
        if self.failed_writes == 1:
            _diagnostic("TREX logging: file write failed; diagnostics are continuing on stderr")

    def close(self) -> None:
        try:
            self.stream.close()
        except OSError:
            self.failed_writes += 1
            _diagnostic("TREX logging: file close failed; diagnostics may be incomplete")
        finally:
            super().close()


def _standalone_showwarning(message: Any, category: type, filename: str, lineno: int,
                            file: Any = None, line: Optional[str] = None) -> None:
    # Arbitrary warning text can contain observations or credentials, even if
    # it was written as a source literal. Only reviewed summaries are emitted.
    text = str(message)
    if text in _KNOWN_WARNINGS:
        summary = text
    elif text.startswith("TREX host limb-darkening requests fell outside the TRICERATOPS TESS grid bounds;"):
        summary = "Limb darkening grid bounds exceeded; nearest nodes used"
    elif text.startswith("progress callback ") and (" failed: " in text or text.startswith("progress callback failed:")):
        summary = "Progress callback failed; details omitted"
    else:
        summary = None
        match = re.fullmatch(
            r"JAX batched (TP|EB) likelihood failed \(([A-Za-z_][A-Za-z0-9_]{0,63}(?:Error|Exception)): "
            r".*\); evaluating these draws with the NumPy reference likelihood instead",
            text, re.DOTALL,
        )
        if match and match.group(2) in _JAX_WARNING_TYPES:
            summary = "JAX batched %s likelihood failed (%s); NumPy reference likelihood used" % match.groups()
        elif match:
            summary = "JAX batch failed; NumPy reference likelihood used"
    summary = summary or _WARNING_OMITTED
    category_name = _allowed(getattr(category, "__name__", None), _WARNING_TYPES)
    line_number = _line(lineno)
    location = "%s:%d" % (_source_file(filename), line_number) if line_number is not None else "<omitted>"
    key = (category_name, location, summary)
    with _standalone_warning_lock:
        if key in _standalone_warnings:
            _standalone_warnings[key]["count"] += 1
        elif len(_standalone_warnings) < WORKER_WARNING_PAYLOAD_LIMIT - 1:
            _standalone_warnings[key] = {"category": category_name, "location": location,
                                         "message": summary, "count": 1}
        else:
            overflow = ("WarningCaptureLimit", "trex.runtime.logging", "Additional distinct warnings omitted")
            entry = _standalone_warnings.setdefault(overflow, {
                "category": overflow[0], "location": overflow[1], "message": overflow[2], "count": 0,
            })
            entry["count"] += 1
            return  # The owning overflow record counts repeats without per-draw log spam.
    host_logger("warnings").warning(
        "Python warning captured", extra={"event": "warning.captured", "warning_category": category_name,
                                           "warning_location": location, "warning_summary": summary},
    )


def _drain_standalone_warnings() -> List[Dict[str, Any]]:
    with _standalone_warning_lock:
        entries = list(_standalone_warnings.values())
        _standalone_warnings.clear()
    return entries


def _adopt_standalone_warnings(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            _diagnostic("TREX logging: malformed worker warning entry omitted")
            continue
        summary = "Pool worker warning reports lost" if entry.get("category") == "WorkerWarningLoss" else entry.get("message")
        host_logger("warnings").warning(
            "Worker warning received", extra={"event": "warning.worker", "warning_category": entry.get("category"),
                                              "warning_location": entry.get("location"),
                                              "warning_summary": summary, "count": entry.get("count")},
        )


def _prime_warning_hook() -> None:
    # Both libraries install their own warning hook on first import.
    for name in ("astropy", "astroquery"):
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - capture can continue even if a library import fails
            host_logger("warnings").warning(
                "Warning capture library setup failed", exc_info=True,
                extra={"event": "warning.setup.degraded", "warning_module": name,
                       "error_type": type(error).__name__},
            )
    warnings.showwarning = _standalone_showwarning


def _standalone_worker_setup() -> Tuple[Any, Tuple[Any, ...]]:
    assert _standalone_state is not None
    return _initialize_standalone_worker, (_standalone_state["invocation_id"], _standalone_state["directory"])


def _configure_standalone(invocation_id: str, directory: Optional[str]) -> None:
    """Install process-local handlers only for an explicitly invoked standalone CLI."""
    global _standalone_state
    if _standalone_state is not None:
        raise RuntimeError("standalone logging is already active")
    logger = logging.getLogger("trex")
    _standalone_state = {
        "invocation_id": invocation_id, "directory": directory, "handlers": list(logger.handlers),
        "level": logger.level, "propagate": logger.propagate, "namespace": _host_namespace,
        "worker_setup": _worker_log_setup, "warning_hooks": _worker_warning_hooks,
        "showwarning": warnings.showwarning, "file": None,
    }
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = _JsonFormatter(invocation_id)
    console = _StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    set_host_namespace("trex")
    set_worker_log_setup(_standalone_worker_setup)
    set_worker_warning_hooks((_drain_standalone_warnings, _adopt_standalone_warnings))
    warnings.showwarning = _standalone_showwarning
    if directory is not None:
        try:
            path = Path(directory) / ("trex-%d-%s.jsonl" % (os.getpid(), uuid4().hex))
            handler = _FileHandler(path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            _standalone_state["file"] = handler
        except OSError:
            _diagnostic("TREX logging: file unavailable; diagnostics are continuing on stderr")


def _shutdown_standalone() -> None:
    global _standalone_state
    state = _standalone_state
    if state is None:
        return
    logger = logging.getLogger("trex")
    for handler in logger.handlers:
        handler.close()
    file_handler = state["file"]
    if file_handler is not None and file_handler.failed_writes:
        _diagnostic("TREX logging: %d file record(s) lost or incomplete" % file_handler.failed_writes)
    logger.handlers = state["handlers"]
    logger.setLevel(state["level"])
    logger.propagate = state["propagate"]
    warnings.showwarning = state["showwarning"]
    set_host_namespace(state["namespace"])
    set_worker_log_setup(state["worker_setup"])
    set_worker_warning_hooks(state["warning_hooks"])
    with _standalone_warning_lock:
        _standalone_warnings.clear()
    _standalone_state = None


def _initialize_standalone_worker(invocation_id: str, directory: Optional[str]) -> None:
    _configure_standalone(invocation_id, directory)
    host_logger("worker").info("TREX worker started", extra={"event": "trex.worker.start"})

    def stop_worker() -> None:
        host_logger("worker").info("TREX worker stopped", extra={"event": "trex.worker.stop"})
        _shutdown_standalone()

    # Warning reporting (+10) must finish before this handler shutdown (-10).
    multiprocessing.util.Finalize(None, stop_worker, exitpriority=-10)
    # Since Python 3.13 a worker runs Finalize handlers from atexit, not right after run();
    # get_logger() re-registers that hook after logging's, so it runs before logging.shutdown
    # closes the handler and the stop record is still written.
    multiprocessing.util.get_logger()
    _prime_warning_hook()


def standalone_cli(module: str) -> int:
    """Run an entry point with a private invocation directory and stderr diagnostics."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--log-dir", type=Path)
    options.add_argument("--no-log-file", action="store_true")
    controls, arguments = parser.parse_known_args()
    invocation_id = uuid4().hex
    directory = None
    if not controls.no_log_file:
        state = Path(os.environ.get("XDG_STATE_HOME", ""))
        root = controls.log_dir or (state if state.is_absolute() else Path.home() / ".local" / "state") / "trex" / "log"
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + invocation_id)
            path.mkdir(mode=0o700)
            directory = str(path)
        except OSError:
            if controls.log_dir is not None:
                _diagnostic("TREX logging: requested log directory unavailable; command not started")
                return 2
            _diagnostic("TREX logging: default log directory unavailable; continuing on stderr")
    _configure_standalone(invocation_id, directory)
    if controls.log_dir is not None and _standalone_state["file"] is None:
        _diagnostic("TREX logging: requested log file unavailable; command not started")
        _shutdown_standalone()
        if directory is not None:
            try:
                Path(directory).rmdir()
            except OSError:
                pass  # Never remove a directory another process has populated.
        return 2
    log = host_logger("cli")
    started = time.monotonic()
    command = module.rsplit(".", 1)[-1]
    exit_code = 1
    log.info("TREX CLI started", extra={"event": "trex.cli.start", "command": command})
    try:
        if not any(arg in ("-h", "--help") for arg in arguments):
            _prime_warning_hook()
        exit_code = importlib.import_module(module).main(arguments)
        exit_code = 0 if exit_code is None else exit_code
        return exit_code
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1
        if exit_code == 0 and any(arg in ("-h", "--help") for arg in arguments):
            print("\nLogging options:\n  --log-dir DIR  write per-process logs under DIR\n"
                  "  --no-log-file  use stderr only, including workers")
        raise
    except KeyboardInterrupt:
        exit_code = 130
        log.warning("TREX CLI stopped", exc_info=True,
                    extra={"event": "trex.cli.stopped", "command": command, "category": "interrupted"})
        raise
    except Exception as exc:
        log.error("TREX CLI stopped", exc_info=True,
                  extra={"event": "trex.cli.stopped", "command": command, "error_type": type(exc).__name__})
        return 1
    finally:
        status = {0: "finished", 2: "refused", 3: "deferred", 75: "paused"}.get(exit_code, "failed")
        log.info("TREX CLI finished", extra={"event": "trex.cli.finish", "command": command,
                                             "exit_code": exit_code, "status": status,
                                             "wall_seconds": time.monotonic() - started})
        _shutdown_standalone()


def start_worker_warning_spool() -> Optional[str]:
    """Create the directory a pool's workers write their final warning payloads into.

    Returns ``None`` when the host has no warning record to adopt from. The spool is a
    short-lived transport, not a log: the parent removes it once it has collected the
    payloads, and a worker that dies first simply leaves no payload behind.
    """
    if _worker_warning_hooks is None:
        return None
    try:
        return tempfile.mkdtemp(prefix="trex-worker-warnings-")
    except OSError as exc:
        message = "pool warning transport unavailable; worker warnings cannot be adopted"
        host_logger("trex").warning(
            message, exc_info=True,
            extra={"event": "warning.transport.unavailable", "category": "WorkerWarningTransportUnavailable",
                   "error_type": type(exc).__name__},
        )
        _worker_warning_hooks[1]([{
            "category": "WorkerWarningTransportUnavailable", "location": "trex.runtime.logging",
            "message": message, "count": 1,
        }])
        return None


def worker_warning_report(spool: Optional[str]) -> Optional[Tuple[str, Callable[[], List[Dict[str, Any]]]]]:
    """The picklable payload a pool initializer installs in each worker, or ``None``."""
    if spool is None or _worker_warning_hooks is None:
        return None
    return spool, _worker_warning_hooks[0]


def install_worker_warning_report(
    report: Optional[Tuple[str, Callable[[], List[Dict[str, Any]]]]],
) -> None:
    """Write this worker's captured warnings to the spool when the process exits.

    Multiprocessing finalization reports what every task accumulated without a
    per-likelihood call. A terminated worker is counted as lost by the parent.
    """
    if report is None:
        return
    spool, drain = report

    try:
        (Path(spool) / "started-{0}".format(os.getpid())).touch()
    except OSError:
        host_logger("trex").warning(
            "Worker warning start marker unavailable", exc_info=True,
            extra={"event": "warning.transport.marker_unavailable"},
        )

    def report_warnings() -> None:
        try:
            entries = list(drain())
            if len(entries) > WORKER_WARNING_PAYLOAD_LIMIT:
                omitted = len(entries) - WORKER_WARNING_PAYLOAD_LIMIT + 1
                entries = entries[:WORKER_WARNING_PAYLOAD_LIMIT - 1] + [{
                    "category": "WorkerWarningCaptureLimit", "location": "trex.runtime.logging",
                    "message": "Additional distinct worker warnings were omitted", "count": omitted,
                }]
            destination = Path(spool) / "worker-{0}.json".format(os.getpid())
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(json.dumps(entries, allow_nan=False), encoding="utf-8")
            os.replace(temporary, destination)
        except Exception:  # noqa: BLE001 - the parent also counts the missing report
            host_logger("trex").warning(
                "Worker warning report failed", exc_info=True,
                extra={"event": "warning.transport.report_failed"},
            )

    multiprocessing.util.Finalize(None, report_warnings, exitpriority=10)


def collect_worker_warnings(spool: Optional[str], expected_workers: int) -> None:
    """Adopt every worker's payload and record the workers that never reported one.

    ``expected_workers`` is the pool size the parent started. A worker that reported
    no warnings counts as reported; one whose payload is missing or damaged counts as
    lost, so a crash cannot silently disappear from the owning record. Valid entries
    from a partially damaged report are retained, alongside its explicit loss marker.
    """
    if spool is None:
        return
    if isinstance(expected_workers, bool) or not isinstance(expected_workers, numbers.Integral) or expected_workers < 0:
        raise ValueError("expected_workers must be a nonnegative integer")
    entries: List[Dict[str, Any]] = []
    reported_pids = set()
    started_pids = set()
    directory = Path(spool)
    try:
        for marker in directory.glob("started-*"):
            started_pids.add(marker.name[len("started-"):])
        for path in sorted(directory.glob("worker-*.json")):
            try:
                with path.open("rb") as handle:
                    text = handle.read(_WORKER_WARNING_BYTES_LIMIT + 1)
                if len(text) > _WORKER_WARNING_BYTES_LIMIT:
                    continue
                payload = json.loads(text)
            except (OSError, UnicodeError, ValueError, TypeError):
                continue  # A damaged payload is lost, not trusted.
            if not isinstance(payload, list):
                continue
            valid = len(payload) <= WORKER_WARNING_PAYLOAD_LIMIT
            for entry in payload[:WORKER_WARNING_PAYLOAD_LIMIT]:
                if not isinstance(entry, dict) or "message" not in entry:
                    valid = False
                    continue
                try:
                    count = entry.get("count", 1)
                    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                        raise ValueError("invalid warning count")
                    if not all(isinstance(entry.get(key, ""), str) for key in ("category", "location", "message")):
                        raise ValueError("invalid warning fields")
                except (TypeError, ValueError):
                    valid = False
                    continue
                entries.append(
                    {
                        "category": str(entry.get("category", "Warning")),
                        "location": str(entry.get("location", "")),
                        "message": str(entry["message"]),
                        "count": count,
                    }
                )
            if valid:
                reported_pids.add(path.name[len("worker-"):-len(".json")])
    finally:
        try:
            shutil.rmtree(spool)
        except OSError:
            host_logger("trex").warning(
                "Worker warning spool cleanup failed", exc_info=True,
                extra={"event": "warning.transport.cleanup_failed"},
            )
    # A worker the pool replaced after a crash has a start marker and no report,
    # which the plain pool-size difference would miss.
    lost = max(
        max(0, int(expected_workers) - len(reported_pids)),
        len(started_pids - reported_pids),
    )
    if lost:
        message = (
            "warnings from {0} pool worker process(es) were lost before they could report "
            "({1} of {2} workers reported)".format(
                lost, len(reported_pids), int(expected_workers)
            )
        )
        entries.append(
            {
                "category": "WorkerWarningLoss",
                "location": "trex.runtime.logging.collect_worker_warnings",
                "message": message,
                "count": lost,
            }
        )
        host_logger("trex").warning(
            "warnings from %d pool worker process(es) were lost (%d of %d reported)",
            lost, len(reported_pids), int(expected_workers),
            extra={"event": "warning.transport.lost", "category": "WorkerWarningLoss", "count": lost},
        )
    if entries and _worker_warning_hooks is not None:
        _worker_warning_hooks[1](entries)


__all__ = [
    "WORKER_WARNING_PAYLOAD_LIMIT", "get_logger", "host_logger", "set_host_namespace",
    "set_worker_log_setup", "worker_log_setup", "set_worker_warning_hooks", "standalone_cli",
    "start_worker_warning_spool", "worker_warning_report", "install_worker_warning_report",
    "collect_worker_warnings",
]

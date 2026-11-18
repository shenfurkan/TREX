"""Actual TREX defensive-refinement caller with native unit-cube evaluation.

Whole-call timing receipts do not establish speedup, warmed compilation cost,
GPU utilization, same-point likelihood agreement or estimator calibration.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pytest
import trex

from trex.evidence import nested_evidence
from trex.accelerators import _jax_models as models
from trex.accelerators import _native_unit_batch
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _profile_value(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported refinement receipt value: {type(value).__name__}")


@pytest.fixture
def refinement_profile(tmp_path, request, record_property):
    root = Path(__file__).resolve().parents[1]
    origins = {module.__name__: Path(module.__file__).resolve()
               for module in (nested_evidence, models, _native_unit_batch)}
    package_root = Path(trex.__file__).resolve().parent
    assert all(path.is_relative_to(package_root) for path in origins.values()), origins
    paths = {__file__, inspect.getsourcefile(synthetic_trex_tp_integrand_case), *map(str, origins.values())}
    files = {(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved.relative_to(root.parent)).as_posix():
             hashlib.sha256(resolved.read_bytes()).hexdigest()
             for resolved in sorted(Path(path).resolve() for path in paths)}
    marker = root / "kaggle_kit_manifest.json"
    kit = None
    if marker.is_file():
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        assert all(manifest["files"].get(name) == digest for name, digest in files.items()), files
        kit = {"src_tree_sha256": manifest["src_tree_sha256"], "created_utc": manifest["created_utc"],
               "manifest_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}
    receipt = {"nodeid": request.node.nodeid, "host": platform.node(), "python": sys.version,
               "numpy_version": np.__version__, "jax_version": models.jax.__version__,
               "trex_n_jobs": os.environ.get("TREX_N_JOBS"),
               "import_origins": {name: str(path) for name, path in origins.items()},
               "selected_file_sha256": files, "kit": kit, "calls": [],
               "snapshot_scope": "Selected imported source/test/fixture hashes; full kit, environment and resource verification belongs to the Kaggle runner",
               "timing_scope": "Single whole call including proposal/density, pilot/compilation, transfers and reduction; not cold/warm decomposition or a speedup measurement",
               "claim_eligible": False, "evidence_calibrated": False}
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = tmp_path
    path = output / ("native_importance_" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16] + ".json")

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, default=_profile_value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @contextmanager
    def phase(mode, integrand, posteriors, parameters):
        inputs = {"integrand": vars(integrand), "proposal_pool": posteriors, "parameters": parameters}
        encoded = json.dumps(inputs, sort_keys=True, default=_profile_value, allow_nan=False, separators=(",", ":"))
        row = {"mode": mode, "inputs": json.loads(encoded), "inputs_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
               "artifact": str(path), "status": "started", "prepared_utc": datetime.now(timezone.utc).isoformat()}
        receipt["calls"].append(row)
        save()
        row["started_utc"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            yield row
        except BaseException as error:
            row["status"] = "raised"
            row["error"] = {"class": type(error).__name__, "message": str(error)}
            raise
        else:
            row["status"] = "returned"
        finally:
            row["wall_seconds"] = time.monotonic() - started
            row["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()

    try:
        yield phase
    finally:
        record_property("native_importance_profile", json.dumps(receipt, default=_profile_value, allow_nan=False))


def _inputs():
    scene = synthetic_trex_tp_integrand_case()
    integrand = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        nsamples=3, bin_integration_days=0.0, use_flat_priors=False, population=None, n_dimensions=4)
    # TEST_DESIGN: fixed target-neutral proposal pool, not a claim that these
    # uniform points are a sampled posterior or an independent physical oracle.
    samples = np.random.default_rng(20261021).uniform(size=(1024, 4))
    posterior = [(samples, np.full(samples.shape[0], 1.0 / samples.shape[0]))]
    return integrand, posterior


def _require_native_contract():
    assert 'native_integrand' in inspect.signature(nested_evidence._run_importance_sampler).parameters, (
        'TREX importance escalation still lacks its native integrand execution boundary')


def test_defensive_refinement_preserves_draws_density_counts_and_reduction_with_native_tp(refinement_profile):
    _require_native_contract()
    integrand, posterior = _inputs()
    kwargs = dict(n_dimensions=4, random_seed=20261022, n_jobs=1, n_draws=2048)
    with refinement_profile("scalar-cpu", integrand, posterior, kwargs) as scalar_row:
        scalar = nested_evidence._run_importance_sampler(
            integrand.prior_transform, integrand.log_likelihood, posterior, **kwargs)
        scalar_row["result"] = scalar._asdict()
    with models.jax.default_device(models.jax.devices('cpu')[0]):
        with refinement_profile("explicit-native-cpu", integrand, posterior, kwargs) as native_row:
            native = nested_evidence._run_importance_sampler(
                integrand.prior_transform, integrand.log_likelihood, posterior, native_integrand=integrand, **kwargs)
            native_row["result"] = native._asdict()
    assert scalar_row["inputs_sha256"] == native_row["inputs_sha256"]
    assert scalar_row["status"] == native_row["status"] == "returned"
    saved = json.loads(Path(native_row["artifact"]).read_text(encoding="utf-8"))
    assert saved["calls"] == [scalar_row, native_row]
    assert not Path(native_row["artifact"]).with_suffix(".json.tmp").exists()
    assert native.execution_provenance["backend"] == "jax-native-unit-cube"
    assert native.execution_provenance["logical_likelihood_calls"] == native.n_likelihood_calls
    dispatches = native.execution_provenance["device_dispatches"]
    assert dispatches and all(row["platform"] == "cpu" and row["executed_kernel_lanes"] > 0 for row in dispatches)
    assert native.n_draws == scalar.n_draws == kwargs['n_draws']
    assert native.n_likelihood_calls == scalar.n_likelihood_calls
    assert native.n_centres == scalar.n_centres and native.defensive_fraction == scalar.defensive_fraction
    assert native.random_seed == scalar.random_seed and native.estimator == scalar.estimator
    # Retained production 0.1-nat integration bar, not a new tolerance fitted
    # to a profile result. Hardware same-native parity is tested separately.
    assert abs(native.log_evidence - scalar.log_evidence) <= 0.1
    np.testing.assert_allclose(native.log_evidence_error, scalar.log_evidence_error, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(native.effective_sample_size, scalar.effective_sample_size, rtol=1e-4, atol=1e-10)


def test_native_defensive_refinement_does_not_hide_supported_numerical_failures(refinement_profile):
    _require_native_contract()
    integrand, posterior = _inputs()
    integrand.sigma = np.finfo(np.float64).tiny
    kwargs = dict(n_dimensions=4, random_seed=20261022, n_jobs=1, n_draws=128)
    with models.jax.default_device(models.jax.devices('cpu')[0]):
        with pytest.raises(RuntimeError, match='numerical|NaN|nonfinite') as caught:
            with refinement_profile("explicit-native-cpu", integrand, posterior, kwargs) as row:
                nested_evidence._run_importance_sampler(
                    integrand.prior_transform, integrand.log_likelihood, posterior, native_integrand=integrand, **kwargs)
    assert row["status"] == "raised"
    assert row["error"] == {"class": "NumericalLikelihoodError", "message": str(caught.value)}
    assert caught.value.count > 0
    assert "result" not in row
    assert row["inputs"]["integrand"]["sigma"] == integrand.sigma
    saved = json.loads(Path(row["artifact"]).read_text(encoding="utf-8"))
    assert saved["calls"] == [row]


@pytest.mark.skipif(
    os.environ.get('EXONYM_RUN_SLOW_VALIDATIONS') != '1' or os.environ.get('EXONYM_REQUIRE_GPU') != '1',
    reason='opt-in production-budget physical-GPU refinement integration acceptance')
def test_production_refinement_caller_routes_all_registered_draws_through_native_execution(refinement_profile):
    _require_native_contract()
    integrand, posterior = _inputs()
    assert len(models.jax.devices('gpu')) == 2, 'approved physical dual-GPU host is required'
    controls = nested_evidence._SamplerControls(400, 0.05, 2_000_000, 20261022, 4)
    kwargs = dict(n_dimensions=4, random_seed=controls.random_seed, n_jobs=controls.n_jobs,
                  n_draws=nested_evidence._IMPORTANCE_DRAWS)
    with refinement_profile("production-auto-native", integrand, posterior, kwargs) as native_row:
        native_row["caller"] = {"function": "_integrate", "scenario": "TP", "controls": controls._asdict()}
        result = nested_evidence._integrate(integrand, controls, 'TP', 4, posterior)
        native_row["result"] = result._asdict()
    assert result.n_draws == nested_evidence._IMPORTANCE_DRAWS == 1_000_000
    assert result.execution_provenance['backend'] == 'jax-native-unit-cube'
    gpu_rows = [row for row in result.execution_provenance['device_dispatches'] if row['platform'] == 'gpu']
    assert len(gpu_rows) == 2, result.execution_provenance
    assert all(row['executed_kernel_lanes'] > 0 and row['float_dtype'] == 'float64' for row in gpu_rows)
    assert result.execution_provenance['logical_likelihood_calls'] == result.n_likelihood_calls
    # Run the same registered proposal/RNG/density/reduction through scalar
    # evaluation; this is fixed-proposal comparison, not sampler calibration.
    with refinement_profile("scalar-cpu", integrand, posterior, kwargs) as scalar_row:
        scalar = nested_evidence._run_importance_sampler(integrand.prior_transform, integrand.log_likelihood,
            posterior, **kwargs)
        scalar_row["result"] = scalar._asdict()
    assert scalar_row["inputs_sha256"] == native_row["inputs_sha256"]
    assert result.n_likelihood_calls == scalar.n_likelihood_calls
    assert abs(result.log_evidence - scalar.log_evidence) <= 0.1
    np.testing.assert_allclose(result.log_evidence_error, scalar.log_evidence_error, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(result.effective_sample_size, scalar.effective_sample_size, rtol=1e-4, atol=1e-10)

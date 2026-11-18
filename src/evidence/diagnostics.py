"""Decision-tree diagnostics for the TREX statistical vetting engine.

Produces FPP/NFPP threshold-screen explanations from Monte Carlo evidence for
human review.

References
----------
* Giacalone et al. (2021), AJ 161, 24, DOI
  ``10.3847/1538-3881/abc6af``, Section 4.2.

``lnZ`` is dimensionless log evidence; FPP, NFPP, and scenario probabilities
are dimensionless.  ``TrexDiagnostic.value`` and ``threshold`` use the unit of
their named diagnostic (probability for FPP/NFPP).  Numerical anomalies and
all-``-inf`` scenarios are explicitly degenerate, never passes.  `TrexResult`
retains ``claim_eligible=False``; these conditional probabilities lack empirical
scene-model calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from trex.evidence.marginal_likelihoods import MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY, MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, NEIGHBOR_SCENARIO_NAMES, TARGET_SCENARIO_NAMES, N_NEIGHBOR_SCENARIOS, N_TARGET_SCENARIOS, _SCENARIO_NAMES, compute_fpp_nfpp, degenerate_evidence_scenarios, scenario_period_log_priors, unconverged_probability_mass

PUBLIC_PROBABILITY_COLUMNS: Tuple[str, ...] = (
    "ID",
    "scenario",
    "M_s",
    "R_s",
    "P_orb",
    "inc",
    "b",
    "ecc",
    "w",
    "R_p",
    "M_EB",
    "R_EB",
    "prob",
)
ALIGNED_FIT_ARRAY_NAMES: Tuple[str, ...] = (
    "star_num",
    "u1",
    "u2",
    "fluxratio_EB",
    "fluxratio_comp",
)

# POLICY_THRESHOLD: Giacalone et al. (2021) used FPP < 0.015 for its
# published classification plane. TREX deliberately applies the stricter 0.01
# routing threshold as its claim boundary. A caller that records a vetting
# decision must use this same value, so a diagnostic pass never conflicts with
# that decision; neither threshold can
# enable ``claim_eligible`` while calibrated scene integration is absent.
# ``MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY`` in ``marginal_likelihoods`` is one
# tenth of this routing boundary by registered convention, not because its
# realized mass bounds FPP error. The heuristic does not rule out missed support
# or guarantee that a routing threshold cannot be crossed.
FPP_THRESHOLD: float = 0.01  # dimensionless probability
NFPP_THRESHOLD: float = 0.001  # dimensionless probability


@dataclass
class TrexDiagnostic:
    """A single diagnostic check result."""
    name: str
    status: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class TrexResult:
    """Complete TREX vetting result with diagnostics.

    Attributes:
        fpp: False-positive probability.
        nfpp: Nearby false-positive probability.
        probs: Per-scenario probability vector; None when required inputs or
            supported numerical evaluations are missing.
        status: Normalization status.
        diagnostics: List of diagnostic messages.
        claim_eligible: Always False (scientific guardrail).
        degenerate: True if numerically degenerate.
        effective_sample_sizes: Kish effective sample size of every evaluated
            scenario's Monte Carlo weights, keyed by scenario name. It is the
            estimator's own convergence evidence and is retained whether or not
            the run met the floor.
        jax_batch_fallbacks: Per-scenario records of JAX batch likelihoods
            that raised and were re-evaluated with the batman reference path,
            keyed by scenario name. They disclose accelerator provenance that
            a transient runtime warning cannot preserve.
        input_identity: Digest of the bound scene and every scientific input
            the engine evaluated. This is the same value the Monte Carlo
            checkpoint carries, so a retained result identifies its inputs without
            that dictionary.
        period_prior_period_days: Observed transit period [days] the G21
            Sect. 2.4.1 scenario prior was evaluated at, or None when the
            caller supplied no period. Every stored ``lnZ`` remains prior-free.
        period_log_priors: Per-scenario natural-log period prior actually
            applied before normalization, keyed by scenario name.
        supplemental_nested_evidence: Estimator-labelled substitution records
            from nested sampling or MC-pilot importance, with disjoint scenario keys.
    """
    fpp: Optional[float] = None
    nfpp: Optional[float] = None
    probs: Optional[np.ndarray] = None
    status: str = "not-run"
    effective_sample_sizes: Dict[str, float] = field(default_factory=dict)
    supplemental_nested_evidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nested_evidence_escalation: Optional[Dict[str, Any]] = None
    jax_batch_fallbacks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # The engine's digest of the bound scene and every scientific input it
    # evaluated. It is the same value its Monte Carlo checkpoint carries, so a
    # retained result identifies its inputs without the checkpoint dictionary.
    input_identity: Optional[str] = None
    period_prior_period_days: Optional[float] = None
    period_log_priors: Dict[str, float] = field(default_factory=dict)
    scenario_fits: List[Dict[str, Any]] = field(default_factory=list)
    fit_evaluation: Dict[str, Any] = field(default_factory=dict)
    # Measured wall seconds per engine phase ("monte_carlo", "nested_escalation"),
    # or None for a phase that did not run in this call. A resumed escalation
    # carries the Monte Carlo phase's measurement from its checkpoint so the
    # accelerator share of a split run stays measurable.
    phase_wall_seconds: Dict[str, Optional[float]] = field(default_factory=dict)
    diagnostics: List[TrexDiagnostic] = field(default_factory=list)
    claim_eligible: bool = False
    claim_block_reason: str = (
        "This FPP/NFPP diagnostic does not create a planet claim. "
        "Claim eligibility is evaluated separately by the C1--C14 claim-readiness "
        "assessment, including calibration and independent review; inspect its "
        "candidate-specific unmet criteria."
    )
    degenerate: bool = False

    def top_scenarios(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-n scenario names and probabilities."""
        if self.probs is None or len(self.probs) == 0:
            return []
        order = np.argsort(self.probs)[::-1]
        results = []
        for idx in order[:n]:
            prob = float(self.probs[idx])
            if prob <= 0.0:
                break
            name = _SCENARIO_NAMES.get(idx, f"N{idx}")
            results.append((name, prob))
        return results

    def _require_completed_fit_state(self) -> None:
        if (
            self.degenerate
            or self.status != "ok"
            or self.probs is None
            or not self.scenario_fits
            or len(self.scenario_fits) != len(self.probs)
            or not any(entry.get("has_representative_draw") for entry in self.scenario_fits)
        ):
            raise ValueError(
                "representative scenario fit state is unavailable for an incomplete or degenerate TREX result"
            )

    def to_probability_dataframe(self):
        """Return the ordered per-scenario probability and representative-fit table.

        Evaluated scenarios populate physical columns from their highest finite
        Monte Carlo log-likelihood draw; unevaluated or zero-support scenarios
        leave unmeasured draw parameters as ``NaN`` rather than fabricating
        zeros or solar defaults.
        """
        from pandas import DataFrame

        self._require_completed_fit_state()
        rows: List[Dict[str, Any]] = []
        for idx, entry in enumerate(self.scenario_fits):
            draw = entry.get("representative_draw") if entry.get("has_representative_draw") else None
            rows.append(
                {
                    "ID": entry.get("source_id"),
                    "scenario": entry.get("scenario"),
                    "M_s": float(draw["M_s"]) if draw is not None else float("nan"),
                    "R_s": float(draw["R_s"]) if draw is not None else float("nan"),
                    "P_orb": float(draw["P_orb"]) if draw is not None else float(entry.get("P_orb", float("nan"))),
                    "inc": float(draw["inc"]) if draw is not None else float("nan"),
                    "b": float(draw["b"]) if draw is not None else float("nan"),
                    "ecc": float(draw["ecc"]) if draw is not None else float("nan"),
                    "w": float(draw["w"]) if draw is not None else float("nan"),
                    "R_p": float(draw["R_p"]) if draw is not None else float("nan"),
                    "M_EB": float(draw["M_EB"]) if draw is not None else float("nan"),
                    "R_EB": float(draw["R_EB"]) if draw is not None else float("nan"),
                    "prob": float(self.probs[idx]),
                }
            )
        return DataFrame(rows, columns=list(PUBLIC_PROBABILITY_COLUMNS))

    def aligned_fit_arrays(self) -> Dict[str, np.ndarray]:
        """Return ``star_num``, ``u1``, ``u2``, ``fluxratio_EB``, and ``fluxratio_comp`` aligned to ``scenario_fits``."""
        self._require_completed_fit_state()
        count = len(self.scenario_fits)
        star_num = np.zeros(count, dtype=np.int64)
        u1 = np.full(count, np.nan, dtype=float)
        u2 = np.full(count, np.nan, dtype=float)
        fluxratio_eb = np.full(count, np.nan, dtype=float)
        fluxratio_comp = np.full(count, np.nan, dtype=float)
        for idx, entry in enumerate(self.scenario_fits):
            star_num[idx] = int(entry.get("star_num", 1))
            draw = entry.get("representative_draw") if entry.get("has_representative_draw") else None
            if draw is not None:
                u1[idx] = float(draw["u1"])
                u2[idx] = float(draw["u2"])
                if "secondary_to_host_flux_ratio" in draw:
                    r = float(draw["secondary_to_host_flux_ratio"])
                    c = float(draw["fluxratio_comp"])
                    secondary = r * c if draw["companion_is_host"] else r
                    target = 1.0 - c if draw["companion_is_host"] else 1.0
                    fluxratio_eb[idx] = secondary / (target + secondary)
                else:
                    fluxratio_eb[idx] = float(draw.get("fluxratio_EB", 0.0))
                fluxratio_comp[idx] = float(draw["fluxratio_comp"])
        return {
            "star_num": star_num,
            "u1": u1,
            "u2": u2,
            "fluxratio_EB": fluxratio_eb,
            "fluxratio_comp": fluxratio_comp,
        }

    def representative_models(self, time: np.ndarray) -> List[Optional[np.ndarray]]:
        """Replay actual MC draws in the observed aperture frame at relative days.

        Uses retained exposure/bin widths and the same planet/EB integration
        rules as the likelihood models. A scenario without a retained draw
        returns ``None``: a nested evidence estimate is not a fitted profile.
        This is model replay, not posterior inference or independent validation.
        """
        from trex.evidence.fit_replay import replay_representative_model

        self._require_completed_fit_state()
        required = ("exptime_days", "bin_integration_days", "nsamples")
        if any(name not in self.fit_evaluation for name in required):
            raise ValueError("retained exposure integration is unavailable for fit replay")
        options = {name: self.fit_evaluation[name] for name in required}
        return [
            replay_representative_model(time, entry, **options)
            if entry.get("has_representative_draw") else None
            for entry in self.scenario_fits
        ]


def _default_scenario_label(index: int) -> str:
    if index < N_TARGET_SCENARIOS:
        return _SCENARIO_NAMES[index]
    offset = (index - N_TARGET_SCENARIOS) % N_NEIGHBOR_SCENARIOS
    return NEIGHBOR_SCENARIO_NAMES[offset]


def generate_diagnostics(
    lnZ: np.ndarray,
    verbose: bool = False,
    effective_sample_sizes: Optional[Mapping[int, float]] = None,
    nested_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    scenario_fits: Optional[Mapping[int, Mapping[str, Any]]] = None,
    importance_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    period_days: Optional[float] = None,
) -> TrexResult:
    """Generate diagnostic interpretations from scenario log-evidences.

    Args:
        lnZ: Log-evidence array per scenario. These are prior-free stored
            evidences; ``period_days`` is applied here, once, at normalization.
        verbose: Include per-scenario probability breakdown.
        effective_sample_sizes: Optional per-scenario Kish effective sample
            sizes from ``calc_target_evidences``. Supplying them lets a
            single-draw-dominated evidence set be reported as degenerate
            instead of being normalized into an FPP.
        nested_provenance: Optional per-scenario nested-sampling records for
            scenarios whose Monte Carlo evidence was re-integrated by the
            nested estimator (``evidence_escalation``). They are retained on
            the result so the report can disclose which estimator produced
            each substituted probability.
        scenario_fits: Retained per-scenario input/numerical status and actual
            Monte Carlo representative draws, indexed like ``lnZ``.
        importance_provenance: Accepted MC-pilot importance substitution records.
            Keys must not overlap ``nested_provenance``: one scenario has one
            final estimator.
        period_days: Optional observed transit period [days].  When supplied,
            the G21 Sect. 2.4.1 scenario prior is applied to every scenario
            before its probability is formed and the per-scenario factor is
            retained.  A resumed or escalated call passes the same prior-free
            evidences, so the factor is never applied twice.

    Returns:
        TrexResult with diagnostics and FPP/NFPP.
    """
    result = TrexResult()
    log_priors = (
        scenario_period_log_priors(period_days, len(lnZ))
        if period_days is not None
        else None
    )
    fpp, nfpp, probs, norm_status = compute_fpp_nfpp(
        lnZ, effective_sample_sizes, period_days=period_days
    )
    unavailable = [
        str(record.get("diagnostic_key", _SCENARIO_NAMES.get(index, f"N{index}")))
        for index, record in (scenario_fits or {}).items()
        if record.get("status") == "unavailable"
    ]
    numerical = {
        str(record.get("diagnostic_key", _SCENARIO_NAMES.get(index, f"N{index}"))): int(record.get("unresolved_numerical_failures", 0))
        for index, record in (scenario_fits or {}).items()
        if record.get("unresolved_numerical_failures", 0)
    }
    if (unavailable or numerical) and norm_status in ("ok", "all_neginf"):
        fpp, nfpp, norm_status = float("nan"), float("nan"), "degenerate_evidence"
    if unavailable or numerical:
        probs = None
    result.fpp = fpp if np.isfinite(fpp) else None
    result.nfpp = nfpp if np.isfinite(nfpp) else None
    result.probs = probs
    result.status = norm_status
    if log_priors is not None:
        result.period_prior_period_days = float(period_days)
        result.period_log_priors = {
            _SCENARIO_NAMES.get(index, "N{0}".format(index)): float(value)
            for index, value in enumerate(log_priors)
        }
    if effective_sample_sizes is not None:
        result.effective_sample_sizes = {
            _SCENARIO_NAMES.get(index, "N{0}".format(index)): float(value)
            for index, value in effective_sample_sizes.items()
        }
    if nested_provenance:
        result.supplemental_nested_evidence = {
            str(name): dict(record) for name, record in nested_provenance.items()
        }
    if importance_provenance:
        if result.supplemental_nested_evidence.keys() & importance_provenance.keys():
            raise ValueError("a scenario cannot have both nested and MC-pilot importance substitutions")
        result.supplemental_nested_evidence.update({
            str(name): dict(record) for name, record in importance_provenance.items()
        })
    if scenario_fits:
        ordered_fits: List[Dict[str, Any]] = []
        for index in range(len(lnZ)):
            diagnostic_key = _SCENARIO_NAMES.get(index, "N{0}".format(index))
            raw = scenario_fits.get(index)
            if raw is None:
                entry: Dict[str, Any] = {
                    "scenario_index": index,
                    "scenario": _default_scenario_label(index),
                    "diagnostic_key": diagnostic_key,
                    "source_id": None,
                    "neighbor_index": None if index < N_TARGET_SCENARIOS else (index - N_TARGET_SCENARIOS) // N_NEIGHBOR_SCENARIOS,
                    "status": "record-unavailable",
                    "reason": "scenario fit record was not supplied",
                    "origin": None,
                    "evidence_origin": "unknown",
                    "has_representative_draw": False,
                    "representative_draw": None,
                    "star_num": 1,
                    "P_orb": float("nan"),
                    "mc_ln_evidence": float(lnZ[index]),
                    "ln_evidence": float(lnZ[index]),
                }
            else:
                entry = dict(raw)
                if entry.get("representative_draw") is not None:
                    entry["representative_draw"] = dict(entry["representative_draw"])
                entry["ln_evidence"] = float(lnZ[index])
            if nested_provenance and diagnostic_key in nested_provenance:
                entry["evidence_origin"] = "nested-escalation"
            elif importance_provenance and diagnostic_key in importance_provenance:
                entry["evidence_origin"] = "mc-pilot-defensive-importance"
            if log_priors is not None and index < log_priors.size:
                entry["period_log_prior"] = float(log_priors[index])
                entry["period_prior"] = float(math.exp(log_priors[index]))
            entry["prob"] = (
                float(probs[index])
                if probs is not None and index < len(probs) and norm_status == "ok"
                else float("nan")
            )
            ordered_fits.append(entry)
        result.scenario_fits = ordered_fits

    diags: List[TrexDiagnostic] = []

    if unavailable:
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="scenario-input-unavailable", status="fail",
            message="Required scenario inputs are unavailable, not zero support: " + ", ".join(unavailable),
        ))
    if numerical:
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="scenario-numerical-failure", status="fail",
            message="Supported likelihood evaluations failed numerically: "
                    + ", ".join(f"{name}={count}" for name, count in numerical.items()),
        ))

    if effective_sample_sizes is not None:
        failures = degenerate_evidence_scenarios(lnZ, effective_sample_sizes)
        if failures:
            worst = sorted(failures.items(), key=lambda item: item[1])
            listed = ", ".join("{0}={1:.3g}".format(name, ess) for name, ess in worst[:5])
            mass = (
                unconverged_probability_mass(probs, effective_sample_sizes, lnZ)
                if probs is not None and len(probs)
                else 0.0
            )
            if norm_status == "degenerate_evidence":
                result.degenerate = True
                zero_names = [
                    _SCENARIO_NAMES.get(index, "N{0}".format(index))
                    for index in effective_sample_sizes
                    if 0 <= index < len(lnZ) and np.isneginf(lnZ[index])
                ]
                diags.append(TrexDiagnostic(
                    name="evidence-effective-sample-size", status="fail",
                    message=(
                        "{0} of {1} evaluated scenarios did not integrate to the "
                        "registered floor of {2:g} effective draws, and they hold "
                        "{3:.3g} of the realized normalized probability (allowance {4:g}). "
                        "Zero finite sampled weight is unbounded, not certified zero: {6}. "
                        "The smallest ESS values are {5}; no FPP is derived.".format(
                            len(failures), len(effective_sample_sizes),
                            MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, mass,
                            MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY, listed,
                            ", ".join(zero_names) or "none",
                        )
                    ),
                    value=mass,
                    threshold=MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY,
                ))
            else:
                # Retain the heuristic and its limitation; small realized mass is not an error bound.
                diags.append(TrexDiagnostic(
                    name="evidence-effective-sample-size", status="info",
                    message=(
                        "{0} of {1} evaluated scenarios stayed below the {2:g}-draw "
                        "floor with estimated normalized mass {3:.3g}, within the "
                        "{4:g} routing allowance. This realized mass is not an "
                        "evidence-error bound and does not establish C5 convergence; "
                        "missed support can change FPP. The smallest are {5}.".format(
                            len(failures), len(effective_sample_sizes),
                            MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, mass,
                            MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY, listed,
                        )
                    ),
                    value=mass,
                    threshold=MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY,
                ))

    if importance_provenance:
        diags.append(TrexDiagnostic(
            name="mc-pilot-importance-substitution", status="info",
            message="MC-pilot defensive importance substituted: " + ", ".join(sorted(importance_provenance)),
        ))

    if nested_provenance:
        names = ", ".join(sorted(nested_provenance))
        substituted_mass = float(
            np.sum(
                [
                    float(record.get("substituted_probability", 0.0))
                    for record in nested_provenance.values()
                ]
            )
        )
        diags.append(TrexDiagnostic(
            name="evidence-estimator-substitution", status="info",
            message=(
                "{0} boundary-supported scenario(s) were re-integrated by nested "
                "sampling (importance-refined where the ensemble missed the "
                "0.1-nat bar) because their Monte Carlo evidence was a single-draw "
                "statistic: {1}. Their substituted probabilities total {2:.3g}; "
                "the per-scenario nested records are retained in the report.".format(
                    len(nested_provenance), names, substituted_mass
                )
            ),
            value=substituted_mass,
        ))

    if norm_status == "anomaly":
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="numerical-stability", status="fail",
            message="Numerical anomaly (NaN/+inf in log-evidences).",
        ))
    elif norm_status == "all_neginf":
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="all-neginf", status="fail",
            message="All lnZ are -inf. FPP=1.0 is a failed computation.",
        ))

    if result.degenerate:
        result.diagnostics = diags
        return result

    if result.fpp is not None:
        fpp_ok = result.fpp < FPP_THRESHOLD
        diags.append(TrexDiagnostic(
            name="fpp", status="pass" if fpp_ok else "fail",
            message=f"FPP = {result.fpp:.6f} {'<' if fpp_ok else '>='} {FPP_THRESHOLD}",
            value=result.fpp, threshold=FPP_THRESHOLD,
        ))

    if result.nfpp is not None:
        nfpp_ok = result.nfpp < NFPP_THRESHOLD
        diags.append(TrexDiagnostic(
            name="nfpp", status="pass" if nfpp_ok else "fail",
            message=f"NFPP = {result.nfpp:.6f} {'<' if nfpp_ok else '>='} {NFPP_THRESHOLD}",
            value=result.nfpp, threshold=NFPP_THRESHOLD,
        ))

    if verbose and result.probs is not None:
        for name, prob in result.top_scenarios(5):
            diags.append(TrexDiagnostic(
                name=f"scenario-{name}", status="info",
                message=f"P({name}) = {prob:.4f}", value=prob,
            ))

    overall = (
        result.fpp is not None and result.nfpp is not None
        and result.fpp < FPP_THRESHOLD and result.nfpp < NFPP_THRESHOLD
    )
    diags.append(TrexDiagnostic(
        name="overall", status="pass" if overall else "fail",
        message=(
            "FPP/NFPP threshold screen PASS (diagnostic only)"
            if overall
            else "FPP/NFPP threshold screen FAIL (diagnostic only)"
        ),
    ))

    result.diagnostics = diags
    return result


# NUMERICAL_POLICY: how far a residual sub-floor scenario's Monte Carlo probability is
# assumed underestimated beyond the largest value any preregistered repeat realization
# recorded for it. A registered decision-robustness margin, not an evidence-error bound: an
# ESS~1 estimate is heavy-tailed, so no number of repeats bounds its mean.
# In a synthetic scene, a resolved-neighbour scenario and SEBx2P (max
# probabilities 5e-6 and 9e-7) varied at most ~5x between seeds; EB
# varied 1e20x but never exceeded 3e-51, and EBx2P varied 1e11x up to 1.6e-6. The factor covers
# the first kind with three orders of magnitude to spare and makes the third decision-relevant,
# so such a scenario is nested-escalated in any run whose own estimate could flip C1/C2
# (``evidence_escalation``); no factor covers it by itself. Unbounded without repeats.
RESIDUAL_UNDERESTIMATE_FACTOR: float = 1e4
# Stands in for an unbounded evidence weight against normalized masses that sum to one.
_UNBOUNDED_WEIGHT: float = 1e200
PLANET_SCENARIO_NAMES = ("TP", "PTP", "DTP")


def residual_could_change_decision(
    scenarios: Mapping[str, float], residual, repeats, factor: float = None, *, unbounded_residual=()
) -> bool:
    """Whether residual sub-floor scenarios could flip the joint C1-and-C2 decision.

    Each residual scenario's weight ranges over ``[0, high]``: ``high`` is the largest
    probability any realization recorded for it times ``RESIDUAL_UNDERESTIMATE_FACTOR``, and
    unbounded without repeats. C1/C2 hold iff ``g1 = sum_false w - t1 sum w < 0`` and
    ``g2 = sum_neighbour w - t2 sum w < 0``, linear in the weights (FPP sums non-planet
    scenarios, NFPP resolved-neighbour ones, as ``compute_fpp_nfpp`` does). Planet residuals
    lower both, neighbour residuals raise both, and target false-positive residuals raise g1 and
    lower g2, so the extremes below are exact.
    """
    t1, t2 = FPP_THRESHOLD, NFPP_THRESHOLD
    factor = RESIDUAL_UNDERESTIMATE_FACTOR if factor is None else factor
    if (isinstance(factor, (bool, np.bool_)) or not math.isfinite(factor) or factor <= 0):
        raise ValueError("residual underestimate factor must be finite and positive")
    if any(name not in scenarios for name in residual):
        raise ValueError("every residual scenario must have a recorded probability")
    if any(isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0.0
           for weights in (scenarios, *repeats) for value in weights.values()):
        raise ValueError("scenario weights must be finite and non-negative")
    if not 0 < t1 < 1 or not 0 < t2 < 1:
        raise ValueError("decision thresholds must lie strictly between zero and one")
    high = {
        name: (max([scenarios[name]] + [other.get(name, 0.0) for other in repeats]) * factor
               if repeats and name not in unbounded_residual else _UNBOUNDED_WEIGHT)
        for name in residual
    }

    def g(weights):
        total = math.fsum(weights.values())
        false = math.fsum(v for k, v in weights.items() if k not in PLANET_SCENARIO_NAMES)
        neighbour = math.fsum(v for k, v in weights.items() if k not in TARGET_SCENARIO_NAMES)
        return false - t1 * total, neighbour - t2 * total

    planet = [name for name in residual if name in PLANET_SCENARIO_NAMES]
    neighbour = [name for name in residual if name not in TARGET_SCENARIO_NAMES]
    target_false = [name for name in residual if name not in planet and name not in neighbour]
    g1, g2 = g(scenarios)
    if g1 < 0 and g2 < 0:
        raise_false = {**scenarios, **{n: (0.0 if n in planet else high[n]) for n in residual}}
        raise_neighbour = {**scenarios, **{n: (high[n] if n in neighbour else 0.0) for n in residual}}
        return g(raise_false)[0] >= 0 or g(raise_neighbour)[1] >= 0
    # Failing now: best case for passing is planets high, neighbours zero, and a total target
    # false-positive weight W in [0, H] with c1 + (1 - t1) W < 0 and c2 - t2 W < 0.
    c1, c2 = g({**scenarios, **{n: (high[n] if n in planet else 0.0) for n in residual}})
    H = math.fsum(high[n] for n in target_false)
    lowest = max(0.0, c2 / t2)
    return (1.0 - t1) * lowest + c1 < 0 and (c2 < 0 or lowest < H)


__all__ = [
    "RESIDUAL_UNDERESTIMATE_FACTOR",
    "residual_could_change_decision",
    "TrexDiagnostic", "TrexResult", "generate_diagnostics",
    "PUBLIC_PROBABILITY_COLUMNS", "ALIGNED_FIT_ARRAY_NAMES",
    "FPP_THRESHOLD", "NFPP_THRESHOLD",
    "MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE",
    "MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY",
]

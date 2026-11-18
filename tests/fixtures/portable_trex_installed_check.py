"""Installed-package probe; inputs are supplied by synthetic_observations only.

Execute in a clean standalone environment with the selected wheel/editable install. This
checks real public inference and replay, not independent physical calibration.
"""

import importlib.abc
import importlib.metadata
import json
import math
from pathlib import Path
import sys


class NoExonym(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "exonym":
            raise ImportError("EXONYM is forbidden in the standalone TREX acceptance probe")


def main():
    sys.meta_path.insert(0, NoExonym())
    try:
        importlib.metadata.distribution("exonym")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("EXONYM must not be installed in this environment")

    import numpy as np
    import trex
    from trex.app.engine import run_trex_vetting
    from trex.app.scene import TargetScene
    from trex.app.target import target

    payload = json.loads(Path(sys.argv[1]).read_text())
    source_root = payload["source_root"]
    expected_root = Path(source_root).resolve() if source_root else Path(sys.prefix).resolve()
    assert Path(trex.__file__).resolve().is_relative_to(expected_root)
    if source_root:
        assert Path(trex.__file__).resolve() == expected_root / "__init__.py"
    scene = TargetScene.from_dict(payload["scene"])
    binding = payload["binding"]
    binding["sector_weights"] = {int(key): value for key, value in binding["sector_weights"].items()}
    obj = target.from_scene(scene, **binding)
    time, flux, sigma = (np.array(payload[name], dtype=float) for name in ("time", "flux", "sigma"))
    obj.calc_probs(time, flux, sigma, payload["period_days"],
                   exptime=binding["exptime_days"], N=4096, nsamples=4)
    assert obj.FPP is not None and obj.NFPP is not None
    # Physical validity at the public result boundary; this is not calibration.
    assert math.isfinite(obj.FPP) and 0.0 <= obj.FPP <= 1.0
    assert math.isfinite(obj.NFPP) and 0.0 <= obj.NFPP <= 1.0
    assert obj.trex_result.claim_eligible is False and obj.trex_result.status == "ok"
    results = Path(sys.argv[1]).parent / "standalone-results"
    obj.write_outputs(results)
    summary = json.loads((results / "summary.json").read_text())
    assert summary["input_identity"] == obj.trex_result.input_identity
    assert obj.trex_result.input_identity is not None
    assert len(obj.probs) == 21 and obj.probs["ID"].nunique() == 3
    models = obj.trex_result.representative_models(time)
    assert any(model is not None and np.ptp(model) > 0 for model in models)
    assert all(model is None or np.isfinite(model).all() for model in models)
    output = Path(sys.argv[1]).parent / "standalone-fits"
    obj.plot_fits(time, flux, sigma, save=True, fname=str(output))
    assert output.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert not any(name.split(".")[0] == "exonym" for name in sys.modules)
    print(json.dumps({
        "trex_module": trex.__file__, "engine_module": run_trex_vetting.__module__,
        "python_executable": sys.executable, "exonym_installed": False,
        "scenario_rows": len(obj.probs), "n_draws_test_only": 4096,
        "status": obj.trex_result.status, "claim_eligible": False,
    }))


if __name__ == "__main__":
    main()

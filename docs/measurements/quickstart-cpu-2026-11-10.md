# Synthetic quick-start: CPU completion, 2026-11-10

This is a public-safe excerpt of a retained Kaggle test record. The input was
the synthetic `quickstart.ipynb`, not an observation. The run used a frozen
TREX snapshot from 2026-11-09; later source may differ. The attempted budget
was 1,000,000 draws per scenario.

The executed notebook reported:

```text
wall time 18312 s on x86_64, 4 CPUs, JAX devices: cpu
status: ok
FPP  = 1.0819255685882508e-05
NFPP = 3.4212592932996246e-10
```

The test that ran the notebook completed in 18,418.3 s, including setup and
earlier work; 18,312 s is the notebook wall time. The run emitted nearest-grid
warnings for host limb-darkening requests outside the TRICERATOPS TESS grid;
the per-draw provenance is retained in the vetting artifact.

An earlier full-budget run of the synthetic notebook on a Kaggle T4 GPU host
reported 2,273 s, FPP 1.0819255685792538e-05 and NFPP 3.421259293268131e-10.
The two records are different code snapshots and are not a controlled speed
benchmark. Their FPPs differ by 9.0e-17 absolute (8.3e-12 relative).

This record does not establish a real-target result. The synthetic scene and
light curve are invented, and the CPU timing is one Kaggle host. The earlier
paused CPU run is retained in
[quickstart-cpu-2026-11-09.md](quickstart-cpu-2026-11-09.md) as history.

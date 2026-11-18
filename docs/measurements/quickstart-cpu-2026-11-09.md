# Synthetic quick-start: CPU pause, 2026-11-09

> **Superseded for the full-budget CPU result.** A later snapshot completed the
> same budget on four Kaggle CPUs in 18,312 s; see
> [quickstart-cpu-2026-11-10.md](quickstart-cpu-2026-11-10.md). This record
> remains the historical pause of the earlier snapshot.

This is an excerpt of the retained private Kaggle test record. The input was
the synthetic `quickstart.ipynb`, not an observation. This run used a frozen
TREX snapshot; it is not a measurement of every later version. The attempted
budget was 1,000,000 draws per scenario. The frozen source set the refinement
wall ceiling to `540 * 60.0` seconds. The environment record reported four CPUs
and the JAX CPU backend.

The test log ended with:

```text
NestedCheckpointPaused: nested wall deadline reached before refinement
1 failed in 34904.40s (9:41:44)
```

The elapsed time is for the test as a whole, including setup and earlier work;
it is not the nine-hour refinement ceiling alone. The test did not produce a
completed CPU FPP. This record does not establish CPU/GPU numerical agreement
or a matched-accuracy speed ratio. The later successful full-budget run is
recorded in [quickstart-cpu-2026-11-10.md](quickstart-cpu-2026-11-10.md).

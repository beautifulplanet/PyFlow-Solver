# Configuration & Hash Contract (v1)

The simulation configuration drives numerical behavior and restart
compatibility. We compute a short deterministic hash (`config_hash`) over only
the *semantic* fields so that checkpoints created under one run can be safely
validated against a restarted run with equivalent physics.

Included fields (examples): `nx`, `ny`, `Re`, `lid_velocity`, `cfl_target`,
`cfl_growth`, `lin_tol`, `lin_maxiter`, advection scheme identifiers,
geometry extents (`lx`, `ly`), and booleans that change numerical operators
like `disable_advection`.

Excluded runtime / presentation / orchestration fields: `force_quiet`,
`log_path`, `enable_jacobi_pc`, `assert_invariants`, `seed`, `progress`.

Rationale: A user should be able to change logging destinations, enable a
quiet mode, attach invariant assertions, or pick a different random seed
without invalidating restart compatibility for a given numerical setup.

If a new field affects the mathematics (stencil, discretization, stability,
termination conditions beyond superficial presentation), it MUST be included
in the hash. To do so, ensure it is not in `EXCLUDED_RUNTIME_FIELDS`. If a
new runtime-only field is added, append it to `EXCLUDED_RUNTIME_FIELDS`.

A regression test (`tests/test_config_hash_regression.py`) enforces this
contract by mutating excluded fields and verifying the hash remains stable.

## Quickstart

Run a short simulation with JSON streaming:

```
python -m pyflow.cli --nx 32 --ny 32 --steps 50 --json-stream > run.jsonl
```

Restart from a checkpoint (hash enforced):

```
python -m pyflow.cli --restart checkpoint.npz --steps 100
```

Benchmark micro performance:

```
python -m benchmarks.micro_step_benchmark --steps 100 --nx 32 --ny 32
```

## Restart Workflow
1. Run with periodic checkpoints: `--checkpoint ckpt.npz --checkpoint-interval 50`.
2. If interrupted, restart: `--restart ckpt.npz`.
3. If configuration changes materially, CLI aborts with mismatch details.



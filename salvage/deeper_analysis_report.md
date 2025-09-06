# Deeper Analysis Report
Generated: 2025-08-29T19:55:59.264834+00:00

## Temporal Trends
Buckets: 2017-01, 2025-06, 2025-07, 2025-08, unknown
Recent bucket failure rates: 2017-01=0.0%, 2025-06=50.0%, 2025-07=50.9%, 2025-08=50.0%, unknown=14.2%

## Function Quality
Median quality score: 96
Top 5 functions: test_residuals_calculation, test_residuals_calculation, plot_results, plot_results, _run_validation_solver
Orphan salvage candidates: 5440 (listed below if any)
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\VS PY\Solver #2\pyflow_solver\test_cpp_cfd.py::test_residuals_calculation score=118
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\lid_a1_ (1).py::plot_results score=116
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\VS PY\Solver #2\pyflow_solver\tests\test_validation.py::_run_validation_solver score=116
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\lid_a1_ (1).py::create_staggered_grid score=114
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\VS PY\Solver #2\pyflow_solver\tests\test_benchmark_quick.py::test_quick_benchmark_re400 score=114
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\_12.py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\cpdail\cpsail_finitude_14.py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\cpdail\finitude_language_12 (1).py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\cpdail\finitude_language_12.py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\cpdail\_12 (1).py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\cpdail\_12.py::get_eos score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\py cfd\pycfdflow2.py::interpolate_face_velocity score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\py cfd\pycfdflow2.py::interpolate_face_pressure score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\py cfd\pycfdflow2.py::interpolate_face_velocity score=112
- C:\Users\Elite\Documents\commands\VS Studio\pythA\VS PY\CFD_Engine\src\Frameworks and fail logs\Colab\py cfd\pycfdflow2.py::interpolate_face_pressure score=112

## Refactor Priorities (Top 10)
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\integrate\_lebedev.py::get_lebedev_sphere (priority 4632) [long no doc]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\optimize\_minimize.py::minimize (priority 1642) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\numpy\lib\_npyio_impl.py::genfromtxt (priority 1627) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\optimize\_lsq\least_squares.py::least_squares (priority 1531) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\sparse\linalg\_eigen\lobpcg\lobpcg.py::lobpcg (priority 1502) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\numpy\f2py\crackfortran.py::analyzeline (priority 1211) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\integrate\_ivp\ivp.py::solve_ivp (priority 1209) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\_lib\cobyqa\main.py::minimize (priority 1194) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\integrate\_quadpack_py.py::quad (priority 1167) [high complexity long]
- C:\Users\Elite\Documents\commands\VS Studio\pythA\.venv\Lib\site-packages\scipy\_lib\pyprima\cobyla\cobylb.py::cobylb (priority 1157) [high complexity long]

## Module Centrality
Top modules by coupling: numpy(1102), scipy(741), pytest(455), sys(312), typing(290), os(287), warnings(227), __future__(223), pip(204), collections(171)

## Evolution Chains
Total chains: 226
- __init__ length=225 funcΔ=0 classΔ=0
- pycfdflow2_ length=12 funcΔ=8 classΔ=0
- __main__ length=8 funcΔ=0 classΔ=0
- test_regression length=8 funcΔ=0 classΔ=0
- setup length=7 funcΔ=0 classΔ=0

## Recommendations
- Extract shared logic from high-coupling central modules into stable APIs to reduce ripple risk.
- Refactor top priority functions before adding new features; high complexity correlates with future failures.
- Promote orphan high-quality functions from failing files into a utilities/core module.
- Monitor failure_rate trend; reduce by instituting pre-commit complexity guardrails.
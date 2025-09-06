# PyFlow Hybrid Solver Benchmarking

This document explains how to use the benchmarking system to compare the performance of the pure Python implementation with the C++/Python hybrid implementation.

## Prerequisites

Before running the benchmarks, ensure you have the following installed:

- Python 3.7 or newer
- NumPy
- Matplotlib
- pybind11
- A C++ compiler (Visual Studio on Windows, GCC/Clang on Linux/macOS)

You can install the Python requirements using:

```bash
pip install numpy matplotlib pybind11
```

## Building the C++ Extensions

Before running the benchmarks, you need to build the C++ extensions:

### On Windows

Run the provided batch script:

```
build_extensions.bat
```

This will:
1. Check if all dependencies are installed
2. Build the C++ extension modules in-place

### On Linux/macOS

Run the setup script directly:

```bash
python setup.py build_ext --inplace
```

## Running the Benchmarks

### Using the Provided Scripts

On Windows, simply run:

```
run_benchmark.bat
```

On Linux/macOS:

```bash
python benchmark_hybrid.py
```

### Customizing the Benchmarks

You can modify the parameters in `benchmark_hybrid.py` to test different configurations:

- Change Reynolds numbers by modifying the `Re_values` list
- Change grid sizes by modifying the `grid_sizes` list
- Adjust the time step or total simulation time in the function calls

## Benchmark Results

The benchmark script will generate:

1. A CSV file (`benchmark_results.csv`) with detailed performance data
2. PNG images showing:
   - Comparison of residual histories between Python and hybrid implementations
   - Visualizations of the computed flow fields
   - Differences between the solutions
   - A speedup graph showing performance gains for different configurations

## Understanding the Results

### Tabular Output

The benchmark will print a summary table with columns:

- `Re`: Reynolds number
- `Grid`: Grid size
- `Python Time (s)`: Execution time for pure Python
- `Hybrid Time (s)`: Execution time for hybrid implementation
- `Speedup`: Performance gain (Python Time / Hybrid Time)
- `u diff`, `v diff`, `p diff`: Maximum absolute differences in solution variables

### Residual History Plots

These plots show how the residuals decrease during the simulation. Lower residuals indicate better convergence. Comparing the Python and hybrid implementations helps verify that both are solving the same problem correctly.

### Solution Field Plots

These compare the flow fields computed by both methods to ensure they're producing the same results.

## Troubleshooting

### "ImportError: No module named 'pyflow_core'"

This error means the C++ extensions have not been built or were not built correctly. Run the build script again and check for any error messages.

### Compilation Errors

If you encounter compilation errors:

1. Ensure you have a compatible C++ compiler installed
2. For Windows: Install Visual Studio with the "Desktop development with C++" workload
3. For Linux: Install GCC with `sudo apt install build-essential`
4. For macOS: Install Xcode Command Line Tools with `xcode-select --install`

### Performance Issues

If you don't see significant speedups:
1. Check if your compiler is using optimization flags (`-O3` or `/O2`)
2. Verify that C++ extensions are actually being loaded (the code should print "Using C++ accelerated components" when running)

## Advanced Customization

For more detailed performance analysis:
- Use the Python `cProfile` module: `python -m cProfile -s cumtime benchmark_hybrid.py`
- Modify the number of pressure iterations in the solver to balance accuracy and speed
- Try different under-relaxation factors in the solver implementations

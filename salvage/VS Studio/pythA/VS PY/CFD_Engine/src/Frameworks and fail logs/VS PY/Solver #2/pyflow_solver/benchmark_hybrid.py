"""
Benchmark script to compare the performance of the pure Python solver
with the hybrid Python/C++ implementation.
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

# Define our own simple table formatting function to avoid requiring the tabulate package
def format_table(data, headers=None, tablefmt="grid"):
    """Simple function to format tabular data without external dependencies"""
    if not data:
        return ""
    
    # Calculate column widths
    widths = []
    if headers:
        for i, header in enumerate(headers):
            col_data = [str(row[i]) for row in data]
            widths.append(max(len(str(header)), max(len(x) for x in col_data)))
    else:
        for i in range(len(data[0])):
            widths.append(max(len(str(row[i])) for row in data))
    
    # Create the table
    result = []
    
    # Add a horizontal line
    hline = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    result.append(hline)
    
    # Add headers if provided
    if headers:
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += " " + str(header).ljust(widths[i]) + " |"
        result.append(header_row)
        result.append(hline)
    
    # Add data rows
    for row in data:
        data_row = "|"
        for i, cell in enumerate(row):
            data_row += " " + str(cell).ljust(widths[i]) + " |"
        result.append(data_row)
    
    # Add final horizontal line
    result.append(hline)
    
    return "\n".join(result)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflow.solver import solve_lid_driven_cavity as solve_python
from pyflow.hybrid_solver import solve_lid_driven_cavity as solve_hybrid
from pyflow.grid import Grid
from pyflow.logging import LiveLogger

def run_benchmark(Re_values=[100], grid_sizes=[33, 65, 97]):
    """
    Run benchmarks comparing Python vs C++ implementation for different
    Reynolds numbers and grid sizes.
    
    Parameters:
    ----------
    Re_values : list
        Reynolds numbers to test
    grid_sizes : list
        Grid sizes to test
    """
    results = []
    
    for Re in Re_values:
        for N in grid_sizes:
            # Configure simulation parameters
            L = 1.0
            grid = Grid(N, L)
            dx = dy = grid.dx
            
            # Adjust dt based on grid size for stability
            dt = min(0.001, 0.25 * dx * dx)
            
            # Set simulation time (longer for higher Re)
            T = 2.0 if Re == 100 else 3.0
            
            # Set pressure solver iterations
            p_iterations = 500 if Re == 100 else 1000
            
            print(f"\n--- Benchmarking Re={Re}, Grid={N}×{N} ---")
            
            # Run Python solver
            print("Running Python solver...")
            logger_py = LiveLogger(N, Re, dt, T, log_interval=1000)
            start_time = time.time()
            try:
                u_py, v_py, p_py, res_py = solve_python(
                    N, dx, dy, Re, dt, T,
                    p_iterations=p_iterations,
                    logger=logger_py
                )
                py_time = time.time() - start_time
                print(f"Python time: {py_time:.3f} seconds")
            except Exception as e:
                print(f"Error in Python solver: {e}")
                print("Skipping this configuration...")
                continue
            
            # Run hybrid solver
            print("Running hybrid solver...")
            logger_hybrid = LiveLogger(N, Re, dt, T, log_interval=1000)
            start_time = time.time()
            try:
                # Handle differences in function signatures between solvers if needed
                u_hybrid, v_hybrid, p_hybrid, res_hybrid = solve_hybrid(
                    N, dx, dy, Re, dt, T,
                    p_iterations=p_iterations,
                    logger=logger_hybrid
                )
                hybrid_time = time.time() - start_time
                print(f"Hybrid time: {hybrid_time:.3f} seconds")
            except Exception as e:
                print(f"Error in hybrid solver: {e}")
                print("Skipping this configuration...")
                continue
            
            # Calculate speedup
            if hybrid_time > 0:
                speedup = py_time / hybrid_time
            else:
                speedup = float('inf')
                
            print(f"Speedup: {speedup:.2f}x")
            
            # Calculate solution differences
            u_diff = np.max(np.abs(u_py - u_hybrid))
            v_diff = np.max(np.abs(v_py - v_hybrid))
            p_diff = np.max(np.abs(p_py - p_hybrid))
            
            print(f"Maximum solution differences:")
            print(f"  u: {u_diff:.8f}")
            print(f"  v: {v_diff:.8f}")
            print(f"  p: {p_diff:.8f}")
            
            # Store results
            results.append({
                'Re': Re,
                'Grid': f"{N}×{N}",
                'Python Time (s)': py_time,
                'Hybrid Time (s)': hybrid_time,
                'Speedup': speedup,
                'u diff': u_diff,
                'v diff': v_diff,
                'p diff': p_diff
            })
            
            # Compare residual history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.semilogy(res_py['u_res'], 'b-', label='Python')
            plt.semilogy(res_hybrid['u_res'], 'r--', label='Hybrid')
            plt.xlabel('Iteration')
            plt.ylabel('u-momentum Residual')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.semilogy(res_py['v_res'], 'b-', label='Python')
            plt.semilogy(res_hybrid['v_res'], 'r--', label='Hybrid')
            plt.xlabel('Iteration')
            plt.ylabel('v-momentum Residual')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.semilogy(res_py['cont_res'], 'b-', label='Python')
            plt.semilogy(res_hybrid['cont_res'], 'r--', label='Hybrid')
            plt.xlabel('Iteration')
            plt.ylabel('Continuity Residual')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"benchmark_Re{Re}_N{N}_residuals.png", dpi=300)
            plt.close()
            
            # Compare solution fields
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.contourf(grid.X, grid.Y, u_hybrid, levels=30, cmap='viridis')
            plt.colorbar(label='u-velocity')
            plt.title(f'Hybrid Solution\nRe={Re}, Grid={N}×{N}')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plt.subplot(1, 3, 2)
            plt.contourf(grid.X, grid.Y, u_py, levels=30, cmap='viridis')
            plt.colorbar(label='u-velocity')
            plt.title('Python Solution')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plt.subplot(1, 3, 3)
            plt.contourf(grid.X, grid.Y, np.abs(u_hybrid - u_py), cmap='hot')
            plt.colorbar(label='|Difference|')
            plt.title('Absolute Difference')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plt.tight_layout()
            plt.savefig(f"benchmark_Re{Re}_N{N}_solutions.png", dpi=300)
            plt.close()
    
    # Print summary table
    print("\n=== Benchmark Results ===")
    headers = ['Re', 'Grid', 'Python Time (s)', 'Hybrid Time (s)', 'Speedup', 'u diff', 'v diff', 'p diff']
    table_data = [[r['Re'], r['Grid'], 
                  f"{r['Python Time (s)']:.3f}", 
                  f"{r['Hybrid Time (s)']:.3f}", 
                  f"{r['Speedup']:.2f}x",
                  f"{r['u diff']:.8f}",
                  f"{r['v diff']:.8f}",
                  f"{r['p diff']:.8f}"] for r in results]
    
    print(format_table(table_data, headers=headers))
    
    # Save results to CSV
    with open("benchmark_results.csv", 'w') as f:
        f.write(','.join(headers) + '\n')
        for r in results:
            f.write(f"{r['Re']},{r['Grid']},{r['Python Time (s)']:.3f},{r['Hybrid Time (s)']:.3f},"
                   f"{r['Speedup']:.2f},{r['u diff']:.8f},{r['v diff']:.8f},{r['p diff']:.8f}\n")
    
    # Create speedup plot
    plt.figure(figsize=(10, 6))
    
    # Group by grid size
    for N in grid_sizes:
        Re_list = []
        speedup_list = []
        
        for r in results:
            if r['Grid'] == f"{N}×{N}":
                Re_list.append(r['Re'])
                speedup_list.append(r['Speedup'])
        
        if Re_list:
            plt.plot(Re_list, speedup_list, 'o-', label=f"Grid {N}×{N}")
    
    plt.xlabel('Reynolds Number')
    plt.ylabel('Speedup (Python Time / Hybrid Time)')
    plt.title('Performance Speedup of Hybrid C++/Python vs Pure Python')
    plt.grid(True)
    plt.legend()
    plt.savefig("benchmark_speedup.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("PyFlow Hybrid Solver Benchmark")
    print("=" * 60)
    print("\nThis benchmark compares the performance of the pure Python solver")
    print("with the C++/Python hybrid implementation.")
    print("\nRESULTS WILL BE SAVED TO:")
    print("- benchmark_results.csv: Summary of all benchmark runs")
    print("- benchmark_Re{Re}_N{N}_residuals.png: Residual history plots")
    print("- benchmark_Re{Re}_N{N}_solutions.png: Solution field comparisons")
    print("- benchmark_speedup.png: Overall performance comparison")
    print("\n" + "=" * 60)
    
    # Default benchmark configuration
    Re_values = [100, 400, 1000]
    grid_sizes = [33, 65, 97]
    
    # Allow command-line customization
    if len(sys.argv) > 1:
        # Simple command-line argument handling
        # Example: python benchmark_hybrid.py --re 100,400 --grid 33,65
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--re" and i+1 < len(sys.argv):
                try:
                    Re_values = [int(x) for x in sys.argv[i+1].split(',')]
                    i += 2
                except ValueError:
                    print(f"Error: Invalid Reynolds numbers: {sys.argv[i+1]}")
                    sys.exit(1)
            elif sys.argv[i] == "--grid" and i+1 < len(sys.argv):
                try:
                    grid_sizes = [int(x) for x in sys.argv[i+1].split(',')]
                    i += 2
                except ValueError:
                    print(f"Error: Invalid grid sizes: {sys.argv[i+1]}")
                    sys.exit(1)
            elif sys.argv[i] == "--help" or sys.argv[i] == "-h":
                print("\nUsage: python benchmark_hybrid.py [options]")
                print("\nOptions:")
                print("  --re RE_LIST     Comma-separated list of Reynolds numbers (default: 100,400,1000)")
                print("  --grid GRID_LIST  Comma-separated list of grid sizes (default: 33,65,97)")
                print("  --help, -h        Show this help message and exit")
                sys.exit(0)
            else:
                i += 1
    
    # Adjust grid sizes based on Reynolds numbers
    # For high Re, we might want to avoid the largest grids for time reasons
    if 1000 in Re_values and 129 in grid_sizes:
        print("Note: Removing grid size 129 for Re=1000 to keep benchmark duration reasonable")
        grid_sizes.remove(129)
    
    print(f"\nRunning benchmarks for:")
    print(f"- Reynolds numbers: {Re_values}")
    print(f"- Grid sizes: {grid_sizes}")
    print("\nThis may take a while depending on the configuration...\n")
    
    # Run the benchmark
    run_benchmark(
        Re_values=Re_values,
        grid_sizes=grid_sizes
    )
    
    print("\nBenchmark completed! Check the output files for detailed results.")

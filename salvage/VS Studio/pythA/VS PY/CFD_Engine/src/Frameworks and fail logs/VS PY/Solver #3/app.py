import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_SCRIPT = os.path.join(PROJECT_ROOT, 'build.ps1')
TEST_SCRIPT = os.path.join(PROJECT_ROOT, 'run_all_tests.ps1')
SOLVER_EXE = os.path.join(PROJECT_ROOT, 'build', 'hpf_cfd.exe')

TEST_EXECUTABLES = [
    os.path.join(PROJECT_ROOT, 'build', exe)
    for exe in [
        'test_bc.exe',
        'test_fields.exe',
        'test_io_utils.exe',
        'test_parameters.exe',
        'test_solver_nan.exe',
    ]
]

def run_powershell(script_path):
    try:
        result = subprocess.run([
            'powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path
        ], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def run_executable(exe_path):
    try:
        result = subprocess.run([exe_path], check=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {exe_path}: {e}")
        return False

def print_menu():
    print("\n==== CFD Solver Application ====")
    print("1. Build Fortran solver and tests")
    print("2. Run main solver")
    print("3. Run all tests")
    print("4. Run individual test")
    print("5. View output files")
    print("0. Exit")

def list_output_files():
    output_dir = os.path.join(PROJECT_ROOT, 'output')
    if not os.path.isdir(output_dir):
        print("No output directory found.")
        return
    files = os.listdir(output_dir)
    if not files:
        print("No output files found.")
        return
    print("\nOutput files:")
    for f in files:
        print(f"- {f}")

def main():
    while True:
        print_menu()
        choice = input("Select an option: ").strip()
        if choice == '1':
            print("\nBuilding...")
            run_powershell(BUILD_SCRIPT)
        elif choice == '2':
            print("\nRunning main solver...")
            run_executable(SOLVER_EXE)
        elif choice == '3':
            print("\nRunning all tests...")
            run_powershell(TEST_SCRIPT)
        elif choice == '4':
            print("\nAvailable tests:")
            for idx, exe in enumerate(TEST_EXECUTABLES, 1):
                print(f"{idx}. {os.path.basename(exe)}")
            t_choice = input("Select test to run (number): ").strip()
            try:
                t_idx = int(t_choice) - 1
                if 0 <= t_idx < len(TEST_EXECUTABLES):
                    run_executable(TEST_EXECUTABLES[t_idx])
                else:
                    print("Invalid test selection.")
            except ValueError:
                print("Invalid input.")
        elif choice == '5':
            list_output_files()
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()

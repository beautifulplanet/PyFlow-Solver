import os
import subprocess
import sys

BUILD_DIR = 'build'
TESTS_DIR = 'tests'

# List all test_*.f90 files in tests/
test_sources = [f for f in os.listdir(TESTS_DIR) if f.startswith('test_') and f.endswith('.f90')]
test_names = [os.path.splitext(f)[0] for f in test_sources]

def build_test(test_name):
    exe = os.path.join(BUILD_DIR, test_name + ('.exe' if os.name == 'nt' else ''))
    obj = os.path.join(BUILD_DIR, test_name + '.o')
    src = os.path.join(TESTS_DIR, test_name + '.f90')
    # Compile object
    subprocess.run(['gfortran', '-O2', '-Wall', '-Wextra', '-fimplicit-none', '-std=f2008', '-c', src, '-o', obj], check=True)
    # Link with core objects
    core = ['parameters.o','fields.o','boundary_conditions.o','io_utils.o','solver.o']
    core_objs = [os.path.join(BUILD_DIR, o) for o in core]
    subprocess.run(['gfortran', '-O2', '-Wall', '-Wextra', '-fimplicit-none', '-std=f2008', '-o', exe] + core_objs + [obj], check=True)
    return exe

def run_test(test_name):
    exe = os.path.join(BUILD_DIR, test_name + ('.exe' if os.name == 'nt' else ''))
    if not os.path.exists(exe):
        print(f'Building {test_name}...')
        build_test(test_name)
    print(f'Running {test_name}...')
    result = subprocess.run([exe], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def menu():
    while True:
        print('\nTest Menu:')
        print('0. Run ALL tests')
        for i, name in enumerate(test_names, 1):
            print(f'{i}. Run {name}')
        print('q. Quit')
        choice = input('Select an option: ').strip().lower()
        if choice == '0':
            for name in test_names:
                run_test(name)
        elif choice == 'q':
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(test_names):
            run_test(test_names[int(choice)-1])
        else:
            print('Invalid choice.')

if __name__ == '__main__':
    # Ensure build dir and core objects exist
    if not os.path.isdir(BUILD_DIR):
        print('Build directory not found. Please build core modules first (e.g., with build.ps1 or Makefile).')
        sys.exit(1)
    core_objs = [os.path.join(BUILD_DIR, o) for o in ['parameters.o','fields.o','boundary_conditions.o','io_utils.o','solver.o']]
    if not all(os.path.exists(o) for o in core_objs):
        print('Core object files missing. Please build core modules first (e.g., with build.ps1 or Makefile).')
        sys.exit(1)
    menu()

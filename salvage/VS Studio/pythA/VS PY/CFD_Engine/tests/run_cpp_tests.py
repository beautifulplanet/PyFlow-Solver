import subprocess
import sys

# Path to the C++ test executable (adjust if needed)
test_exe = '../build/runTests.exe'  # or './runTests' on Linux/Mac

try:
    result = subprocess.run([test_exe], capture_output=True, text=True, check=True)
    print("C++ Test Output:")
    print(result.stdout)
    if result.stderr:
        print("C++ Test Errors:")
        print(result.stderr)
    sys.exit(result.returncode)
except FileNotFoundError:
    print(f"Test executable not found: {test_exe}")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print("Test run failed:")
    print(e.stdout)
    print(e.stderr)
    sys.exit(e.returncode)

import json
import os
import sys

# Read the configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    raw = f.read()
    try:
        config = json.loads(raw) if raw.strip() else {}
    except Exception:
        config = {}

# Extract the solver type
solver = config.get('solver', 'unknown')

print(f"Running simulation with {solver} solver...")

# Simulate the solver running
if solver == "pyfoam":
    print("PyFOAM simulation completed successfully.")
elif solver == "finitude":
    print("Finitude simulation completed successfully.")
else:
    print(f"Unknown solver: {solver}")
    sys.exit(1)

print("Simulation completed successfully.")
sys.exit(0)

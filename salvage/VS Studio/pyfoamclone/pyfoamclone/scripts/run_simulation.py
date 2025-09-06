import json
import os
import sys
from pyfoamclone.logging_utils import get_logger

# Read the configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Extract the solver type
solver = config.get('solver', 'unknown')

logger = get_logger()
logger.info(f"Running simulation with {solver} solver...")

# Simulate the solver running
if solver == "pyfoam":
    logger.info("PyFOAM simulation completed successfully.")
elif solver == "finitude":
    logger.info("Finitude simulation completed successfully.")
else:
    logger.error(f"Unknown solver: {solver}")
    sys.exit(1)

logger.info("Simulation completed successfully.")
sys.exit(0)
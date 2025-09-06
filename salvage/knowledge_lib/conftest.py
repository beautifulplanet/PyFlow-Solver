import sys, os
# Ensure project root path is on sys.path for 'framework' package imports
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

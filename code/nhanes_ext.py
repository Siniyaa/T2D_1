"""Compatibility entry point for the unified harmonisation/evaluation pipeline.

The former cached-pickle path used a different mapping; those caches must not be
used to reproduce the harmonised models or the revised external evaluation.
"""
from nhanes_val import main

if __name__ == '__main__':
    main()

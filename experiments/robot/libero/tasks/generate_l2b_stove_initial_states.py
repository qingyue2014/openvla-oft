"""
Canonical entrypoint for current L2-B2 stove initial-state generation.

The implementation lives in generate_l2b1_stove_initial_states.py for backward
compatibility with older L2-B1 pilots.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import main


if __name__ == "__main__":
    main()

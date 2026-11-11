"""
Photometry, Tracking, and Spectral Coherence Analysis for Mouse 311 (Young)

Usage:
    python analyze_Mouse_311.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if os.path.exists(os.path.join(REPO_ROOT, "analyze_mouse.py")):
    sys.path.insert(0, REPO_ROOT)
    from analyze_mouse import process_single_mouse
else:
    raise FileNotFoundError("Place this folder inside the mea-social-photometry repository to execute.")


def main():
    process_single_mouse(mouse_id="311", age_group="Young", output_dir=HERE, generate_plots=True)
    print("Analysis complete for Mouse 311 (Young).")


if __name__ == "__main__":
    main()

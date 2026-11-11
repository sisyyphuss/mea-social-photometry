"""
Master One-Command Runner: Executes Core Cohort Analysis, Extended Controls, and 8-Pass Verification

Usage:
    python run_all.py [--raw_zip PATH] [--tracking_dir PATH] [--output_dir PATH]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cohort_analysis import run_pipeline
from run_extended_analysis import run_extended
from verify_pipeline import run_verification


def main():
    parser = argparse.ArgumentParser(description="Execute full photometry, tracking & statistical pipeline")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to Data_2.zip or extracted Data_2 folder")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to results_social tracking directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom results output directory")
    args = parser.parse_args()

    print("=== STEP 1/3: CORE COHORT PIPELINE (FIGURES 1-6 & INDIVIDUAL ANIMALS) ===")
    run_pipeline(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, output_dir=args.output_dir)

    print("\n=== STEP 2/3: EXTENDED QC, TRANSIENTS, LMM & SELECTIVITY (FIGURES 7-10) ===")
    run_extended(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, results_dir=args.output_dir)

    print("\n=== STEP 3/3: 8-PASS VERIFICATION AUDIT ===")
    run_verification(results_dir=args.output_dir)


if __name__ == "__main__":
    main()

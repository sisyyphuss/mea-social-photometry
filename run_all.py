"""
Run the full MeA fiber photometry and behavioral analysis pipeline.

Usage:
    python run_all.py [--raw_zip PATH] [--tracking_dir PATH] [--output_dir PATH]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cohort_analysis import run_pipeline
from run_extended_analysis import run_extended


def main():
    parser = argparse.ArgumentParser(description="Run full photometry and behavioral analysis pipeline")
    parser.add_argument("--raw_zip", type=str, default=None, help="Path to Data_2.zip or extracted Data_2 folder")
    parser.add_argument("--tracking_dir", type=str, default=None, help="Path to results_social tracking directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom results output directory")
    args = parser.parse_args()

    run_pipeline(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, output_dir=args.output_dir)
    run_extended(raw_zip=args.raw_zip, tracking_dir=args.tracking_dir, results_dir=args.output_dir)


if __name__ == "__main__":
    main()

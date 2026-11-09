r"""Master End-to-End Quantitative Trading Research Pipeline.

DSA5205 Machine Learning in Quantitative Finance — Project 2
Group 28 (Yu Jie, Marco, Diana)

By default, executing `python main.py` runs the entire multi-strategy research
pipeline end-to-end across all 5 strategies and both benchmarks, printing the
master comparative scorecard matching Table 1 in the final report and saving
all comparison figures into figures/.

Usage:
    python main.py                  # Run full master showdown (All strategies + benchmarks)
    python main.py --all            # Explicit alias for master showdown
    python main.py --strategy 1     # Run standalone Strategy 1 (Ridge Mean Reversion)
    python main.py --strategy 2     # Run standalone Strategy 2 (Macro Trend & Momentum)
    python main.py --strategy 3     # Run standalone Strategy 3 (Volatility Targeting)
"""

import argparse
import sys
import warnings
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_consolidated_showdown import run_showdown, plot_showdown_figures


def run_strategy1():
    """Execute standalone Strategy 1 (Daily Dip-Buying via Ridge Regression)."""
    from scripts.run_extension1_mean_reversion import run_strategy1 as _run_s1
    _run_s1()


def run_strategy2():
    """Execute standalone Strategy 2 (Macro Trend & ML Momentum)."""
    from scripts.run_extension2_momentum import (
        fetch_momentum_data,
        engineer_momentum_features,
        run_strategy2_models,
        plot_extension2_figures,
    )
    df_raw = fetch_momentum_data()
    df_m = engineer_momentum_features(df_raw)
    test_m, results = run_strategy2_models(df_m)
    plot_extension2_figures(df_m, test_m)


def run_strategy3():
    """Execute standalone Strategy 3 (Dynamic Volatility Targeting)."""
    from scripts.run_extension3_volatility import run_strategy3 as _run_s3
    _run_s3()


def run_master_pipeline():
    """Execute the full end-to-end multi-strategy showdown."""
    df_showdown, results = run_showdown()
    plot_showdown_figures(df_showdown, results)
    print("\nMaster Pipeline execution completed successfully.")
    print("All comparison figures and metric scorecards have been generated.")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(
        description="DSA5205 Project 2 — Master Quantitative Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Full master showdown (All 5 strategies + 2 benchmarks)
  python main.py --strategy 1     # Strategy 1 standalone (Ridge Mean Reversion)
  python main.py --strategy 2     # Strategy 2 standalone (Macro Trend & Momentum)
  python main.py --strategy 3     # Strategy 3 standalone (Volatility Targeting)
        """,
    )
    parser.add_argument(
        "--strategy",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run an individual standalone strategy (1, 2, or 3). Defaults to running all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly run the master multi-strategy showdown across all strategies and benchmarks (default behavior).",
    )

    args = parser.parse_args()

    if args.strategy == 1:
        run_strategy1()
    elif args.strategy == 2:
        run_strategy2()
    elif args.strategy == 3:
        run_strategy3()
    else:
        run_master_pipeline()


if __name__ == "__main__":
    main()

r"""Extension 3 Standalone Pipeline: Volatility-Targeted Position Sizing.

DSA5205 Machine Learning in Quantitative Finance — Project 2
Group 28 (Diana — Strategy 3)

This script executes the complete standalone pipeline for Strategy 3:
1. Ingests unified NVDA & SPY market data (2015–2023)
2. Computes 20-day backward-looking realized volatility
3. Dynamically scales position exposure w_t = min(1.0, sigma_target / sigma_t)
4. Backtests across the 449-day out-of-sample test window with 10 bps friction
5. Runs target volatility sensitivity ablation (20% to 60%)
6. Generates publication tear sheets in figures/ext3_*.png
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy_volatility import main as run_strategy3

if __name__ == "__main__":
    run_strategy3()

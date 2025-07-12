# Kalman-Filter Pairs-Trading Demo

A small, self-contained project that shows how to:

1. pull daily prices from Yahoo Finance;
2. estimate a **time-varying hedge ratio** (beta) with a Kalman filter;
3. create mean-reversion signals from a 60-day rolling z-score;
4. run a dollar-neutral back-test with 5 bp transaction costs;
5. save equity-curve PNG files and print a one-line performance table.

---

## How to run


# install the only libraries we need
pip install numpy pandas matplotlib yfinance statsmodels

# option 1 – Jupyter
jupyter lab kalman_pairs_scanner.ipynb   # press “Run all”

# option 2 – command line
python scan_pairs.py


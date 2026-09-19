# Equity Signal Backtest

An end-to-end machine learning backtest for daily equity trading signals, built from scratch to learn how systematic strategies are researched and evaluated. It pulls historical market data, engineers features, trains a classifier to predict next-day direction, and backtests a probability-thresholded strategy against buy-and-hold.

## What it does

For a set of tickers (AAPL, MSFT, GOOG by default), the pipeline:

1. **Pulls data** — daily OHLCV from Yahoo Finance (2020–2024), indexed by ticker and date.
2. **Engineers features** — daily and 5/10-day lagged returns, 10- and 30-day simple moving averages and their ratio, 10-day rolling volatility, a volume ratio, and SPY's daily return as a market factor.
3. **Labels** — a binary target: 1 if the next day's return is positive, 0 otherwise.
4. **Splits by date** — trains on 2020–2022, tests on 2023–2024, with no shuffling, so the model never sees the future.
5. **Trains** — a class-balanced random forest classifier on standardized features (scaler fit on the training set only).
6. **Backtests** — takes a long position when predicted probability clears a threshold (0.6), and compares strategy cumulative return against buy-and-hold.
7. **Evaluates** — total return, annualized volatility, Sharpe ratio, model accuracy, and days in market, plus price/SMA and equity-curve plots per ticker.

## Running it

```bash
pip install pandas numpy yfinance scikit-learn matplotlib
python main.py
```

Edit `ticker_list` in `main.py` to change the universe.

## Structure

- `main.py` — the pipeline: data, split, preprocessing, training, inference, backtest, reporting.
- `helper.py` — feature engineering, strategy returns, evaluation metrics, and plotting.

## Known limitations

These are deliberate — this is a signal-research prototype, not a live-tradeable system — but worth being explicit about:

- **No transaction costs or slippage.** The backtest trades the next-day return at zero cost; real costs would eat into the returns shown.
- **Predicting next-day direction is inherently hard.** Daily direction is close to a coin flip, so accuracy near 50% is expected; strong results should be treated with suspicion rather than celebration.
- **The 0.6 probability threshold is fixed, not tuned.** A fuller version would calibrate it or sweep it.
- **A random forest ignores temporal ordering** within its inputs; it's a reasonable first model, not a sequence-aware one.

## Possible next steps

Model transaction costs, tune the entry threshold, add walk-forward validation, and compare the random forest against a sequence model.

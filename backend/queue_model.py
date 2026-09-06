"""Queue intelligence: cell occupancy -> headcount, M/M/c, and smoothing.

Three pieces, all pure functions so they can be unit-tested against textbook
values and fitted on the queue-clip ground truth in W3/W5:

- `fit_regression` / `predict_count`: least-squares `count = a*sum(cells) + b`.
  Before the fit lands (W3), the naive fallback is `0.8 * cells_with_occ > 40`.
- `ema`: exponential smoothing, the running headcount smoother.
- `erlang_c` / `erlang_wait`: the M/M/c steady-state delay formulas, plus
  `holt` for forecasting the arrival rate 15 minutes ahead.
"""
from __future__ import annotations

import math


def fit_regression(xs, ys):
    """Least-squares slope/intercept for count ~ a*sum(cells) + b.

    xs: list of sum(cells) values (the regressor)
    ys: list of true counts (the target)
    Returns (a, b, r2, mae).
    """
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty and the same length")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        a, b = 0.0, my
    else:
        a = sxy / sxx
        b = my - a * mx

    preds = [a * x + b for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    mae = sum(abs(y - p) for y, p in zip(ys, preds)) / n
    return a, b, r2, mae


def predict_count(cells, a, b):
    """Headcount from the raw cell occupancy list (16 cells, 0..100)."""
    s = sum(cells)
    return max(0.0, a * s + b)


def naive_count(cells, threshold=40, scale=0.8):
    """Fallback before the regression is fitted: 0.8 * cells over 40."""
    return scale * sum(1 for c in cells if c > threshold)


def ema(prev, value, alpha=0.3):
    """Exponential moving average; `prev` may be None to seed."""
    if prev is None:
        return float(value)
    return alpha * float(value) + (1.0 - alpha) * float(prev)


def erlang_c(rho, c):
    """Probability an M/M/c arrival must wait (Erlang C formula).

    rho = lambda / mu (total offered load), c = number of servers.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if rho >= c:
        return 1.0                      # unstable / fully saturated: everyone waits

    inv_sum = sum((rho ** k) / math.factorial(k) for k in range(c))
    inv_sum += (rho ** c) / (math.factorial(c) * (1.0 - rho / c))
    p0 = 1.0 / inv_sum
    return (rho ** c) / (math.factorial(c) * (1.0 - rho / c)) * p0


def erlang_wait(rho, c, mu):
    """Expected wait Wq = P(wait) / (c*mu - lambda), in the same time units as mu.

    mu is the per-server service rate (customers/time). lambda = rho * mu.
    """
    if mu <= 0:
        raise ValueError("mu must be positive")
    if rho >= c:
        return float("inf")
    return erlang_c(rho, c) / (c * mu - rho * mu)


def forecast_wait(rho, c, mu, naive_wait):
    """Blend the M/M/c number with a naive count/mu floor.

    The plan blends with `count / mu` and takes the max so the prediction is
    never absurdly low when the queue is visibly long but lambda looks small.
    """
    mm = erlang_wait(rho, c, mu)
    return max(mm, naive_wait)


def holt(values, alpha=0.3, beta=0.1, horizon=1):
    """Holt's linear trend forecast, `horizon` steps ahead.

    Returns (level, trend, forecast). Forecast is level + horizon*trend.
    """
    if not values:
        raise ValueError("values must be non-empty")
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
    return level, trend, level + horizon * trend

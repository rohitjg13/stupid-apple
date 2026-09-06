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


class QueueEngine:
    """Stateful queue intelligence: cells in, `queue_estimate` payloads out.

    One instance per store. It holds the three numbers the M/M/c model needs
    and nothing else:

    - `count` per lane, smoothed with an EMA (alpha=0.3) off the regression
    - `lambda`, arrivals to checkout per second, over a 10-minute window
    - `mu`, per-lane service rate, from POS inter-departure times

    The published wait is `max(Erlang-C Wq, count/mu)`: the plan's blend, so a
    visibly long queue never reports a two-second wait because lambda happened
    to read low. Holt's linear trend projects lambda 15 minutes ahead, which is
    what makes the staffing alert fire *before* the peak rather than during it.
    """

    PUBLISH_INTERVAL_S = 2.0
    LAMBDA_WINDOW_S = 600.0         # N = 10 min, per the plan
    MIN_LAMBDA_WINDOW_S = 60.0      # never divide by less than this (see below)
    LAMBDA_SAMPLE_S = 60.0          # one Holt sample a minute
    PROJECT_S = 900.0               # forecast horizon: 15 min
    DEFAULT_MU_PER_S = 1.0 / 45.0   # 45 s a customer until the POS says otherwise

    def __init__(self, counters=2, target_wait_s=180.0, a=None, b=None,
                 alpha=0.3, publish_interval_s=None, default_mu_per_s=None):
        self.counters = int(counters)
        self.target_wait_s = float(target_wait_s)
        self.a, self.b = a, b           # fitted regression, or None for naive
        self.alpha = float(alpha)
        self.publish_interval_s = float(publish_interval_s or self.PUBLISH_INTERVAL_S)
        self.default_mu_per_s = float(default_mu_per_s or self.DEFAULT_MU_PER_S)

        self.counts = {}                # lane -> smoothed headcount
        self._levels = {}               # lane -> that count as a whole number
        self._arrivals = []             # store-wide arrival times to checkout
        self._departures = {}           # lane -> [txn times]
        self._lambda_series = []        # per-minute lambda, for Holt
        self._last_sample_t = None
        self._last_publish_t = None

    # ---- fitting -----------------------------------------------------------
    def fit(self, sums, truths):
        """Least-squares fit on the annotated clip; returns (a, b, r2, mae)."""
        self.a, self.b, r2, mae = fit_regression(sums, truths)
        return self.a, self.b, r2, mae

    def raw_count(self, cells):
        if self.a is None or self.b is None:
            return naive_count(cells)
        return predict_count(cells, self.a, self.b)

    # ---- inputs ------------------------------------------------------------
    def on_lane_occ(self, t, lane, cells):
        """A lane_occ event: smooth the headcount, count any new arrivals.

        An arrival is a *whole person* joining, so the trigger is the smoothed
        count crossing to a higher integer. Treating any upward wobble as an
        arrival made lambda track the lane_occ sample rate rather than the
        queue: on a 45-second clip it read 120 arrivals a minute and asked for
        sixteen counters to be opened.
        """
        lane = int(lane)
        prev = self.counts.get(lane)
        value = ema(prev, self.raw_count(cells), self.alpha)
        self.counts[lane] = value
        level = int(value)
        prev_level = self._levels.get(lane)
        self._levels[lane] = level
        if prev_level is not None and level > prev_level:
            # The plan allows lane-count increases as the arrival signal when
            # the door tripwire is the noisier of the two.
            self._arrivals.extend([float(t)] * (level - prev_level))
        self._trim(t)
        return value

    def on_arrival(self, t):
        """A tripwire 'out' (walking towards checkout), the preferred signal."""
        self._arrivals.append(float(t))
        self._trim(t)

    def on_departure(self, t, lane):
        """A pos_txn: one customer served on this lane."""
        self._departures.setdefault(int(lane), []).append(float(t))
        self._trim(t)

    def _trim(self, t):
        cut = float(t) - self.LAMBDA_WINDOW_S
        self._arrivals = [x for x in self._arrivals if x >= cut]
        for lane, ts in self._departures.items():
            self._departures[lane] = [x for x in ts if x >= cut]

    # ---- parameters --------------------------------------------------------
    def lambda_rate(self, t):
        """Arrivals per second over the trailing window.

        The divisor never drops below a minute. Straight after warm-up the
        observed span is a second or two, and dividing a handful of arrivals by
        it extrapolated to 60 arrivals a minute and an "open counter 16" alert
        before anyone had queued. Under-reading a rate you have not watched
        long enough is the right way to be wrong about staffing.
        """
        self._trim(t)
        observed = float(t) - self._t0(t)
        window = min(self.LAMBDA_WINDOW_S, max(self.MIN_LAMBDA_WINDOW_S, observed))
        return len(self._arrivals) / window

    def _t0(self, t):
        return self._arrivals[0] if self._arrivals else float(t)

    def mu(self, lane=None):
        """Per-server service rate from POS inter-departure times."""
        if lane is not None:
            ts = sorted(self._departures.get(int(lane), []))
        else:
            ts = sorted(x for v in self._departures.values() for x in v)
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if not gaps:
            return self.default_mu_per_s
        return 1.0 / (sum(gaps) / len(gaps))

    def sample_lambda(self, t):
        """Record one lambda sample a minute; Holt forecasts off this series."""
        if self._last_sample_t is not None and t - self._last_sample_t < self.LAMBDA_SAMPLE_S:
            return None
        self._last_sample_t = float(t)
        value = self.lambda_rate(t)
        self._lambda_series.append(value)
        del self._lambda_series[:-60]        # an hour of samples is plenty
        return value

    def projected_lambda(self, t):
        """Lambda projected `PROJECT_S` ahead, so staffing beats the peak."""
        if len(self._lambda_series) < 2:
            return self.lambda_rate(t)
        steps = self.PROJECT_S / self.LAMBDA_SAMPLE_S
        _level, _trend, forecast = holt(self._lambda_series, horizon=steps)
        return max(0.0, forecast)

    # ---- output ------------------------------------------------------------
    def due(self, t):
        return (self._last_publish_t is None
                or float(t) - self._last_publish_t >= self.publish_interval_s)

    def estimates(self, t):
        """`queue_estimate` payloads for every lane. Publish these every 2 s."""
        self._last_publish_t = float(t)
        self.sample_lambda(t)
        lam = self.lambda_rate(t)
        mu_all = self.mu()
        rho = lam / mu_all if mu_all > 0 else float("inf")
        store_wq = erlang_wait(rho, self.counters, mu_all) if mu_all > 0 else float("inf")

        out = []
        for lane in sorted(self.counts):
            count = self.counts[lane]
            mu_lane = self.mu(lane)
            naive = count / mu_lane if mu_lane > 0 else float("inf")
            wait = max(store_wq, naive)
            if not math.isfinite(wait):
                # Saturated model: fall back to the observable queue rather
                # than publishing infinity into the dashboard.
                wait = naive if math.isfinite(naive) else 0.0
            out.append({"lane": lane, "count": round(float(count), 2),
                        "pred_wait_s": round(float(wait), 1)})
        return out

    def advice(self, t):
        """Staffing advice on the *projected* lambda, or None if all is well."""
        from backend.staffing import staffing_advice
        return staffing_advice(self.counters, self.projected_lambda(t), self.mu(),
                               self.target_wait_s)


def cells_by_lane(lane_table, lane_occupancy):
    """Split a FrameResult's 16-cell `lane_occupancy` into per-lane 16-cell lists.

    `lane_table` is lanes.json in table order, so entry i of the config is
    cell i of the hardware array (pl/regs.py packs it that way). Each lane gets
    a 16-long list because that is what the frozen `lane_occ` payload and the
    16-byte blob in the schema expect; unused cells stay zero.
    """
    out = {}
    for i, entry in enumerate(lane_table):
        if i >= len(lane_occupancy):
            break
        lane = int(entry["lane"])
        cell = int(entry["cell"])
        cells = out.setdefault(lane, [0] * 16)
        if 0 <= cell < 16:
            cells[cell] = int(lane_occupancy[i])
    return out

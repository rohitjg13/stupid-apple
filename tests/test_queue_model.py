"""backend/queue_model.py and backend/staffing.py — textbook M/M/c, EMA, Holt."""
import math

import pytest

from backend.queue_model import (ema, erlang_c, erlang_wait, fit_regression,
                                 forecast_wait, holt, naive_count, predict_count)
from backend.staffing import required_servers, staffing_advice


# ---- regression -----------------------------------------------------------

def test_fit_regression_recovers_known_slope():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]          # y = 2x + 1
    a, b, r2, mae = fit_regression(xs, ys)
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
    assert mae == pytest.approx(0.0)


def test_fit_regression_reports_error():
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 1.0, 1.0]               # constant target -> perfect mean model
    a, b, r2, mae = fit_regression(xs, ys)
    assert a == pytest.approx(0.0) and b == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_predict_count_never_negative():
    assert predict_count([0] * 16, 0.1, -10.0) == 0.0
    assert predict_count([50] * 16, 0.1, 0.0) == pytest.approx(80.0)


def test_naive_count_matches_plan():
    cells = [10, 50, 90, 30, 70] + [0] * 11
    assert naive_count(cells) == pytest.approx(0.8 * 3)


# ---- smoothing ------------------------------------------------------------

def test_ema_seeds_on_first_value():
    assert ema(None, 10.0) == 10.0


def test_ema_converges():
    prev = ema(None, 0.0, alpha=0.5)
    prev = ema(prev, 100.0, alpha=0.5)
    assert prev == 50.0
    assert ema(50.0, 100.0, alpha=0.5) == 75.0


# ---- Erlang C -------------------------------------------------------------

def test_erlang_c_m11_with_low_load():
    # M/M/1 at rho=0.5: P(wait) = rho = 0.5
    assert erlang_c(0.5, 1) == pytest.approx(0.5)


def test_erlang_c_saturates_at_one_when_rho_ge_c():
    assert erlang_c(1.0, 1) == 1.0
    assert erlang_c(2.0, 2) == 1.0
    assert erlang_c(3.0, 2) == 1.0


def test_erlang_c_rejects_bad_inputs():
    with pytest.raises(ValueError):
        erlang_c(0.5, 0)
    with pytest.raises(ValueError):
        erlang_c(-0.1, 1)


def test_erlang_wait_m11_textbook():
    # M/M/1: Wq = rho / (mu*(1-rho)); lambda=1, mu=2 -> rho=0.5, Wq=0.5
    assert erlang_wait(0.5, 1, 2.0) == pytest.approx(0.5)


def test_erlang_wait_m12_textbook():
    # lambda=2, mu=1, c=2 -> rho=2.0 = c, saturated -> inf
    assert erlang_wait(2.0, 2, 1.0) == float("inf")


def test_erlang_wait_light_m2():
    # lambda=0.6, mu=0.5, c=2 -> rho=1.2. Not saturated (1.2<2). Wq finite.
    wq = erlang_wait(1.2, 2, 0.5)
    assert wq > 0 and math.isfinite(wq)


def test_forecast_wait_takes_max_of_model_and_naive():
    # naive floor wins when the queue is long but lambda reads small.
    assert forecast_wait(0.1, 1, 1.0, 120.0) == 120.0
    # model wins when naive is tiny.
    w = forecast_wait(0.5, 1, 2.0, 0.1)
    assert w == pytest.approx(0.5)


# ---- Holt -----------------------------------------------------------------

def test_holt_constant_series():
    level, trend, fc = holt([5.0, 5.0, 5.0, 5.0])
    assert level == pytest.approx(5.0)
    assert trend == pytest.approx(0.0, abs=1e-6)
    assert fc == pytest.approx(5.0)


def test_holt_linear_trend():
    level, trend, fc = holt([1.0, 2.0, 3.0, 4.0], alpha=0.5, beta=0.5)
    assert trend > 0.0
    assert fc > 4.0


def test_holt_single_value():
    level, trend, fc = holt([7.0])
    assert (level, trend, fc) == (7.0, 0.0, 7.0)


def test_holt_rejects_empty():
    with pytest.raises(ValueError):
        holt([])


# ---- staffing -------------------------------------------------------------

def test_required_servers_returns_one_when_idle():
    assert required_servers(0.0, 1.0, 180.0) == 1


def test_required_servers_m11_target_is_reachable():
    # lambda=0.9, mu=1.0 -> rho=0.9; M/M/1 Wq=9. One counter suffices if target>=9.
    assert required_servers(0.9, 1.0, 180.0) == 1
    assert required_servers(0.9, 1.0, 5.0) == 2    # need a second counter


def test_required_servers_respects_max():
    # wildly overloaded: needs more than max, clamps to max
    assert required_servers(100.0, 0.1, 1.0, max_servers=4) == 4


def test_staffing_advice_only_adds_capacity():
    # lambda=0.9, mu=0.7 -> rho=1.286; needs 2 counters for a 60 s target.
    a = staffing_advice(1, 0.9, 0.7, 60.0)
    assert a is not None and a["open"] == 2
    # current 2 already open -> no advice
    assert staffing_advice(2, 0.9, 0.7, 60.0) is None


# ---- cells_by_lane: FrameResult.lane_occupancy -> per-lane 16-cell lists ----

def test_cells_by_lane_splits_the_hardware_array():
    from backend.queue_model import cells_by_lane
    from core.config import load_clipset
    cfg = load_clipset("config/sim")
    # lanes.json is L1C0..L1C3 then L2C0..L2C3, in hardware table order.
    out = cells_by_lane(cfg.lanes, list(range(16)))
    assert out[1] == [0, 1, 2, 3] + [0] * 12
    assert out[2] == [4, 5, 6, 7] + [0] * 12


def test_cells_by_lane_always_returns_sixteen_cells():
    from backend.queue_model import cells_by_lane
    table = [{"lane": 1, "cell": 0}, {"lane": 1, "cell": 2}]
    out = cells_by_lane(table, [80, 90])
    assert len(out[1]) == 16
    assert out[1][0] == 80 and out[1][2] == 90 and out[1][1] == 0


def test_cells_by_lane_ignores_cells_past_the_hardware_array():
    from backend.queue_model import cells_by_lane
    table = [{"lane": 1, "cell": 0}, {"lane": 2, "cell": 0}]
    assert cells_by_lane(table, [50]) == {1: [50] + [0] * 15}


# ---- QueueEngine -----------------------------------------------------------

def _engine(**kw):
    from backend.queue_model import QueueEngine
    return QueueEngine(**kw)


def test_engine_uses_the_naive_rule_before_a_fit():
    e = _engine()
    cells = [90, 90, 90, 10] + [0] * 12
    assert e.raw_count(cells) == pytest.approx(0.8 * 3)


def test_engine_uses_the_regression_once_fitted():
    e = _engine()
    # counts are exactly sum(cells)/100, so a = 0.01, b = 0
    sums = [100.0, 200.0, 300.0]
    truths = [1.0, 2.0, 3.0]
    a, b, r2, mae = e.fit(sums, truths)
    assert a == pytest.approx(0.01) and r2 == pytest.approx(1.0) and mae == pytest.approx(0.0)
    assert e.raw_count([50] * 4 + [0] * 12) == pytest.approx(2.0)


def test_engine_smooths_the_count_with_an_ema():
    e = _engine(alpha=0.5)
    full = [100] * 16
    first = e.on_lane_occ(0.0, 1, [0] * 16)
    second = e.on_lane_occ(1.0, 1, full)
    assert first == 0.0
    # naive_count of a full lane is 0.8*16 = 12.8; EMA at alpha 0.5 from 0.
    assert second == pytest.approx(6.4)


def test_engine_counts_a_lengthening_lane_as_an_arrival():
    """An empty lane filling up is arrivals — one per person, not per sample."""
    e = _engine()
    e.on_lane_occ(0.0, 1, [0] * 16)
    e.on_lane_occ(1.0, 1, [90] * 16)
    e.on_lane_occ(2.0, 1, [90] * 16)      # still rising under the EMA
    # naive_count of a full lane is 12.8; the EMA climbs 0 -> 3.8 -> 6.5, so
    # the whole-number count crosses 6 boundaries in total.
    assert len(e._arrivals) == 6
    assert int(e.counts[1]) == 6


def test_engine_mu_is_one_over_mean_inter_departure_time():
    e = _engine()
    for t in (0.0, 20.0, 40.0):
        e.on_departure(t, 1)
    assert e.mu(1) == pytest.approx(1 / 20.0)


def test_engine_mu_falls_back_to_a_default_before_any_pos_data():
    from backend.queue_model import QueueEngine
    e = _engine()
    assert e.mu(1) == pytest.approx(QueueEngine.DEFAULT_MU_PER_S)


def test_engine_publishes_every_two_seconds():
    e = _engine()
    assert e.due(0.0) is True
    e.estimates(0.0)
    assert e.due(1.0) is False
    assert e.due(2.0) is True


def test_engine_estimate_payload_matches_the_frozen_contract():
    from core.events import Event
    e = _engine()
    e.on_lane_occ(0.0, 1, [90] * 16)
    est = e.estimates(1.0)
    assert len(est) == 1
    # The payload must construct a valid Event: exact keys, nothing invented.
    Event(1.0, "demo-01", "overhead", None, "queue_estimate", est[0])


def test_engine_wait_takes_the_max_of_model_and_naive():
    """A visibly long queue never reports a two-second wait."""
    e = _engine(counters=2)
    for t in (0.0, 10.0, 20.0):
        e.on_departure(t, 1)              # mu = 0.1/s
    e.on_lane_occ(0.0, 1, [100] * 16)     # naive count 12.8 -> 12.8/0.1 = 128 s
    est = e.estimates(30.0)[0]
    assert est["pred_wait_s"] >= 100.0


def test_engine_never_publishes_an_infinite_wait():
    e = _engine(counters=1)
    for t in range(0, 200, 2):
        e.on_arrival(float(t))            # lambda far above mu -> saturated
    e.on_lane_occ(0.0, 1, [90] * 16)
    est = e.estimates(200.0)[0]
    assert math.isfinite(est["pred_wait_s"])


def test_engine_reports_a_lane_per_row_sorted():
    e = _engine()
    e.on_lane_occ(0.0, 2, [90] * 16)
    e.on_lane_occ(0.0, 1, [90] * 16)
    assert [r["lane"] for r in e.estimates(1.0)] == [1, 2]


def test_engine_trims_arrivals_to_the_ten_minute_window():
    from backend.queue_model import QueueEngine
    e = _engine()
    e.on_arrival(0.0)
    e.on_arrival(1.0)
    e.on_arrival(QueueEngine.LAMBDA_WINDOW_S + 100.0)
    assert len(e._arrivals) == 1


def test_engine_projects_lambda_forward_when_it_is_rising():
    e = _engine()
    e._lambda_series = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert e.projected_lambda(0.0) > 0.5


def test_engine_projection_never_goes_negative():
    e = _engine()
    e._lambda_series = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert e.projected_lambda(0.0) >= 0.0


def test_engine_advice_asks_for_a_counter_before_the_peak():
    """Rising arrivals must trigger the advice on the *projection*, not on the
    rate as it stands, which is the whole point of the Holt step."""
    e = _engine(counters=1, target_wait_s=30.0)
    for t in (0.0, 10.0, 20.0):
        e.on_departure(t, 1)              # mu = 0.1/s
    e._lambda_series = [0.01, 0.03, 0.05, 0.07]
    advice = e.advice(30.0)
    assert advice is not None and advice["open"] >= 2


def test_engine_advice_is_none_when_one_counter_copes():
    e = _engine(counters=2, target_wait_s=600.0)
    e._lambda_series = [0.001, 0.001]
    assert e.advice(10.0) is None


# ---- arrivals are whole people, not EMA wobble -----------------------------

def test_a_wobbling_count_is_not_a_stream_of_arrivals():
    """Regression: on real footage the smoothed count jitters every sample.
    Counting each uptick as an arrival read 120 arrivals/min off a 45 s clip
    and asked for sixteen counters."""
    e = _engine()
    e.a, e.b = 1.0, 0.0                   # count == sum(cells), for a clean test
    for i in range(40):
        # Hovers around two people, never crossing to three.
        e.on_lane_occ(float(i), 1, [2 if i % 2 else 1] + [0] * 15)
    assert e.counts[1] < 3.0
    assert len(e._arrivals) <= 2, f"{len(e._arrivals)} arrivals from a flat queue"


def test_a_whole_person_joining_is_an_arrival():
    e = _engine(alpha=1.0)                # no smoothing, so levels move at once
    e.a, e.b = 1.0, 0.0
    e.on_lane_occ(0.0, 1, [1] + [0] * 15)
    before = len(e._arrivals)
    e.on_lane_occ(1.0, 1, [2] + [0] * 15)
    e.on_lane_occ(2.0, 1, [3] + [0] * 15)
    assert len(e._arrivals) == before + 2


def test_a_jump_of_two_counts_two_arrivals():
    e = _engine(alpha=1.0)
    e.a, e.b = 1.0, 0.0
    e.on_lane_occ(0.0, 1, [1] + [0] * 15)
    e.on_lane_occ(1.0, 1, [4] + [0] * 15)
    assert len(e._arrivals) == 3


def test_a_shrinking_queue_is_never_an_arrival():
    e = _engine(alpha=1.0)
    e.a, e.b = 1.0, 0.0
    for v in (5, 4, 3, 2, 1, 0):
        e.on_lane_occ(float(v), 1, [v] + [0] * 15)
    assert e._arrivals == []


def test_lambda_stays_sane_on_a_flat_queue():
    """The number that drove the bogus staffing alert."""
    e = _engine()
    e.a, e.b = 1.0, 0.0
    for i in range(120):                  # 60 s of 2 Hz lane_occ
        e.on_lane_occ(i * 0.5, 1, [3 if i % 3 else 2] + [0] * 15)
    assert e.lambda_rate(60.0) * 60 < 5.0, "lambda should not track the sample rate"


def test_lambda_is_not_extrapolated_from_a_two_second_window():
    """Regression: the first seconds after warm-up produced lambda=60/min and
    an 'open counter 16' alert before anyone had actually queued."""
    from backend.queue_model import QueueEngine
    e = _engine()
    for i in range(6):
        e.on_arrival(float(i) * 0.3)      # 6 arrivals inside two seconds
    lam = e.lambda_rate(2.0)
    assert lam == pytest.approx(6.0 / QueueEngine.MIN_LAMBDA_WINDOW_S)
    assert lam * 60 <= 6.0


def test_lambda_uses_the_real_window_once_it_is_long_enough():
    e = _engine()
    for i in range(10):
        e.on_arrival(float(i) * 12.0)     # 10 arrivals over 108 s
    assert e.lambda_rate(120.0) == pytest.approx(10.0 / 120.0, rel=0.15)


def test_no_staffing_panic_in_the_first_seconds():
    e = _engine(counters=1, target_wait_s=180.0)
    for i in range(8):
        e.on_arrival(float(i) * 0.25)
        e.on_departure(float(i) * 0.25, 1)
    advice = e.advice(2.0)
    assert advice is None or advice["open"] <= 2

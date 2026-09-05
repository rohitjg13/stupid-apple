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

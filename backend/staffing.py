"""Staffing rule: smallest server count that keeps the projected wait on target.

The plan (05-backend-queue.md W3) wants "open counter N" before the peak, so
this is deliberately small and pure: project the arrival rate with Holt, then
walk up from the current open counters until Erlang C says Wq <= target_wait_s.
"""
from __future__ import annotations

from backend.queue_model import erlang_wait


def required_servers(lambda_rate, mu, target_wait_s, max_servers=16):
    """Smallest c >= 1 with Wq <= target_wait_s for M/M/c.

    lambda_rate: arrivals per second
    mu:          per-server service rate (departures per second)
    """
    if lambda_rate <= 0 or mu <= 0:
        return 1                        # nobody waiting; one counter is enough
    for c in range(1, max_servers + 1):
        rho = lambda_rate / mu
        if rho >= c:
            continue                    # unstable at this server count
        if erlang_wait(rho, c, mu) <= target_wait_s:
            return c
    return max_servers


def staffing_advice(current_c, lambda_rate, mu, target_wait_s, max_servers=16):
    """Return the open-counter message, or None if nothing needs to open.

    current_c is the number of counters already open. We only ever *add*
    capacity; closing counters is an operational decision, not an alert.
    """
    need = required_servers(lambda_rate, mu, target_wait_s, max_servers)
    if need <= current_c:
        return None
    return {
        "open": need,
        "lambda_per_min": round(lambda_rate * 60, 2),
        "mu_per_min": round(mu * 60, 2),
        "projected_wait_s": erlang_wait(lambda_rate / mu, need, mu),
    }

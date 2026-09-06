import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from core.events import Event
from shelf.fill import FacingReading, ShelfStatus

@dataclass
class FacingPlan:
    facing: str
    sku: str
    name: str
    unit_price: float
    expected_sales_per_hour: float

@dataclass
class StockoutState:
    facing: str
    sku: str
    is_out: bool = False
    out_since_t: Optional[float] = None
    is_low: bool = False
    low_since_t: Optional[float] = None
    total_out_duration_s: float = 0.0

class StockoutTracker:
    def __init__(
        self,
        planogram: List[dict],
        store_id: str = "demo-01",
    ):
        self.store_id = store_id
        self.catalog: Dict[str, FacingPlan] = {
            p["facing"]: FacingPlan(
                facing=p["facing"],
                sku=p["sku"],
                name=p["name"],
                unit_price=float(p["unit_price"]),
                expected_sales_per_hour=float(p["expected_sales_per_hour"]),
            )
            for p in planogram
        }
        self.states: Dict[str, StockoutState] = {
            f: StockoutState(facing=f, sku=p.sku)
            for f, p in self.catalog.items()
        }

    def update(self, t: float, readings: Dict[str, FacingReading]) -> List[Event]:
        events = []
        for facing, reading in readings.items():
            if facing not in self.states:
                continue
            state = self.states[facing]
            plan = self.catalog[facing]

            if reading.status == ShelfStatus.EMPTY:
                if not state.is_out:
                    state.is_out = True
                    state.out_since_t = t
                    events.append(
                        Event(
                            t=t,
                            store_id=self.store_id,
                            stream="shelf",
                            zone_id=facing,
                            event_type="stockout_start",
                            payload={"sku": state.sku, "facing": facing},
                        )
                    )
                    # Trigger alert event
                    events.append(
                        Event(
                            t=t,
                            store_id=self.store_id,
                            stream="shelf",
                            zone_id=facing,
                            event_type="alert",
                            payload={
                                "severity": "critical",
                                "rule": "stockout_detected",
                                "message": f"OUT OF STOCK: {plan.name} ({state.sku}) at facing {facing}",
                            },
                        )
                    )
            else:
                if state.is_out:
                    state.is_out = False
                    if state.out_since_t is not None:
                        state.total_out_duration_s += max(0.0, t - state.out_since_t)
                    state.out_since_t = None
                    events.append(
                        Event(
                            t=t,
                            store_id=self.store_id,
                            stream="shelf",
                            zone_id=facing,
                            event_type="stockout_end",
                            payload={"sku": state.sku, "facing": facing},
                        )
                    )

            if reading.status == ShelfStatus.LOW:
                if not state.is_low:
                    state.is_low = True
                    state.low_since_t = t
                    events.append(
                        Event(
                            t=t,
                            store_id=self.store_id,
                            stream="shelf",
                            zone_id=facing,
                            event_type="alert",
                            payload={
                                "severity": "warning",
                                "rule": "low_stock_detected",
                                "message": f"LOW STOCK: {plan.name} ({state.sku}) at facing {facing} (fill {reading.smoothed_fill})",
                            },
                        )
                    )
            elif reading.status == ShelfStatus.NORMAL:
                state.is_low = False
                state.low_since_t = None

        return events

    def get_lost_revenue_estimate(self, facing: str, now_t: float) -> float:
        if facing not in self.states or facing not in self.catalog:
            return 0.0
        state = self.states[facing]
        plan = self.catalog[facing]

        duration = state.total_out_duration_s
        if state.is_out and state.out_since_t is not None:
            duration += max(0.0, now_t - state.out_since_t)

        hours = duration / 3600.0
        lost_units = hours * plan.expected_sales_per_hour
        return round(lost_units * plan.unit_price, 2)

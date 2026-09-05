from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import cv2

from core.events import Event
from shelf.fill import ShelfStatus

@dataclass
class ItemCountState:
    facing: str
    sku: str
    name: str
    initial_stock: int
    current_stock: int
    removed_count: int = 0
    added_count: int = 0
    unit_price: float = 0.0
    low_threshold: int = 2

@dataclass
class RemovalEvent:
    t: float
    facing: str
    sku: str
    quantity: int
    remaining_stock: int
    estimated_revenue: float

class InventoryTracker:
    def __init__(
        self,
        planogram: List[dict],
        store_id: str = "demo-01",
        default_initial_stock: int = 10,
    ):
        self.store_id = store_id
        self.items: Dict[str, ItemCountState] = {}
        for p in planogram:
            facing = p["facing"]
            init_qty = int(p.get("initial_stock", default_initial_stock))
            low_thresh = max(1, int(p.get("low_threshold", int(init_qty * 0.3))))
            self.items[facing] = ItemCountState(
                facing=facing,
                sku=p["sku"],
                name=p["name"],
                initial_stock=init_qty,
                current_stock=init_qty,
                unit_price=float(p.get("unit_price", 0.0)),
                low_threshold=low_thresh,
            )
        self.removal_history: List[RemovalEvent] = []

    def register_removal(self, t: float, facing: str, quantity: int = 1) -> List[Event]:
        if facing not in self.items or quantity <= 0:
            return []
        item = self.items[facing]
        actual_removed = min(quantity, item.current_stock)
        item.current_stock -= actual_removed
        item.removed_count += actual_removed

        rev = actual_removed * item.unit_price
        self.removal_history.append(
            RemovalEvent(
                t=t,
                facing=facing,
                sku=item.sku,
                quantity=actual_removed,
                remaining_stock=item.current_stock,
                estimated_revenue=rev,
            )
        )

        events = []
        # Dispatches alert event
        if item.current_stock == 0:
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
                        "message": f"OUT OF STOCK: {item.name} ({item.sku}) at {facing}. Removed: {item.removed_count}, Remaining: 0",
                    },
                )
            )
            events.append(
                Event(
                    t=t,
                    store_id=self.store_id,
                    stream="shelf",
                    zone_id=facing,
                    event_type="stockout_start",
                    payload={"sku": item.sku, "facing": facing},
                )
            )
        elif item.current_stock <= item.low_threshold:
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
                        "message": f"LOW STOCK: {item.name} ({item.sku}) at {facing}. Remaining: {item.current_stock}/{item.initial_stock}",
                    },
                )
            )
        return events

    def update_from_fractional_fill(
        self,
        t: float,
        facing: str,
        fill_val: float,
        max_fill_expected: float = 180.0,
        empty_fill_threshold: float = 50.0,
    ) -> List[Event]:
        if facing not in self.items:
            return []
        item = self.items[facing]
        # Map fill 0..255 to expected items remaining
        span = max(1.0, max_fill_expected - empty_fill_threshold)
        fraction = np.clip((fill_val - empty_fill_threshold) / span, 0.0, 1.0)
        estimated_qty = int(round(fraction * item.initial_stock))

        if estimated_qty < item.current_stock:
            diff = item.current_stock - estimated_qty
            return self.register_removal(t=t, facing=facing, quantity=diff)
        return []

    def get_summary(self) -> Dict[str, dict]:
        return {
            f: {
                "sku": it.sku,
                "name": it.name,
                "initial": it.initial_stock,
                "current": it.current_stock,
                "removed": it.removed_count,
                "unit_price": it.unit_price,
                "lost_or_taken_val": round(it.removed_count * it.unit_price, 2),
            }
            for f, it in self.items.items()
        }

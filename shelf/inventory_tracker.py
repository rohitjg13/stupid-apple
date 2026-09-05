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
    low_threshold: int = 3
    confirmed_history: List[int] = field(default_factory=list)

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
        confirmation_frames: int = 15, # Requires sustained drop across 15 frames (~1-2 sec) before committing
    ):
        self.store_id = store_id
        self.confirmation_frames = confirmation_frames
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
        self._pending_candidates: Dict[str, List[int]] = {f: [] for f in self.items}

    def register_removal(self, t: float, facing: str, quantity: int = 1) -> List[Event]:
        if facing not in self.items or quantity <= 0:
            return []
        item = self.items[facing]
        actual_removed = min(quantity, item.current_stock)
        if actual_removed == 0:
            return []
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

    def update_from_measurement(
        self,
        t: float,
        facing: str,
        measured_qty: int,
    ) -> List[Event]:
        if facing not in self.items:
            return []
        item = self.items[facing]
        measured_qty = max(0, min(item.initial_stock, measured_qty))

        candidates = self._pending_candidates[facing]
        candidates.append(measured_qty)
        if len(candidates) > self.confirmation_frames:
            candidates.pop(0)

        # Only change stock when confirmed consistently across confirmation window
        if len(candidates) >= self.confirmation_frames:
            # Check if all or vast majority agree
            mode_qty = max(set(candidates), key=candidates.count)
            agreement_ratio = candidates.count(mode_qty) / float(len(candidates))

            if agreement_ratio >= 0.85 and mode_qty < item.current_stock:
                diff = item.current_stock - mode_qty
                # Confirmed actual persistent removal
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

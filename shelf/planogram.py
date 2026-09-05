from dataclasses import dataclass
from typing import Dict, List, Optional
from core.events import Event
from shelf.fill import FacingReading, ShelfStatus

@dataclass
class PlanogramEntry:
    facing: str
    sku: str
    name: str
    unit_price: float
    expected_sales_per_hour: float

@dataclass
class ComplianceResult:
    facing: str
    expected_sku: str
    observed: str
    is_compliant: bool
    status: ShelfStatus

class PlanogramAuditor:
    def __init__(self, planogram: List[dict], store_id: str = "demo-01"):
        self.store_id = store_id
        self.planogram: Dict[str, PlanogramEntry] = {
            p["facing"]: PlanogramEntry(
                facing=p["facing"],
                sku=p["sku"],
                name=p["name"],
                unit_price=float(p["unit_price"]),
                expected_sales_per_hour=float(p["expected_sales_per_hour"]),
            )
            for p in planogram
        }
        self._last_violation: Dict[str, str] = {}

    def audit_facings(
        self,
        t: float,
        readings: Dict[str, FacingReading],
        detected_skus: Optional[Dict[str, str]] = None,
    ) -> List[Event]:
        events = []
        for facing, entry in self.planogram.items():
            reading = readings.get(facing)
            if reading is None:
                continue

            observed = "ok"
            if reading.status == ShelfStatus.EMPTY:
                observed = "empty"
            elif reading.status == ShelfStatus.LOW:
                observed = "low"
            elif detected_skus and facing in detected_skus:
                det = detected_skus[facing]
                if det != entry.sku:
                    observed = "unknown"

            if observed in ("empty", "low", "unknown"):
                last = self._last_violation.get(facing)
                if last != observed:
                    self._last_violation[facing] = observed
                    events.append(
                        Event(
                            t=t,
                            store_id=self.store_id,
                            stream="shelf",
                            zone_id=facing,
                            event_type="planogram_violation",
                            payload={"expected_sku": entry.sku, "observed": observed},
                        )
                    )
            else:
                self._last_violation.pop(facing, None)

        return events

    def get_compliance_score(self, readings: Dict[str, FacingReading]) -> float:
        if not self.planogram:
            return 100.0
        compliant_count = sum(
            1 for facing in self.planogram
            if facing in readings and readings[facing].status == ShelfStatus.NORMAL
        )
        return round((compliant_count / len(self.planogram)) * 100.0, 1)

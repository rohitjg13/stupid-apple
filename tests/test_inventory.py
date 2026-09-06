import pytest
from shelf.inventory_tracker import InventoryTracker

PLANOGRAM = [
    {"facing": "A1", "sku": "NOODLE-CUP-01", "name": "Nissin Cup Noodles", "unit_price": 50, "expected_sales_per_hour": 10, "initial_stock": 8, "low_threshold": 3},
    {"facing": "A2", "sku": "SNACK-01", "name": "Chips", "unit_price": 20, "expected_sales_per_hour": 5, "initial_stock": 10, "low_threshold": 2},
]

def test_inventory_item_removals_and_stock_tracking():
    inv = InventoryTracker(planogram=PLANOGRAM, default_initial_stock=10)
    summary = inv.get_summary()
    assert summary["A1"]["current"] == 8
    assert summary["A1"]["removed"] == 0

    # Customer removes 2 items
    evs = inv.register_removal(t=100.0, facing="A1", quantity=2)
    assert len(evs) == 0  # 6 remaining > low_threshold 3
    summary = inv.get_summary()
    assert summary["A1"]["current"] == 6
    assert summary["A1"]["removed"] == 2
    assert summary["A1"]["lost_or_taken_val"] == 100.0

    # Customer removes 3 more (3 remaining -> triggers low stock alert)
    evs = inv.register_removal(t=110.0, facing="A1", quantity=3)
    assert any(e.event_type == "alert" and e.payload["severity"] == "warning" for e in evs)
    summary = inv.get_summary()
    assert summary["A1"]["current"] == 3
    assert summary["A1"]["removed"] == 5

    # Customer removes last 3 (0 remaining -> triggers stockout and critical alert)
    evs = inv.register_removal(t=120.0, facing="A1", quantity=3)
    types = [e.event_type for e in evs]
    assert "stockout_start" in types
    assert "alert" in types
    summary = inv.get_summary()
    assert summary["A1"]["current"] == 0
    assert summary["A1"]["removed"] == 8

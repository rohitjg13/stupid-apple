"""core/bus.py: thread-safe, wildcard, and a slow subscriber can never stall analytics."""
import threading
import time

from core.bus import Bus
from core.events import Event


def ev(event_type="occupancy", payload=None, **kw):
    return Event(t=time.time(), store_id="demo-01", stream="overhead", zone_id=None,
                 event_type=event_type, payload=payload or {"count": 1}, **kw)


def test_subscribe_by_type():
    bus, seen = Bus(), []
    bus.subscribe("occupancy", seen.append)
    bus.publish(ev("occupancy"))
    bus.publish(ev("tripwire", {"dir": "in", "tripwire_id": "door"}))
    bus.drain()
    assert [e.event_type for e in seen] == ["occupancy"]


def test_wildcard_gets_everything():
    bus, seen = Bus(), []
    bus.subscribe("*", seen.append)
    bus.publish(ev("occupancy"))
    bus.publish(ev("tripwire", {"dir": "in", "tripwire_id": "door"}))
    bus.drain()
    assert len(seen) == 2


def test_order_preserved_per_subscriber():
    bus, seen = Bus(), []
    bus.subscribe("*", seen.append)
    for i in range(50):
        bus.publish(ev(payload={"count": i}))
    bus.drain()
    assert [e.payload["count"] for e in seen] == list(range(50))


def test_slow_subscriber_does_not_block_publisher():
    """The dashboard websocket stalls; analytics must keep running.

    Publishes on a worker thread so a bus that *does* block fails this test
    instead of hanging the whole suite.
    """
    bus = Bus(maxsize=4)
    release = threading.Event()
    bus.subscribe("*", lambda e: release.wait(10))

    done = threading.Event()

    def publisher():
        for _ in range(200):
            bus.publish(ev())
        done.set()

    threading.Thread(target=publisher, daemon=True).start()
    finished = done.wait(2.0)
    release.set()

    assert finished, "publish() blocked on a slow subscriber"
    assert bus.dropped > 0          # bounded queue drops rather than blocks


def test_subscriber_exception_does_not_kill_the_bus():
    bus, seen = Bus(), []
    bus.subscribe("*", lambda e: 1 / 0)
    bus.subscribe("*", seen.append)
    bus.publish(ev())
    bus.publish(ev())
    bus.drain()
    assert len(seen) == 2


def test_publish_is_thread_safe():
    bus, seen = Bus(), []
    bus.subscribe("*", seen.append)
    threads = [threading.Thread(target=lambda: [bus.publish(ev()) for _ in range(100)])
               for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    bus.drain()
    assert len(seen) == 400

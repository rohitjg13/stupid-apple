# Board bring-up (PYNQ-Z2)

## First time

```bash
BOARD=xilinx@192.168.2.99 tools/deploy.sh --no-follow   # copies the tree
ssh $BOARD "cd /home/xilinx/retail && sudo tools/setup_board.sh"
ssh $BOARD "sudo systemctl start retail && journalctl -u retail -f"
```

`setup_board.sh` installs the board dependency set (no torch, no pandas),
creates `/var/lib/retail`, installs and enables `retail.service`, and caps the
journal at 200 MB so a full SD card cannot take the demo down.

## Every day

```bash
tools/deploy.sh          # rsync, restart, tail the log
```

## Checks

| What | Command | Expect |
|---|---|---|
| Service alive | `systemctl is-active retail` | `active` |
| Health | `cat /var/lib/retail/health.json` | `clock_unsynced: false`, frames rising |
| Overlay loaded | `python3 -c "from pl import driver; print(driver.available())"` | `True` |
| Throughput | `python3 -m tools.perf --backends reference pl --frames 200` | PL ≤ 5 ms/frame |
| PL vs reference | `python3 -m tools.compare_backends --frames 300` | `PASS` |

## The board has no RTC

With no NTP it boots near 1970. `core/clock.py` continues from the last time
recorded in `/var/lib/retail/clock.json` and reports `clock_unsynced` until
something hands it a real epoch. Cloud sync waits on that flag; everything
else runs normally. Never "fix" this by blocking startup on NTP — the demo has
to come up with no network.

## When it goes wrong on the day

1. `journalctl -u retail -n 100` — the logs are JSON lines, one per event.
2. DMA hang: `driver` watchdog reloads the overlay; if it repeats, drop to
   `--backend reference` on the board and say so.
3. Board dead: laptop fallback, `--source file --backend reference --config
   config/demo`, same dashboard. Rehearsed twice in W6 for exactly this.
4. Camera unplugged: `CameraSource` reopens it by itself; give it 2 s before
   touching anything.

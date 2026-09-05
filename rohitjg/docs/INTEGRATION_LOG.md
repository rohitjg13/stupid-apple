# Integration log

One entry per day during W4, and any day something breaks on the board.
Terse. What was tried, what happened, what is next. Rohit writes it.

Template:

```
## YYYY-MM-DD
- Board: <bitstream / commit>
- Ran: <command>
- Result: <numbers, not adjectives>
- Broke: <what, and the actual error>
- Fixed by: <change, or "not yet">
- Next: <one thing>
```

---

## 2026-09-05
- Board: none yet.
- Ran: `pytest -q` (155 passing), `python main.py --source sim --frames 300 --headless`.
- Result: contracts, bus, config, sim, reference, file/camera sources, clock,
  perf and compare tools all green on the laptop. `--source file --backend
  reference` produces blobs from a synthetic clip end to end.
- Broke: `docs/SHARED.md` §4 register map was arithmetically impossible —
  ROI_TABLE at `0x100–0x1FF` cannot hold 64 × 8 bytes and overran LANE_TABLE.
- Fixed by: ROI_TABLE `0x100–0x2FF`, LANE_TABLE `0x300–0x37F`, 1 KB AXI-Lite
  aperture. `pl/regs.py` and both plan docs updated.
- Next: Khushwant confirms `0x300` in the block design and generates
  `hls/regs.h` from `pl/regs.py`; add the header-vs-`Reg` drift test once it
  exists.

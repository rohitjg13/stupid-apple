"""AXI-Lite register map + table packing. docs/SHARED.md §4."""
import pytest

from pl.regs import (LANE_TABLE_BYTES, PL_H, PL_W, ROI_TABLE_BYTES, Reg,
                     pack_lane_table, pack_roi_table, q0_16, q8_8)


def test_offsets_are_frozen():
    assert Reg.CTRL == 0x00
    assert Reg.STREAM_ID == 0x04
    assert Reg.MOG2_VAR_THRESH == 0x08
    assert Reg.MOG2_LR == 0x0C
    assert Reg.MIN_BLOB_AREA == 0x10
    assert Reg.MORPH_ITERS == 0x14
    assert Reg.LATENCY == 0x18
    assert Reg.DEBUG_STAGE == 0x1C
    assert Reg.ROI_TABLE == 0x100
    assert Reg.LANE_TABLE == 0x300   # see pl/regs.py: §4 offset overruns the ROI table


def test_table_regions_do_not_overlap():
    assert ROI_TABLE_BYTES == 64 * 8          # 64 x {x,y,w,h} u16
    assert LANE_TABLE_BYTES == 16 * 8
    assert Reg.ROI_TABLE + ROI_TABLE_BYTES <= Reg.LANE_TABLE


def test_q8_8_defaults_and_rounding():
    assert q8_8(16.0) == 0x1000
    assert q8_8(1.0) == 0x0100
    assert q8_8(0.5) == 0x0080
    assert q8_8(16.1) == 4122          # rounds, never truncates: 16.1*256 = 4121.6
    assert q8_8(0.999) == 256
    with pytest.raises(ValueError):
        q8_8(256.0)


def test_q0_16_default():
    assert q0_16(0.005) == round(0.005 * 65536)
    assert q0_16(1.0) == 0xFFFF               # saturates, never wraps to 0
    with pytest.raises(ValueError):
        q0_16(-0.1)


def test_pack_roi_table_scales_to_320x240():
    # rois.json is authored in 320x240 PL scale already
    rois = [{"id": "A1", "stream": "shelf", "x": 10, "y": 20, "w": 30, "h": 40}]
    buf = pack_roi_table(rois)
    assert len(buf) == ROI_TABLE_BYTES
    assert buf[:8] == bytes([10, 0, 20, 0, 30, 0, 40, 0])   # little-endian u16
    assert buf[8:] == bytes(ROI_TABLE_BYTES - 8)            # unused entries zeroed


def test_pack_tables_reject_overflow():
    with pytest.raises(ValueError):
        pack_roi_table([{"id": str(i), "x": 0, "y": 0, "w": 1, "h": 1} for i in range(65)])
    with pytest.raises(ValueError):
        pack_lane_table([{"id": f"L{i}", "x": 0, "y": 0, "w": 1, "h": 1} for i in range(17)])


def test_pack_lane_table_preserves_json_order():
    lanes = [{"id": "L1C0", "x": 1, "y": 2, "w": 3, "h": 4},
             {"id": "L1C1", "x": 5, "y": 6, "w": 7, "h": 8}]
    buf = pack_lane_table(lanes)
    assert len(buf) == LANE_TABLE_BYTES
    assert buf[8:16] == bytes([5, 0, 6, 0, 7, 0, 8, 0])


@pytest.mark.parametrize("bad", [
    {"id": "neg", "x": -1, "y": 0, "w": 10, "h": 10},        # would wrap to 65535 as u16
    {"id": "zero_w", "x": 0, "y": 0, "w": 0, "h": 10},
    {"id": "full_res", "x": 0, "y": 0, "w": 640, "h": 480},   # authored at 640x480 by mistake
    {"id": "off_right", "x": PL_W - 5, "y": 0, "w": 10, "h": 10},
    {"id": "off_bottom", "x": 0, "y": PL_H - 5, "w": 10, "h": 10},
])
def test_tables_reject_entries_outside_the_pl_frame(bad):
    with pytest.raises(ValueError):
        pack_roi_table([bad])
    with pytest.raises(ValueError):
        pack_lane_table([bad])


def test_full_frame_roi_is_allowed():
    assert pack_roi_table([{"id": "all", "x": 0, "y": 0, "w": PL_W, "h": PL_H}])[:8] == \
        bytes([0, 0, 0, 0, 0x40, 0x01, 0xF0, 0x00])

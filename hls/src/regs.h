// AXI-Lite register map. Written from pl/regs.py -- that file is the truth.
// tests/test_regs.py diffs these #defines against pl.regs.Reg; if you change
// one, change both or CI fails.
#ifndef RETAIL_REGS_H
#define RETAIL_REGS_H

#define REG_APERTURE          0x400

#define REG_CTRL              0x00
#define REG_STREAM_ID         0x04
#define REG_MOG2_VAR_THRESH   0x08
#define REG_MOG2_LR           0x0C
#define REG_MIN_BLOB_AREA     0x10
#define REG_MORPH_ITERS       0x14
#define REG_LATENCY           0x18
#define REG_DEBUG_STAGE       0x1C
#define REG_ROI_TABLE         0x100   // 0x100-0x2FF, 64 x {x,y,w,h} u16
#define REG_LANE_TABLE        0x300   // 0x300-0x37F, 16 x {x,y,w,h} u16

#define CTRL_START            (1u << 0)
#define CTRL_RESET_BG         (1u << 1)
#define CTRL_DUMP_MASK        (1u << 2)
#define CTRL_RESULT_MASK_MODE (1u << 3)

#define STREAM_OVERHEAD       0
#define STREAM_SHELF          1

#define DEBUG_STAGE_GRAY      0
#define DEBUG_STAGE_MASK      1
#define DEBUG_STAGE_MORPH     2

#define DEFAULT_VAR_THRESH_Q88 4096   // 16.0
#define DEFAULT_LR_Q016        328    // 0.005
#define DEFAULT_MIN_BLOB_AREA  150
#define DEFAULT_MORPH_ITERS    1

#endif

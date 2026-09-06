#ifndef RETAIL_TYPES_H
#define RETAIL_TYPES_H

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "regs.h"

// Geometry. Video enters full-res BGR8 and is halved once, up front.
#define FULL_W 640
#define FULL_H 480
#define PROC_W 320
#define PROC_H 240
#define PROC_PIX (PROC_W * PROC_H)

#define N_ROI 64
#define N_LANE 16
#define MAX_BLOBS 64

// FrameResult is 856 B = 214 words of 32 bit. See pl/contract.py.
#define RESULT_WORDS 214
#define WARMUP_FRAMES 100

// ---- MOG2 model sizing ------------------------------------------------------
// MOG2_K = 2 (option A): mean u8 + var Q4.4 + weight Q0.8 = 24 b per Gaussian,
//   48 b/px * 76800 = 461 KB of the 630 KB on a 7z020 (~73 %).
// MOG2_K = 1 (option B): single running Gaussian, mean Q8.8 + var Q4.4 = 24 b/px
//   = 230 KB (~37 %). Build with -DMOG2_K=1 if csynth puts BRAM over 85 %.
#ifndef MOG2_K
#define MOG2_K 2
#endif

#if MOG2_K == 1
#define MEAN_BITS 16          // Q8.8
#define MEAN_FRAC 8
#define GAUSS_BITS 24
#else
#define MEAN_BITS 8           // integer gray
#define MEAN_FRAC 0
#define GAUSS_BITS 24
#endif
#define MODEL_BITS (GAUSS_BITS * MOG2_K)

typedef ap_axiu<32, 0, 0, 0> axi_word;
typedef ap_uint<8>  u8_t;
typedef ap_uint<1>  mask_t;
typedef ap_uint<32> u32_t;
typedef ap_uint<16> u16_t;

struct Blob {
    ap_uint<16> x, y, w, h;
    ap_uint<32> area;
};

struct Rect {
    ap_uint<16> x, y, w, h;
};

#endif

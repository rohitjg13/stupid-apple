# config/sim — synthetic clipset

Floor is 8.0 m x 6.0 m. `homography_overhead.npy` is the inverse of an exact
metres->pixel scale (80 px/m, Y flipped), so sim ground truth and `geometry/`
agree by construction. Real clipsets get a calibrated one instead
(W2, `geometry/tools/calibrate.py`).

Shelf ROIs are a 3x2 grid at 320x240 PL scale. Six facings, six SKUs.
Two checkout lanes, four queue cells each.

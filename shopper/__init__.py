"""Shopper analytics: blobs -> tracks -> counts, visits, dwell and the heatmap.

Consumes FrameResult.blobs (from sim, reference or PL -- this package never knows
which) and publishes the shopper events in docs/SHARED.md §5.

Nothing here imports cv2: it has to stay importable on a board with no display
stack. Drawing lives in tools/viz_tracks.py.
"""

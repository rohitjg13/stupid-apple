import json
from pathlib import Path
from typing import List

def create_default_shelf_rois(
    num_cols: int = 4,
    num_rows: int = 2,
    start_x: int = 20,
    start_y: int = 30,
    roi_w: int = 60,
    roi_h: int = 80,
    gap_x: int = 15,
    gap_y: int = 20,
    empty_below: int = 60,
    low_below: int = 110,
) -> List[dict]:
    rois = []
    rows = ["A", "B", "C", "D", "E"]
    for r in range(num_rows):
        row_letter = rows[r] if r < len(rows) else f"R{r+1}"
        for c in range(num_cols):
            x = start_x + c * (roi_w + gap_x)
            y = start_y + r * (roi_h + gap_y)
            rois.append({
                "id": f"{row_letter}{c+1}",
                "stream": "shelf",
                "x": x,
                "y": y,
                "w": roi_w,
                "h": roi_h,
                "empty_below": empty_below,
                "low_below": low_below,
            })
    return rois

def save_rois(rois: List[dict], out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rois, f, indent=2)

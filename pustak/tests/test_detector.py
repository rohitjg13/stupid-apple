import numpy as np
import pytest
from shelf.detector import ShelfEdgeDetector

def test_shelf_edge_detector_fast_cv():
    detector = ShelfEdgeDetector(model_type="fast_cv")
    
    # Create test synthetic image: left half textured, right half blank
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    # Add noise / texture on left
    img[:, :160] = np.random.randint(0, 255, (240, 160, 3), dtype=np.uint8)

    rois = [
        {"id": "A1", "stream": "shelf", "x": 20, "y": 40, "w": 60, "h": 70},
        {"id": "A2", "stream": "shelf", "x": 200, "y": 40, "w": 60, "h": 70},
    ]

    results = detector.analyze_rois(img, rois, scale_from_320=False)
    assert "A1" in results and "A2" in results
    assert results["A1"]["edge_density"] > results["A2"]["edge_density"]
    assert results["A2"]["edge_density"] == 0.0

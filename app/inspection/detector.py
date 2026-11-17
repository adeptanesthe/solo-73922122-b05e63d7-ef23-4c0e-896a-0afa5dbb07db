"""End-to-end re-inspection pipeline.

This module is the *only* entry point used by both the API and the
independent evaluator.  Its sole inputs are two decoded grayscale numpy
arrays; it never reads files, generation parameters, or ground truth.

Pipeline
--------
1. Shared trusted front-end (see :mod:`alignment`): content trust gates,
   multi-scale robust ECC affine registration, warp / correlation /
   overlap gates, and a robust Huber-IRLS gain/bias exposure fit.  The
   returned warp ``W`` is golden->inspected (forward); inspected content is
   rendered into the golden frame with the inverse mapping.
2. Residual defect detection inside the eroded overlap region, then the
   mask and boxes are forward-mapped into *inspected-image* coordinates.
3. Any failed trust gate yields status ``undeterminable`` -- never ``pass``.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry
from .alignment import AlignmentFailure, align_and_correct, correction_payload
from .exposure import detect_defects
from .io_utils import IMAGE_SIZE, encode_png_base64

MIN_BOX_AREA = 36


def _undeterminable(code: str, reason: str, metrics: dict | None = None) -> dict:
    return {
        "status": "undeterminable",
        "reason_code": code,
        "reason": reason,
        "metrics": metrics or {},
        "correction": None,
        "defect_mask_png_base64": None,
        "defect_boxes": [],
        "debug_png_base64": None,
    }


def inspect_images(
    golden: np.ndarray, inspected: np.ndarray, include_debug: bool = True
) -> dict:
    """Run the full re-inspection. See module docstring for the contract."""
    if golden.shape != (IMAGE_SIZE, IMAGE_SIZE) or \
            inspected.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("both images must be 512x512")

    # ---- 1-4. Trusted registration + exposure compensation ----------------
    aln = align_and_correct(golden, inspected)
    if isinstance(aln, AlignmentFailure):
        return _undeterminable(aln.reason_code, aln.reason, aln.metrics)

    warp = aln.warp
    coverage_g, coverage_i = aln.coverage_g, aln.coverage_i
    i_in_g = aln.inspected_in_golden

    # ---- 5. Defect detection in golden frame, then map to inspected -------
    # detect_defects(inspected-frame image, golden reference, overlap mask).
    result = detect_defects(i_in_g, golden, coverage_g, aln.fit)
    mask_g = result.mask
    mask_i = geometry.render_golden_in_inspected(mask_g, warp, nearest=True)
    # Restrict to the inspected-frame overlap and clean resampling artifacts.
    mask_i = cv2.bitwise_and(mask_i, coverage_i)
    kernel = np.ones((3, 3), np.uint8)
    mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_i, connectivity=8
    )
    boxes: list[list[int]] = []
    keep = np.zeros_like(mask_i)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_BOX_AREA:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, w, h])
        keep[labels == label] = 255
    mask_i = keep

    defect_pixels = int((mask_i > 0).sum())
    metrics = dict(aln.metrics)
    metrics["defect_pixels"] = defect_pixels
    metrics["defect_area_ratio"] = defect_pixels / float(IMAGE_SIZE ** 2)
    metrics["residual_threshold_gray"] = result.threshold_gray

    mask_b64 = encode_png_base64(mask_i)
    debug_b64 = None
    if include_debug:
        debug_b64 = encode_png_base64(
            _debug_panel(golden, aln.corrected_golden, mask_g, result.boxes,
                         result.residual, result.eroded_valid_mask)
        )

    status = "defect" if defect_pixels > 0 else "pass"
    return {
        "status": status,
        "reason_code": None,
        "reason": None,
        "metrics": metrics,
        "correction": correction_payload(aln),
        "defect_mask_png_base64": mask_b64,
        "defect_boxes": boxes,
        "debug_png_base64": debug_b64,
    }


def _debug_panel(
    golden: np.ndarray,
    corrected_g: np.ndarray,
    mask_g: np.ndarray,
    boxes_g: list[list[int]],
    residual: np.ndarray,
    valid_g: np.ndarray,
) -> np.ndarray:
    """2x2 BGR panel in golden-frame geometry.

    golden | exposure-corrected inspected ; residual heat-map | overlay.
    """
    def to_bgr(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    overlay = to_bgr(corrected_g)
    overlay[mask_g > 0] = (0, 0, 255)
    for x, y, w, h in boxes_g:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 1)

    r_show = np.zeros_like(residual)
    r_show[valid_g > 0] = residual[valid_g > 0]
    if (valid_g > 0).any():
        lim = max(1.0, float(np.percentile(np.abs(r_show[valid_g > 0]), 99)))
    else:
        lim = 1.0
    r_norm = np.clip(128 + 127 * r_show / lim, 0, 255).astype(np.uint8)
    r_norm[valid_g == 0] = 128
    heat = cv2.applyColorMap(r_norm, cv2.COLORMAP_JET)
    heat[valid_g == 0] = (60, 60, 60)

    top = np.hstack([to_bgr(golden), to_bgr(corrected_g)])
    bottom = np.hstack([heat, overlay])
    return np.vstack([top, bottom])

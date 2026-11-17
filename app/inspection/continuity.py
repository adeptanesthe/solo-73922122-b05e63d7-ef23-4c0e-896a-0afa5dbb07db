"""Two-net electrical continuity analysis (open / short detection).

Given the golden board, the inspected board and a validated golden-frame
annotation (two label masks 1/2 and two pad centers per net), this module:

1. Reuses the shared trusted registration + linear exposure compensation
   (:mod:`alignment`) and renders the inspected board into golden-frame
   geometry on the golden intensity scale.
2. Reconstructs the inspected *conductor* mask with a midpoint threshold
   between robust copper / background intensities.  Only isolated 1-2 px
   specks are removed -- no morphological opening -- so a genuine 3 px-wide
   bridge or neck cannot be erased by denoising.
3. Judges each net independently inside its Voronoi (nearest-net) region:
   the two pads must sit on one 8-connected conductor component.  Because
   the open test never leaves the net's own region, a cross-net detour
   created by a short cannot mask an intra-net break.
4. Judges cross-net shorts on the *full* conductor mask: a component that
   approaches both labels within the registration tolerance and has a
   >=3 px-wide neck in the inter-net zone is a short.

Any failed trust gate returns ``status="undeterminable"`` with a
``reason_code`` -- the service never reports ``normal`` when it cannot
decide.  All masks are returned in **golden-frame coordinates**.
"""
from __future__ import annotations

import cv2
import numpy as np

from .alignment import AlignmentFailure, align_and_correct, correction_payload
from .annotation import Annotation, NET_LABELS
from .io_utils import IMAGE_SIZE, encode_png_base64

# Registration residual is sub-pixel after ECC; allow this many px of slack
# when deciding whether reconstructed copper still "belongs" to a net.
ALIGN_TOLERANCE_PX = 2
# Gray-level separation required between copper and background after the
# exposure correction (the input guarantees >=40 on the original scale;
# plausible gains keep the corrected separation comfortably above this).
MIN_INTENSITY_SEPARATION = 30.0
# Golden self-consistency: thresholding the golden image itself must
# reproduce the supplied label masks at this Dice level.
MIN_GOLDEN_DICE = 0.90
# Per-net reconstructed area must stay within this band of the label area.
NET_AREA_RATIO_BAND = (0.55, 1.55)
# Only 1-2 px isolated specks are dropped; anything >=3 px may be part of a
# real 3 px-wide neck/bridge and is preserved.
MAX_SPECK_AREA = 2
# A >=3 px-wide bridge contains a 3x3 block of conductor pixels in the open
# gap between the nets; a 1-2 px noise link never survives 3x3 erosion.
WIDTH_CHECK_KERNEL = np.ones((3, 3), np.uint8)
MIN_GAP_PIXELS = 3
MIN_COMP_PIXELS_NO_GAP = 9


def _undeterminable(code: str, reason: str, metrics: dict | None = None) -> dict:
    return {
        "status": "undeterminable",
        "reason_code": code,
        "reason": reason,
        "metrics": metrics or {},
        "correction": None,
        "nets": None,
        "short": None,
        "conductor_mask_png_base64": None,
        "short_mask_png_base64": None,
        "debug_png_base64": None,
    }


def _median_in(gray: np.ndarray, mask: np.ndarray) -> float:
    vals = gray[mask > 0]
    if vals.size == 0:
        raise RuntimeError("empty sampling region")
    return float(np.median(vals))


def _threshold_mask(gray: np.ndarray, copper_med: float, bg_med: float,
                    valid: np.ndarray) -> np.ndarray:
    """Midpoint threshold; polarity follows whichever class is brighter."""
    thr = 0.5 * (copper_med + bg_med)
    if copper_med >= bg_med:
        mask = gray >= thr
    else:
        mask = gray <= thr
    return (mask & (valid > 0)).astype(np.uint8) * 255


def _drop_specks(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only 1-2 px isolated components; never open a real neck."""
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    dropped = 0
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= MAX_SPECK_AREA:
            dropped += 1
            continue
        keep[cc == label] = 255
    return keep, dropped


def _dice(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    pa = (a > 0) & (valid > 0)
    pb = (b > 0) & (valid > 0)
    inter = float(np.logical_and(pa, pb).sum())
    total = float(pa.sum()) + float(pb.sum())
    return 2.0 * inter / total if total > 0 else 1.0


def _voronoi_regions(net_masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Assign every pixel to the nearest net label (ties go to net 1)."""
    dist_to: dict[int, np.ndarray] = {}
    for label in NET_LABELS:
        # distanceTransform of the inverted mask = distance to the nearest
        # copper pixel of that net for every pixel outside the mask.
        dist_to[label] = cv2.distanceTransform(
            cv2.bitwise_not(net_masks[label]), cv2.DIST_L2, 5
        )
    assign = np.where(dist_to[1] <= dist_to[2], 1, 2).astype(np.uint8)
    return {label: (assign == label).astype(np.uint8) * 255
            for label in NET_LABELS}


def _component_at(cc: np.ndarray, px: int, py: int,
                  allowed: np.ndarray) -> int:
    """Resolve the conductor component under a pad, searching a small window.

    Returns 0 when no conductor component covers the pad neighborhood.
    """
    if cc[py, px] > 0 and allowed[py, px] > 0:
        return int(cc[py, px])
    x0, x1 = max(0, px - ALIGN_TOLERANCE_PX), min(IMAGE_SIZE, px + 3)
    y0, y1 = max(0, py - ALIGN_TOLERANCE_PX), min(IMAGE_SIZE, py + 3)
    window = cc[y0:y1, x0:x1]
    allow = allowed[y0:y1, x0:x1]
    candidates = window[(window > 0) & (allow > 0)]
    if candidates.size == 0:
        return 0
    return int(np.bincount(candidates.ravel()).argmax())


def analyze_continuity(
    golden: np.ndarray,
    inspected: np.ndarray,
    annotation: Annotation,
    include_debug: bool = True,
) -> dict:
    """Run the full open/short analysis. See module docstring."""
    labels = annotation.labels
    net_masks = annotation.net_masks
    metrics: dict = {}

    # ---- 1. Trusted registration + exposure compensation ------------------
    aln = align_and_correct(golden, inspected)
    if isinstance(aln, AlignmentFailure):
        return _undeterminable(aln.reason_code, aln.reason, aln.metrics)
    metrics.update(aln.metrics)
    coverage_g = aln.coverage_g
    corrected = aln.corrected_golden

    # ---- 2. Sampling regions ----------------------------------------------
    erode_kernel = np.ones((3, 3), np.uint8)
    copper_region = np.zeros_like(coverage_g)
    for label in NET_LABELS:
        copper_region = cv2.bitwise_or(copper_region, net_masks[label])
    copper_region = cv2.erode(copper_region, erode_kernel, iterations=1)
    copper_region = cv2.bitwise_and(copper_region, coverage_g)
    if int((copper_region > 0).sum()) < 100:
        return _undeterminable(
            "SEGMENTATION_NOT_TRUSTED",
            "too little labeled copper survives inside the overlap",
            metrics,
        )
    # Background samples: coverage pixels well away from either net.
    near_copper = cv2.dilate(
        cv2.bitwise_or(net_masks[1], net_masks[2]), erode_kernel, iterations=2
    )
    bg_region = cv2.bitwise_and(coverage_g, cv2.bitwise_not(near_copper))
    if int((bg_region > 0).sum()) < 100:
        return _undeterminable(
            "SEGMENTATION_NOT_TRUSTED",
            "not enough background inside the overlap to set a threshold",
            metrics,
        )

    # ---- 3. Golden self-consistency gate ----------------------------------
    g_copper = _median_in(golden, copper_region)
    g_bg = _median_in(golden, bg_region)
    if abs(g_copper - g_bg) < MIN_INTENSITY_SEPARATION:
        return _undeterminable(
            "SEGMENTATION_NOT_TRUSTED",
            "golden copper/background gray difference below the supported "
            "contrast; the label mask does not match a bimodal board",
            metrics,
        )
    golden_conductor = _threshold_mask(golden, g_copper, g_bg, coverage_g)
    label_union = cv2.bitwise_or(net_masks[1], net_masks[2])
    dice = _dice(golden_conductor, label_union, coverage_g)
    metrics["golden_label_dice"] = float(dice)
    if dice < MIN_GOLDEN_DICE:
        return _undeterminable(
            "SEGMENTATION_NOT_TRUSTED",
            f"thresholding the golden image reproduces the label masks only "
            f"at Dice={dice:.3f} (< {MIN_GOLDEN_DICE}); annotation does not "
            "match the board image",
            metrics,
        )

    # ---- 4. Reconstruct the inspected conductor ---------------------------
    c_copper = _median_in(corrected, copper_region)
    c_bg = _median_in(corrected, bg_region)
    separation = c_copper - c_bg
    metrics["conductor_intensity"] = c_copper
    metrics["background_intensity"] = c_bg
    metrics["intensity_separation"] = float(abs(separation))
    if abs(separation) < MIN_INTENSITY_SEPARATION:
        return _undeterminable(
            "SEGMENTATION_NOT_TRUSTED",
            f"corrected copper/background separation is only "
            f"{abs(separation):.1f} gray levels; conductor segmentation is "
            "not trustworthy",
            metrics,
        )
    threshold = 0.5 * (c_copper + c_bg)
    metrics["conductor_threshold_gray"] = float(threshold)
    conductor = _threshold_mask(corrected, c_copper, c_bg, coverage_g)
    conductor, dropped = _drop_specks(conductor)
    metrics["dropped_speck_components"] = dropped

    # All four pads must lie inside the judged (overlap) region.
    near_band_kernel = np.ones(
        (2 * ALIGN_TOLERANCE_PX + 1,) * 2, np.uint8
    )
    for label in NET_LABELS:
        for px, py in annotation.pads[label]:
            window = coverage_g[
                max(0, py - ALIGN_TOLERANCE_PX):py + ALIGN_TOLERANCE_PX + 1,
                max(0, px - ALIGN_TOLERANCE_PX):px + ALIGN_TOLERANCE_PX + 1,
            ]
            if int((window > 0).sum()) == 0:
                return _undeterminable(
                    "PAD_OUTSIDE_OVERLAP",
                    f"net {label} pad ({px},{py}) falls outside the "
                    "trustworthy overlap; continuity cannot be judged",
                    metrics,
                )

    # ---- 5. Per-net area plausibility -------------------------------------
    voronoi = _voronoi_regions(net_masks)
    for label in NET_LABELS:
        area_label = float((net_masks[label] > 0).sum())
        area_observed = float(
            ((conductor > 0) & (voronoi[label] > 0)).sum()
        )
        ratio = area_observed / max(area_label, 1.0)
        metrics[f"net{label}_area_ratio"] = float(ratio)
        if not (NET_AREA_RATIO_BAND[0] <= ratio <= NET_AREA_RATIO_BAND[1]):
            return _undeterminable(
                "SEGMENTATION_NOT_TRUSTED",
                f"net {label} reconstructed area ratio {ratio:.2f} outside "
                f"plausible band {NET_AREA_RATIO_BAND}; conductor "
                "segmentation is not trustworthy",
                metrics,
            )

    # ---- 6. Intra-net open test (Voronoi-restricted) ----------------------
    net_findings: dict[str, dict] = {}
    open_labels: list[int] = []
    for label in NET_LABELS:
        allowed = voronoi[label]
        sub = cv2.bitwise_and(conductor, allowed)
        num, cc = cv2.connectedComponents(sub, connectivity=8)
        comp_ids = [
            _component_at(cc, px, py, allowed)
            for px, py in annotation.pads[label]
        ]
        pad_ok = [cid > 0 for cid in comp_ids]
        connected = bool(pad_ok[0] and pad_ok[1] and
                         comp_ids[0] == comp_ids[1])
        if not connected:
            open_labels.append(label)
        net_findings[f"net{label}"] = {
            "open": not connected,
            "pad_components": comp_ids,
            "pads_on_conductor": pad_ok,
            "component_count": int(num - 1),
        }

    # ---- 7. Cross-net short test (full conductor mask) --------------------
    num_full, cc_full, stats_full, _ = cv2.connectedComponentsWithStats(
        conductor, connectivity=8
    )
    near: dict[int, np.ndarray] = {
        label: cv2.dilate(net_masks[label], near_band_kernel, iterations=1)
        for label in NET_LABELS
    }
    # Pixels in the open gap between the two nets. The corridor excludes a
    # 3 px band hugging either label so that a wide trace junction can never
    # masquerade as a wide bridge neck: only the bridge's own mid-span
    # width is measured.
    label_union = cv2.bitwise_or(net_masks[1], net_masks[2])
    label_hug = cv2.dilate(label_union, near_band_kernel, iterations=1)
    label_reach = cv2.dilate(label_union, near_band_kernel, iterations=8)
    gap_corridor = cv2.bitwise_and(
        label_reach,
        cv2.bitwise_and(
            cv2.bitwise_not(label_hug), coverage_g
        ),
    )
    corridor_pixels = int((gap_corridor > 0).sum())

    short_mask = np.zeros_like(conductor)
    bridge_components: list[int] = []
    for comp in range(1, num_full):
        comp_mask = (cc_full == comp).astype(np.uint8) * 255
        touches_1 = int(cv2.bitwise_and(comp_mask, near[1]).sum())
        touches_2 = int(cv2.bitwise_and(comp_mask, near[2]).sum())
        if touches_1 == 0 or touches_2 == 0:
            continue
        # Width check on the mid-gap part of the component: erosion with a
        # 3x3 kernel leaves pixels only where the conductor is >=3 px wide,
        # so a genuine >=3 px bridge passes while a 1-2 px link fails even
        # though both connect the same wide traces at their junctions.
        corridor_part = cv2.bitwise_and(comp_mask, gap_corridor)
        wide = cv2.erode(corridor_part, WIDTH_CHECK_KERNEL, iterations=1)
        wide_ok = int((wide > 0).sum()) >= MIN_GAP_PIXELS
        # Nets whose labels nearly touch leave almost no corridor; in that
        # case require a substantial connected component instead.
        size_ok = (
            int(stats_full[comp, cv2.CC_STAT_AREA]) >= MIN_COMP_PIXELS_NO_GAP
            if corridor_pixels < MIN_GAP_PIXELS else False
        )
        if wide_ok or size_ok:
            bridge_components.append(int(comp))
            short_mask = cv2.bitwise_or(short_mask, comp_mask)
    is_short = bool(bridge_components)
    metrics["short_component_ids"] = bridge_components
    metrics["short_pixels"] = int((short_mask > 0).sum())

    # ---- 8. Verdict --------------------------------------------------------
    if open_labels and is_short:
        status = "open_and_short"
    elif open_labels:
        status = "open"
    elif is_short:
        status = "short"
    else:
        status = "normal"

    debug_b64 = None
    if include_debug:
        debug_b64 = encode_png_base64(
            _debug_panel(golden, corrected, annotation, conductor,
                         short_mask, voronoi)
        )

    return {
        "status": status,
        "reason_code": None,
        "reason": None,
        "metrics": metrics,
        "correction": correction_payload(aln),
        "nets": net_findings,
        "short": {
            "short": is_short,
            "between_nets": [1, 2] if is_short else [],
            "component_ids": bridge_components,
        },
        "open_nets": open_labels,
        "coordinate_space": "golden",
        "conductor_mask_png_base64": encode_png_base64(conductor),
        "short_mask_png_base64": encode_png_base64(short_mask),
        "debug_png_base64": debug_b64,
    }


def _debug_panel(
    golden: np.ndarray,
    corrected: np.ndarray,
    annotation: Annotation,
    conductor: np.ndarray,
    short_mask: np.ndarray,
    voronoi: dict[int, np.ndarray],
) -> np.ndarray:
    """2x2 BGR diagnostic panel in golden-frame coordinates."""
    def to_bgr(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Top row: raw golden, exposure-corrected inspected.
    top = np.hstack([to_bgr(golden), to_bgr(corrected)])

    # Bottom-left: labels + pads over corrected image.
    overlay = to_bgr(corrected)
    overlay[annotation.net_masks[1] > 0] = (0, 0, 200)   # red = net 1
    overlay[annotation.net_masks[2] > 0] = (0, 180, 0)   # green = net 2
    for label in NET_LABELS:
        for px, py in annotation.pads[label]:
            cv2.drawMarker(
                overlay, (px, py), (0, 255, 255),
                cv2.MARKER_CROSS, markerSize=12, thickness=2,
            )

    # Bottom-right: reconstructed conductor, bridges highlighted.
    recon = to_bgr(corrected)
    recon[conductor > 0] = (220, 220, 220)
    # Tint each reconstructed conductor by nearest net.
    recon[(conductor > 0) & (voronoi[1] > 0)] = (60, 60, 220)
    recon[(conductor > 0) & (voronoi[2] > 0)] = (60, 200, 60)
    recon[short_mask > 0] = (0, 0, 255)
    for label in NET_LABELS:
        for px, py in annotation.pads[label]:
            cv2.drawMarker(
                recon, (px, py), (0, 255, 255),
                cv2.MARKER_CROSS, markerSize=12, thickness=2,
            )
    bottom = np.hstack([overlay, recon])
    return np.vstack([top, bottom])

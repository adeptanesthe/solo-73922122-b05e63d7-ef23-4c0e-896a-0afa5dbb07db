"""Shared trusted registration + exposure-compensation front-end.

Both pipelines (defect re-inspection and continuity analysis) must rely on
the *same* trustworthy correction, so the common stages live here:

1. Content trust gates (solid / low-texture / periodic ambiguity).
2. Multi-scale robust ECC affine registration with residual-mask refits.
3. Warp plausibility, correlation and overlap gates.
4. Robust (Huber-IRLS) linear gain/bias exposure fit in golden-frame geometry.

The entry point :func:`align_and_correct` works on decoded numpy arrays
only and never touches the file system.  A failed trust gate is returned as
an :class:`AlignmentFailure` -- callers must translate it into an
``undeterminable`` response and never report a healthy/pass verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry, trust
from .exposure import (
    ExposureFit,
    correct_to_reference,
    exposure_plausible,
    fit_exposure,
)
from .io_utils import IMAGE_SIZE

ROBUST_REFITS = 2
RESIDUAL_REJECT_K = 3.0


@dataclass
class AlignmentResult:
    """A trustworthy correction, all geometry expressed in golden frame."""

    warp: np.ndarray                      # 2x3, golden -> inspected
    params: dict
    cc: float
    level_correlations: list[float]
    inlier_ratios: list[float]
    overlap_ratio: float
    coverage_g: np.ndarray                # uint8 0/255, golden frame
    coverage_i: np.ndarray                # uint8 0/255, inspected frame
    fit: ExposureFit
    inspected_in_golden: np.ndarray       # raw inspected rendered to golden
    corrected_golden: np.ndarray          # exposure-corrected, uint8
    metrics: dict = field(default_factory=dict)


@dataclass
class AlignmentFailure:
    """A trust-gate failure; callers must answer undeterminable."""

    reason_code: str
    reason: str
    metrics: dict = field(default_factory=dict)


def _build_inlier_mask_golden_frame(
    golden: np.ndarray,
    inspected_in_golden: np.ndarray,
    coverage_g: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Inlier mask in golden-frame geometry, using normalized differences.

    A per-pixel z-score difference keeps a global gain/bias mismatch from
    turning well-aligned pixels into outliers before the exposure fit.
    """
    g = golden.astype(np.float32)
    y = inspected_in_golden.astype(np.float32)
    valid = coverage_g > 0
    if valid.sum() < 100:
        return np.zeros_like(coverage_g), 0.0
    gz = np.zeros_like(g)
    yz = np.zeros_like(y)
    gz[valid] = (g[valid] - g[valid].mean()) / max(g[valid].std(), 1e-6)
    yz[valid] = (y[valid] - y[valid].mean()) / max(y[valid].std(), 1e-6)
    diff = np.abs(gz - yz)
    med = float(np.median(diff[valid]))
    mad = float(np.median(np.abs(diff[valid] - med)))
    sigma = max(1.4826 * mad, 0.1)
    thr = med + RESIDUAL_REJECT_K * sigma
    mask_g = ((diff < thr) & valid).astype(np.uint8) * 255
    return mask_g, float((mask_g[valid] > 0).mean())


def align_and_correct(
    golden: np.ndarray,
    inspected: np.ndarray,
) -> AlignmentResult | AlignmentFailure:
    """Run the shared trust -> registration -> exposure pipeline.

    Both inputs must be 512x512 uint8 arrays.  Returns either a fully
    validated :class:`AlignmentResult` or an :class:`AlignmentFailure`
    whose ``reason_code`` is safe to surface to the API.
    """
    if golden.shape != (IMAGE_SIZE, IMAGE_SIZE) or \
            inspected.shape != (IMAGE_SIZE, IMAGE_SIZE):
        return AlignmentFailure(
            "BAD_IMAGE_SIZE", "both images must be 512x512"
        )
    golden = golden.astype(np.uint8)
    inspected = inspected.astype(np.uint8)
    metrics: dict = {}

    # ---- 1. Content trust gates -------------------------------------------
    for img, role in ((golden, "golden"), (inspected, "inspected")):
        verdict = trust.assess_content(img, role)
        metrics.update(verdict.metrics)
        if not verdict.decidable:
            return AlignmentFailure(
                verdict.reason_code or "CONTENT_REJECTED",
                verdict.reason or "content not decidable",
                metrics,
            )

    # ---- 2. Robust multi-scale ECC ----------------------------------------
    try:
        reg = geometry.estimate_affine(golden, inspected)
        warp = reg.warp  # golden -> inspected (forward)
        inlier_ratios: list[float] = []
        for _ in range(ROBUST_REFITS):
            i_in_g = geometry.render_inspected_in_golden(inspected, warp)
            coverage_g, _ = geometry.overlap_masks(warp)
            mask_g, inlier_ratio = _build_inlier_mask_golden_frame(
                golden, i_in_g, coverage_g
            )
            inlier_ratios.append(inlier_ratio)
            if inlier_ratio < trust.MIN_INLIER_RATIO:
                metrics["ecc_inlier_ratio"] = inlier_ratios
                return AlignmentFailure(
                    "LOW_INLIER_RATIO",
                    f"only {inlier_ratio:.1%} of overlap pixels support the "
                    "alignment; the correction is not trustworthy",
                    metrics,
                )
            # ECC's inputImage mask must be expressed in inspected frame.
            mask_i = geometry.render_golden_in_inspected(
                mask_g, warp, nearest=True
            )
            # A refit can diverge on a degraded mask; keep the previous warp.
            try:
                refit = geometry.robust_refit_with_mask(
                    golden, inspected, mask_i, warp
                )
                warp = refit.warp
                reg.cc = refit.cc
                reg.levels = refit.levels
            except geometry.RegistrationError:
                break
        metrics["ecc_inlier_ratio"] = inlier_ratios
    except geometry.RegistrationError as exc:
        return AlignmentFailure("ECC_FAILED", str(exc), metrics)

    metrics["ecc_correlation"] = float(reg.cc)
    metrics["ecc_level_correlations"] = [float(x) for x in reg.levels]
    params = geometry.decompose_affine(warp)
    metrics["warp_params"] = params

    # ---- 3. Plausibility / correlation / overlap gates --------------------
    ok, why = geometry.warp_plausible(warp)
    if not ok:
        return AlignmentFailure(
            "IMPLAUSIBLE_WARP", why or "implausible warp", metrics
        )
    if reg.cc < trust.MIN_CC:
        return AlignmentFailure(
            "LOW_CORRELATION",
            f"alignment correlation {reg.cc:.3f} below trust threshold "
            f"{trust.MIN_CC}",
            metrics,
        )

    coverage_g, coverage_i = geometry.overlap_masks(warp)
    overlap_ratio = float((coverage_i > 0).mean())
    metrics["overlap_ratio"] = overlap_ratio
    if overlap_ratio < trust.MIN_OVERLAP_RATIO:
        return AlignmentFailure(
            "INSUFFICIENT_OVERLAP",
            f"overlap covers only {overlap_ratio:.1%} of the frame; "
            "non-overlap regions cannot be judged",
            metrics,
        )

    # ---- 4. Robust exposure fit in golden-frame geometry ------------------
    i_in_g = geometry.render_inspected_in_golden(inspected, warp)
    fit_kernel = np.ones((3, 3), np.uint8)
    coverage_fit = cv2.erode(coverage_g, fit_kernel, iterations=2)
    try:
        fit = fit_exposure(i_in_g, golden, coverage_fit)
    except ValueError as exc:
        return AlignmentFailure("EXPOSURE_FIT_FAILED", str(exc), metrics)
    metrics["exposure"] = {
        "gain": fit.gain,
        "bias": fit.bias,
        "robust_sigma": fit.robust_sigma,
        "inlier_ratio": fit.inlier_ratio,
        "irls_iterations": fit.iterations,
    }
    ok, why = exposure_plausible(fit)
    if not ok:
        return AlignmentFailure(
            "IMPLAUSIBLE_EXPOSURE", why or "bad exposure", metrics
        )

    corrected = correct_to_reference(i_in_g, fit)
    return AlignmentResult(
        warp=warp,
        params=params,
        cc=float(reg.cc),
        level_correlations=[float(x) for x in reg.levels],
        inlier_ratios=inlier_ratios,
        overlap_ratio=overlap_ratio,
        coverage_g=coverage_g,
        coverage_i=coverage_i,
        fit=fit,
        inspected_in_golden=i_in_g,
        corrected_golden=corrected,
        metrics=metrics,
    )


def correction_payload(aln: AlignmentResult) -> dict:
    """Build the public ``correction`` block shared by both API responses."""
    fit = aln.fit
    return {
        "warp_matrix": aln.warp.tolist(),
        "warp_convention": "golden_coords -> inspected_coords",
        "warp_params": aln.params,
        "gain": fit.gain,
        "bias": fit.bias,
        "exposure_model": "observed I = gain * G + bias; "
                          "corrected = (I - bias) / gain",
        "correction_gain": 1.0 / fit.gain,
        "correction_bias": -fit.bias / fit.gain,
        "robust_sigma": fit.robust_sigma,
        "ecc_correlation": aln.cc,
        "overlap_ratio": aln.overlap_ratio,
    }

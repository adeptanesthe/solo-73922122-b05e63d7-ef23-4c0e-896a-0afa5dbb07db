"""FastAPI application for PCB golden-board re-inspection.

Endpoints
---------
``POST /api/v1/inspect``
    Multipart upload of two 512x512 grayscale PNG files (``golden`` and
    ``inspected``). Returns registration / exposure evidence plus the
    defect mask and bounding boxes in *inspected-image* coordinates.
``POST /api/v1/continuity``
    Multipart upload of the same two PNGs, a golden-coordinate label mask
    (0 background / 1-2 the two nets) and four pad centers. Returns the
    per-net open-circuit verdict and the cross-net short-circuit verdict
    in golden-frame coordinates.
``GET  /health``
    Liveness probe.
``GET  /docs``
    Swagger UI with upload forms (Try it out).

A content-level inability to decide is HTTP 200 with
``status="undeterminable"`` (it is a valid inspection outcome). Bad input
is HTTP 422/400.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .inspection.annotation import AnnotationError, parse_pads, validate_annotation
from .inspection.continuity import analyze_continuity
from .inspection.detector import inspect_images
from .inspection.io_utils import (
    IMAGE_SIZE,
    ImageValidationError,
    decode_gray_png,
    decode_label_png,
)

app = FastAPI(
    title="PCB Golden-Board Re-inspection API",
    version="1.1.0",
    description=(
        "Upload a 512x512 grayscale golden-board PNG and an inspected PNG. "
        "The service estimates a global affine registration and linear "
        "exposure compensation (robust to defects), then either localizes "
        "defects (/api/v1/inspect) or judges two-net electrical continuity "
        "(/api/v1/continuity). Ambiguous content (solid color, periodic "
        "texture, untrusted correction or conductor segmentation) returns "
        "status **undeterminable**, never pass/normal."
    ),
)


class WarpParams(BaseModel):
    tx_px: float
    ty_px: float
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_deg: float


class Correction(BaseModel):
    warp_matrix: list[list[float]] = Field(
        description="2x3 affine, maps golden coords to inspected coords"
    )
    warp_convention: str
    warp_params: WarpParams
    gain: float = Field(description="Fitted generation gain: I = gain*G + bias")
    bias: float = Field(description="Fitted generation bias")
    exposure_model: str
    correction_gain: float = Field(
        description="Multiplier that maps inspected intensities back to golden"
    )
    correction_bias: float = Field(
        description="Offset that maps inspected intensities back to golden"
    )
    robust_sigma: float
    ecc_correlation: float
    overlap_ratio: float


class InspectResponse(BaseModel):
    status: str = Field(
        description="pass | defect | undeterminable"
    )
    reason_code: str | None = Field(
        default=None,
        description=(
            "Set when undeterminable: SOLID_COLOR, LOW_TEXTURE, "
            "PERIODIC_TEXTURE, ECC_FAILED, LOW_CORRELATION, "
            "IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, LOW_INLIER_RATIO, "
            "EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE"
        ),
    )
    reason: str | None = None
    metrics: dict
    correction: Correction | None = None
    defect_mask_png_base64: str | None = Field(
        default=None, description="PNG-encoded binary defect mask, inspected coordinates",
    )
    defect_boxes: list[list[int]] = Field(
        default_factory=list,
        description="[x, y, w, h] bounding boxes in inspected coordinates",
    )
    debug_png_base64: str | None = Field(
        default=None, description="2x2 diagnostic panel (PNG, base64)"
    )


class NetContinuity(BaseModel):
    open: bool = Field(description="True when the two pads are not connected")
    pad_components: list[int]
    pads_on_conductor: list[bool]
    component_count: int


class ShortVerdict(BaseModel):
    short: bool
    between_nets: list[int]
    component_ids: list[int]


class ContinuityResponse(BaseModel):
    status: str = Field(
        description=(
            "normal | open | short | open_and_short | undeterminable"
        )
    )
    reason_code: str | None = Field(
        default=None,
        description=(
            "Set when undeterminable: SOLID_COLOR, LOW_TEXTURE, "
            "PERIODIC_TEXTURE, ECC_FAILED, LOW_CORRELATION, "
            "IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, LOW_INLIER_RATIO, "
            "EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE, "
            "SEGMENTATION_NOT_TRUSTED, PAD_OUTSIDE_OVERLAP"
        ),
    )
    reason: str | None = None
    metrics: dict
    correction: Correction | None = None
    nets: dict[str, NetContinuity] | None = None
    short: ShortVerdict | None = None
    open_nets: list[int] = Field(
        default_factory=list,
        description="Labels of nets with an internal open circuit",
    )
    coordinate_space: str = Field(
        default="golden", description="Coordinate frame of all returned masks"
    )
    conductor_mask_png_base64: str | None = Field(
        default=None,
        description="Reconstructed inspected conductor mask (golden frame)",
    )
    short_mask_png_base64: str | None = Field(
        default=None,
        description="Conductor components bridging the two nets (golden frame)",
    )
    debug_png_base64: str | None = Field(
        default=None, description="2x2 diagnostic panel (PNG, base64)"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "image_size": IMAGE_SIZE}


@app.post("/api/v1/inspect", response_model=InspectResponse)
async def inspect(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    include_debug: bool = Form(
        default=True, description="Include the 2x2 diagnostic panel"
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return inspect_images(g, i, include_debug=include_debug)


@app.post("/api/v1/continuity", response_model=ContinuityResponse)
async def continuity(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    labels: UploadFile = File(
        ...,
        description=(
            "512x512 label mask PNG in golden coordinates: "
            "0=background, 1/2=the two nets"
        ),
    ),
    pads: str = Form(
        ...,
        description=(
            "JSON array of four [x, y] pad centers in golden coordinates: "
            "first two belong to net 1, last two to net 2"
        ),
    ),
    include_debug: bool = Form(
        default=True, description="Include the 2x2 diagnostic panel"
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    labels_bytes = await labels.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
        label_img = decode_label_png(labels_bytes)
        pad_map = parse_pads(pads)
        annotation = validate_annotation(label_img, pad_map)
    except (ImageValidationError, AnnotationError) as exc:
        # Bad annotations / uploads are client errors, not undeterminable.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return analyze_continuity(g, i, annotation, include_debug=include_debug)

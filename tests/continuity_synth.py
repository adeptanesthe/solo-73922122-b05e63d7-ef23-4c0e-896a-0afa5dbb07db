"""Fixed-seed synthetic two-net boards for continuity testing.

The boards obey the production input envelope:

* 512x512 grayscale, non-periodic content;
* copper/background gray difference >= 40;
* trace width >= 5 px;
* open gaps and cross-net bridges are >= 3 px wide;
* the two nets never touch on the golden board and each net's two pads are
  connected by its own copper.

Defects (open cut, cross-net bridge, or both) are painted before the same
affine warp + linear exposure + sensor-noise disturbance used by the defect
generator.  The label mask and pad centers stay in golden coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 512
BASE_SEED = 20261117
BG_GRAY = 80.0
COPPER_GRAY = 205.0
TRACE_WIDTH = 7
PAD_SIDE = 14
NOISE_SIGMA_GOLDEN = 3.0
NOISE_SIGMA_INSPECTED = 2.5

# Deterministic geometry (golden-frame pixel coordinates).
NET1_PADS = ((60, 150), (450, 150))
NET2_PADS = ((60, 360), (450, 360))
NET1_TRACE_Y = 150
NET2_TRACE_Y = 360
# Open cut on net 1, left of the bridge so a cross-net detour cannot heal it.
CUT_X = 150
CUT_HALF_W = 5          # total gap 10 px >= the 3 px minimum
# Vertical bridge joining both traces.
BRIDGE_X = 280
BRIDGE_WIDTH = 4        # >= 3 px minimum bridge width


@dataclass
class ContinuitySample:
    name: str
    golden: np.ndarray
    inspected: np.ndarray
    labels: np.ndarray               # uint8 0/1/2, golden coordinates
    pads: list[list[int]]            # four [x, y] points, net1 then net2
    gain: float
    bias: float
    warp: np.ndarray
    expect_status: str
    expect_open_nets: list[int]
    expect_short: bool


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + 700 + index)


def _paint_pad(img: np.ndarray, labels: np.ndarray,
               cx: int, cy: int, label: int, gray: float) -> None:
    half = PAD_SIDE // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half),
                  gray, thickness=-1)
    cv2.rectangle(labels, (cx - half, cy - half), (cx + half, cy + half),
                  label, thickness=-1)


def make_two_net_board(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Paint the defect-free golden board and its label mask."""
    img = np.full((SIZE, SIZE), BG_GRAY, np.float32)
    labels = np.zeros((SIZE, SIZE), np.uint8)

    # Smooth non-repeating shading so the content is non-periodic.
    small = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
    shading = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    img += shading * 10.0

    copper = COPPER_GRAY + float(rng.uniform(-8.0, 8.0))
    for (pads, y, label) in (
        (NET1_PADS, NET1_TRACE_Y, 1),
        (NET2_PADS, NET2_TRACE_Y, 2),
    ):
        x0, x1 = pads[0][0], pads[1][0]
        cv2.line(img, (x0, y), (x1, y), copper, thickness=TRACE_WIDTH)
        cv2.line(labels, (x0, y), (x1, y), label, thickness=TRACE_WIDTH)
        for cx, cy in pads:
            _paint_pad(img, labels, cx, cy, label, copper)

    img += rng.normal(0.0, NOISE_SIGMA_GOLDEN, (SIZE, SIZE)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), labels


def _apply_open_cut(board: np.ndarray) -> None:
    """Paint a background-colored gap across net 1 of the inspected board.

    The golden label mask is intentionally untouched: it always describes
    the intact golden board.
    """
    y = NET1_TRACE_Y
    half = TRACE_WIDTH // 2 + 3
    cv2.rectangle(
        board,
        (CUT_X - CUT_HALF_W, y - half),
        (CUT_X + CUT_HALF_W, y + half),
        BG_GRAY, thickness=-1,
    )


def _apply_bridge(board: np.ndarray, width: int = BRIDGE_WIDTH) -> None:
    """Paint a vertical copper bridge between the two nets."""
    cv2.line(
        board,
        (BRIDGE_X, NET1_TRACE_Y),
        (BRIDGE_X, NET2_TRACE_Y),
        COPPER_GRAY, thickness=width,
    )


def _random_warp(rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    angle = float(rng.uniform(-2.5, 2.5))
    tx = float(rng.uniform(-6.0, 6.0))
    ty = float(rng.uniform(-6.0, 6.0))
    warp = cv2.getRotationMatrix2D(
        ((SIZE - 1) / 2.0, (SIZE - 1) / 2.0), angle, 1.0
    )
    warp[0, 2] += tx
    warp[1, 2] += ty
    return warp.astype(np.float64), angle, tx, ty


def make_continuity_sample(
    index: int,
    defect: str = "none",
    bridge_width: int = BRIDGE_WIDTH,
) -> ContinuitySample:
    """``defect`` is one of none | open | short | both."""
    rng = _rng(index)
    golden, labels = make_two_net_board(rng)
    disturbed = golden.copy()

    expect_open: list[int] = []
    expect_short = False
    if defect in ("open", "both"):
        _apply_open_cut(disturbed)
        expect_open = [1]
    if defect in ("short", "both"):
        _apply_bridge(disturbed, width=bridge_width)
        expect_short = True

    if defect == "none":
        name = f"cont_{index:02d}_normal"
    else:
        name = f"cont_{index:02d}_{defect}"

    warp, _, _, _ = _random_warp(rng)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-12.0, 12.0))
    warped = cv2.warpAffine(
        disturbed, warp.astype(np.float32), (SIZE, SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    inspected = gain * warped.astype(np.float32) + bias
    inspected += rng.normal(
        0.0, NOISE_SIGMA_INSPECTED, (SIZE, SIZE)
    ).astype(np.float32)
    inspected = np.clip(inspected, 0, 255).astype(np.uint8)

    if expect_open and expect_short:
        expect_status = "open_and_short"
    elif expect_open:
        expect_status = "open"
    elif expect_short:
        expect_status = "short"
    else:
        expect_status = "normal"

    pads = [list(NET1_PADS[0]), list(NET1_PADS[1]),
            list(NET2_PADS[0]), list(NET2_PADS[1])]
    return ContinuitySample(
        name=name, golden=golden, inspected=inspected, labels=labels,
        pads=pads, gain=gain, bias=bias, warp=warp,
        expect_status=expect_status,
        expect_open_nets=expect_open, expect_short=expect_short,
    )


def make_periodic_continuity_sample(index: int) -> ContinuitySample:
    """Periodic board: registration is non-unique -> undeterminable."""
    rng = _rng(9000 + index)
    period = 16
    coords = np.arange(SIZE)
    xx = np.tile(coords, (SIZE, 1))
    pattern = np.where((xx // period) % 2 == 0, 70, 180).astype(np.uint8)
    golden = np.clip(
        pattern + rng.normal(0, 1.0, (SIZE, SIZE)), 0, 255
    ).astype(np.uint8)
    warp = np.array([[1.0, 0.0, float(period)], [0.0, 1.0, 0.0]])
    inspected = cv2.warpAffine(
        golden, warp.astype(np.float32), (SIZE, SIZE),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )

    # Structurally valid annotation even though the content is ambiguous.
    labels = np.zeros((SIZE, SIZE), np.uint8)
    cv2.rectangle(labels, (40, 40), (200, 90), 1, -1)
    cv2.rectangle(labels, (40, 300), (200, 350), 2, -1)
    pads = [[60, 65], [180, 65], [60, 325], [180, 325]]
    return ContinuitySample(
        name=f"cont_{index:02d}_periodic", golden=golden, inspected=inspected,
        labels=labels, pads=pads, gain=1.0, bias=0.0, warp=warp,
        expect_status="undeterminable", expect_open_nets=[],
        expect_short=False,
    )

"""Golden-board annotation validation for continuity analysis.

The caller supplies, in golden-board coordinates:

* a label mask PNG whose pixels are 0 (background), 1 or 2 (the two nets);
* two pad centers per net (four points total).

A faulty annotation is a *client error*, not an undeterminable content
verdict: :class:`AnnotationError` carries a message the API maps to HTTP
400/422.  Validation enforces the structural guarantees the rest of the
pipeline relies on:

* both labels exist and only values 0/1/2 are used;
* the two nets do not touch (8-connectivity) on the golden board;
* each net is internally 8-connected between its own two pads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from .io_utils import IMAGE_SIZE

NET_LABELS = (1, 2)
# Pads may be this close to the nominal label border and still count as
# "on copper": a pad center rounding to one pixel off is tolerated.
PAD_TOLERANCE_PX = 1


class AnnotationError(ValueError):
    """Raised when the label mask or pad centers are invalid."""


@dataclass
class Annotation:
    labels: np.ndarray                  # uint8 512x512, values 0/1/2
    net_masks: dict[int, np.ndarray]    # label -> uint8 0/255 mask
    pads: dict[int, list[tuple[int, int]]]


def _parse_point(value: object, index: int) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnnotationError(
            f"pad #{index} must be an [x, y] pair, got {value!r}"
        )
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"pad #{index} coordinates must be numeric") from exc
    if not (np.isfinite(x) and np.isfinite(y)):
        raise AnnotationError(f"pad #{index} coordinates must be finite")
    if not (0.0 <= x <= IMAGE_SIZE - 1 and 0.0 <= y <= IMAGE_SIZE - 1):
        raise AnnotationError(
            f"pad #{index} ({x:.1f},{y:.1f}) is outside the "
            f"{IMAGE_SIZE}x{IMAGE_SIZE} frame"
        )
    return x, y


def parse_pads(text: str | None) -> dict[int, list[tuple[int, int]]]:
    """Parse the JSON ``pads`` form field.

    Expected layout::

        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    points 1-2 are the two pads of net 1, points 3-4 the two pads of net 2.
    """
    if not text:
        raise AnnotationError("missing 'pads' form field")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"'pads' must be valid JSON: {exc}") from exc
    if not isinstance(data, list) or len(data) != 4:
        raise AnnotationError(
            "'pads' must be a JSON array of exactly four [x, y] points "
            "(two per net, ordered net1, net1, net2, net2)"
        )
    points = [_parse_point(v, i) for i, v in enumerate(data)]
    rounded = [(int(round(x)), int(round(y))) for x, y in points]
    if len(set(rounded)) != 4:
        raise AnnotationError("the four pad centers must be distinct")
    return {1: rounded[0:2], 2: rounded[2:4]}


def validate_annotation(
    labels: np.ndarray, pads: dict[int, list[tuple[int, int]]]
) -> Annotation:
    """Validate label values, net separation and pad connectivity."""
    if labels.shape != (IMAGE_SIZE, IMAGE_SIZE) or labels.dtype != np.uint8:
        raise AnnotationError(
            f"label mask must be a {IMAGE_SIZE}x{IMAGE_SIZE} uint8 image"
        )
    unique = set(int(v) for v in np.unique(labels))
    if not unique.issubset({0, 1, 2}):
        raise AnnotationError(
            "label mask may only contain values 0 (background), 1 and 2 "
            f"(nets); found {sorted(unique)}"
        )

    net_masks: dict[int, np.ndarray] = {}
    for label in NET_LABELS:
        mask = (labels == label).astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            raise AnnotationError(f"label mask contains no pixels for net {label}")
        net_masks[label] = mask

    # Golden-board guarantee: the two nets must not touch (8-connectivity).
    kernel3 = np.ones((3, 3), np.uint8)
    dilated_1 = cv2.dilate(net_masks[1], kernel3, iterations=1)
    if int(cv2.bitwise_and(dilated_1, net_masks[2]).sum()) > 0:
        raise AnnotationError(
            "nets 1 and 2 are in contact in the golden label mask; "
            "they must be separate networks"
        )

    annotation = Annotation(labels=labels, net_masks=net_masks, pads=pads)
    for label in NET_LABELS:
        _validate_net_pads(annotation, label)
    return annotation


def _validate_net_pads(annotation: Annotation, label: int) -> None:
    mask = annotation.net_masks[label]
    num, cc = cv2.connectedComponents(mask, connectivity=8)
    if num <= 1:
        raise AnnotationError(f"net {label} has no connected copper region")

    tolerance_kernel = np.ones(
        (2 * PAD_TOLERANCE_PX + 1,) * 2, np.uint8
    )
    near = cv2.dilate(mask, tolerance_kernel, iterations=1)
    pad_components: list[int] = []
    for px, py in annotation.pads[label]:
        if near[py, px] == 0:
            raise AnnotationError(
                f"pad ({px},{py}) is not on net {label} copper"
            )
        pad_components.append(int(cc[py, px]))

    # A pad center one pixel off the mask lands on background (cc == 0);
    # locate the adjacent copper component in that case.
    resolved: list[int] = []
    for (px, py), comp in zip(annotation.pads[label], pad_components):
        if comp == 0:
            window = cc[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
            copper = window[window > 0]
            if copper.size == 0:
                raise AnnotationError(
                    f"pad ({px},{py}) is not connected to net {label} copper"
                )
            comp = int(np.bincount(copper.ravel()).argmax())
        resolved.append(comp)
    if resolved[0] != resolved[1]:
        raise AnnotationError(
            f"the two pads of net {label} are not connected by its label "
            "mask; each net must be a single connected copper region"
        )

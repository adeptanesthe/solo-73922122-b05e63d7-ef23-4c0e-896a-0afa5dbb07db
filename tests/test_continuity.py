"""Functional tests for the two-net continuity API and analysis pipeline."""
from __future__ import annotations

import base64
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inspection.annotation import (
    AnnotationError,
    parse_pads,
    validate_annotation,
)
from app.inspection.continuity import analyze_continuity
from app.inspection.io_utils import decode_png_base64, encode_png
from app.main import app
from tests.continuity_synth import (
    make_continuity_sample,
    make_periodic_continuity_sample,
)

client = TestClient(app)


def _annotation_files(sample):
    return {
        "golden": ("golden.png", io.BytesIO(encode_png(sample.golden)),
                   "image/png"),
        "inspected": ("inspected.png", io.BytesIO(encode_png(sample.inspected)),
                      "image/png"),
        "labels": ("labels.png", io.BytesIO(encode_png(sample.labels)),
                   "image/png"),
    }


def _upload(sample, debug: bool = True, pads_text: str | None = None):
    data = {
        "pads": pads_text or json.dumps(sample.pads),
        "include_debug": "true" if debug else "false",
    }
    return client.post(
        "/api/v1/continuity",
        files=_annotation_files(sample),
        data=data,
    )


# --------------------------------------------------------------------------
# Pipeline-level verdicts (array entry point)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "defect,expected",
    [("none", "normal"), ("open", "open"),
     ("short", "short"), ("both", "open_and_short")],
)
def test_continuity_verdicts(defect, expected):
    for idx in range(4):
        sample = make_continuity_sample(idx, defect=defect)
        annotation = validate_annotation(
            sample.labels, parse_pads(json.dumps(sample.pads))
        )
        result = analyze_continuity(
            sample.golden, sample.inspected, annotation, include_debug=False
        )
        assert result["status"] == expected, (
            idx, defect, result["status"], result["reason_code"]
        )
        assert result["coordinate_space"] == "golden"


def test_normal_board_reports_no_open_or_short():
    sample = make_continuity_sample(2, defect="none")
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    assert result["status"] == "normal"
    assert result["open_nets"] == []
    assert result["short"]["short"] is False
    assert result["correction"] is not None


def test_open_circuit_is_localized_per_net():
    sample = make_continuity_sample(1, defect="open")
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    assert result["open_nets"] == [1]
    assert result["nets"]["net1"]["open"] is True
    assert result["nets"]["net2"]["open"] is False
    assert result["short"]["short"] is False


def test_short_circuit_bridges_both_nets():
    sample = make_continuity_sample(1, defect="short")
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    assert result["status"] == "short"
    assert result["short"]["short"] is True
    assert result["short"]["between_nets"] == [1, 2]
    short_mask = decode_png_base64(result["short_mask_png_base64"])
    assert (short_mask > 0).sum() > 0


def test_short_detour_cannot_mask_intra_net_open():
    """The headline safety property, checked over multiple random seeds."""
    for idx in range(8):
        sample = make_continuity_sample(idx, defect="both")
        annotation = validate_annotation(
            sample.labels, parse_pads(json.dumps(sample.pads))
        )
        result = analyze_continuity(
            sample.golden, sample.inspected, annotation, include_debug=False
        )
        assert result["status"] == "open_and_short"
        assert result["open_nets"] == [1]
        assert result["short"]["short"] is True


@pytest.mark.parametrize("width", [3, 4, 5])
def test_minimum_three_pixel_bridges_are_detected(width):
    # The supported envelope: bridges >= 3 px wide must never be erased.
    sample = make_continuity_sample(4, defect="short", bridge_width=width)
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    assert result["short"]["short"] is True
    assert result["status"] == "short"


def test_one_pixel_speck_link_is_not_a_short():
    sample = make_continuity_sample(4, defect="short", bridge_width=1)
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    # A sub-3px link is outside the guaranteed envelope; the safe behavior
    # is that it never overrides the per-net open test nor crashes.
    assert result["status"] in ("normal", "short")
    assert result["open_nets"] == []


def test_periodic_board_is_undeterminable_never_normal():
    sample = make_periodic_continuity_sample(0)
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=False
    )
    assert result["status"] == "undeterminable"
    assert result["reason_code"] == "PERIODIC_TEXTURE"
    assert result["correction"] is None
    assert result["nets"] is None


def test_masks_are_golden_frame_pngs():
    sample = make_continuity_sample(0, defect="both")
    annotation = validate_annotation(
        sample.labels, parse_pads(json.dumps(sample.pads))
    )
    result = analyze_continuity(
        sample.golden, sample.inspected, annotation, include_debug=True
    )
    conductor = decode_png_base64(result["conductor_mask_png_base64"])
    short_mask = decode_png_base64(result["short_mask_png_base64"])
    debug = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(result["debug_png_base64"]), dtype=np.uint8
        ),
        cv2.IMREAD_COLOR,
    )
    assert conductor.shape == (512, 512)
    assert short_mask.shape == (512, 512)
    # Debug panel is a 2x2 BGR composition.
    assert debug.shape == (1024, 1024, 3)
    # Short mask is a subset of the conductor mask.
    assert np.logical_and(short_mask > 0, conductor == 0).sum() == 0


# --------------------------------------------------------------------------
# Annotation validation (client errors)
# --------------------------------------------------------------------------

def test_bad_label_values_rejected():
    sample = make_continuity_sample(0)
    bad = sample.labels.copy()
    bad[10, 10] = 7
    with pytest.raises(AnnotationError, match="0 .*1 and 2|values"):
        validate_annotation(bad, parse_pads(json.dumps(sample.pads)))


def test_missing_net_label_rejected():
    sample = make_continuity_sample(0)
    only_one = sample.labels.copy()
    only_one[only_one == 2] = 0
    with pytest.raises(AnnotationError, match="net 2"):
        validate_annotation(only_one, parse_pads(json.dumps(sample.pads)))


def test_touching_nets_rejected():
    sample = make_continuity_sample(0)
    touching = sample.labels.copy()
    # Draw a 1-wide label corridor between the two traces.
    touching[151:360, 250] = 1
    with pytest.raises(AnnotationError, match="contact"):
        validate_annotation(touching, parse_pads(json.dumps(sample.pads)))


def test_disconnected_net_rejected():
    sample = make_continuity_sample(0)
    broken = sample.labels.copy()
    # Remove a slice of net 1's trace between its two pads.
    broken[146:155, 250:260] = 0
    with pytest.raises(AnnotationError, match="not connected"):
        validate_annotation(broken, parse_pads(json.dumps(sample.pads)))


def test_pad_off_copper_rejected():
    sample = make_continuity_sample(0)
    bad_pads = [list(p) for p in sample.pads]
    bad_pads[0] = [256, 256]  # image center, background
    with pytest.raises(AnnotationError, match="not on net 1"):
        validate_annotation(sample.labels, parse_pads(json.dumps(bad_pads)))


def test_pad_parse_errors():
    with pytest.raises(AnnotationError):
        parse_pads("not json")
    with pytest.raises(AnnotationError):
        parse_pads(json.dumps([[1, 2], [3, 4]]))  # only two points
    with pytest.raises(AnnotationError):
        parse_pads(json.dumps([[1, 2], [3, 4], [5, 6], [7, 999]]))
    with pytest.raises(AnnotationError):
        parse_pads(None)


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------

def test_continuity_endpoint_normal():
    sample = make_continuity_sample(0, defect="none")
    resp = _upload(sample)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "normal"
    assert body["open_nets"] == []
    assert body["short"]["short"] is False
    assert body["conductor_mask_png_base64"]


def test_continuity_endpoint_open_and_short():
    sample = make_continuity_sample(0, defect="both")
    resp = _upload(sample, debug=False)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "open_and_short"
    assert body["open_nets"] == [1]
    assert body["short"]["short"] is True
    assert body["debug_png_base64"] is None


def test_continuity_endpoint_undeterminable():
    sample = make_periodic_continuity_sample(0)
    resp = _upload(sample)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "PERIODIC_TEXTURE"


def test_continuity_endpoint_rejects_bad_annotation():
    sample = make_continuity_sample(0)
    bad = sample.labels.copy()
    bad[10, 10] = 9
    files = _annotation_files(sample)
    files["labels"] = ("labels.png", io.BytesIO(encode_png(bad)), "image/png")
    resp = client.post(
        "/api/v1/continuity",
        files=files,
        data={"pads": json.dumps(sample.pads)},
    )
    assert resp.status_code == 400
    assert "label" in resp.json()["detail"].lower()


def test_continuity_endpoint_rejects_bad_pads_json():
    sample = make_continuity_sample(0)
    resp = client.post(
        "/api/v1/continuity",
        files=_annotation_files(sample),
        data={"pads": "{not json"},
    )
    assert resp.status_code == 400


def test_continuity_endpoint_rejects_wrong_size():
    sample = make_continuity_sample(0)
    small = np.zeros((256, 256), np.uint8)
    files = _annotation_files(sample)
    files["labels"] = ("labels.png", io.BytesIO(encode_png(small)), "image/png")
    resp = client.post(
        "/api/v1/continuity",
        files=files,
        data={"pads": json.dumps(sample.pads)},
    )
    assert resp.status_code == 400


def test_legacy_inspect_endpoint_still_works():
    sample = make_continuity_sample(0, defect="none")
    resp = client.post(
        "/api/v1/inspect",
        files={
            "golden": ("g.png", io.BytesIO(encode_png(sample.golden)),
                       "image/png"),
            "inspected": ("i.png", io.BytesIO(encode_png(sample.inspected)),
                          "image/png"),
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("pass", "defect", "undeterminable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for PPE detection filtering and glove validation.

Requirements:
1. If gloves are detected, only validate if 2 non-overlapping boxes are detected.
2. At all times, show a max of 2 boxes for gloves and 1 for mask, coverall, and goggles.
"""
import pytest

from ppe_filter import (
    compute_iou,
    are_boxes_overlapping,
    filter_and_validate_ppe,
)
import unified_config as cfg


class TestIoUCalculation:
    def test_disjoint_boxes_have_zero_iou(self):
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == 0.0

    def test_identical_boxes_have_one_iou(self):
        box = (10, 10, 60, 60)
        assert pytest.approx(compute_iou(box, box), 0.001) == 1.0

    def test_partial_overlap_iou(self):
        # 10x10 box and shifted 5x5 overlap:
        # box1: (0, 0, 10, 10) area = 100
        # box2: (5, 0, 15, 10) area = 100
        # intersection: (5, 0, 10, 10) w=5, h=10 -> area = 50
        # union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ~ 0.3333
        box1 = (0, 0, 10, 10)
        box2 = (5, 0, 15, 10)
        assert pytest.approx(compute_iou(box1, box2), 0.01) == 0.3333

    def test_touching_edges_have_zero_iou(self):
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)
        assert compute_iou(box1, box2) == 0.0


class TestBoxOverlapCheck:
    def test_disjoint_boxes_do_not_overlap(self):
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert are_boxes_overlapping(box1, box2, threshold=0.1) is False

    def test_overlapping_boxes_detected(self):
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        assert are_boxes_overlapping(box1, box2, threshold=0.1) is True

    def test_nested_box_considered_overlapping(self):
        # Small box inside a large box
        box1 = (0, 0, 200, 200)
        box2 = (50, 50, 100, 100)
        assert are_boxes_overlapping(box1, box2, threshold=0.1) is True


class TestMaxBoxesDisplayLimits:
    def test_max_boxes_limits_per_class(self):
        """At all times, show max 2 for gloves and 1 for mask, coverall, goggles."""
        raw_detections = [
            # 3 masks with different confidences
            (3, 0.95, 10, 10, 50, 50),
            (3, 0.85, 15, 15, 55, 55),
            (3, 0.70, 20, 20, 60, 60),
            # 2 coveralls
            (0, 0.90, 0, 0, 200, 300),
            (0, 0.80, 5, 5, 205, 305),
            # 2 goggles
            (2, 0.88, 30, 30, 80, 50),
            (2, 0.86, 35, 35, 85, 55),
            # 3 non-overlapping gloves
            (1, 0.92, 10, 150, 40, 190),   # hand 1
            (1, 0.89, 120, 150, 150, 190), # hand 2
            (1, 0.75, 200, 150, 230, 190), # extraneous
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            max_boxes_per_class={0: 1, 1: 2, 2: 1, 3: 1},
            glove_iou_threshold=0.1,
        )

        # Count boxes per class in display_boxes
        counts = {}
        for cls_id, conf, x1, y1, x2, y2 in display_boxes:
            counts[cls_id] = counts.get(cls_id, 0) + 1

        assert counts.get(0, 0) == 1, "Coverall must show max 1 box"
        assert counts.get(1, 0) == 2, "Gloves must show max 2 boxes"
        assert counts.get(2, 0) == 1, "Goggles must show max 1 box"
        assert counts.get(3, 0) == 1, "Mask must show max 1 box"

    def test_top_confidence_box_selected_when_multiple(self):
        """Top confidence box is preserved for single-box classes."""
        raw_detections = [
            (3, 0.70, 0, 0, 50, 50),
            (3, 0.99, 10, 10, 60, 60),
            (3, 0.80, 20, 20, 70, 70),
        ]
        _, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            max_boxes_per_class={3: 1},
        )
        assert len(display_boxes) == 1
        assert display_boxes[0][1] == 0.99, "Must select highest confidence box"


class TestGlovesValidation:
    def test_zero_gloves_not_validated(self):
        raw_detections = []
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
        )
        assert "gloves" not in detected_names
        assert len(display_boxes) == 0

    def test_single_glove_not_validated_but_displayed(self):
        """1 glove box detected: must display it (<= 2) but NOT validate gloves."""
        raw_detections = [
            (1, 0.85, 10, 100, 50, 150),
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
        )
        assert "gloves" not in detected_names, "Single glove box must NOT be validated"
        assert len(display_boxes) == 1, "Single glove box should still be displayed"

    def test_two_overlapping_gloves_not_validated(self):
        """2 overlapping glove boxes (e.g. duplicate on same hand) must NOT validate."""
        raw_detections = [
            (1, 0.90, 10, 100, 50, 150),
            (1, 0.85, 12, 102, 52, 152),  # heavy overlap with first box
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            glove_iou_threshold=0.1,
        )
        assert "gloves" not in detected_names, "2 overlapping gloves must NOT validate"
        assert len(display_boxes) <= 2, "Must show at most 2 boxes"

    def test_two_non_overlapping_gloves_validated(self):
        """2 distinct non-overlapping glove boxes (left and right hands) must validate."""
        raw_detections = [
            (1, 0.90, 10, 100, 50, 150),   # Hand 1
            (1, 0.85, 150, 100, 190, 150), # Hand 2 (disjoint)
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            glove_iou_threshold=0.1,
        )
        assert "gloves" in detected_names, "2 non-overlapping gloves MUST validate"
        assert len(display_boxes) == 2, "Must display both glove boxes"

    def test_three_gloves_with_duplicate_selects_non_overlapping_pair(self):
        """If right hand has duplicate detections and left hand has one detection,
        it should select the non-overlapping pair (right + left) and validate."""
        raw_detections = [
            (1, 0.92, 10, 100, 50, 150),   # Hand 1 detection A
            (1, 0.88, 12, 102, 52, 152),   # Hand 1 detection B (overlaps A)
            (1, 0.80, 200, 100, 240, 150), # Hand 2 (non-overlapping)
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            glove_iou_threshold=0.1,
        )
        assert "gloves" in detected_names, "Should find non-overlapping pair and validate"
        assert len(display_boxes) == 2, "Must show max 2 boxes"

    def test_three_gloves_all_overlapping_not_validated(self):
        """If 3 detections all overlap on the same spot, gloves cannot be validated."""
        raw_detections = [
            (1, 0.92, 10, 100, 50, 150),
            (1, 0.88, 12, 102, 52, 152),
            (1, 0.80, 15, 105, 55, 155),
        ]
        detected_names, display_boxes = filter_and_validate_ppe(
            raw_detections,
            class_names=cfg.CLASS_NAMES,
            class_conf=cfg.CLASS_CONF,
            default_conf=cfg.DEFAULT_CONF,
            glove_iou_threshold=0.1,
        )
        assert "gloves" not in detected_names
        assert len(display_boxes) <= 2


class TestConfigTunables:
    def test_config_has_max_boxes_tunable(self):
        assert hasattr(cfg, "MAX_BOXES_PER_CLASS"), "unified_config must define MAX_BOXES_PER_CLASS"
        limits = cfg.MAX_BOXES_PER_CLASS
        assert limits[1] == 2 or limits.get('gloves') == 2
        assert limits[0] == 1 or limits.get('coverall') == 1
        assert limits[2] == 1 or limits.get('goggles') == 1
        assert limits[3] == 1 or limits.get('mask') == 1

    def test_config_has_glove_iou_threshold(self):
        assert hasattr(cfg, "GLOVE_OVERLAP_IOU_THRESHOLD")
        assert 0.0 <= cfg.GLOVE_OVERLAP_IOU_THRESHOLD <= 1.0


class TestUnifiedEntryMonitorDetectionIntegration:
    def test_detect_ppe_enforces_limits_and_validation(self, monkeypatch):
        import unified_entry_monitor as uem
        import numpy as np

        class MockBox:
            def __init__(self, cls_id, conf, xyxy):
                self.cls = [cls_id]
                self.conf = [conf]
                self.xyxy = [xyxy]

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class MockModel:
            def __init__(self, boxes):
                self._boxes = boxes
            def predict(self, frame, **kwargs):
                return [MockResult(self._boxes)]

        mock_boxes = [
            MockBox(3, 0.95, [10, 10, 50, 50]),     # mask 1
            MockBox(3, 0.85, [15, 15, 55, 55]),     # mask 2
            MockBox(1, 0.90, [10, 100, 50, 150]),   # glove hand 1
            MockBox(1, 0.85, [150, 100, 190, 150]), # glove hand 2
            MockBox(1, 0.70, [200, 100, 240, 150]), # glove 3
            MockBox(0, 0.92, [0, 0, 100, 200]),     # coverall 1
            MockBox(0, 0.80, [5, 5, 105, 205]),     # coverall 2
        ]
        monkeypatch.setattr(uem, "_get_yolo", lambda: MockModel(mock_boxes))

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detected_names, filtered_boxes = uem._detect_ppe(dummy_frame)

        assert "gloves" in detected_names
        assert "mask" in detected_names
        assert "coverall" in detected_names

        counts = {}
        for cls_id, conf, x1, y1, x2, y2 in filtered_boxes:
            counts[cls_id] = counts.get(cls_id, 0) + 1

        assert counts[1] == 2, "Max 2 glove boxes"
        assert counts[3] == 1, "Max 1 mask box"
        assert counts[0] == 1, "Max 1 coverall box"


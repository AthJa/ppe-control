"""PPE detection filtering and validation logic.

Provides bounding box deduplication, display limits, and glove overlap validation:
- Gloves are validated ONLY IF at least 2 non-overlapping boxes are detected.
- Bounding box display is constrained at all times:
    - Gloves: maximum 2 boxes
    - Mask: maximum 1 box
    - Coverall: maximum 1 box
    - Goggles: maximum 1 box
"""
from typing import Iterable, Sequence

try:
    import unified_config as cfg
except ImportError:
    cfg = None


def compute_iou(box_a: Sequence[int | float], box_b: Sequence[int | float]) -> float:
    """Compute Intersection-over-Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def are_boxes_overlapping(
    box_a: Sequence[int | float],
    box_b: Sequence[int | float],
    threshold: float = 0.1,
) -> bool:
    """Determine whether two boxes overlap beyond the given threshold.

    Checks both standard IoU and intersection-over-minimum-area (to catch
    nested / heavily enclosing boxes).
    """
    iou = compute_iou(box_a, box_b)
    if iou > threshold:
        return True

    # Check intersection over min area (nested / containment check)
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return False

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    min_area = min(area_a, area_b)
    if min_area > 0.0 and (inter_area / min_area) > threshold:
        return True

    return False


def filter_and_validate_ppe(
    raw_detections: Iterable[tuple[int, float, int, int, int, int]],
    class_names: dict | None = None,
    class_conf: dict | None = None,
    default_conf: float | None = None,
    max_boxes_per_class: dict | None = None,
    glove_iou_threshold: float | None = None,
) -> tuple[set[str], list[tuple[int, float, int, int, int, int]]]:
    """Filter raw YOLO PPE detections and apply validation constraints.

    Parameters
    ----------
    raw_detections:
        Sequence of (cls_id, conf, x1, y1, x2, y2).
    class_names:
        Mapping of class ID to class name.
    class_conf:
        Per-class minimum confidence overrides.
    default_conf:
        Default minimum confidence threshold.
    max_boxes_per_class:
        Maximum bounding boxes to display per class (default: gloves=2, others=1).
    glove_iou_threshold:
        Maximum IoU/overlap allowed between two glove boxes to consider them non-overlapping.

    Returns
    -------
    detected_names : set of validated PPE class names
    display_boxes  : list of (cls_id, conf, x1, y1, x2, y2) to render on the frame
    """
    if class_names is None:
        class_names = cfg.CLASS_NAMES if cfg else {0: 'coverall', 1: 'gloves', 2: 'goggles', 3: 'mask'}
    if class_conf is None:
        class_conf = cfg.CLASS_CONF if cfg else {}
    if default_conf is None:
        default_conf = cfg.DEFAULT_CONF if cfg else 0.5
    if max_boxes_per_class is None:
        max_boxes_per_class = cfg.MAX_BOXES_PER_CLASS if cfg else {0: 1, 1: 2, 2: 1, 3: 1}
    if glove_iou_threshold is None:
        glove_iou_threshold = cfg.GLOVE_OVERLAP_IOU_THRESHOLD if cfg else 0.1

    # Group candidate detections by class ID (only those passing confidence)
    by_class: dict[int, list[tuple[int, float, int, int, int, int]]] = {}
    for item in raw_detections:
        cls_id = int(item[0])
        conf = float(item[1])
        min_conf = class_conf.get(cls_id, default_conf)
        if cls_id in class_names and conf >= min_conf:
            by_class.setdefault(cls_id, []).append(item)

    # Sort each class by confidence descending
    for cls_id in by_class:
        by_class[cls_id].sort(key=lambda d: d[1], reverse=True)

    detected_names: set[str] = set()
    display_boxes: list[tuple[int, float, int, int, int, int]] = []

    for cls_id, candidates in by_class.items():
        cls_name = class_names[cls_id]
        limit = max_boxes_per_class.get(cls_id, max_boxes_per_class.get(cls_name, 1))

        if cls_name == "gloves" or cls_id == 1:
            # GLOVE LOGIC:
            # 1. Must validate ONLY IF at least 2 non-overlapping boxes are detected.
            # 2. Show a max of 2 boxes at all times.
            non_overlapping_pair = None
            n = len(candidates)
            for i in range(n):
                for j in range(i + 1, n):
                    box_i = candidates[i][2:6]
                    box_j = candidates[j][2:6]
                    if not are_boxes_overlapping(box_i, box_j, threshold=glove_iou_threshold):
                        non_overlapping_pair = (candidates[i], candidates[j])
                        break
                if non_overlapping_pair:
                    break

            if non_overlapping_pair:
                # Validated: 2 non-overlapping boxes detected
                detected_names.add(cls_name)
                # Keep the non-overlapping pair for display (max 2 boxes)
                display_boxes.extend(non_overlapping_pair)
            else:
                # Not validated (either < 2 boxes or all pairs overlap)
                # Display up to `limit` (max 2) highest-confidence boxes
                display_boxes.extend(candidates[:limit])

        else:
            # OTHER CLASSES (mask, coverall, goggles):
            # If at least 1 box is detected, class is validated.
            # Show a max of `limit` (1) box at all times (highest confidence).
            if candidates:
                detected_names.add(cls_name)
                display_boxes.extend(candidates[:limit])

    return detected_names, display_boxes

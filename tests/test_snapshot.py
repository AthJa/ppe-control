"""Tests for face bounding box snapshotting in unified_entry_monitor.py."""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

import unified_config as cfg
from unified_entry_monitor import _save_snapshot


@pytest.fixture
def temp_snapshot_dir(monkeypatch):
    """Temporary directory for snapshots during testing."""
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(cfg, "ENTRY_SNAPSHOTS_DIR", tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestFaceSnapshot:
    def test_save_snapshot_crops_face_box(self, temp_snapshot_dir):
        """Must save only the bounding box detected for the face, not the whole image."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Face bounding box: x1=100, y1=50, x2=200, y2=170 -> width=100, height=120
        face_box = (100, 50, 200, 170)
        # Fill face area with a distinct color
        frame[50:170, 100:200] = (0, 255, 0)

        path = _save_snapshot(frame, "AUTHORIZED", "101", face_box=face_box)
        assert path is not None
        assert os.path.exists(path)

        saved = cv2.imread(path)
        assert saved is not None
        # Verify dimensions match the face box crop (height, width), NOT full frame (480, 640)
        assert saved.shape == (120, 100, 3), f"Expected (120, 100, 3), got {saved.shape}"

    def test_save_snapshot_with_pre_cropped_face(self, temp_snapshot_dir):
        """If a pre-cropped face image is passed, save that crop."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_crop = np.ones((80, 90, 3), dtype=np.uint8) * 128

        path = _save_snapshot(frame, "AUTHORIZED", "102", face_crop=face_crop)
        assert path is not None
        assert os.path.exists(path)

        saved = cv2.imread(path)
        assert saved.shape == (80, 90, 3)

    def test_save_snapshot_clips_boundary_coordinates(self, temp_snapshot_dir):
        """Bounding box coordinates extending beyond frame borders must be safely clipped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_box = (-20, -10, 100, 80)

        path = _save_snapshot(frame, "AUTHORIZED", "103", face_box=face_box)
        assert path is not None
        assert os.path.exists(path)

        saved = cv2.imread(path)
        # Clipped from (0, 0) to (100, 80) -> height=80, width=100
        assert saved.shape == (80, 100, 3)

    def test_save_snapshot_fallback_to_frame_if_no_face_detected(self, temp_snapshot_dir):
        """If no face box or crop is available, fallback to full frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        path = _save_snapshot(frame, "AUTHORIZED", "104", face_box=None, face_crop=None)
        assert path is not None
        saved = cv2.imread(path)
        assert saved.shape == (480, 640, 3)

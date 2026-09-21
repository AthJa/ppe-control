"""Tests for unified_config.py — written BEFORE the implementation (RED phase).

Run with: .\\venv\\Scripts\\python.exe -m pytest tests/test_unified_config.py -v
"""
import importlib
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    """Import unified_config from project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Reload to avoid stale module state across test runs
    if "unified_config" in sys.modules:
        return importlib.reload(sys.modules["unified_config"])
    return importlib.import_module("unified_config")


# ---------------------------------------------------------------------------
# Face-recognition tunables
# ---------------------------------------------------------------------------

class TestFaceRecognitionTunables:
    def test_match_threshold_present(self):
        cfg = load_config()
        assert hasattr(cfg, "MATCH_THRESHOLD"), "MATCH_THRESHOLD must be present"

    def test_match_threshold_is_float_between_0_and_1(self):
        cfg = load_config()
        assert isinstance(cfg.MATCH_THRESHOLD, float)
        assert 0.0 < cfg.MATCH_THRESHOLD < 1.0

    def test_detector_score_threshold_present(self):
        cfg = load_config()
        assert hasattr(cfg, "DETECTOR_SCORE_THRESHOLD")

    def test_camera_index_is_int(self):
        cfg = load_config()
        assert hasattr(cfg, "CAMERA_INDEX")
        assert isinstance(cfg.CAMERA_INDEX, int)

    def test_log_cooldown_seconds_present(self):
        cfg = load_config()
        assert hasattr(cfg, "LOG_COOLDOWN_SECONDS")
        assert isinstance(cfg.LOG_COOLDOWN_SECONDS, (int, float))

    def test_detection_downscale_present(self):
        cfg = load_config()
        assert hasattr(cfg, "DETECTION_DOWNSCALE")
        assert 0.0 < cfg.DETECTION_DOWNSCALE <= 1.0

    def test_face_db_path_present(self):
        cfg = load_config()
        assert hasattr(cfg, "FACE_DB_PATH")

    def test_yunet_model_path_present(self):
        cfg = load_config()
        assert hasattr(cfg, "YUNET_MODEL_PATH")


# ---------------------------------------------------------------------------
# PPE tunables
# ---------------------------------------------------------------------------

class TestPPETunables:
    def test_mandated_set_present(self):
        cfg = load_config()
        assert hasattr(cfg, "MANDATED_SET")

    def test_mandated_set_is_set_or_frozenset(self):
        cfg = load_config()
        assert isinstance(cfg.MANDATED_SET, (set, frozenset))

    def test_mandated_set_non_empty(self):
        cfg = load_config()
        assert len(cfg.MANDATED_SET) > 0, "MANDATED_SET must contain at least one PPE class"

    def test_history_len_present_and_positive(self):
        cfg = load_config()
        assert hasattr(cfg, "HISTORY_LEN")
        assert isinstance(cfg.HISTORY_LEN, int)
        assert cfg.HISTORY_LEN > 0

    def test_min_detect_count_present_and_positive(self):
        cfg = load_config()
        assert hasattr(cfg, "MIN_DETECT_COUNT")
        assert isinstance(cfg.MIN_DETECT_COUNT, int)
        assert cfg.MIN_DETECT_COUNT > 0

    def test_eval_timeout_present_and_positive(self):
        cfg = load_config()
        assert hasattr(cfg, "EVAL_TIMEOUT")
        assert isinstance(cfg.EVAL_TIMEOUT, (int, float))
        assert cfg.EVAL_TIMEOUT > 0

    def test_class_conf_is_dict(self):
        cfg = load_config()
        assert hasattr(cfg, "CLASS_CONF")
        assert isinstance(cfg.CLASS_CONF, dict)

    def test_class_names_present(self):
        cfg = load_config()
        assert hasattr(cfg, "CLASS_NAMES")
        assert isinstance(cfg.CLASS_NAMES, dict)

    def test_ppe_model_path_present(self):
        cfg = load_config()
        assert hasattr(cfg, "PPE_MODEL_PATH")
        assert cfg.PPE_MODEL_PATH.replace("\\", "/").endswith("model/best.pt")
        assert os.path.exists(cfg.PPE_MODEL_PATH), f"Model file not found at {cfg.PPE_MODEL_PATH}"

    def test_mandated_set_values_in_class_names(self):
        """Every mandated PPE class must have a corresponding CLASS_NAMES entry."""
        cfg = load_config()
        class_name_values = set(cfg.CLASS_NAMES.values())
        for item in cfg.MANDATED_SET:
            assert item in class_name_values, (
                f"'{item}' in MANDATED_SET has no matching entry in CLASS_NAMES"
            )


# ---------------------------------------------------------------------------
# Combined system tunables
# ---------------------------------------------------------------------------

class TestCombinedTunables:
    def test_granted_hold_seconds_present(self):
        cfg = load_config()
        assert hasattr(cfg, "GRANTED_HOLD_SECONDS")
        assert isinstance(cfg.GRANTED_HOLD_SECONDS, (int, float))
        assert cfg.GRANTED_HOLD_SECONDS > 0

    def test_default_conf_present(self):
        cfg = load_config()
        assert hasattr(cfg, "DEFAULT_CONF")
        assert 0.0 < cfg.DEFAULT_CONF < 1.0

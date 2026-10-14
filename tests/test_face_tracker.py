"""Tests for face tracking, bounding box smoothing, and decision hysteresis.

Requirements:
1. Prevent bounding box flickering and dropouts using temporal smoothing & persistence.
2. Prevent flip-flopping between UNKNOWN and AUTHORIZED using hysteresis and similarity smoothing.
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACERECG_APP = os.path.join(PROJECT_ROOT, "facerecg-main", "app")
sys.path.insert(0, FACERECG_APP)
sys.path.insert(0, PROJECT_ROOT)

from face_engine import MatchResult
from face_tracker import FaceTracker



class TestFaceTrackerSmoothing:
    def test_single_face_first_frame(self):
        tracker = FaceTracker(threshold=0.60)
        m = MatchResult(1, "Alice", 0.70, True, "AUTHORIZED")
        results = tracker.update([(100, 100, 200, 200, m)])

        assert len(results) == 1
        x1, y1, x2, y2, res = results[0]
        assert (x1, y1, x2, y2) == (100, 100, 200, 200)
        assert res.decision == "AUTHORIZED"
        assert res.name == "Alice"

    def test_box_coordinate_smoothing(self):
        """Coordinates should be smoothed with EMA rather than jumping abruptly."""
        tracker = FaceTracker(threshold=0.60, smoothing_alpha=0.7)
        m1 = MatchResult(1, "Alice", 0.70, True, "AUTHORIZED")
        tracker.update([(100, 100, 200, 200, m1)])

        # Frame 2: small jitter to (110, 110, 210, 210)
        m2 = MatchResult(1, "Alice", 0.72, True, "AUTHORIZED")
        results = tracker.update([(110, 110, 210, 210, m2)])

        x1, y1, x2, y2, _ = results[0]
        # EMA: 0.7 * 110 + 0.3 * 100 = 77 + 30 = 107
        assert x1 == 107
        assert y1 == 107
        assert x2 == 207
        assert y2 == 207


class TestFaceTrackerPersistence:
    def test_preserves_box_on_temporary_dropout(self):
        """When face detector misses a frame, box must be held to avoid strobe/flicker."""
        tracker = FaceTracker(threshold=0.60, max_lost_frames=3)
        m = MatchResult(1, "Alice", 0.70, True, "AUTHORIZED")
        tracker.update([(100, 100, 200, 200, m)])

        # Frame 2: Detector missed the face (empty list)
        results = tracker.update([])
        assert len(results) == 1, "Must retain previous face box on 1-frame dropout"
        assert results[0][4].name == "Alice"

        # Frame 3: Another missed frame
        results = tracker.update([])
        assert len(results) == 1

        # Frame 4: Another missed frame (lost_frames reaches 3)
        results = tracker.update([])
        assert len(results) == 1

        # Frame 5: Exceeded max_lost_frames -> pruned
        results = tracker.update([])
        assert len(results) == 0, "Should prune track after max_lost_frames"


class TestDecisionHysteresis:
    def test_authorized_stays_authorized_on_minor_dip(self):
        """AUTHORIZED status must not flip to UNKNOWN on a minor similarity dip."""
        tracker = FaceTracker(threshold=0.60, hysteresis_margin=0.06, hysteresis_frames=3)
        m_auth = MatchResult(1, "Alice", 0.65, True, "AUTHORIZED")
        tracker.update([(100, 100, 200, 200, m_auth)])

        # Frame 2: Similarity dips to 0.58 (below 0.60 threshold, but within margin 0.54)
        m_dip = MatchResult(1, "Alice", 0.58, True, "UNKNOWN")
        results = tracker.update([(100, 100, 200, 200, m_dip)])

        assert len(results) == 1
        _, _, _, _, res = results[0]
        assert res.decision == "AUTHORIZED", "Must hold AUTHORIZED across minor similarity dip"
        assert res.name == "Alice"

    def test_authorized_transitions_to_unknown_if_consistently_low(self):
        """If similarity stays consistently low beyond hysteresis_frames, transition to UNKNOWN."""
        tracker = FaceTracker(threshold=0.60, hysteresis_margin=0.06, hysteresis_frames=3)
        m_auth = MatchResult(1, "Alice", 0.65, True, "AUTHORIZED")
        tracker.update([(100, 100, 200, 200, m_auth)])

        # Dips for 3 frames
        m_low = MatchResult(None, None, 0.40, False, "UNKNOWN")
        for _ in range(3):
            tracker.update([(100, 100, 200, 200, m_low)])

        # Frame 4: now it should be UNKNOWN
        results = tracker.update([(100, 100, 200, 200, m_low)])
        assert len(results) == 1
        assert results[0][4].decision == "UNKNOWN"

    def test_stranger_stays_unknown(self):
        """An unknown person with low similarity must never flip to AUTHORIZED."""
        tracker = FaceTracker(threshold=0.60)
        m_stranger = MatchResult(None, None, 0.35, False, "UNKNOWN")
        results = tracker.update([(100, 100, 200, 200, m_stranger)])

        assert len(results) == 1
        assert results[0][4].decision == "UNKNOWN"

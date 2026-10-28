"""Temporal Face Tracking, Coordinate Smoothing, and Decision Hysteresis.

Prevents:
1. Bounding box flicker & dropout: Holds bounding boxes across temporary
   detector misses and applies Exponential Moving Average (EMA) to coordinates.
2. Label flip-flopping: Applies hysteresis to cosine similarity so an authorized
   face does not flap between AUTHORIZED and UNKNOWN on small frame-to-frame dips.
"""
from collections import namedtuple
from typing import Sequence

from ppe_filter import compute_iou

MatchResult = namedtuple("MatchResult", "staff_id name similarity authorized decision")


class _Track:
    def __init__(self, track_id: int, box: Sequence[int], match_result: MatchResult):
        self.track_id = track_id
        self.box = [int(v) for v in box]
        self.match_result = match_result
        self.smoothed_sim = float(match_result.similarity)
        self.lost_frames = 0
        self.consecutive_low_sim = 0
        self.decision = match_result.decision
        self.staff_id = match_result.staff_id
        self.name = match_result.name
        self.authorized = match_result.authorized


class FaceTracker:
    """Maintains face tracks across video frames to eliminate flickering and jitter."""

    def __init__(
        self,
        threshold: float = 0.60,
        smoothing_alpha: float = 0.7,
        max_lost_frames: int = 4,
        hysteresis_margin: float = 0.06,
        hysteresis_frames: int = 4,
        iou_match_threshold: float = 0.3,
    ):
        self.threshold = threshold
        self.smoothing_alpha = smoothing_alpha
        self.max_lost_frames = max_lost_frames
        self.hysteresis_margin = hysteresis_margin
        self.hysteresis_frames = hysteresis_frames
        self.iou_match_threshold = iou_match_threshold

        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def update(
        self,
        raw_detections: list[tuple[int, int, int, int, MatchResult]],
    ) -> list[tuple[int, int, int, int, MatchResult]]:
        """Update tracker with detections from the current frame.

        Parameters
        ----------
        raw_detections:
            List of (x1, y1, x2, y2, MatchResult) from face detector and embedder.

        Returns
        -------
        Stabilized list of (x1, y1, x2, y2, MatchResult).
        """
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        matched_detections = set()

        if self._tracks and raw_detections:
            # Greedy matching by highest IoU
            pairs = []
            for t_idx, track in enumerate(self._tracks):
                for d_idx, det in enumerate(raw_detections):
                    det_box = det[:4]
                    iou = compute_iou(track.box, det_box)
                    if iou >= self.iou_match_threshold:
                        pairs.append((iou, t_idx, d_idx))

            pairs.sort(key=lambda p: p[0], reverse=True)

            for iou, t_idx, d_idx in pairs:
                if t_idx not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)
                    self._update_track(self._tracks[t_idx], raw_detections[d_idx])

        # Handle unmatched tracks (dropout persistence)
        surviving_tracks = []
        for t_idx, track in enumerate(self._tracks):
            if t_idx in matched_tracks:
                surviving_tracks.append(track)
            else:
                track.lost_frames += 1
                if track.lost_frames <= self.max_lost_frames:
                    surviving_tracks.append(track)

        # Handle unmatched detections (new faces)
        for d_idx, det in enumerate(raw_detections):
            if d_idx not in matched_detections:
                box = det[:4]
                m = det[4]
                track = _Track(self._next_track_id, box, m)
                self._next_track_id += 1
                surviving_tracks.append(track)

        self._tracks = surviving_tracks


        # Produce stabilized output
        results = []
        for t in self._tracks:
            res = MatchResult(
                staff_id=t.staff_id,
                name=t.name,
                similarity=float(t.smoothed_sim),
                authorized=bool(t.authorized),
                decision=str(t.decision),
            )
            results.append((t.box[0], t.box[1], t.box[2], t.box[3], res))

        return results

    def _update_track(self, track: _Track, det: tuple[int, int, int, int, MatchResult]):
        det_box = det[:4]
        new_match: MatchResult = det[4]

        track.lost_frames = 0

        # 1. Smooth coordinates with EMA
        a = self.smoothing_alpha
        for k in range(4):
            track.box[k] = int(round(a * det_box[k] + (1.0 - a) * track.box[k]))

        # 2. Smooth similarity
        track.smoothed_sim = 0.6 * new_match.similarity + 0.4 * track.smoothed_sim

        # 3. Decision Hysteresis
        if track.decision == "AUTHORIZED":
            if new_match.decision == "AUTHORIZED":
                track.consecutive_low_sim = 0
                track.staff_id = new_match.staff_id
                track.name = new_match.name
                track.authorized = new_match.authorized
            else:
                # Similarity is below threshold; check hysteresis margin
                if new_match.similarity >= (self.threshold - self.hysteresis_margin):
                    # Minor dip: hold AUTHORIZED
                    pass
                else:
                    track.consecutive_low_sim += 1
                    if track.consecutive_low_sim >= self.hysteresis_frames or new_match.similarity < 0.45:
                        track.decision = "UNKNOWN"
                        track.name = None
                        track.staff_id = None
                        track.authorized = False
        else:
            # Currently UNKNOWN
            if new_match.decision == "AUTHORIZED":
                track.decision = "AUTHORIZED"
                track.staff_id = new_match.staff_id
                track.name = new_match.name
                track.authorized = new_match.authorized
                track.consecutive_low_sim = 0
            else:
                track.decision = "UNKNOWN"
                track.name = None
                track.staff_id = None
                track.authorized = False

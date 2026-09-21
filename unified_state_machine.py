"""Pure state-machine logic for the Unified Face Recognition + PPE Detection system.

Extracted from the camera loop so it can be unit-tested without OpenCV or a webcam.

States
------
IDLE            Waiting for a recognized (AUTHORIZED) face.
EVALUATING_PPE  A person has been identified; checking PPE compliance over time.
GRANTED         All mandated PPE validated within the window; access is granted.

After GRANTED the system holds the state for ``granted_hold_seconds``, then resets
to IDLE.  On timeout in EVALUATING_PPE without full compliance the machine records
the violation and also returns to IDLE.

Usage
-----
    sm = UnifiedStateMachine(mandated_set={'mask'}, ...)

    # called once per frame from the camera loop:
    sm.update(face_decision, face_name, ppe_detections)

    # poll for a pending log row (set after GRANTED or VIOLATION):
    log = sm.consume_log()
    if log:
        db.insert_entry_log(**log)
"""
import enum
import time
from collections import deque


class State(enum.Enum):
    IDLE = "IDLE"
    EVALUATING_PPE = "EVALUATING_PPE"
    GRANTED = "GRANTED"


class UnifiedStateMachine:
    """State machine that combines face identity check with temporal PPE validation.

    Parameters
    ----------
    mandated_set:
        Set of PPE class names that must all be validated (e.g. ``{'mask', 'gloves'}``).
    history_len:
        Rolling window size in frames for temporal PPE validation.
    min_detect_count:
        Minimum number of frames a PPE item must appear in to be "validated".
    eval_timeout:
        Seconds allowed for the subject to show all required PPE before a VIOLATION.
    granted_hold_seconds:
        Seconds the GRANTED state is held before returning to IDLE.
    """

    def __init__(
        self,
        mandated_set: set,
        history_len: int = 100,
        min_detect_count: int = 5,
        eval_timeout: float = 10.0,
        granted_hold_seconds: float = 5.0,
    ):
        self.mandated_set = frozenset(mandated_set)
        self.history_len = history_len
        self.min_detect_count = min_detect_count
        self.eval_timeout = eval_timeout
        self.granted_hold_seconds = granted_hold_seconds

        # --- runtime state ---
        self.state: State = State.IDLE
        self.recognized_person: str | None = None   # name from face matching
        self.recognized_staff_id: int | None = None
        self.recognized_similarity: float = 0.0

        self._history: deque[set] = deque(maxlen=history_len)
        self._eval_start_time: float = 0.0
        self._granted_start_time: float = 0.0

        self.validated_set: set = set()
        self.last_missing_ppe: set = set()

        self.pending_log: dict | None = None        # written on GRANTED or VIOLATION

        # public summary text for HUD rendering
        self.hud_eval_remaining: float = 0.0
        self.hud_missing: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        face_decision: str,         # "AUTHORIZED" | "DENIED" | "UNKNOWN"
        face_name: str | None,
        ppe_detections: set,        # set of PPE class names detected in this frame
        face_staff_id: int | None = None,
        face_similarity: float = 0.0,
        now: float | None = None,
    ) -> None:
        """Advance the state machine by one frame.

        Parameters
        ----------
        face_decision:  Result from face_engine.match().decision
        face_name:      Matched staff name (or None)
        ppe_detections: Set of PPE class names that passed confidence threshold
        face_staff_id:  Staff DB id (or None)
        face_similarity: Cosine similarity score from face matching
        now:            Override for current time (useful in tests)
        """
        t = now if now is not None else time.time()

        if self.state == State.IDLE:
            self._handle_idle(face_decision, face_name, face_staff_id, face_similarity, t)

        elif self.state == State.EVALUATING_PPE:
            self._handle_evaluating(face_decision, face_name, ppe_detections, t)

        elif self.state == State.GRANTED:
            self._handle_granted(t)

    def consume_log(self) -> dict | None:
        """Return and clear the pending log entry (if any)."""
        log = self.pending_log
        self.pending_log = None
        return log

    # ------------------------------------------------------------------
    # Private state handlers
    # ------------------------------------------------------------------

    def _handle_idle(self, face_decision, face_name, face_staff_id, face_similarity, t):
        if face_decision == "AUTHORIZED":
            self.recognized_person = face_name
            self.recognized_staff_id = face_staff_id
            self.recognized_similarity = face_similarity
            self._history.clear()
            self.validated_set = set()
            self._eval_start_time = t
            self.state = State.EVALUATING_PPE

    def _handle_evaluating(self, face_decision, face_name, ppe_detections, t):
        # Accumulate PPE detections into rolling history
        self._history.append(set(ppe_detections))

        # Temporal validation: count occurrences in history window
        counts: dict[str, int] = {}
        for frame_dets in self._history:
            for item in frame_dets:
                counts[item] = counts.get(item, 0) + 1

        self.validated_set = {
            cls for cls, cnt in counts.items()
            if cnt >= self.min_detect_count
        }

        missing = self.mandated_set - self.validated_set
        elapsed = t - self._eval_start_time
        self.hud_eval_remaining = max(0.0, self.eval_timeout - elapsed)
        self.hud_missing = missing

        if not missing:
            # All mandated PPE validated → GRANTED
            self.last_missing_ppe = set()
            self.pending_log = {
                'staff_id': self.recognized_staff_id,
                'name': self.recognized_person,
                'similarity': self.recognized_similarity,
                'decision': 'AUTHORIZED',
                'ppe_status': 'COMPLIANT',
                'ppe_missing': '',
            }
            self._granted_start_time = t
            self.state = State.GRANTED

        elif elapsed > self.eval_timeout:
            # Timeout → VIOLATION
            self.last_missing_ppe = missing
            self.pending_log = {
                'staff_id': self.recognized_staff_id,
                'name': self.recognized_person,
                'similarity': self.recognized_similarity,
                'decision': 'AUTHORIZED',
                'ppe_status': 'NON_COMPLIANT',
                'ppe_missing': ','.join(sorted(missing)),
            }
            self._reset()

    def _handle_granted(self, t):
        if t - self._granted_start_time > self.granted_hold_seconds:
            self._reset()

    def _reset(self):
        self.state = State.IDLE
        self._history.clear()
        self.validated_set = set()
        self.recognized_person = None
        self.recognized_staff_id = None
        self.recognized_similarity = 0.0
        self.hud_eval_remaining = 0.0
        self.hud_missing = set()

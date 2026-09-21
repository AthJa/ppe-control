"""Tests for the extended db.py (PPE status columns) — RED phase.

Run with: .\\venv\\Scripts\\python.exe -m pytest tests/test_db_ppe_extension.py -v
"""
import os
import sqlite3
import sys
import tempfile

import pytest

# Add facerecg-main/app to path for db import
FACERECG_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "facerecg-main", "app"
)
sys.path.insert(0, FACERECG_APP)

import db
import config as face_config


# ---------------------------------------------------------------------------
# Fixtures — isolated in-memory/temp DB per test
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB and schema paths to a temporary directory."""
    db_path = str(tmp_path / "test_face.db")
    schema_path = face_config.SCHEMA_PATH

    monkeypatch.setattr(face_config, "DB_PATH", db_path)
    monkeypatch.setattr(face_config, "DATA_STORE_DIR", str(tmp_path))
    db.init_db()
    yield db_path


# ---------------------------------------------------------------------------
# Tests: new columns exist after init_db()
# ---------------------------------------------------------------------------

class TestSchemaExtension:
    def test_entry_logs_has_ppe_status_column(self, tmp_db):
        """entry_logs table must have a ppe_status column after init."""
        conn = sqlite3.connect(tmp_db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(entry_logs)")}
        conn.close()
        assert "ppe_status" in cols, "entry_logs must have ppe_status column"

    def test_entry_logs_has_ppe_missing_column(self, tmp_db):
        """entry_logs table must have a ppe_missing column after init."""
        conn = sqlite3.connect(tmp_db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(entry_logs)")}
        conn.close()
        assert "ppe_missing" in cols, "entry_logs must have ppe_missing column"


# ---------------------------------------------------------------------------
# Tests: insert_entry_log() backwards-compatible signature
# ---------------------------------------------------------------------------

class TestInsertEntryLogBackwardsCompat:
    def test_old_call_signature_still_works(self, tmp_db):
        """Calling insert_entry_log without ppe_status/ppe_missing must not raise."""
        # Insert a minimal staff row first (foreign key may be NULL per schema)
        db.insert_entry_log(None, "Unknown", 0.0, "UNKNOWN")

    def test_old_call_populates_ppe_status_as_null(self, tmp_db):
        """When called without ppe_status, the column value must be NULL."""
        db.insert_entry_log(None, "OldStyle", 0.5, "AUTHORIZED")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT ppe_status, ppe_missing FROM entry_logs WHERE matched_name='OldStyle'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is None, "ppe_status should be NULL when not provided"
        assert row[1] is None, "ppe_missing should be NULL when not provided"


# ---------------------------------------------------------------------------
# Tests: insert_entry_log() new ppe_status / ppe_missing args
# ---------------------------------------------------------------------------

class TestInsertEntryLogPPEStatus:
    def test_insert_with_ppe_status_compliant(self, tmp_db):
        """ppe_status='COMPLIANT' is stored and retrievable."""
        db.insert_entry_log(None, "Alice", 0.75, "AUTHORIZED", ppe_status="COMPLIANT")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT ppe_status FROM entry_logs WHERE matched_name='Alice'"
        ).fetchone()
        conn.close()
        assert row[0] == "COMPLIANT"

    def test_insert_with_ppe_status_non_compliant(self, tmp_db):
        """ppe_status='NON_COMPLIANT' is stored and retrievable."""
        db.insert_entry_log(None, "Bob", 0.7, "AUTHORIZED", ppe_status="NON_COMPLIANT")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT ppe_status FROM entry_logs WHERE matched_name='Bob'"
        ).fetchone()
        conn.close()
        assert row[0] == "NON_COMPLIANT"

    def test_insert_with_ppe_missing_list(self, tmp_db):
        """ppe_missing is stored as a comma-separated string."""
        db.insert_entry_log(
            None, "Carol", 0.65, "AUTHORIZED",
            ppe_status="NON_COMPLIANT",
            ppe_missing="mask,gloves"
        )
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT ppe_missing FROM entry_logs WHERE matched_name='Carol'"
        ).fetchone()
        conn.close()
        assert row[0] == "mask,gloves"

    def test_insert_with_ppe_status_not_checked(self, tmp_db):
        """ppe_status='NOT_CHECKED' is stored when PPE wasn't evaluated."""
        db.insert_entry_log(None, "Dave", 0.0, "UNKNOWN", ppe_status="NOT_CHECKED")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT ppe_status FROM entry_logs WHERE matched_name='Dave'"
        ).fetchone()
        conn.close()
        assert row[0] == "NOT_CHECKED"

    def test_insert_with_snapshot_and_ppe_status(self, tmp_db):
        """snapshot_path + ppe_status can both be set at the same time."""
        db.insert_entry_log(
            None, "Eve", 0.8, "AUTHORIZED",
            snapshot_path="/tmp/snap.jpg",
            ppe_status="COMPLIANT"
        )
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT snapshot_path, ppe_status FROM entry_logs WHERE matched_name='Eve'"
        ).fetchone()
        conn.close()
        assert row[0] == "/tmp/snap.jpg"
        assert row[1] == "COMPLIANT"

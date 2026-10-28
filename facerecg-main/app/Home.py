"""Streamlit entrypoint. Run with: streamlit run app/Home.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import face_engine

import streamlit as st

st.set_page_config(page_title="Staff Entry Recognition", page_icon="\U0001FA7A", layout="wide")

db.init_db()


@st.cache_resource(show_spinner="Loading face detection + recognition models (one-time)...")
def _warm_up_models():
    face_engine.warm_up()
    return True


_warm_up_models()

st.title("Masked-Face Staff Entry Recognition + PPE Detection")
st.markdown(
    """
Use the sidebar to:
- **Register Staff** - enroll a new staff member via webcam burst capture (mask + hair cap on).
- **Manage Authorization** - authorize/deauthorize or remove staff.
- **Entry Logs** - review entry attempts recorded by the entry monitor (includes PPE status).

Run the **unified** entry camera (face recognition + PPE detection in one loop):
```
python unified_entry_monitor.py
```

Or run the original standalone monitors separately:
```
python facerecg-main/app/entry_monitor.py   # face recognition only
python step4_compliance_arbiter_medical_ppe.py  # PPE detection only
```
"""
)

staff_df = db.list_staff()
logs_df = db.query_entry_logs(limit=1000)

col1, col2, col3 = st.columns(3)
col1.metric("Registered staff", len(staff_df))
col2.metric("Authorized", int(staff_df["authorized"].sum()) if not staff_df.empty else 0)
col3.metric("Deauthorized", int((staff_df["authorized"] == 0).sum()) if not staff_df.empty else 0)

# PPE compliance metrics (only shown if unified monitor has logged entries)
if not logs_df.empty and "ppe_status" in logs_df.columns and logs_df["ppe_status"].notna().any():
    st.divider()
    st.subheader("PPE Compliance (recent 1000 entries)")
    ppe_col1, ppe_col2, ppe_col3 = st.columns(3)
    compliant = int((logs_df["ppe_status"] == "COMPLIANT").sum())
    non_compliant = int((logs_df["ppe_status"] == "NON_COMPLIANT").sum())
    checked = compliant + non_compliant
    rate = f"{100 * compliant / checked:.0f}%" if checked > 0 else "N/A"
    ppe_col1.metric("PPE Compliant", compliant)
    ppe_col2.metric("PPE Non-Compliant", non_compliant)
    ppe_col3.metric("Compliance Rate", rate)


"""
Masked-Face Staff Entry Recognition — PPT
Mirrors the Progress_Report.pdf exactly: same headings, same content, same slide count.
"""

from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

TITLE   = prs.slide_layouts[0]
CONTENT = prs.slide_layouts[1]
BLANK   = prs.slide_layouts[6]


def title_slide(title, subtitle):
    s = prs.slides.add_slide(TITLE)
    s.shapes.title.text = title
    s.placeholders[1].text = subtitle


def bullet_slide(title, bullets, levels=None):
    s = prs.slides.add_slide(CONTENT)
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame
    tf.text = bullets[0]
    if levels and 0 in levels:
        tf.paragraphs[0].level = levels[0]
    for i, b in enumerate(bullets[1:], 1):
        p = tf.add_paragraph()
        p.text = b
        p.level = levels.get(i, 0) if levels else 0


def table_slide(title, headers, rows, col_widths=None, footer=None):
    s = prs.slides.add_slide(BLANK)
    tx = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.7))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    nr = len(rows) + 1
    nc = len(headers)
    shape = s.shapes.add_table(nr, nc, Inches(0.8), Inches(1.2),
                               Inches(11.7), Inches(min(0.45 * nr, 5.5)))
    tbl = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        for pr in c.text_frame.paragraphs:
            pr.font.size = Pt(12); pr.font.bold = True
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.cell(i + 1, j)
            c.text = v
            for pr in c.text_frame.paragraphs:
                pr.font.size = Pt(11)
    if footer:
        ft = s.shapes.add_textbox(Inches(0.8), Inches(1.2 + 0.45 * nr + 0.15),
                                  Inches(11), Inches(0.4))
        fp = ft.text_frame.paragraphs[0]
        fp.text = footer
        fp.font.size = Pt(12)
        fp.font.italic = True
    return s


def table_and_bullets_slide(title, headers, rows, col_widths, bullets):
    """Table followed by bullet points on the same slide."""
    s = prs.slides.add_slide(BLANK)
    tx = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.7))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    nr = len(rows) + 1
    nc = len(headers)
    tbl_height = 0.42 * nr
    shape = s.shapes.add_table(nr, nc, Inches(0.8), Inches(1.2),
                               Inches(11.7), Inches(tbl_height))
    tbl = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        for pr in c.text_frame.paragraphs:
            pr.font.size = Pt(12); pr.font.bold = True
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.cell(i + 1, j)
            c.text = v
            for pr in c.text_frame.paragraphs:
                pr.font.size = Pt(11)

    # Bullets below table
    bullet_top = 1.2 + tbl_height + 0.3
    bx = s.shapes.add_textbox(Inches(0.8), Inches(bullet_top), Inches(11), Inches(7.0 - bullet_top))
    tf = bx.text_frame
    tf.word_wrap = True
    # sub-heading
    tf.paragraphs[0].text = "Interim System Metrics"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    for b in bullets:
        bp = tf.add_paragraph()
        bp.text = b
        bp.font.size = Pt(12)
    return s


# =====================================================================
# PAGE 1 — Title
# =====================================================================
title_slide(
    "Masked-Face Staff Entry Recognition",
    "Progress Report -- Proof of Concept\n\nRecognising individuals from partial or occluded face regions"
)

# =====================================================================
# PAGE 2 — 1. Executive Summary
# =====================================================================
bullet_slide("1. Executive Summary", [
    "Staff entry-control system that recognises individuals from partial or occluded face regions (surgical mask, hair cap).",
    "Current phase delivers a working proof-of-concept covering: face detection, embedding, open-set matching, staff enrollment, a live entry-monitoring camera, and an administrative dashboard for authorization management and audit logging.",
    "Approved scope implemented and executable end-to-end: 70%",
    "Remaining work: threshold calibration against real staff data, handling of concurrent unknown detections, and expanded field testing.",
])

# =====================================================================
# PAGE 3 — 2. Implementation - Modules Delivered
# =====================================================================
table_slide(
    "2. Implementation and Functional Progress — Modules Delivered",
    ["Module", "File", "Function"],
    [
        ["Detection + Embedding + Matching", "face_engine.py",
         "Core recognition logic: RetinaFace detection, FaceNet embedding, cosine-similarity open-set matching"],
        ["Storage layer", "db.py, schema.sql",
         "SQLite persistence for staff, embeddings, and entry logs (WAL mode)"],
        ["Configuration", "config.py",
         "Centralized tunables: thresholds, camera settings, capture parameters"],
        ["Enrollment capture", "capture_utils.py",
         "Standalone webcam burst-capture window (8 photos), subprocess"],
        ["Live entry monitor", "entry_monitor.py",
         "Real-time camera loop: detect, embed, match, draw decision, debounced logging"],
        ["Admin dashboard", "Home.py",
         "Streamlit entrypoint, model warm-up, summary metrics"],
        ["Staff registration UI", "pages/1_Register_Staff.py",
         "Guided burst-capture enrollment flow"],
        ["Authorization management UI", "pages/2_Manage_Authorization.py",
         "List staff, toggle authorization, delete"],
        ["Entry log viewer", "pages/3_Entry_Logs.py",
         "Filterable log table with decision counts and snapshot preview"],
    ],
    col_widths=[2.5, 2.8, 6.4],
    footer="Working modules representing ~50% of the approved project scope, all executable end-to-end.",
)

# =====================================================================
# PAGE 4 — 2.2 System Architecture
# =====================================================================
bullet_slide("2.2 System Architecture", [
    "Webcam burst capture (enrollment) or live camera feed (entry)",
    "Face detection -- RetinaFace, occlusion-tuned confidence threshold",
    "Embedding generation -- FaceNet InceptionResnetV1, 512-d, L2-normalized",
    "Open-set matching -- cosine similarity, per-person maximum",
    "Decision (AUTHORIZED / DENIED / UNKNOWN) logged to SQLite with a snapshot",
    "Reviewed via the Streamlit admin dashboard",
    "",
    "Streamlit admin app and standalone entry-monitor camera loop run concurrently against the same SQLite database, coordinated via Write-Ahead Logging (WAL) mode.",
])

# =====================================================================
# PAGE 5 — 2.3 Remaining Scope
# =====================================================================
bullet_slide("2.3 Remaining Scope", [
    "Calibration of the match threshold against real enrolled-staff data (tool built; awaiting sufficient enrolled staff).",
    "Handling of multiple concurrent unrecognized individuals (currently a single global cooldown bucket).",
    "Extended field testing across varied lighting, angles, and occlusion types.",
])

# =====================================================================
# PAGE 6 — 3. Technical Accuracy and Use of Best Practices
# =====================================================================
table_slide(
    "3. Technical Accuracy and Use of Best Practices",
    ["Decision", "Rationale"],
    [
        ["RetinaFace for detection",
         "Selected specifically for robustness to facial occlusion (mask, cap), rather than a generic detector."],
        ["Detector confidence threshold lowered to 0.6",
         "Occluded faces consistently scored below the typical 0.9 default, causing missed detections."],
        ["FaceNet (InceptionResnetV1, pretrained VGGFace2)",
         "Carried forward from prior model comparison as the highest-performing embedding model."],
        ["Input pipeline: 160x160 resize, (pixel-127.5)/128.0 normalization, L2-normalized 512-d output",
         "Matches facenet-pytorch's expected input specification exactly."],
        ["Open-set cosine-similarity matching (max similarity per person)",
         "Allows new staff to be enrolled without retraining any model, unlike a closed-set classifier."],
        ["Background-thread detection worker",
         "Keeps the live video preview rendering at camera frame rate while the slower, CPU-bound detect+embed pass runs asynchronously."],
        ["Centralized configuration module",
         "All tunables (thresholds, paths, capture parameters) live in one place rather than as scattered magic numbers."],
    ],
    col_widths=[4.5, 7.2],
)

# =====================================================================
# PAGE 7 — 4.1 Interim Results - Baseline
# =====================================================================
s = prs.slides.add_slide(BLANK)
tx = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.7))
p = tx.text_frame.paragraphs[0]
p.text = "4.1 Interim Results -- Baseline (Prior Research)"
p.font.size = Pt(28)
p.font.bold = True

# Intro bullets
bx = s.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11), Inches(1.2))
tf = bx.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "Five model architectures evaluated for partial-face recognition on a custom dataset (423 original images, augmented to 2,202 via rotation, shear, zoom, horizontal flip, and brightness range adjustments)."
tf.paragraphs[0].font.size = Pt(14)
bp = tf.add_paragraph()
bp.text = "FaceNet achieved the strongest result and was carried forward into this implementation."
bp.font.size = Pt(14)

# Metrics table
shape = s.shapes.add_table(6, 2, Inches(0.8), Inches(2.7), Inches(7.0), Inches(3.0))
tbl = shape.table
tbl.columns[0].width = Inches(3.0)
tbl.columns[1].width = Inches(4.0)
data = [
    ["Metric", "Value"],
    ["Original dataset size", "423 images"],
    ["Augmented dataset size", "2,202 images"],
    ["Best-performing model", "FaceNet (InceptionResnetV1)"],
    ["Training accuracy", "97.7%"],
    ["Validation accuracy", "85.5%"],
]
for i, row in enumerate(data):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.text = val
        for pr in c.text_frame.paragraphs:
            pr.font.size = Pt(12)
            pr.font.bold = (i == 0)

# =====================================================================
# PAGE 8 — 4.2-4.3 Current POC Configuration & Interim Metrics
# =====================================================================
table_and_bullets_slide(
    "4.2-4.3 Current POC Configuration & Interim Metrics",
    ["Parameter", "Value", "Status"],
    [
        ["Embedding dimension",           "512",                       "Fixed (FaceNet output)"],
        ["Detector confidence threshold",  "0.6",                      "Tuned for occlusion, empirically set"],
        ["Minimum images to enroll",       "3 successful detections",  "Configured"],
        ["Burst capture",                  "8 photos at 1.2s interval","Configured"],
    ],
    col_widths=[3.5, 3.5, 4.7],
    bullets=[
        "Entry-logs dashboard produces live, interpretable metrics for every recognition attempt: AUTHORIZED / DENIED / UNKNOWN counts, filterable by date and staff name, each with an associated snapshot image for audit and verification.",
        "Dedicated calibration tool computes genuine-pair vs. impostor-pair cosine-similarity statistics (min, mean, max, std) from enrolled staff embeddings, producing a data-driven threshold recommendation.",
        "Scheduled as the next data-collection milestone once a representative set of staff has been enrolled.",
    ],
)

# =====================================================================
# PAGE 9 — 5. Problem-Solving and Technical Refinement
# =====================================================================
bullet_slide("5. Problem-Solving and Technical Refinement", [
    "Occluded faces missed by the detector",
    "Problem: Default RetinaFace confidence threshold (0.9) rejected masked/capped faces, which scored lower confidence than unoccluded faces.",
    "Fix: Lowered detector confidence threshold to 0.6, restoring detection recall.",
    "",
    "Streamlit's built-in camera input unsuitable for burst enrollment",
    "Problem: st.camera_input captures only a single shot per rerun, too limited for an eight-photo enrollment burst with live feedback.",
    "Fix: Built standalone OpenCV subprocess (capture_utils.py) with live preview and countdown.",
    "",
    "Live video preview freezing during recognition",
    "Problem: Synchronous detect+embed on every frame would block the video feed (CPU-bound, slow).",
    "Fix: Moved detection/embedding to a background thread; main loop always renders latest completed result.",
], levels={0: 0, 1: 1, 2: 1, 4: 0, 5: 1, 6: 1, 8: 0, 9: 1, 10: 1})

# =====================================================================
# PAGE 10 — Open Issue
# =====================================================================
bullet_slide("Open Issue -- Identified, Not Yet Resolved", [
    "A single global cooldown bucket for unrecognized individuals means two different unknown people appearing in quick succession may only be logged once.",
    "",
    "Documented as a known limitation -- planned for the next iteration.",
])

# =====================================================================
# PAGE 11 — Thank You
# =====================================================================
title_slide("Thank You", "Questions?")


# ── Save ─────────────────────────────────────────────────────────────
out = r"D:\DOWN LOADS\PPE\Masked_Face_Recognition_v3.pptx"
prs.save(out)
print(f"Saved -> {out}")

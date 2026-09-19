"""
Medical PPE Detection Engine — PPT
Mirrors the Progress_Report structure: same headings, same slide count pattern.
Includes result image slides for Section 4.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
import os

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

TITLE   = prs.slide_layouts[0]
CONTENT = prs.slide_layouts[1]
BLANK   = prs.slide_layouts[6]

# ── Path to training-run images ──────────────────────────────────────
RUNS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "runs", "detect", "runs", "detect")


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

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


def table_and_bullets_slide(title, headers, rows, col_widths, bullets, sub_heading="Interim System Metrics"):
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

    bullet_top = 1.2 + tbl_height + 0.3
    bx = s.shapes.add_textbox(Inches(0.8), Inches(bullet_top), Inches(11), Inches(7.0 - bullet_top))
    tf = bx.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = sub_heading
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    for b in bullets:
        bp = tf.add_paragraph()
        bp.text = b
        bp.font.size = Pt(12)
    return s


def image_slide(title, img_path):
    """Add a slide with a title and a single image."""
    s = prs.slides.add_slide(BLANK)
    txBox = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.7))
    p = txBox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    if os.path.exists(img_path):
        s.shapes.add_picture(img_path, Inches(1.0), Inches(1.2), Inches(11.3), Inches(5.8))
    else:
        t2 = s.shapes.add_textbox(Inches(2), Inches(3), Inches(9), Inches(1))
        t2.text_frame.paragraphs[0].text = f"[Image not found: {img_path}]"
    return s


def two_image_slide(title, img1, img2, cap1="", cap2=""):
    """Add a slide with two side-by-side images."""
    s = prs.slides.add_slide(BLANK)
    txBox = s.shapes.add_textbox(Inches(0.8), Inches(0.2), Inches(11), Inches(0.6))
    p = txBox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    img_top, img_h, img_w = Inches(1.0), Inches(5.2), Inches(5.8)
    if os.path.exists(img1):
        s.shapes.add_picture(img1, Inches(0.5), img_top, img_w, img_h)
    if os.path.exists(img2):
        s.shapes.add_picture(img2, Inches(6.8), img_top, img_w, img_h)
    if cap1:
        c1 = s.shapes.add_textbox(Inches(0.5), Inches(6.3), img_w, Inches(0.4))
        c1.text_frame.paragraphs[0].text = cap1
        c1.text_frame.paragraphs[0].font.size = Pt(12)
        c1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    if cap2:
        c2 = s.shapes.add_textbox(Inches(6.8), Inches(6.3), img_w, Inches(0.4))
        c2.text_frame.paragraphs[0].text = cap2
        c2.text_frame.paragraphs[0].font.size = Pt(12)
        c2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    return s


# =====================================================================
# PAGE 1 — Title
# =====================================================================
title_slide(
    "Medical PPE Detection Engine",
    "Progress Report -- Proof of Concept\n\nReal-time compliance verification using YOLOv11"
)

# =====================================================================
# PAGE 2 — 1. Executive Summary
# =====================================================================
bullet_slide("1. Executive Summary", [
    "Automated PPE compliance system that detects medical protective equipment (mask, gloves, goggles, coverall) in real time using a webcam and YOLOv11.",
    "Current phase delivers a working proof-of-concept covering: dataset acquisition and merging from multiple sources, data augmentation (blur, greyscale, noise), YOLO model training across three dataset configurations, and a real-time Compliance Arbiter with temporal validation.",
    "Approved scope implemented and executable end-to-end: 50%",
    "Remaining work: pilot deployment at a real entry-point, per-person PPE tracking, and multi-camera support.",
])

# =====================================================================
# PAGE 3 — 2. Implementation - Modules Delivered
# =====================================================================
table_slide(
    "2. Implementation and Functional Progress — Modules Delivered",
    ["Module", "File", "Function"],
    [
        ["Dataset acquisition", "step1_download.py",
         "Automated download from Kaggle API and Roboflow API with authentication"],
        ["Dataset merging", "step2_merge.py",
         "Unifies heterogeneous datasets: VOC→YOLO conversion, class name mapping (15+ variants → 3 classes), filename deduplication"],
        ["Training (merged)", "step3_train.py",
         "YOLOv11n on merged 3-class dataset (mask, glove, cap), 50 epochs"],
        ["Training (medical)", "step3_train_medical_ppe.py",
         "YOLOv11n on dedicated 4-class dataset (Coverall, Gloves, Goggles, Mask), 100 epochs"],
        ["Training (augmented) ★", "step3_train_roboflow.py",
         "YOLOv11n on augmented 4-class dataset, 50 epochs — best performer (mAP@50 = 97.5%)"],
        ["Compliance arbiter", "step4_compliance_arbiter.py",
         "Real-time webcam inference with temporal validation state machine (IDLE → EVALUATING → GRANTED)"],
        ["Medical PPE arbiter", "step4_compliance_arbiter_medical_ppe.py",
         "4-class arbiter for medical PPE with per-class confidence tuning"],
        ["Colab training", "train_colab.ipynb",
         "Google Colab GPU training notebook with Drive mount for cloud training"],
    ],
    col_widths=[2.3, 3.0, 6.4],
    footer="Working modules representing ~50% of the approved project scope, all executable end-to-end.",
)

# =====================================================================
# PAGE 4 — 2.2 System Architecture
# =====================================================================
bullet_slide("2.2 System Architecture", [
    "Webcam captures live video feed at entry point",
    "YOLOv11 inference — detects PPE items (mask, gloves, goggles, coverall) with bounding boxes and confidence scores",
    "Detection parsing — extracts detected class names from YOLO output per frame",
    "Temporal history buffer — rolling deque of last 15–100 frames' detections",
    "Temporal validation — a class is 'validated' only when detected in ≥ 5 frames within the buffer",
    "State machine — three states: IDLE → EVALUATING (10s window) → GRANTED",
    "",
    "On compliance success: door-open signal held for 5 seconds. On timeout: set-difference identifies exactly which PPE items are missing and sends targeted dispense signals (dispense_mask, dispense_glove, etc.).",
])

# =====================================================================
# PAGE 5 — 2.3 Remaining Scope
# =====================================================================
bullet_slide("2.3 Remaining Scope", [
    "Pilot deployment at a real hospital/lab entry-point with live webcam.",
    "Per-person PPE tracking — currently validates PPE presence in frame, not on a specific individual.",
    "Multi-camera support — multiple entry points writing to a central compliance log.",
    "Edge deployment on NVIDIA Jetson Nano with TensorRT quantisation for low-cost hardware.",
])

# =====================================================================
# PAGE 6 — 3. Technical Accuracy and Use of Best Practices
# =====================================================================
table_slide(
    "3. Technical Accuracy and Use of Best Practices",
    ["Decision", "Rationale"],
    [
        ["YOLOv11 Nano (yolo11n.pt) for detection",
         "2.6M parameters, fits in 4 GB VRAM, real-time inference (~30+ FPS at 416×416), COCO-pretrained for strong transfer learning."],
        ["Image size 416 instead of 640",
         "~2.5× faster training per epoch; sufficient resolution for PPE items (mask, gloves are large objects); negligible accuracy trade-off."],
        ["Data augmentation: blur, greyscale, noise, brightness, flip",
         "Forces the model to learn shape and texture features rather than relying on colour or clean conditions. Proved decisive: +29% mAP improvement."],
        ["Dataset merging with class harmonisation",
         "15+ label variations (face_mask, with_mask, medical_mask, etc.) mapped to 3 canonical classes via a unified TARGET_MAPPING dictionary."],
        ["VOC XML → YOLO format conversion",
         "SH17 and Face Mask datasets used Pascal VOC annotations; built convert_voc_to_yolo() with bounding box normalisation and clamping to [0, 1]."],
        ["Temporal validation (rolling history buffer)",
         "Prevents single-frame false positives from triggering compliance decisions; smooths noisy detections across 15–100 frames."],
        ["Three-state finite state machine",
         "Clean separation of IDLE / EVALUATING / GRANTED states with deterministic transitions; set-difference logic for missing-item identification."],
    ],
    col_widths=[4.5, 7.2],
)

# =====================================================================
# PAGE 7 — 4.1 Interim Results: Baseline (Datasets + Training Runs)
# =====================================================================
s = prs.slides.add_slide(BLANK)
tx = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.7))
p = tx.text_frame.paragraphs[0]
p.text = "4.1 Interim Results -- Datasets and Training Runs"
p.font.size = Pt(28)
p.font.bold = True

# Intro bullets
bx = s.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11), Inches(1.4))
tf = bx.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "Multiple datasets collected from Kaggle and Roboflow Universe. Three training runs compared, all using YOLOv11 Nano pretrained on COCO. Data augmentation (blur, greyscale, noise, brightness, flip) applied via Roboflow before export."
tf.paragraphs[0].font.size = Pt(14)
bp = tf.add_paragraph()
bp.text = "The Roboflow Augmented run achieved the best results and was selected as the production model."
bp.font.size = Pt(14)

# Comparison table
shape = s.shapes.add_table(5, 5, Inches(0.8), Inches(2.8), Inches(11.5), Inches(2.5))
tbl = shape.table
tbl.columns[0].width = Inches(2.5)
tbl.columns[1].width = Inches(1.5)
tbl.columns[2].width = Inches(2.5)
tbl.columns[3].width = Inches(2.5)
tbl.columns[4].width = Inches(2.5)
data = [
    ["Metric", "Merged PPE", "Medical PPE", "Roboflow Aug. ★"],
    ["Dataset size (train)", "9,630", "3,352", "8,304"],
    ["Classes", "3 (mask, glove, cap)", "4 (Coverall, Gloves, Goggles, Mask)", "4 (Coverall, Gloves, Goggles, Mask)"],
    ["Best mAP@50", "—", "68.5%", "97.5%"],
    ["Best mAP@50-95", "—", "40.0%", "97.3%"],
]
for i, row in enumerate(data):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.text = val
        for pr in c.text_frame.paragraphs:
            pr.font.size = Pt(12)
            pr.font.bold = (i == 0)

# Key insight
kx = s.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11), Inches(0.8))
kf = kx.text_frame
kf.word_wrap = True
kf.paragraphs[0].text = "Key Insight: Same source images, same model, same training setup — augmentation alone boosted mAP@50 by +29%. The un-augmented model overfit and generalised poorly."
kf.paragraphs[0].font.size = Pt(14)
kf.paragraphs[0].font.bold = True

# =====================================================================
# PAGE 8 — 4.1 Results: Training Curves (★ Best Run)
# =====================================================================
image_slide(
    "4.1 Results — Training Curves (★ Roboflow Augmented)",
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "results.png"),
)

# =====================================================================
# PAGE 9 — 4.1 Results: Confusion Matrix (★ Best Run)
# =====================================================================
two_image_slide(
    "4.1 Results — Confusion Matrix (★ Roboflow Augmented)",
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "confusion_matrix.png"),
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "confusion_matrix_normalized.png"),
    "Raw Counts",
    "Normalised",
)

# =====================================================================
# PAGE 10 — 4.1 Results: F1 & PR Curves (★ Best Run)
# =====================================================================
two_image_slide(
    "4.1 Results — F1 and Precision-Recall Curves (★ Roboflow Augmented)",
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "BoxF1_curve.png"),
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "BoxPR_curve.png"),
    "F1-Confidence Curve",
    "Precision-Recall Curve",
)

# =====================================================================
# PAGE 11 — 4.1 Results: Validation Predictions (★ Best Run)
# =====================================================================
two_image_slide(
    "4.1 Results — Validation Predictions vs Ground Truth",
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "val_batch0_labels.jpg"),
    os.path.join(RUNS_BASE, "ppe_engine_roboflow", "val_batch0_pred.jpg"),
    "Ground Truth Labels",
    "Model Predictions",
)

# =====================================================================
# PAGE 12 — 4.2-4.3 Current POC Configuration & Interim Metrics
# =====================================================================
table_and_bullets_slide(
    "4.2-4.3 Current POC Configuration & Interim Metrics",
    ["Parameter", "Value", "Status"],
    [
        ["Base model",                 "YOLOv11 Nano (yolo11n.pt)",    "COCO-pretrained, 2.6M params"],
        ["Image size",                 "416×416",                       "Reduced from 640 for ~2.5× speed"],
        ["Batch size",                 "80",                            "Maximises RTX 3050 GPU utilisation"],
        ["Inference confidence",       "0.5",                           "Default threshold for live arbiter"],
        ["Temporal history buffer",    "15–100 frames",                 "Configurable per arbiter variant"],
        ["Min. detection count",       "5 frames",                      "Class must appear in ≥5 frames to validate"],
        ["Evaluation timeout",         "10 seconds",                    "Compliance window before dispensing"],
    ],
    col_widths=[3.5, 3.0, 5.2],
    bullets=[
        "★ Best model: mAP@50 = 97.5%, mAP@50-95 = 97.3%, Precision = 94.9%, Recall = 92.8%.",
        "Final box loss 0.0198, class loss 0.0678 — near-perfect localisation and classification.",
        "Converged in 50 epochs (~41 minutes on RTX 3050); no early stopping triggered.",
        "Augmentation (blur, greyscale, noise) was the decisive factor — +29% mAP improvement over un-augmented training.",
    ],
)

# =====================================================================
# PAGE 13 — 5. Problem-Solving and Technical Refinement
# =====================================================================
bullet_slide("5. Problem-Solving and Technical Refinement", [
    "Heterogeneous annotation formats across datasets",
    "Problem: Kaggle datasets used Pascal VOC XML; Roboflow datasets used YOLO format — cannot train on mixed formats.",
    "Fix: Built convert_voc_to_yolo() in step2_merge.py with bounding box normalisation and coordinate clamping to [0, 1].",
    "",
    "Class label fragmentation (15+ variations for 3 concepts)",
    "Problem: Different datasets used different names for the same PPE item (e.g., 'face_mask', 'with_mask', 'medical_mask' all mean mask).",
    "Fix: Created a unified TARGET_MAPPING dictionary that maps all known label variations to 3 canonical class IDs.",
    "",
    "Model overfitting without augmentation",
    "Problem: Medical PPE model trained without augmentation reached only 68.5% mAP@50 — overfit to training distribution.",
    "Fix: Applied blur, greyscale, salt-and-pepper noise, brightness/exposure variation, and horizontal flip via Roboflow, boosting mAP@50 to 97.5%.",
    "",
    "Windows pagefile errors during training",
    "Problem: workers=8 in the data loader caused Error 1455 (insufficient pagefile) on Windows.",
    "Fix: Reduced workers to 2 with no impact on GPU utilisation.",
], levels={0: 0, 1: 1, 2: 1, 4: 0, 5: 1, 6: 1, 8: 0, 9: 1, 10: 1, 12: 0, 13: 1, 14: 1})

# =====================================================================
# PAGE 14 — Open Issue
# =====================================================================
bullet_slide("Open Issue -- Identified, Not Yet Resolved", [
    "No per-person PPE tracking: the system validates PPE presence in the frame, not on a specific individual. A compliant and non-compliant person side-by-side could grant access incorrectly.",
    "",
    "Planned fix: integrate YOLOv11 person detection to pair each detected PPE item with the nearest person bounding box, enabling per-individual compliance checking.",
    "",
    "Documented as a known limitation — planned for the next iteration.",
])

# =====================================================================
# PAGE 15 — Thank You
# =====================================================================
title_slide("Thank You", "Questions?")


# ── Save ─────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PPE_Detection_Presentation.pptx")
prs.save(out)
print(f"Saved -> {out}")

# -*- coding: utf-8 -*-
"""
Project configuration: ROI names, metrics, feature columns, paths, colors.
"""

import os

# ===================== Paths =====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ===================== Global Seed =====================
SEED = 2026

# ===================== 14 ROIs =====================
ROIS = [
    "Pec_Major", "Pec_Minor", "Triceps", "Deltoid",
    "Supraspinatus", "Infraspinatus", "Subscapularis", "Teres_Major",
    "Trapezius", "Latissimus",
    "Humerus", "Scapula", "Vertebrae_T1", "Vertebrae_C7",
]

# 5 metrics per ROI (excluding HU_Raw)
METRICS = ["Vol", "MaxCSA", "HU_Pure", "Entropy", "Contrast"]

# 70-dimensional feature vector
FEATURE_COLS = [f"{roi}_{metric}" for roi in ROIS for metric in METRICS]

# ===================== ROI Groups (for ablation) =====================
ROI_GROUPS = {
    "Rotator_Cuff": ["Supraspinatus", "Infraspinatus", "Subscapularis", "Teres_Major"],
    "Non_RC_Muscles": ["Trapezius", "Latissimus", "Triceps", "Pec_Major", "Pec_Minor", "Deltoid"],
    "Bone": ["Humerus", "Scapula", "Vertebrae_T1", "Vertebrae_C7"],
}


def get_feature_cols_by_group(group_name: str) -> list:
    """Return feature column names for a given ROI group."""
    rois = ROI_GROUPS[group_name]
    return [f"{roi}_{metric}" for roi in rois for metric in METRICS]


# ===================== Model Colors (for plots) =====================
COLORS = {
    "Radiomics-ML": {"main": "#E69F00", "light": "#FFE4B5"},
    "Deep-Learning": {"main": "#0072B2", "light": "#C7DFF0"},
    "Hybrid-Fusion": {"main": "#CC79A7", "light": "#F4D4E5"},
}

# ===================== CT HU Clipping =====================
HU_CLIP_LOW = -200
HU_CLIP_HIGH = 800
HU_WINDOW_CENTER = 300
HU_WINDOW_WIDTH = 1000

# ===================== Pyradiomics Settings =====================
RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "geometryTolerance": 10000,
    "correctMask": True,
    "verbose": False,
}

RADIOMICS_FEATURES_ENABLED = {
    "firstorder": ["Entropy"],
    "glcm": ["Contrast"],
}

# ===================== TotalSegmentator Tasks =====================
TOTALSEG_TASKS = ["total", "thigh_shoulder_muscles", "abdominal_muscles"]

# ROI keyword mapping for TotalSegmentator label lookup
ROI_KEYWORDS = {
    "Pec_Major": "pectoralis_major",
    "Pec_Minor": "pectoralis_minor",
    "Triceps": "triceps_brachii",
    "Deltoid": "deltoid",
    "Supraspinatus": "supraspinatus",
    "Infraspinatus": "infraspinatus",
    "Subscapularis": "subscapularis",
    "Teres_Major": "teres_major",
    "Trapezius": "trapezius",
    "Latissimus": "latissimus_dorsi",
    "Humerus": "humerus",
    "Scapula": "scapula",
    "Vertebrae_T1": "vertebrae_T1",
    "Vertebrae_C7": "vertebrae_C7",
}

# Bone ROIs (for HU clipping)
BONE_ROIS = ["Humerus", "Scapula", "Vertebrae_T1", "Vertebrae_C7"]

"""
MalnutriSense project configuration.
Single source of truth for file paths, NFHS column names, and model constants.
 
Usage in any module:
    from src.config import NFHS5_PATH, NFHS_COLS, MISSING_CODES, RANDOM_STATE
"""
import os
from pathlib import Path
from dotenv import load_dotenv
 
# Load .env file (Codespace local config — not committed to git)
load_dotenv()
 
# ── Project root ──────────────────────────────────────────────────────────
# Resolves to /workspaces/malnutrisense regardless of where code runs from.
ROOT = Path(__file__).parent.parent
 
# ── Data directories ──────────────────────────────────────────────────────
DATA_DIR       = ROOT / 'data'
RAW_DIR        = DATA_DIR / 'raw'
PROCESSED_DIR  = DATA_DIR / 'processed'
INTERIM_DIR    = DATA_DIR / 'interim'
TRAIN_TEST_DIR = PROCESSED_DIR / 'train_test_splits'
 
# ── Raw data file paths ────────────────────────────────────────────────────
NFHS5_PATH        = RAW_DIR / 'nfhs5' / 'IAKR7EFL.DTA'
NFHS4_PATH        = RAW_DIR / 'nfhs4' / 'IAKR74FL.DTA'
SHAPEFILE_PATH    = RAW_DIR / 'shapefiles' / 'gadm41_IND_2.shp'
ASPIRATIONAL_PATH = RAW_DIR / 'external' / 'aspirational_districts.csv'
 
# ── Processed data paths ──────────────────────────────────────────────────
NFHS5_CLEANED_PATH  = PROCESSED_DIR / 'nfhs5_cleaned.csv'
NFHS5_FEATURES_PATH = PROCESSED_DIR / 'nfhs5_features.csv'
DISTRICTS_GEOJSON   = PROCESSED_DIR / 'india_districts.geojson'
ASPIRATIONAL_CLEAN  = PROCESSED_DIR / 'aspirational_districts_clean.csv'
 
# ── Model and report directories ──────────────────────────────────────────
MODELS_DIR  = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
TABLES_DIR  = REPORTS_DIR / 'tables'
 
# ── NFHS column selection ─────────────────────────────────────────────────
# Load ONLY these columns from the 1,300+ column NFHS DTA file.
# Using usecols saves ~70% memory and ~80% load time.
NFHS_COLS = [
    'HW1',   # Child age in months (0-59)
    'HW2',   # Child weight in kg x10
    'HW3',   # Child height in cm x10
    'HW70',  # Height-for-age Z-score x100 (divide by 100 to get HAZ)
    'HW71',  # Weight-for-age Z-score x100 (divide by 100 to get WAZ)
    'HW72',  # Weight-for-height Z-score x100 (divide by 100 to get WHZ)
    'V106',  # Mother education: 0=None 1=Primary 2=Secondary 3=Higher
    'V025',  # Residence type: 1=Urban 2=Rural
    'V024',  # State code (numeric)
    'HV001', # Cluster number (for district mapping)
    'HV270', # Wealth quintile: 1=Poorest to 5=Richest
    'H11',   # Diarrhoea in last 2 weeks (0=No 1=Yes)
    'M4',    # Duration of breastfeeding in months
    'M19',   # Birth weight in grams (9996=missing)
    'HV201', # Drinking water source (coded)
    'HV205', # Toilet facility type (coded)
    'V130',  # Religion (coded)
    'B4',    # Sex of child: 1=Male 2=Female
]
 
# ── DHS missing value codes ────────────────────────────────────────────────
# DHS encodes 'not applicable', 'missing', and 'unknown' as these integers.
# Replace all of them with float('nan') before any analysis.
MISSING_CODES = [9996, 9997, 9998, 9999, 99996, 99997, 99998, 99999]
 
# ── WHO malnutrition classification thresholds ────────────────────────────
STUNTING_THRESHOLD    = -2.0  # HAZ < -2.0 -> stunted
UNDERWEIGHT_THRESHOLD = -2.0  # WAZ < -2.0 -> underweight
WASTING_THRESHOLD     = -2.0  # WHZ < -2.0 -> wasted
 
# ── Model constants ────────────────────────────────────────────────────────
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
TEST_SIZE    = float(os.getenv('TEST_SIZE', 0.20))
CV_FOLDS     = int(os.getenv('CV_FOLDS', 5))
 

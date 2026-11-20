"""
PredAI Global Configuration
Autonomous Turbofan Predictive Maintenance & Fleet Intelligence
Emperor Syndicate
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
PRIMARY_DATA_DIR = ROOT_DIR / "02_NASA_CMAPSS_Turbofan"
SECONDARY_DATA_DIR = ROOT_DIR / "data"
DATA_DIR = PRIMARY_DATA_DIR if PRIMARY_DATA_DIR.exists() else SECONDARY_DATA_DIR
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"
ASSETS_DIR = ROOT_DIR / "assets"

# ── C-MAPSS dataset settings ─────────────────────────────────────────────────
# Primary dataset set to 02_NASA_CMAPSS_Turbofan (FD001)
DATASETS = [
    "02_NASA_CMAPSS_Turbofan (FD001 - Primary)",
    "FD002 (Multi-Regime Six Conditions)",
    "FD003 (Two Fault Modes)",
    "FD004 (Multi-Regime + Two Fault Modes)",
]
RAW_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

RAW_COLUMNS = [
    "unit_id", "cycle",
    "setting1", "setting2", "setting3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21",
]

# Sensors with zero (or near-zero) variance — pruned from all analysis
PRUNED_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SETTING_COLS = ["setting1", "setting2", "setting3"]

ACTIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15",
    "s17", "s20", "s21",
]

# ── Regime clustering ────────────────────────────────────────────────────────
N_REGIMES = 6
REGIME_RANDOM_STATE = 42

# ── Rolling-feature windows ─────────────────────────────────────────────────
ROLLING_WINDOWS = [5, 15, 30]

# ── RUL target ───────────────────────────────────────────────────────────────
RUL_CAP = 125

# ── Autoencoder ─────────────────────────────────────────────────────────────
AE_PRISTINE_CYCLES = 30          # Only cycles ≤ this used for AE training
AE_FTTP_CONSEC = 3               # Consecutive cycles above threshold → FTTP
AE_FTTP_SIGMA = 3.0              # Threshold = µ + k*σ

# ── Quantile model ──────────────────────────────────────────────────────────
QUANTILES = [0.10, 0.50, 0.90]

LGBM_BASE_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Triage FSM thresholds ───────────────────────────────────────────────────
TRIAGE_P0_RUL_THRESHOLD = 10
TRIAGE_P0_HI_THRESHOLD = 0.20
TRIAGE_P1_RUL_THRESHOLD = 25
TRIAGE_P2_RUL_THRESHOLD = 45
TRIAGE_HITL_INTERVAL_THRESHOLD = 35   # (p90 - p10) > this → HITL

# ── Drift monitoring ─────────────────────────────────────────────────────────
PSI_ALERT_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05
PSI_BINS = 10

# ── Champion-Challenger gating ───────────────────────────────────────────────
CC_IMPROVEMENT_PCT = 3.0  # Challenger must beat champion NASA score by this %

# ── MLflow experiment name ───────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "aura_fleet_rul"

# PredAI: Autonomous Predictive Maintenance Platform for Turbofan Engines

> **Emperor Syndicate · Turbofan PHM System**
> Zero-LLM Deterministic Triage · NASA C-MAPSS (`02_NASA_CMAPSS_Turbofan`) · Quantile Regression · ISO 13381-1

---

## 🏗️ Architecture

PredAI implements a **4-tier architecture** for predictive maintenance of turbofan engines:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Layer 1: Data Engineering                       │
│  Ingestion → Sensor Pruning → K-Means Regime Clustering (k=6)     │
│  → Within-Regime Z-Score Normalization → Rolling Features          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      Layer 2: ML Engine                             │
│  Autoencoder Health Index (D→32→16→8→16→32→D)                     │
│  → FTTP Detection (µ+3σ, 3 consecutive)                           │
│  → LightGBM Quantile Regression (p10, p50, p90)                   │
│  → NASA Asymmetric Scoring                                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   Layer 3: Triage Agent                             │
│  Deterministic FSM: P0 → P1 → P2 → P3 (+ HITL Exception)         │
│  → ERP-Ready JSON Work Orders with BOM & Downtime                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      Layer 4: MLOps                                 │
│  PSI + KS-Test Drift Detection → Champion-Challenger Gating        │
│  → Auto-Retraining Pipeline                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
PredAI/
├── config.py                       # Global configuration & thresholds
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
│
├── data/                           # NASA C-MAPSS datasets (FD001-FD004)
│   ├── download_cmapss.py          # Download script with synthetic fallback
│   ├── train_FD001.txt ... FD004   # Training data
│   ├── test_FD001.txt  ... FD004   # Test data
│   └── RUL_FD001.txt   ... FD004   # Ground-truth RUL labels
│
├── src/                            # Core source code
│   ├── data_engineering/           # Layer 1
│   │   ├── ingestion.py            # C-MAPSS file loader
│   │   ├── pruner.py               # Zero-variance sensor removal
│   │   ├── regime.py               # K-Means regime clustering (k=6)
│   │   ├── normalizer.py           # Within-regime Z-score normalization
│   │   └── features.py             # Rolling OLS slope, std, EWMA
│   │
│   ├── ml_engine/                  # Layer 2
│   │   ├── rul_target.py           # Piecewise RUL (cap=125)
│   │   ├── autoencoder.py          # Health Index (PyTorch/NumPy fallback)
│   │   ├── quantile_model.py       # LightGBM/NumPy quantile regression
│   │   └── nasa_loss.py            # NASA asymmetric score + pinball loss
│   │
│   ├── triage_agent/               # Layer 3
│   │   ├── fsm.py                  # Deterministic FSM (P0-P3 + HITL)
│   │   ├── schemas.py              # Work order data schemas
│   │   └── work_order.py           # ERP JSON work order generator
│   │
│   └── mlops/                      # Layer 4
│       ├── drift_monitor.py        # PSI + KS-test drift detection
│       ├── model_registry.py       # Champion-challenger gating (>3%)
│       └── trainer.py              # Auto-retrainer
│
├── pipelines/                      # Orchestration
│   ├── train_pipeline.py           # Full training pipeline
│   └── inference_pipeline.py       # Real-time inference pipeline
│
├── dashboard/                      # Streamlit UI
│   └── app.py                      # Fleet monitoring console
│
├── tests/                          # Test suite
│   ├── test_data_engineering.py    # Layer 1 tests
│   ├── test_ml_engine.py           # Layer 2 tests
│   └── test_triage_agent.py        # Layer 3 tests
│
├── models/                         # Saved model artifacts
└── mlruns/                         # MLflow experiment tracking
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Resilient Design:** The entire codebase works with **only `numpy` and `pandas`** installed. Optional packages (scipy, scikit-learn, LightGBM, PyTorch) enhance performance but are not required.

### 2. Download Data

```bash
py data/download_cmapss.py
```

Downloads the NASA C-MAPSS turbofan engine degradation dataset (FD001–FD004). Falls back to synthetic data generation if download fails.

### 3. Train Models

```bash
py pipelines/train_pipeline.py --dataset FD001
```

Trains all 4 layers end-to-end:
- Fits K-Means regime clusterer (k=6)
- Trains autoencoder on pristine cycles (t ≤ 30)
- Fits quantile regressors (τ ∈ {0.10, 0.50, 0.90})
- Establishes drift baselines
- Saves all artifacts to `models/`

### 4. Run Inference

```bash
py pipelines/inference_pipeline.py --dataset FD001 --engine 1
```

Produces ERP-ready JSON work orders with:
- Priority classification (P0–P3 + HITL)
- Flagged sensors with descriptions
- Recommended parts (BOM)
- Estimated downtime hours

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Interactive fleet monitoring with:
- Real-time health overview table
- RUL prediction interval charts (p10/p50/p90 bands)
- Priority distribution visualization
- Drift monitoring panel
- Work order generation

### 6. Run Tests

```bash
py -m unittest discover tests/ -v
```

## 🔬 Technical Details

### Sensor Pruning
Removes 7 zero-variance sensors: `s1, s5, s6, s10, s16, s18, s19` → 14 active sensors remaining.

### Operating Regime Normalization
- K-Means clustering (k=6) on `[setting1, setting2, setting3]`
- Within-regime Z-score standardization: `x_norm = (x - µ_regime) / σ_regime`

### Rolling Feature Engineering
For each active sensor over windows w ∈ {5, 15, 30}:
- **OLS Slope**: Linear trend of sensor over the window
- **Rolling Std**: Variability within the window
- **EWMA**: Exponentially weighted moving average (α = 2/(w+1))

### Health Index (Autoencoder)
- Architecture: D → 32 → 16 → 8 → 16 → 32 → D
- Trained on pristine cycles only (cycle ≤ 30)
- HI(t) = 1 − MSE(t) / MSE_max, clipped to [0, 1]

### First Time To Predict (FTTP)
- Triggered when MSE > µ + 3σ for 3 consecutive cycles
- Marks the onset of detectable degradation

### Quantile RUL Prediction
- Three separate models for τ ∈ {0.10, 0.50, 0.90}
- Uses LightGBM pinball loss (NumPy gradient descent fallback)
- Monotonicity enforced: p10 ≤ p50 ≤ p90

### NASA Asymmetric Score
```
d = predicted − true
s = exp(−d/13) − 1    if d < 0  (late prediction → heavy penalty)
s = exp(d/10) − 1     if d ≥ 0  (early prediction → light penalty)
S = Σ sᵢ              (lower is better)
```

### Triage FSM
| Priority | Condition | Action |
|----------|-----------|--------|
| **P0 CRITICAL** | RUL_p10 ≤ 10 OR HI < 0.20 | Emergency grounding & rapid overhaul |
| **P1 URGENT** | 10 < RUL_p10 ≤ 25 | Depot HPC overhaul within deadline |
| **P2 ADVISORY** | 25 < RUL_p10 ≤ 45 | Borescope inspection & part pre-order |
| **P3 NORMAL** | RUL_p10 > 45 | Continue normal operation |
| **HITL** | (p90−p10) > 35 | Escalate to human engineering team |

### Drift Detection
- **PSI** (Population Stability Index) > 0.20 → distribution shift
- **KS-test** p-value < 0.05 → feature drift confirmed
- Triggers champion-challenger evaluation and potential retraining

### Champion-Challenger
- Challenger model must beat champion's NASA score by > 3% to be promoted
- Hot-swap mechanism for zero-downtime model updates

## 📋 Datasets

| ID | Conditions | Fault Modes | Train Engines | Test Engines |
|----|-----------|-------------|---------------|--------------|
| FD001 | 1 | 1 (HPC) | 100 | 100 |
| FD002 | 6 | 1 (HPC) | 260 | 259 |
| FD003 | 1 | 2 (HPC + Fan) | 100 | 100 |
| FD004 | 6 | 2 (HPC + Fan) | 249 | 248 |

## 🛡️ Resilience

Every module includes pure NumPy/stdlib fallbacks:

| Package | Used For | Fallback |
|---------|----------|----------|
| PyTorch | Autoencoder | NumPy MLP with Adam optimizer |
| LightGBM | Quantile regression | NumPy subgradient descent |
| scikit-learn | K-Means, StandardScaler | NumPy Lloyd's algorithm |
| scipy | KS-test | NumPy ECDF approximation |
| Pydantic | Schema validation | stdlib dataclasses |
| Streamlit | Dashboard | CLI text summary |


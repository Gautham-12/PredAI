"""
tests.test_data_engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Layer 1: Data Engineering.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import PRUNED_SENSORS, ACTIVE_SENSORS, RAW_COLUMNS, RUL_CAP
from src.data_engineering.pruner import prune_sensors, get_feature_columns
from src.data_engineering.regime import RegimeClusterer
from src.data_engineering.normalizer import RegimeNormalizer
from src.data_engineering.features import FeatureEngineer
from src.ml_engine.rul_target import add_rul_target


class TestDataEngineering(unittest.TestCase):
    """Test suite for Data Engineering components."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n_rows = 150
        data = {
            "unit_id": np.repeat(np.arange(1, 4), 50),
            "cycle": np.tile(np.arange(1, 51), 3),
            "setting1": rng.choice([0.0, 10.0, 20.0, 25.0, 35.0, 42.0], size=n_rows),
            "setting2": rng.choice([0.0, 0.25, 0.50, 0.62, 0.70, 0.84], size=n_rows),
            "setting3": rng.choice([100.0, 60.0, 80.0], size=n_rows),
        }
        # Add 21 sensors
        for s in range(1, 22):
            col = f"s{s}"
            if col in PRUNED_SENSORS:
                data[col] = np.full(n_rows, 500.0 + s)
            else:
                data[col] = rng.normal(500.0 + s, 2.0, size=n_rows)

        self.df = pd.DataFrame(data)

    def test_pruner_removes_correct_sensors(self):
        """Verify the 7 zero-variance sensors are removed."""
        pruned = prune_sensors(self.df)
        for s in PRUNED_SENSORS:
            self.assertNotIn(s, pruned.columns)
        for s in ACTIVE_SENSORS:
            self.assertIn(s, pruned.columns)
        self.assertEqual(len(pruned.columns), len(self.df.columns) - len(PRUNED_SENSORS))

    def test_regime_clusterer_k_equals_6(self):
        """Verify K-Means clusters operational settings into k=6 regimes."""
        clusterer = RegimeClusterer(n_regimes=6)
        labels = clusterer.fit_predict(self.df)
        self.assertEqual(len(labels), len(self.df))
        unique_labels = np.unique(labels)
        self.assertTrue(len(unique_labels) <= 6)
        self.assertEqual(len(clusterer.cluster_centers_), 6)

    def test_zscore_normalization_within_regime(self):
        """Verify sensor standardization within each operational regime."""
        df_copy = self.df.copy()
        df_copy["regime"] = np.random.choice(6, size=len(df_copy))

        normalizer = RegimeNormalizer(sensor_cols=["s2", "s3"])
        norm = normalizer.fit_transform(df_copy, regime_col="regime")

        for r in np.unique(df_copy["regime"]):
            grp = norm[norm["regime"] == r]
            if len(grp) > 1:
                self.assertAlmostEqual(grp["s2"].mean(), 0.0, places=4)
                self.assertAlmostEqual(grp["s2"].std(ddof=1), 1.0, places=2)

    def test_rolling_features_windows(self):
        """Verify rolling windows (5, 15, 30) generate trend, std, EWMA."""
        feat_eng = FeatureEngineer(sensor_cols=["s2"], windows=[5, 15, 30])
        res = feat_eng.transform(self.df)

        expected_cols = [
            "s2_slope_5", "s2_std_5", "s2_ewma_5",
            "s2_slope_15", "s2_std_15", "s2_ewma_15",
            "s2_slope_30", "s2_std_30", "s2_ewma_30",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns)
        self.assertFalse(res[expected_cols].isna().any().any())

    def test_rul_target_capped_at_125(self):
        """Verify piecewise linear RUL target is capped at 125 cycles."""
        df_long = pd.DataFrame({
            "unit_id": [1] * 200,
            "cycle": list(range(1, 201)),
        })
        target_df = add_rul_target(df_long, rul_cap=125)
        self.assertIn("RUL", target_df.columns)
        self.assertLessEqual(target_df["RUL"].max(), 125)
        # Cycle 1 should have capped value 125 (since 199 > 125)
        self.assertEqual(target_df.loc[0, "RUL"], 125)
        # Last cycle should have RUL 0
        self.assertEqual(target_df.loc[199, "RUL"], 0)


if __name__ == "__main__":
    unittest.main()

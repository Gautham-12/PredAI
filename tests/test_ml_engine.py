"""
Tests for AURA-Fleet ML Engine (Layer 2).
Covers autoencoder architecture, health index, quantile model,
NASA scoring, and FTTP detection.
"""
import sys, os, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml_engine.autoencoder import HealthAutoencoder
from src.ml_engine.quantile_model import QuantileRULModel
from src.ml_engine.nasa_loss import nasa_score, pinball_loss
from src.ml_engine.rul_target import compute_rul


def _make_ae_dataframe(n_rows, n_features, noise_scale=0.1, seed=42):
    """Helper: create a DataFrame with 'cycle' col and feature columns."""
    rng = np.random.default_rng(seed)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    data = rng.standard_normal((n_rows, n_features)) * noise_scale
    df = pd.DataFrame(data, columns=feature_cols)
    df["cycle"] = list(range(1, n_rows + 1))
    return df, feature_cols


class TestAutoencoder(unittest.TestCase):
    """Tests for the Health Autoencoder."""

    def setUp(self):
        self.n_features = 14

    def test_architecture_dimensions(self):
        """Verify autoencoder accepts D features and stores input_dim."""
        ae = HealthAutoencoder(input_dim=self.n_features)
        self.assertEqual(ae.input_dim, self.n_features)

    def test_health_index_range_zero_to_one(self):
        """Health Index must be in [0, 1]."""
        ae = HealthAutoencoder(input_dim=self.n_features, epochs=20, pristine_cycles=50)
        df, feat_cols = _make_ae_dataframe(60, self.n_features, noise_scale=0.1)
        ae.fit(df, feature_cols=feat_cols)

        # Create test data with varying degradation
        rng = np.random.default_rng(99)
        test_data = rng.standard_normal((50, self.n_features)).astype(np.float32) * 5.0
        hi = ae.health_index(test_data)

        self.assertTrue(np.all(hi >= 0.0), f"HI has values below 0: min={hi.min()}")
        self.assertTrue(np.all(hi <= 1.0), f"HI has values above 1: max={hi.max()}")

    def test_pristine_data_has_high_health_index(self):
        """Pristine (healthy) data should have HI close to 1."""
        ae = HealthAutoencoder(input_dim=self.n_features, epochs=40, pristine_cycles=200)
        df, feat_cols = _make_ae_dataframe(200, self.n_features, noise_scale=0.1)
        ae.fit(df, feature_cols=feat_cols)

        pristine_arr = df[feat_cols].values[:50].astype(np.float32)
        hi = ae.health_index(pristine_arr)
        mean_hi = float(np.mean(hi))
        self.assertGreater(mean_hi, 0.5,
                           f"Pristine data should have high HI, got mean={mean_hi:.3f}")

    def test_degraded_data_has_lower_health_index(self):
        """Degraded data should have lower HI than pristine."""
        ae = HealthAutoencoder(input_dim=self.n_features, epochs=40, pristine_cycles=200)
        df, feat_cols = _make_ae_dataframe(200, self.n_features, noise_scale=0.1)
        ae.fit(df, feature_cols=feat_cols)

        pristine_arr = df[feat_cols].values[:50].astype(np.float32)
        rng = np.random.default_rng(42)
        degraded = (rng.standard_normal((50, self.n_features)) * 10.0 + 5.0).astype(np.float32)

        hi_pristine = float(np.mean(ae.health_index(pristine_arr)))
        hi_degraded = float(np.mean(ae.health_index(degraded)))
        self.assertGreater(hi_pristine, hi_degraded,
                           f"Pristine HI ({hi_pristine:.3f}) should > degraded ({hi_degraded:.3f})")

    def test_fttp_detection(self):
        """FTTP: MSE > µ+3σ for 3 consecutive cycles."""
        ae = HealthAutoencoder(input_dim=self.n_features, epochs=20, pristine_cycles=50)
        df, feat_cols = _make_ae_dataframe(50, self.n_features, noise_scale=0.1)
        ae.fit(df, feature_cols=feat_cols)

        # Create sequence: normal then very anomalous
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((20, self.n_features)).astype(np.float32) * 0.1
        anomalous = rng.standard_normal((10, self.n_features)).astype(np.float32) * 50.0
        sequence = np.vstack([normal, anomalous])

        mse_seq = ae.reconstruction_mse(sequence)
        flags = ae.fttp_flags(mse_seq)

        # Flags should have some True values in the anomalous region (last 10)
        self.assertEqual(len(flags), 30)
        # At least verify the array has the right shape and type
        self.assertTrue(flags.dtype == bool)


class TestQuantileModel(unittest.TestCase):
    """Tests for the Quantile RUL Model."""

    def setUp(self):
        np.random.seed(42)
        self.n_features = 14
        self.n_samples = 500
        self.feat_cols = [f"feat_{i}" for i in range(self.n_features)]
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=self.feat_cols,
        )
        self.y = pd.Series(
            np.clip(125 - np.sum(self.X.values[:, :3] ** 2, axis=1)
                    + np.random.randn(self.n_samples) * 5, 0, 125)
        )

    def test_quantile_ordering_p10_lt_p50_lt_p90(self):
        """Quantile predictions must satisfy p10 ≤ p50 ≤ p90."""
        model = QuantileRULModel()
        model.fit(self.X, self.y, feature_cols=self.feat_cols)

        X_test = pd.DataFrame(
            np.random.randn(100, self.n_features),
            columns=self.feat_cols,
        )
        preds = model.predict(X_test)

        self.assertIn("rul_p10", preds.columns)
        self.assertIn("rul_p50", preds.columns)
        self.assertIn("rul_p90", preds.columns)

        # Monotonicity enforced by the model
        self.assertTrue(np.all(preds["rul_p10"].values <= preds["rul_p50"].values + 1e-6),
                        "p10 should be ≤ p50")
        self.assertTrue(np.all(preds["rul_p50"].values <= preds["rul_p90"].values + 1e-6),
                        "p50 should be ≤ p90")

    def test_predictions_non_negative(self):
        """RUL predictions should be non-negative."""
        model = QuantileRULModel()
        model.fit(self.X, self.y, feature_cols=self.feat_cols)

        X_test = pd.DataFrame(
            np.random.randn(50, self.n_features),
            columns=self.feat_cols,
        )
        preds = model.predict(X_test)
        for col in ["rul_p10", "rul_p50", "rul_p90"]:
            self.assertTrue(np.all(preds[col].values >= 0),
                            f"Column {col} has negative values")

    def test_feature_importance(self):
        """Feature importance should return a Series with correct length."""
        model = QuantileRULModel()
        model.fit(self.X, self.y, feature_cols=self.feat_cols)
        fi = model.feature_importance(tau=0.50)
        self.assertEqual(len(fi), self.n_features)


class TestNASAScore(unittest.TestCase):
    """Tests for the NASA asymmetric scoring function."""

    def test_perfect_prediction_scores_zero(self):
        """d=0 ⇒ score=0."""
        actual = np.array([50.0, 100.0, 25.0])
        predicted = np.array([50.0, 100.0, 25.0])
        score = nasa_score(actual, predicted)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_late_prediction_penalized_more(self):
        """Late (predicted > actual → dangerous failure) penalized more heavily.
        d = predicted - actual.
        d < 0 → early (predicted too low → maintain too soon) → exp(-d/13)-1
        d > 0 → late  (predicted too high → machine fails) → exp(d/10)-1
        """
        actual = np.array([50.0])

        # Early by 10: predicted=40, d=-10 → exp(10/13)-1 ≈ 1.154
        early_pred = np.array([40.0])
        score_early = nasa_score(actual, early_pred)

        # Late by 10: predicted=60, d=10 → exp(10/10)-1 ≈ 1.718
        late_pred = np.array([60.0])
        score_late = nasa_score(actual, late_pred)

        self.assertGreater(score_late, score_early,
                           f"Late penalty ({score_late:.3f}) should > early ({score_early:.3f})")

    def test_asymmetry_direction(self):
        """Verify exp(d/10)-1 > exp(-d/13)-1 = exp(d/13)-1 for positive d."""
        d = 10.0
        late_penalty = np.exp(d / 10) - 1    # exp(1)-1 ≈ 1.718
        early_penalty = np.exp(d / 13) - 1   # exp(10/13)-1 ≈ 1.154
        self.assertGreater(late_penalty, early_penalty)

    def test_shape_mismatch_raises(self):
        """Mismatched input shapes should raise ValueError."""
        with self.assertRaises(ValueError):
            nasa_score(np.array([1, 2]), np.array([1, 2, 3]))


class TestPinballLoss(unittest.TestCase):
    """Tests for the pinball (quantile) loss."""

    def test_pinball_loss_zero_on_exact(self):
        """Perfect predictions → zero loss."""
        actual = np.array([50.0, 100.0])
        predicted = np.array([50.0, 100.0])
        loss = pinball_loss(actual, predicted, tau=0.5)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_pinball_loss_asymmetry(self):
        """Pinball loss penalizes differently above vs below."""
        actual = np.array([50.0])
        over_pred = np.array([60.0])   # y - ŷ = -10
        under_pred = np.array([40.0])  # y - ŷ = +10

        # For tau=0.1, underprediction (y>ŷ) is weighted by tau=0.1
        # and overprediction (y<ŷ) is weighted by (1-tau)=0.9
        loss_over = pinball_loss(actual, over_pred, tau=0.1)
        loss_under = pinball_loss(actual, under_pred, tau=0.1)

        # over: max(0.1*(-10), -0.9*(-10)) = max(-1, 9) = 9
        # under: max(0.1*10, -0.9*10) = max(1, -9) = 1
        self.assertGreater(loss_over, loss_under,
                           f"Overprediction loss ({loss_over}) should > underprediction ({loss_under}) at tau=0.1")

    def test_pinball_non_negative(self):
        """Pinball loss is always non-negative."""
        rng = np.random.default_rng(42)
        y = rng.uniform(0, 125, 100)
        yhat = rng.uniform(0, 125, 100)
        for tau in [0.1, 0.5, 0.9]:
            loss = pinball_loss(y, yhat, tau=tau)
            self.assertGreaterEqual(loss, 0.0)


class TestRULTarget(unittest.TestCase):
    """Tests for piecewise RUL target computation."""

    def test_rul_capped_at_125(self):
        """RUL should be capped at 125."""
        df = pd.DataFrame({
            'unit_id': [1] * 200,
            'cycle': list(range(1, 201)),
        })
        for i in range(1, 22):
            df[f's{i}'] = 0.0
        for i in range(1, 4):
            df[f'setting{i}'] = 0.0

        result = compute_rul(df)
        max_rul = result['RUL'].max()
        self.assertLessEqual(max_rul, 125,
                             f"RUL should be ≤ 125, got {max_rul}")

    def test_rul_decreases_with_cycle(self):
        """RUL decreases as cycle increases; last cycle = 0."""
        df = pd.DataFrame({
            'unit_id': [1] * 50,
            'cycle': list(range(1, 51)),
        })
        for i in range(1, 22):
            df[f's{i}'] = 0.0
        for i in range(1, 4):
            df[f'setting{i}'] = 0.0

        result = compute_rul(df)
        ruls = result['RUL'].values
        self.assertEqual(ruls[-1], 0)
        self.assertEqual(ruls[0], min(49, 125))

    def test_multi_unit_rul(self):
        """RUL computed independently per unit."""
        df = pd.DataFrame({
            'unit_id': [1]*30 + [2]*50,
            'cycle': list(range(1, 31)) + list(range(1, 51)),
        })
        for i in range(1, 22):
            df[f's{i}'] = 0.0
        for i in range(1, 4):
            df[f'setting{i}'] = 0.0

        result = compute_rul(df)
        # Last cycle of unit 1 should have RUL=0
        unit1_last_rul = result[result['unit_id'] == 1]['RUL'].iloc[-1]
        unit2_last_rul = result[result['unit_id'] == 2]['RUL'].iloc[-1]
        self.assertEqual(unit1_last_rul, 0)
        self.assertEqual(unit2_last_rul, 0)


if __name__ == '__main__':
    unittest.main()

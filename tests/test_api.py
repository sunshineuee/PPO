import unittest
import pandas as pd
from ppo_agent.agent import PPOAgent
from ppo_agent.features import compute_features

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PPOAgent()
        self.dummy_data = pd.DataFrame({
            "open": [100, 102, 104, 103, 105, 107],
            "high": [102, 105, 106, 106, 108, 110],
            "low": [99, 101, 102, 100, 104, 106],
            "close": [101, 104, 105, 104, 107, 109],
            "volume": [1000, 1200, 1300, 1250, 1400, 1500],
        })

    def test_compute_features(self):
        features = compute_features(self.dummy_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)
        self.assertIn("rsi", features.columns)
        self.assertIn("macd", features.columns)

    def test_predict_on_features(self):
        features = compute_features(self.dummy_data)
        prediction = self.agent.predict_on_features(features)
        self.assertIsInstance(prediction, dict)
        self.assertIn("signal", prediction)
        self.assertIn("confidence", prediction)
        self.assertIsInstance(prediction["signal"], float)
        self.assertIsInstance(prediction["confidence"], float)

    def test_train_on_dataframe(self):
        features = compute_features(self.dummy_data)
        features["figi"] = "TEST"
        features["timeframe"] = "1d"
        features["timestamp"] = pd.Timestamp.utcnow()
        try:
            self.agent.train_on_dataframe(features)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()

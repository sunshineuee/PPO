import unittest
import pandas as pd
from ppo_agent.features import compute_features

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Пример искусственных свечей
        self.df = pd.DataFrame({
            "time": pd.date_range(start="2025-01-01", periods=100, freq="D"),
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100 + i for i in range(100)],
            "volume": [1000 + 10 * i for i in range(100)]
        }).set_index("time")

    def test_compute_features_columns_exist(self):
        result = compute_features(self.df)
        expected_columns = ["rsi", "atr", "macd", "macd_signal", "macd_hist"]
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Отсутствует колонка {col}")

    def test_compute_features_not_empty(self):
        result = compute_features(self.df)
        self.assertGreater(len(result), 0, "Результат должен содержать строки")

    def test_compute_features_drops_nan(self):
        result = compute_features(self.df)
        self.assertFalse(result.isnull().values.any(), "Результат не должен содержать NaN")

    def test_compute_features_returns_dataframe(self):
        result = compute_features(self.df)
        self.assertIsInstance(result, pd.DataFrame, "Результат должен быть DataFrame")

if __name__ == '__main__':
    unittest.main()
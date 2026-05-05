import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clean_data import standardize_categorical_text


class StandardizeCategoricalTextTests(unittest.TestCase):
    def test_preserves_missing_categorical_values(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "gender": [" male ", pd.NA],
                "age": [25, 31],
                "country": [" us ", pd.NA],
                "subscription_type": [" premium ", pd.NA],
                "listening_time": [100, 120],
                "songs_played_per_day": [20, 24],
                "skip_rate": [0.2, 0.3],
                "device_type": [" mobile ", pd.NA],
                "ads_listened_per_week": [4, 0],
                "offline_listening": [1, 0],
                "is_churned": [0, 1],
            }
        )

        cleaned = standardize_categorical_text(df)

        self.assertEqual(cleaned.loc[0, "gender"], "Male")
        self.assertEqual(cleaned.loc[0, "country"], "US")
        self.assertEqual(cleaned.loc[0, "subscription_type"], "Premium")
        self.assertEqual(cleaned.loc[0, "device_type"], "Mobile")

        self.assertTrue(pd.isna(cleaned.loc[1, "gender"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "country"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "subscription_type"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "device_type"]))


if __name__ == "__main__":
    unittest.main()

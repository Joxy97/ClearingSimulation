import unittest
import numpy as np
import pandas as pd
from clearing import gray_codes

class TestGrayCodes(unittest.TestCase):
    def setUp(self):
        # Load test.csv as DataFrame
        self.test_df = pd.read_csv("test.csv")
        # Extract asset names from columns
        self.assets = sorted(set(col.split('_g')[0] for col in self.test_df.columns))
        # Create dummy returns for fitting quantizer
        # Here, we use random data for each asset
        np.random.seed(42)
        returns = pd.DataFrame({asset: np.random.normal(0, 0.01, len(self.test_df)) for asset in self.assets})
        self.quantizer = gray_codes.GrayQuantizer(K=8)
        self.quantizer.fit(returns, self.assets)

    def test_encode_dataframe(self):
        # Decode the test.csv binary to bin indices
        binary_df = self.test_df
        decoded_df = self.quantizer.decode_dataframe(binary_df)
        # Encode back to binary
        reencoded_df, states_df = self.quantizer.encode_dataframe(decoded_df)
        # Cast both DataFrames to the same dtype for comparison
        reencoded_df_cast = reencoded_df.astype('int64')
        binary_df_cast = binary_df.astype('int64')
        pd.testing.assert_frame_equal(reencoded_df_cast, binary_df_cast)

    def test_gray_encode_decode_int(self):
        # Test round-trip for all bin indices
        for n in range(8):
            g = gray_codes._gray_encode_int(n)
            n2 = gray_codes._gray_decode_int(g)
            self.assertEqual(n, n2)

    def test_bits_conversion(self):
        # Test conversion between int and bits
        for n in range(8):
            bits = gray_codes._int_to_bits(n, 3)
            n2 = gray_codes._bits_to_int(bits)
            self.assertEqual(n, n2)

if __name__ == "__main__":
    unittest.main()

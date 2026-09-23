from pathlib import Path
import os
os.environ.setdefault('BIOTOX_PACKAGE_ALIAS_ROOT', str(Path(__file__).resolve().parent / 'methodology_audit/pkg_alias'))
import unittest
import numpy as np
import pandas as pd
from methods_residual_posthoc import compound_statistics, _assert_close

class CompoundMethodsTests(unittest.TestCase):

    def make_group(self, ik, y, deltas):
        return pd.DataFrame({'ik': ik, 'repeat_id': np.arange(len(deltas)), 'y': y, 'p_chem_only': np.full(len(deltas), 0.4), 'p_chem_bio': 0.4 + np.asarray(deltas)})

    def test_effect_stability_and_directional_significance(self):
        data = pd.concat([self.make_group('rescue', 1, np.linspace(0.06, 0.08, 20)), self.make_group('correct', 0, -np.linspace(0.03, 0.05, 20)), self.make_group('unstable', 1, [0.09] * 14 + [0.01] * 6), self.make_group('zero', 0, np.zeros(20))], ignore_index=True)
        stats = compound_statistics(data).set_index('ik')
        self.assertEqual(stats.loc['rescue', 'classification'], 'rescued')
        self.assertEqual(stats.loc['correct', 'classification'], 'corrected')
        self.assertEqual(stats.loc['unstable', 'classification'], 'not_retained')
        self.assertEqual(stats.loc['unstable', 'n_repeats_exceeding_threshold'], 14)
        self.assertEqual(stats.loc['zero', 'wilcoxon_p_value'], 1.0)
        self.assertAlmostEqual(stats.loc['rescue', 'wilcoxon_p_value'], 2.0 ** (-20))
        self.assertTrue((stats.bh_q_value >= stats.wilcoxon_p_value).all())

    def test_threshold_and_fifteen_repeat_boundary(self):
        data = pd.concat([self.make_group('fifteen', 1, [0.08] * 15 + [0.01] * 5), self.make_group('below_mean', 1, [0.051] * 15 + [-0.01] * 5), self.make_group('wrong_direction', 0, [0.08] * 20)], ignore_index=True)
        stats = compound_statistics(data).set_index('ik')
        self.assertEqual(stats.loc['fifteen', 'classification'], 'rescued')
        self.assertEqual(stats.loc['below_mean', 'classification'], 'not_retained')
        self.assertEqual(stats.loc['wrong_direction', 'classification'], 'not_retained')

    def test_missing_and_duplicate_repeats_fail(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            compound_statistics(self.make_group('x', 1, [0.1] * 19))
        data = self.make_group('x', 1, [0.1] * 20)
        data.loc[1, 'repeat_id'] = 0
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compound_statistics(data)

    def test_reconstruction_check_rejects_intercept_contamination(self):
        with self.assertRaisesRegex(ValueError, 'maximum absolute error'):
            _assert_close([1.1, 2.1], [1.0, 2.0], 'transcriptomic term')
if __name__ == '__main__':
    unittest.main()

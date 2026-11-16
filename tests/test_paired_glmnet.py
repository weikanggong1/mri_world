"""Software invariants; fixtures/backend stub are not scientific benchmarks."""
import importlib.util
import subprocess
import sys
import unittest
from unittest.mock import patch

from brain_mri_world_model import paired_glmnet as paired

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class PairedInterfaceTests(unittest.TestCase):
    def test_import_and_help_need_no_numerical_backend(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys;from brain_mri_world_model import paired_glmnet;assert 'numpy' not in sys.modules and 'torch' not in sys.modules and 'glmnet' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "brain_mri_world_model.paired_glmnet", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--minimum-subjects", result.stdout)

    def test_wrong_glmnet_version_never_falls_back(self):
        with patch("importlib.metadata.version", return_value="0.0"):
            with self.assertRaisesRegex(paired.MethodHold, "EXACT_INSTALLED_PYTHON_GLMNET_2_2_1_REQUIRED"):
                paired._backend()


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for numerical software tests")
class PairedNumericalTests(unittest.TestCase):
    def test_training_only_imputation_and_column_drop(self):
        import numpy as np
        processor = paired.FoldPreprocessor()
        training = np.array([[1., np.nan, np.nan], [3., 8., np.nan]])
        fitted = processor.fit(training, np.empty((2, 0)))
        np.testing.assert_array_equal(processor.columns, [0, 1])
        np.testing.assert_allclose(fitted, [[1., 8.], [3., 8.]])
        heldout = processor.transform(np.array([[1000., np.nan, 900.]]), np.empty((1, 0)))
        np.testing.assert_allclose(heldout, [[1000., 8.]])

    def test_top200_exact_ties_keep_original_order(self):
        import numpy as np
        target = np.arange(30, dtype=float)
        predictors = np.tile(target[:, None], (1, 205))
        np.testing.assert_array_equal(paired.top_training_correlations(predictors, target), np.arange(200))

    def test_missing_target_does_not_reassign_folds_or_baseline_intersection(self):
        import numpy as np
        n = 50; ids = ["software-" + str(i) for i in range(n)]
        baseline = np.ones((n, 3)); available = np.arange(n) % 5 != 0
        target = np.arange(n, dtype=float); changed = target.copy(); changed[::7] = np.nan
        options = dict(expected_availability={"m": np.ones(n, dtype=bool)},
                       completed_availability={"m": np.ones(n, dtype=bool)},
                       baseline_available=available, minimum_subjects=1001)
        first = paired.fit_paired(ids, target, baseline, {"m": baseline}, **options)
        second = paired.fit_paired(ids, changed, baseline, {"m": baseline}, **options)
        np.testing.assert_array_equal(first["outer_fold_ids"], second["outer_fold_ids"])
        np.testing.assert_array_equal(first["cv_scope_mask"], available)
        self.assertEqual(first["counts"]["CV_subjects"], 40)
        self.assertFalse(first["paired_comparison_complete"])

    def test_raw_unit_fusion_requires_every_expected_modality(self):
        import numpy as np
        predictions = {"a": np.array([10., 20., 30.]), "b": np.array([100., np.nan, 300.])}
        expected = {name: np.ones(3, dtype=bool) for name in predictions}
        fused, complete, count = paired.mean_raw_oof(predictions, expected, np.ones(3, dtype=bool))
        np.testing.assert_allclose(fused[[0, 2]], [55., 165.])
        self.assertTrue(np.isnan(fused[1]))
        np.testing.assert_array_equal(complete, [True, False, True])
        expected["b"][1] = False
        fused, complete, count = paired.mean_raw_oof(predictions, expected, np.ones(3, dtype=bool))
        self.assertEqual(fused[1], 20.)
        self.assertTrue(complete.all())

    def test_pending_computation_does_not_drop_people_or_fit(self):
        import numpy as np
        n = 30; ids = ["software-" + str(i) for i in range(n)]
        values = np.arange(n * 3, dtype=float).reshape(n, 3)
        done = np.ones(n, dtype=bool); done[4] = False
        with patch.object(paired, "_backend", side_effect=AssertionError("must not fit pending arms")):
            result = paired.fit_paired(ids, np.arange(n, dtype=float), values, {"m": values},
                expected_availability={"m": np.ones(n, dtype=bool)}, completed_availability={"m": done},
                baseline_complete=done, minimum_subjects=1)
        self.assertEqual(result["counts"]["target_observed_subjects"], n)
        self.assertEqual(result["arms"]["m"]["pending_subjects"], 1)
        self.assertFalse(result["paired_comparison_complete"])

    def test_missing_explicit_availability_mask_is_rejected(self):
        import numpy as np
        n = 30; ids = ["software-" + str(i) for i in range(n)]
        values = np.ones((n, 3))
        with self.assertRaisesRegex(paired.MethodHold, "EXPLICIT_EXPECTED_AND_COMPLETED_MODALITY_MASKS_REQUIRED"):
            paired.fit_paired(ids, np.arange(n, dtype=float), values, {"m": values},
                expected_availability={"m": None}, completed_availability={"m": np.ones(n, dtype=bool)})

    def test_backend_stub_receives_reference_kwargs_and_preserves_rng(self):
        import numpy as np
        constructors = []

        class BackendSoftwareStub:
            def __init__(self, **kwargs):
                constructors.append(kwargs)
            def fit(self, X, y):
                np.random.random()  # The public wrapper must restore caller RNG.
                self.coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
            def predict(self, X):
                return np.column_stack([np.ones(len(X)), X]) @ self.coefficients

        n = 50; ids = ["software-" + str(i) for i in range(n)]
        x = np.arange(n, dtype=float)
        features = np.column_stack([x, x * x, np.sin(x)])
        target = 2.5 * x + 10.
        mask = np.ones(n, dtype=bool)
        state = np.random.get_state()
        with patch.object(paired, "_backend", return_value=BackendSoftwareStub):
            result = paired.fit_paired(ids, target, features, {"m": features},
                expected_availability={"m": mask}, completed_availability={"m": mask}, minimum_subjects=10)
        self.assertTrue(result["paired_comparison_complete"])
        self.assertEqual(len(constructors), 2 * 5 * (5 * 3 + 1))
        self.assertTrue(all(set(row) == {"alpha", "n_lambda"} and row["n_lambda"] == 30 for row in constructors))
        np.testing.assert_allclose(result["feature_mean_predictions"], target, rtol=0, atol=1e-9)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0]); np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit


RUN_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "BIOTOX_PACKAGE_ALIAS_ROOT", str(RUN_DIR / "methodology_audit" / "pkg_alias")
)
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))


def import_calibrated_module():
    return importlib.import_module("nested_cv_offset_logistic_calibrated")


class MethodsResidualLearningTests(unittest.TestCase):
    def test_calibration_is_fit_once_per_training_partition_and_shared_across_alphas(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(0.01, 1.0, 100.0))
        with mock.patch.object(calibrated, "fit_offset_intercept_lbfgs", wraps=calibrated.fit_offset_intercept_lbfgs) as calibrate:
            outcome = calibrated.run_outer_fold(job)
        self.assertTrue(outcome.success, outcome.error)
        self.assertEqual(calibrate.call_count, 4)
        inner_folds, _ = calibrated._find_binary_evaluable_inner_folds()
        expected_indices = [job.outer_train_idx[train] for train, _ in inner_folds] + [job.outer_train_idx]
        for call, indices in zip(calibrate.call_args_list, expected_indices):
            np.testing.assert_array_equal(call.args[0], prepared.y[indices])
            np.testing.assert_array_equal(call.args[1], prepared.offset[indices])
        for inner_id in range(3):
            intercepts = [row["inner_calibration_intercept"] for row in outcome.alpha_rows if row["inner_fold"] == inner_id]
            self.assertEqual(len(set(intercepts)), 1)

    def test_outer_test_labels_do_not_affect_fitted_parameters(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(0.01, 1.0))
        first = calibrated.run_outer_fold(job)
        changed_y = prepared.y.copy()
        changed_y[job.outer_test_idx] = 1 - changed_y[job.outer_test_idx]
        second = calibrated.run_outer_fold(replace(job, prepared=replace(prepared, y=changed_y)))
        self.assertTrue(first.success, first.error)
        self.assertTrue(second.success, second.error)
        np.testing.assert_array_equal(first.beta, second.beta)
        self.assertEqual(first.fold_row["null_intercept"], second.fold_row["null_intercept"])
        self.assertEqual(first.fold_row["alpha_star_outer"], second.fold_row["alpha_star_outer"])

    def test_statsmodels_pipeline_uses_same_fixed_calibration(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(0.1,))
        lbfgs = calibrated.run_outer_fold(job)
        reference = calibrated.run_outer_fold(replace(job, solver="statsmodels"))
        self.assertTrue(lbfgs.success, lbfgs.error)
        self.assertTrue(reference.success, reference.error)
        self.assertEqual(reference.fold_row["null_intercept"], reference.fold_row["bio_intercept"])
        np.testing.assert_allclose(lbfgs.beta, reference.beta, atol=5e-5)

    def test_class_weighting_is_rejected(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(1.0,))
        outcome = calibrated.run_outer_fold(replace(job, use_sample_weight=True))
        self.assertFalse(outcome.success)
        self.assertIn("unweighted BCE", outcome.error)

    def test_bce_remains_exact_for_extreme_logits(self):
        calibrated = import_calibrated_module()
        self.assertEqual(calibrated._mean_bce_from_logits(np.array([0, 1]), np.array([1000., -1000.])), 1000.)

    def test_one_se_uses_sample_standard_error_and_global_eligibility(self):
        calibrated = import_calibrated_module()
        losses = {0.1: [0.1, 0.2, 0.3], 1.: [0.4]*3, 10.: [0.25]*3, 100.: [0.4]*3}
        selected = calibrated._select_alpha_from_losses(losses)
        self.assertAlmostEqual(selected["cv_loss_se_at_alpha_min"], 0.1/np.sqrt(3))
        self.assertEqual(selected["alpha_star"], 10.)

    def test_one_standard_error_rule_selects_largest_globally_eligible_alpha(self):
        calibrated = import_calibrated_module()
        losses_by_alpha = {
            0.1: [0.30, 0.30, 0.30],
            1.0: [0.20, 0.20, 0.20],
            10.0: [0.40, 0.40, 0.40],
            100.0: [0.20, 0.20, 0.20],
        }

        selected = calibrated._select_alpha_from_losses(losses_by_alpha)

        self.assertEqual(selected["alpha_min_loss"], 1.0)
        self.assertEqual(selected["alpha_star"], 100.0)

    def test_fixed_offset_beta_gradient_matches_central_finite_difference(self):
        from offset_ridge_fixed import fixed_offset_ridge_objective_gradient

        X = np.array(
            [
                [-1.2, 0.3, 0.5],
                [-0.4, -0.7, 1.1],
                [0.1, 0.2, -0.3],
                [0.8, -1.0, -0.9],
                [1.4, 0.6, 0.7],
                [2.0, -0.2, -1.2],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 1, 0, 1, 1], dtype=float)
        offset = np.array([-0.8, -0.2, 0.1, 0.3, 0.7, 1.0], dtype=float)
        beta = np.array([0.25, -0.15, 0.08], dtype=float)
        alpha = 0.37

        _, analytic = fixed_offset_ridge_objective_gradient(beta, X, y, offset, alpha)
        numeric = np.zeros_like(beta)
        step = 1e-6
        for index in range(beta.size):
            delta = np.zeros_like(beta)
            delta[index] = step
            plus, _ = fixed_offset_ridge_objective_gradient(
                beta + delta, X, y, offset, alpha
            )
            minus, _ = fixed_offset_ridge_objective_gradient(
                beta - delta, X, y, offset, alpha
            )
            numeric[index] = (plus - minus) / (2.0 * step)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_fixed_offset_beta_optimizer_matches_statsmodels_without_intercept(self):
        from offset_ridge_fixed import fit_fixed_offset_ridge_lbfgs

        X = np.array(
            [
                [-1.1, 0.2],
                [-0.7, -0.8],
                [-0.1, 0.5],
                [0.2, -0.4],
                [0.9, 0.7],
                [1.5, -0.1],
                [1.9, 0.3],
                [2.3, -0.6],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=float)
        offset = np.array([-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1])
        alpha = 0.21

        fit = fit_fixed_offset_ridge_lbfgs(
            X, y, offset, alpha, maxiter=1000, retry_maxiter=2000
        )
        model = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
        reference = model.fit_regularized(
            method="elastic_net",
            alpha=alpha,
            L1_wt=0.0,
            maxiter=1000,
            cnvrg_tol=1e-10,
        )

        self.assertTrue(fit.success, fit.message)
        np.testing.assert_allclose(fit.beta, reference.params, rtol=5e-5, atol=5e-5)

    def test_calibration_intercept_score_matches_observed_prevalence(self):
        from offset_ridge_lbfgs import fit_offset_intercept_lbfgs

        y = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
        offset = np.array([-2.0, -1.0, -0.3, 0.1, 0.4, 1.2, 1.9], dtype=float)

        fit = fit_offset_intercept_lbfgs(y, offset)
        calibrated_probability = expit(offset + fit.intercept)

        self.assertAlmostEqual(
            float(np.mean(calibrated_probability)), float(np.mean(y)), places=11
        )

    def test_stage2_standardization_uses_training_statistics_and_constant_genes(self):
        calibrated = import_calibrated_module()
        X_train_raw = np.array(
            [
                [1.0, 4.0, 7.0],
                [2.0, 4.0, 7.0 + 2e-9],
                [3.0, 4.0, 7.0 - 2e-9],
            ]
        )
        X_test_raw = np.array([[4.0, 100.0, 9.0]])

        X_train, X_test, scaling = calibrated._fit_apply_stage2_feature_scaler(
            "standardized",
            X_train_raw,
            X_test_raw,
            fit_matrix="raw_inner_training",
        )

        self.assertEqual(scaling.n_zero_sd_genes, 2)
        np.testing.assert_allclose(X_train[:, 1:], 0.0, atol=3e-9)
        np.testing.assert_allclose(X_test[0, 0], (4.0 - 2.0) / np.std([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(X_test[0, 1:], [96.0, 2.0], atol=1e-12)

    def test_unweighted_methods_path_does_not_request_inverse_prevalence_weights(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(1.0,))

        outcome = calibrated.run_outer_fold(job)

        self.assertTrue(outcome.success, outcome.error)
        self.assertFalse(outcome.fold_row["use_sample_weight"])
        self.assertEqual(
            outcome.fold_row["stage2_likelihood"], "unweighted_publication_primary"
        )
        self.assertTrue(all("validation_bce" in row for row in outcome.alpha_rows))

    def test_outer_fold_freezes_calibration_intercept_for_bio_prediction(self):
        calibrated = import_calibrated_module()
        prepared = self._prepared_outer_fold(calibrated)
        job = self._outer_fold_job(calibrated, prepared, alpha_grid=(0.01,))

        outcome = calibrated.run_outer_fold(job)

        self.assertTrue(outcome.success, outcome.error)
        self.assertAlmostEqual(
            outcome.fold_row["bio_intercept"], outcome.fold_row["null_intercept"], places=10
        )

        X_train_raw = prepared.X_raw[job.outer_train_idx]
        X_test_raw = prepared.X_raw[job.outer_test_idx]
        _, X_test, _ = calibrated._fit_apply_stage2_feature_scaler(
            prepared.variant,
            X_train_raw,
            X_test_raw,
            fit_matrix="raw_outer_training",
        )
        expected_transcriptomic_logit = X_test @ outcome.beta
        observed_transcriptomic_logit = np.array(
            [
                logit(row["p_chem_bio"]) - logit(row["p_chem_only"])
                for row in outcome.oof_rows
            ]
        )
        np.testing.assert_allclose(
            observed_transcriptomic_logit,
            expected_transcriptomic_logit,
            rtol=1e-8,
            atol=1e-8,
        )

    def _prepared_outer_fold(self, calibrated):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], dtype=float)
        X_raw = np.array(
            [
                [-2.0, 0.2, 1.0],
                [1.7, -0.1, 1.0],
                [-1.5, 0.5, 1.0],
                [1.3, -0.4, 1.0],
                [-1.0, 0.7, 1.0],
                [1.1, -0.6, 1.0],
                [-0.8, 0.9, 1.0],
                [0.9, -0.8, 1.0],
                [-1.8, 0.0, 1.0],
                [-1.2, 0.1, 1.0],
                [1.2, -0.2, 1.0],
                [1.8, -0.3, 1.0],
            ],
            dtype=float,
        )
        offset = np.array(
            [-0.4, 0.2, -0.3, 0.1, -0.2, 0.0, -0.1, 0.3, -0.5, -0.4, 0.2, 0.4],
            dtype=float,
        )
        return calibrated.PreparedTaskCell(
            task="toy_task",
            cell="toy_cell",
            variant="standardized",
            X_raw=X_raw,
            y=y,
            offset=offset,
            smiles=tuple(f"C{i}" for i in range(y.size)),
            ik=tuple(f"IK{i}" for i in range(y.size)),
            gene_names=("gene_a", "gene_b", "constant_gene"),
            outer_folds=(self._outer_indices(),),
            outer_seed_used=123,
        )

    def _outer_fold_job(self, calibrated, prepared, alpha_grid):
        train_idx, test_idx = self._outer_indices()
        job = calibrated.OuterFoldJob(
            prepared=prepared,
            fold_id=0,
            outer_train_idx=train_idx,
            outer_test_idx=test_idx,
            solver="lbfgs",
            use_sample_weight=False,
            alpha_grid=tuple(alpha_grid),
            inner_cv_k=3,
            inner_cv_seed=17,
            maxiter=1000,
            retry_maxiter=2000,
            gtol=1e-7,
            acceptance_gradient=1e-5,
            ftol=1e-14,
        )
        inner_folds = [
            (np.array([3, 4, 5, 6, 7]), np.array([0, 1, 2])),
            (np.array([0, 1, 2, 6, 7]), np.array([3, 4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]
        patcher = mock.patch.object(
            calibrated,
            "_find_binary_evaluable_inner_folds",
            return_value=(inner_folds, job.inner_cv_seed),
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        return job

    @staticmethod
    def _outer_indices():
        return np.arange(8), np.arange(8, 12)


if __name__ == "__main__":
    unittest.main()

"""Methods-specified gene means, compound tests, and held-out gene attributions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
import nested_cv_offset_logistic_calibrated as calibrated
from publication.aggregate_offset_logistic import benjamini_hochberg
from publication.utils import build_task_cell_frame, build_scaffold_dict, load_matched_split
from publication.summarize_repeated_cv_publication import _validate_group

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def compound_statistics(oof: pd.DataFrame, expected_repeats: int=20) -> pd.DataFrame:
    """Directional Wilcoxon tests precede effect/stability filtering and BH."""
    if oof.duplicated(['repeat_id', 'ik']).any():
        raise ValueError('Duplicate held-out compound observations in a repeat')
    rows = []
    for (ik, group) in oof.groupby('ik', sort=True):
        if len(group) != expected_repeats or group.repeat_id.nunique() != expected_repeats:
            raise ValueError(f'{ik}: incomplete held-out repetitions')
        if group.y.nunique() != 1 or group.y.iloc[0] not in (0, 1):
            raise ValueError(f'{ik}: inconsistent/nonbinary outcome')
        delta = (group.p_chem_bio - group.p_chem_only).to_numpy(float)
        if not np.all(np.isfinite(delta)):
            raise ValueError(f'{ik}: nonfinite probability differences')
        y = int(group.y.iloc[0])
        alternative = 'greater' if y == 1 else 'less'
        p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta, alternative=alternative, zero_method='wilcox', method='auto').pvalue)
        threshold = 0.05 if y == 1 else -0.02
        exceed = delta > threshold if y == 1 else delta < threshold
        mean = float(delta.mean())
        rows.append({'ik': str(ik), 'y': y, 'n_repeats': len(group), 'mean_probability_chem': float(group.p_chem_only.mean()), 'mean_probability_bio': float(group.p_chem_bio.mean()), 'mean_delta_probability': mean, 'n_repeats_exceeding_threshold': int(exceed.sum()), 'effect_threshold': threshold, 'wilcoxon_alternative': alternative, 'wilcoxon_p_value': p, 'effect_and_stability_pass': bool((mean > threshold if y == 1 else mean < threshold) and exceed.sum() >= 15)})
    result = pd.DataFrame(rows)
    result['bh_q_value'] = benjamini_hochberg(result.wilcoxon_p_value)
    result['retained'] = result.effect_and_stability_pass & (result.bh_q_value < 0.05)
    result['classification'] = np.where(result.retained, np.where(result.y == 1, 'rescued', 'corrected'), 'not_retained')
    return result

def _assert_close(actual, expected, name: str, atol: float=1e-10) -> float:
    (a, b) = (np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))
    if a.shape != b.shape or not np.all(np.isfinite(a)) or (not np.all(np.isfinite(b))):
        raise ValueError(f'{name}: invalid arrays')
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if error > atol:
        raise ValueError(f'{name}: maximum absolute error {error:.3g} > {atol:.3g}')
    return error

def run(input_dir: Path, output_dir: Path, expected_repeats: int=20) -> pd.DataFrame:
    (input_dir, output_dir) = (input_dir.resolve(), output_dir.resolve())
    manifest = json.loads((input_dir / 'RUN_COMPLETE.json').read_text())
    if manifest['stage2_model_spec'] != calibrated.STAGE2_MODEL_SPEC or not manifest.get('calibration_intercept_frozen'):
        raise ValueError('Post-hoc analysis requires the fixed-calibration Methods model')
    if manifest['n_repeats'] != expected_repeats or manifest['outer_folds_per_repeat'] != 5:
        raise ValueError('Unexpected repeat/fold count')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite post-hoc outputs: {output_dir}')
    output_dir.mkdir(parents=True)
    offsets_path = Path(manifest['chemical_offset_path'])
    if sha256(offsets_path) != manifest['chemical_offset_sha256']:
        raise ValueError('Frozen chemical offset checksum changed')
    offsets = pd.read_csv(offsets_path).set_index('ik', verify_integrity=True)
    (cohort_dir, feature_dir) = (Path(manifest['cohort_dir']), Path(manifest['gene_features_dir']))
    (summaries, gene_means, compound_frames) = ([], [], [])
    for cell in manifest['cells']:
        cohort_path = cohort_dir / f'tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl'
        if not cohort_path.exists():
            cohort_path = cohort_dir / f'tox21_scaffold_df_split_{cell}_8to12uM_paired6h24h.pkl'
        compounds = load_matched_split(str(cohort_path))
        for variant in manifest['variants']:
            suffix = calibrated.VARIANT_SUFFIX[variant]
            feature_path = feature_dir / f'{cell}_gene_features_{suffix}.csv'
            if sha256(feature_path) != manifest['gene_feature_files_sha256'][feature_path.name]:
                raise ValueError(f'Feature checksum changed: {feature_path}')
            features = pd.read_csv(feature_path)
            if features.ik.duplicated().any():
                raise ValueError('Duplicate feature compound IDs')
            for task in manifest['tasks']:
                stem = f'{variant}_{task}__{cell}'
                fold_path = input_dir / f'repeated_nested_cv_{stem}.csv'
                folds = pd.read_csv(fold_path)
                _validate_group(folds, fold_path, expected_repeats, 5)
                oof = pd.read_csv(input_dir / f'repeated_nested_cv_oof_predictions_{stem}.csv')
                beta_frame = pd.read_csv(input_dir / f'repeated_nested_cv_beta_{stem}.csv')
                (frame, genes) = build_task_cell_frame(compounds, features, task)
                if len(genes) != 978:
                    raise ValueError(f'{stem}: expected 978 genes, got {len(genes)}')
                frame = frame.set_index('ik', verify_integrity=True).sort_index()
                ids = frame.index.astype(str)
                if set(ids) != set(oof.ik):
                    raise ValueError(f'{stem}: OOF IDs do not equal the labeled feature cohort')
                X = frame[genes].to_numpy(float)
                y = frame[task].to_numpy(float)
                raw_offset = offsets.loc[ids, f'{task}_chem_logit'].to_numpy(float)
                scaffolds = build_scaffold_dict(list(enumerate(frame.smiles_canon.astype(str))))
                scaffold_ids = np.full(len(ids), -1, dtype=int)
                for (scaffold_id, indices) in enumerate(scaffolds.values()):
                    scaffold_ids[indices] = scaffold_id
                if np.any(scaffold_ids < 0):
                    raise ValueError(f'{stem}: cannot verify scaffolds for invalid molecular structures')
                if beta_frame.duplicated(['repeat_id', 'fold_id', 'gene']).any():
                    raise ValueError(f'{stem}: duplicate gene coefficients')
                betas = beta_frame.pivot(index=['repeat_id', 'fold_id'], columns='gene', values='beta')
                if set(betas.columns) != set(genes) or len(betas) != expected_repeats * 5 or betas.isna().any().any():
                    raise ValueError(f'{stem}: incomplete coefficient matrix')
                psi_sum = np.zeros_like(X)
                heldout_counts = np.zeros(len(ids), dtype=int)
                max_prediction_error = max_logit_error = max_score_error = 0.0
                seen = set()
                for ((repeat, fold), group) in oof.groupby(['repeat_id', 'fold_id'], sort=True):
                    positions = ids.get_indexer(group.ik)
                    if len(set(positions)) != len(positions) or np.any(positions < 0):
                        raise ValueError(f'{stem}: duplicate/unknown held-out IDs')
                    train = np.ones(len(ids), dtype=bool)
                    train[positions] = False
                    if set(scaffold_ids[train]) & set(scaffold_ids[positions]):
                        raise ValueError(f'{stem}: scaffold leakage')
                    row = folds.loc[(folds.repeat_id == repeat) & (folds.fold_id == fold)].iloc[0]
                    if int(row.n_outer_train) != int(train.sum()) or int(row.n_outer_test) != len(positions):
                        raise ValueError(f'{stem}: fold-size mismatch')
                    a = float(row.null_intercept)
                    _assert_close(group.null_intercept, np.full(len(group), a), 'shared OOF calibration')
                    _assert_close(group.bio_intercept, group.null_intercept, 'shared bio/chem calibration', atol=0.0)
                    _assert_close(group.y, y[positions], 'observed outcomes', atol=0.0)
                    (mu, sd) = (X[train].mean(axis=0), X[train].std(axis=0))
                    sd = np.where(sd < 1e-08, 1.0, sd)
                    beta = betas.loc[(repeat, fold), genes].to_numpy(float)
                    psi = (X[positions] - mu) / sd * beta
                    gene_logit = psi.sum(axis=1)
                    chem_logit = raw_offset[positions] + a
                    reconstructed_log_losses = [float(np.mean(np.logaddexp(0.0, logits) - y[positions] * logits)) for logits in (chem_logit, chem_logit + gene_logit)]
                    _assert_close([row.log_loss_chem, row.log_loss_bio], reconstructed_log_losses, 'fold LogLoss')
                    _assert_close([row.delta_log_loss], [reconstructed_log_losses[1] - reconstructed_log_losses[0]], 'fold delta LogLoss')
                    score_error = abs(float(np.mean(expit(raw_offset[train] + a) - y[train])))
                    if score_error > 1e-10:
                        raise ValueError(f'{stem}: calibration score is not zero')
                    max_score_error = max(max_score_error, score_error)
                    max_logit_error = max(max_logit_error, _assert_close(group.logit_chem_only, chem_logit, 'chemical logit'), _assert_close(group.logit_chem_bio, chem_logit + gene_logit, 'biological logit'), _assert_close(group.transcriptomic_logit, gene_logit, 'sum of gene attributions'))
                    max_prediction_error = max(max_prediction_error, _assert_close(group.p_chem_only, expit(chem_logit), 'chemical probability'), _assert_close(group.p_chem_bio, expit(chem_logit + gene_logit), 'biological probability'))
                    psi_sum[positions] += psi
                    heldout_counts[positions] += 1
                    seen.add((repeat, fold))
                if len(seen) != expected_repeats * 5 or not np.all(heldout_counts == expected_repeats):
                    raise ValueError(f'{stem}: incomplete held-out attribution coverage')
                stats = compound_statistics(oof, expected_repeats)
                stats.insert(0, 'cell', cell)
                stats.insert(0, 'task', task)
                stats.insert(0, 'variant', variant)
                compound_frames.append(stats)
                means = pd.DataFrame({'gene': genes, 'mean_beta': betas[genes].mean(axis=0).to_numpy(), 'sd_beta': betas[genes].std(axis=0, ddof=1).to_numpy(), 'n_outer_fits': len(betas)})
                means.insert(0, 'cell', cell)
                means.insert(0, 'task', task)
                means.insert(0, 'variant', variant)
                gene_means.append(means)
                np.savez_compressed(output_dir / f'mean_gene_attributions_{stem}.npz', ik=np.asarray(ids, dtype=str), gene=np.asarray(genes, dtype=str), mean_psi=psi_sum / heldout_counts[:, None], n_heldout=heldout_counts)
                summaries.append({'variant': variant, 'task': task, 'cell': cell, 'n_compounds': len(ids), 'n_genes': len(genes), 'n_outer_fits': len(betas), 'n_heldout_evaluations': len(oof), 'n_rescued': int((stats.classification == 'rescued').sum()), 'n_corrected': int((stats.classification == 'corrected').sum()), 'max_probability_reconstruction_error': max_prediction_error, 'max_logit_reconstruction_error': max_logit_error, 'max_calibration_score_error': max_score_error, 'all_outer_scaffolds_disjoint': True})
                print(f'[methods-posthoc] {stem}: {len(oof)} held-out attributions verified', flush=True)
    pd.concat(gene_means, ignore_index=True).to_csv(output_dir / 'gene_coefficient_means.csv', index=False)
    compounds_all = pd.concat(compound_frames, ignore_index=True)
    compounds_all.to_csv(output_dir / 'compound_statistics.csv', index=False)
    compounds_all.loc[compounds_all.retained].to_csv(output_dir / 'retained_rescued_corrected_compounds.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / 'reconstruction_and_counts.csv', index=False)
    completed = {'source_run': str(input_dir), 'source_run_manifest_sha256': sha256(input_dir / 'RUN_COMPLETE.json'), 'stage2_model_spec': calibrated.STAGE2_MODEL_SPEC, 'analysis_complete': True, 'full_methods_20x5': expected_repeats == 20, 'n_repeats': expected_repeats, 'n_groups': len(summary), 'gene_coefficients': 'arithmetic mean over all outer training fits', 'attributions': 'mean of fold-specific standardized feature times coefficient over held-out repeats', 'rescue_rule': 'y=1; mean delta >0.05; delta >0.05 in at least15 repeats; BH q<0.05', 'correction_rule': 'y=0; mean delta <-0.02; delta <-0.02 in at least15 repeats; BH q<0.05', 'wilcoxon_null': 'paired probability difference equals zero; greater for toxic, less for non-toxic', 'bh_family': 'all compounds within each variant/endpoint/cell before effect and stability filtering', 'wilcoxon_zero_method': 'wilcox; all-zero differences assigned p=1', 'wilcoxon_method': 'scipy auto', 'input_feature_hashes_verified': True, 'chemical_offset_hash_verified': True}
    (output_dir / 'RUN_COMPLETE.json').write_text(json.dumps(completed, indent=2) + '\n')
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expected-repeats', type=int, default=20)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.input_dir, a.output_dir, a.expected_repeats)
if __name__ == '__main__':
    main()

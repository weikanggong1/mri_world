"""Paired subject-level, nested GLMnet regression on caller-owned matrices.

Numerical defaults: outer5/inner3, shared shuffle42 folds, training-only mean
imputation and top200 correlation selection, Python glmnet2.2.1. No tensor
backend, model, label, participant, or server is loaded at import time.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

METHOD = dict(backend="python_glmnet",backend_version="2.2.1",outer_splits=5,
    inner_splits=3,outer_shuffle_seed=42,inner_shuffle_seed=42,
    alpha_grid=[.01,.1,.5,.8,.99],n_lambda=30,top_training_correlations=200,
    alpha_selection="mean_inner3_validation_Pearson",
    lambda_selection="original_ElasticNet_default_internal_CV_and_default_predict",
    target_transform="NONE",confounds="NONE",residualization="NONE",
    fusion="arithmetic_mean_of_expected_raw_continuous_OOF_predictions")

class MethodHold(ValueError):pass
class UnscorableTarget(MethodHold):pass

def need(value,reason):
    if not value:raise MethodHold(reason)

def subject_group_folds(subject_ids,n_splits,seed):
    """KFold-style shuffle of unique subjects in first-occurrence order."""
    import numpy as np
    unique=list(dict.fromkeys(subject_ids))
    need(len(unique)>=n_splits and n_splits>=2,'SUFFICIENT_DISTINCT_SUBJECT_GROUPS_REQUIRED')
    order=np.arange(len(unique));np.random.RandomState(seed).shuffle(order)
    mapping={unique[int(i)]:fold for fold,part in enumerate(np.array_split(order,n_splits))for i in part}
    return np.asarray([mapping[group]for group in subject_ids],dtype=np.int64)

def pearson(y,prediction):
    import numpy as np
    a=np.asarray(y,dtype=np.float64).reshape(-1);b=np.asarray(prediction,dtype=np.float64).reshape(-1)
    need(len(a)==len(b)and np.isfinite(a).all()and np.isfinite(b).all(),'PEARSON_COMPLETE_FINITE_SAME_ROWS_REQUIRED')
    if len(a)<2:return None
    a=a-a.mean();b=b-b.mean();den=float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    return None if den==0 else float(np.clip(np.dot(a,b)/den,-1.,1.))

class FoldPreprocessor:
    """Training-fold predictor imputation; the new reference passes no confounds."""
    def fit(self,X,C):
        import numpy as np
        X=np.asarray(X,dtype=np.float64);C=np.asarray(C,dtype=np.float64)
        need(X.ndim==2 and C.shape==(len(X),0),'UNADJUSTED_MODEL_REQUIRES_ZERO_CONFOUND_COLUMNS')
        need(not np.isinf(X).any(),'INFINITE_INPUT_VALUES_ARE_NOT_MISSINGNESS')
        self.columns=np.flatnonzero(np.any(~np.isnan(X),axis=0))
        need(len(self.columns)>0,'NO_OBSERVED_TRAINING_PREDICTOR')
        xx=X[:,self.columns];self.x_mean=np.nanmean(xx,axis=0)
        xx=np.where(np.isnan(xx),self.x_mean,xx)
        need(np.isfinite(xx).all(),'FINITE_TRAINING_FOLD_PREDICTORS')
        return xx
    def transform(self,X,C):
        import numpy as np
        X=np.asarray(X,dtype=np.float64);C=np.asarray(C,dtype=np.float64)
        need(C.shape==(len(X),0),'HELDOUT_UNADJUSTED_CONFOUND_CONTRACT_CHANGED')
        xx=X[:,self.columns];need(not np.isinf(xx).any(),'INFINITE_HELDOUT_VALUE')
        xx=np.where(np.isnan(xx),self.x_mean,xx)
        need(np.isfinite(xx).all(),'FINITE_HELDOUT_TRAINING_TRANSFORM_REQUIRED');return xx

def top_training_correlations(X,y):
    import numpy as np
    yy=np.asarray(y,dtype=np.float64);yc=yy-yy.mean();ys=float(np.dot(yc,yc))
    if ys==0:raise UnscorableTarget('CONSTANT_TRAINING_TARGET_NO_CORRELATION_SCORE')
    xc=X-X.mean(axis=0);squares=np.einsum('ij,ij->j',xc,xc);valid=np.flatnonzero(squares>0)
    need(len(valid)>0,'NO_VARIABLE_TRAINING_PREDICTOR_AFTER_RESIDUALIZATION')
    scores=(xc[:,valid].T@yc)/np.sqrt(squares[valid]*ys)
    need(np.isfinite(scores).all(),'FINITE_TRAIN_ONLY_CORRELATION_RANKING')
    # Stable original column order resolves exact correlation ties.
    order=np.lexsort((valid,-np.abs(scores)));return valid[order[:min(200,len(order))]]

def _backend():
    import importlib.metadata
    need(importlib.metadata.version('glmnet')=='2.2.1','EXACT_INSTALLED_PYTHON_GLMNET_2_2_1_REQUIRED')
    from glmnet import ElasticNet
    return ElasticNet

def _fit_original_wrapper(ElasticNet,X,y,alpha,seed):
    import numpy as np
    state=np.random.get_state();np.random.seed(seed)
    try:
        # These are exactly the original Notebook constructor kwargs. Keep its
        # own lambda CV/lambda_best/cutpoint and default predict unchanged.
        model=ElasticNet(alpha=alpha,n_lambda=30);model.fit(X,y)
    finally:np.random.set_state(state)
    return model

def _lambda_metadata(model):
    import numpy as np
    result={}
    for key in('lambda_best_','lambda_max_'):
        value=getattr(model,key,None)
        result[key]=None if value is None else np.asarray(value).reshape(-1).tolist()
    getter=getattr(model,'get_params',None)
    if callable(getter):
        params=getter(deep=False)
        result['actual_wrapper_defaults']={key:params.get(key)for key in('n_splits','cut_point','standardize','fit_intercept','scoring','random_state')}
    return result

def _outer_oof(X,y,C,groups,folds,shared_inner_folds,protocol,ElasticNet):
    import numpy as np
    prediction=np.full(len(y),np.nan);seen=np.zeros(len(y),dtype=np.int64);records=[]
    for outer in range(METHOD['outer_splits']):
        test=np.flatnonzero(folds==outer);train=np.flatnonzero(folds!=outer)
        if not len(test):continue
        need(len(set(groups[i]for i in train))>=3 and not(set(groups[i]for i in train)&set(groups[i]for i in test)),
             'OUTER_SUBJECT_GROUPS_DISJOINT_AND_INNER3_POSSIBLE')
        inner=shared_inner_folds[outer][train];need(np.isin(inner,[0,1,2]).all(),'FROZEN_SHARED_INNER3_ASSIGNMENT_REQUIRED');validation=[]
        for inner_fold in range(3):
            a=train[inner!=inner_fold];b=train[inner==inner_fold]
            need(len(a)>0 and len(b)>1 and not(set(groups[i]for i in a)&set(groups[i]for i in b)),
                 'INNER_SUBJECT_GROUPS_DISJOINT')
            pre=FoldPreprocessor();xx=pre.fit(X[a],C[a]);selected=top_training_correlations(xx,y[a])
            validation.append((xx[:,selected],pre.transform(X[b],C[b])[:,selected],y[a],y[b]))
        alpha_scores=[]
        for alpha in METHOD['alpha_grid']:
            fold_scores=[]
            for inner_fold,(xx,xv,yy,yv)in enumerate(validation):
                seed=(protocol['wrapper_numpy_seed']+outer*100+inner_fold)%2**32
                fitted=_fit_original_wrapper(ElasticNet,xx,yy,alpha,seed)
                pp=np.asarray(fitted.predict(xv),dtype=np.float64).reshape(-1)
                value=pearson(yv,pp)
                fold_scores.append(value)
            alpha_scores.append(None if any(value is None for value in fold_scores)else float(np.mean(fold_scores)))
        # Undefined validation Pearson invalidates that alpha, never becomes0.
        valid=[i for i,score in enumerate(alpha_scores)if score is not None]
        if not valid:raise UnscorableTarget('ALL_ALPHA_INNER_VALIDATION_PEARSON_UNDEFINED_NO_ZERO_SUBSTITUTION')
        # First grid value wins an exact finite tie.
        chosen=max(valid,key=lambda i:alpha_scores[i]);alpha=METHOD['alpha_grid'][chosen]
        pre=FoldPreprocessor();xx=pre.fit(X[train],C[train]);selected=top_training_correlations(xx,y[train])
        fitted=_fit_original_wrapper(ElasticNet,xx[:,selected],y[train],alpha,
                                     (protocol['wrapper_numpy_seed']+outer*100+99)%2**32)
        pp=np.asarray(fitted.predict(pre.transform(X[test],C[test])[:,selected]),dtype=np.float64).reshape(-1)
        need(len(pp)==len(test)and np.isfinite(pp).all(),'FINITE_ORIGINAL_WRAPPER_PREDICTIONS_IN_RAW_TARGET_UNITS')
        prediction[test]=pp;seen[test]+=1
        records.append(dict(outer_fold=outer,training_subjects=len(set(groups[i]for i in train)),
            validation_subjects=len(set(groups[i]for i in test)),chosen_alpha=alpha,
            alpha_mean_inner3_Pearson=alpha_scores,selected_predictors=len(selected),
            selected_original_columns=pre.columns[selected].tolist(),wrapper=_lambda_metadata(fitted)))
    need((seen==1).all()and np.isfinite(prediction).all(),'EVERY_ELIGIBLE_SUBJECT_HAS_ONE_TRUE_OUTER_OOF_PREDICTION')
    metric=pearson(y,prediction)
    if metric is None:raise UnscorableTarget('OOF_PEARSON_UNDEFINED_NO_ZERO_SUBSTITUTION')
    return prediction,metric,records


def shared_subject_folds(subject_ids, cv_scope):
    """Assign outer and inner folds before target missingness/arm availability."""
    import numpy as np
    ids = list(subject_ids)
    scope_mask = np.asarray(cv_scope)
    need(scope_mask.shape == (len(ids),) and scope_mask.dtype.kind == "b", "BOOLEAN_CV_SCOPE_REQUIRED")
    scope = np.flatnonzero(scope_mask)
    outer = np.full(len(ids), -1, dtype=np.int64)
    outer[scope] = subject_group_folds([ids[i] for i in scope], 5, 42)
    inner = {}
    for fold in range(5):
        training_scope = scope[outer[scope] != fold]
        labels = np.full(len(ids), -1, dtype=np.int64)
        labels[training_scope] = subject_group_folds([ids[i] for i in training_scope], 3, 42)
        inner[fold] = labels
    return outer, inner


def mean_raw_oof(predictions, expected, common_mask):
    """Require all expected modality OOFs; retain incomplete subjects as NaN."""
    import numpy as np
    names = list(predictions)
    need(bool(names) and set(names) == set(expected), "ONE_EXPECTED_MASK_PER_MODALITY_REQUIRED")
    common = np.asarray(common_mask)
    need(common.dtype.kind == "b" and common.ndim == 1, "BOOLEAN_COMMON_TARGET_MASK_REQUIRED")
    n = len(common)
    values, masks = [], []
    for name in names:
        row = np.asarray(predictions[name], dtype=np.float64)
        mask = np.asarray(expected[name])
        need(row.shape == mask.shape == (n,) and mask.dtype.kind == "b", "MATCHING_OOF_AND_EXPECTED_MASK_SHAPES")
        values.append(row); masks.append(mask)
    expected_count = np.sum(np.stack(masks), axis=0)
    available = np.stack([masks[i] & np.isfinite(values[i]) for i in range(len(names))])
    complete = common & (expected_count > 0) & (available.sum(axis=0) == expected_count)
    summed = np.sum(np.stack([np.where(available[i], values[i], 0.0) for i in range(len(names))]), axis=0)
    fused = np.full(n, np.nan)
    fused[complete] = summed[complete] / expected_count[complete]
    return fused, complete, expected_count


def fit_paired(subject_ids, target, baseline_features, feature_blocks, *,
               expected_availability, completed_availability,
               scientific_eligible=None, baseline_available=None, baseline_complete=None,
               outer_fold_ids=None, minimum_subjects=1001):
    """Fit one continuous target against a baseline and declared modality arms.

    All matrices use the same explicit participant row order. The CV cohort is
    scientific_eligible & baseline_available; target missingness is applied only
    after the shared folds are fixed. Missing expected computations remain
    pending rather than changing the evaluated cohort.
    """
    import numpy as np
    ids = list(subject_ids); n = len(ids)
    need(n > 0 and all(isinstance(v, str) and v for v in ids) and len(set(ids)) == n,
         "UNIQUE_EXPLICIT_CANONICAL_SUBJECT_IDS_REQUIRED")
    need(type(minimum_subjects) is int and minimum_subjects >= 1, "POSITIVE_MINIMUM_SUBJECT_COUNT_REQUIRED")
    y = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline_features)
    need(y.shape == (n,) and not np.isinf(y).any(), "ONE_CONTINUOUS_TARGET_PER_SUBJECT_NO_INFINITY")
    need(baseline.ndim == 2 and baseline.shape[0] == n and baseline.shape[1] > 0
         and baseline.dtype.kind in "fi", "NUMERIC_BASELINE_MATRIX_REQUIRED")

    def boolean(value, name):
        result = np.ones(n, dtype=bool) if value is None else np.asarray(value)
        need(result.shape == (n,) and result.dtype.kind == "b", name + "_MUST_BE_BOOLEAN_SUBJECT_MASK")
        return result

    scientific = boolean(scientific_eligible, "SCIENTIFIC_ELIGIBLE")
    baseline_expected = boolean(baseline_available, "BASELINE_AVAILABLE")
    baseline_done = boolean(baseline_complete, "BASELINE_COMPLETE")
    cv_scope = scientific & baseline_expected
    folds, shared_inner = shared_subject_folds(ids, cv_scope)
    if outer_fold_ids is not None:
        supplied = np.asarray(outer_fold_ids)
        need(supplied.shape == (n,) and supplied.dtype.kind in "iu" and np.array_equal(supplied, folds),
             "SUPPLIED_FOLDS_MUST_MATCH_SHARED_SHUFFLE42_BEFORE_TARGET_MISSINGNESS")
    names = list(feature_blocks)
    need(bool(names) and "baseline" not in names and all(isinstance(v, str) and v for v in names),
         "NONEMPTY_NAMED_FEATURE_MODALITIES_REQUIRED_BASELINE_IS_RESERVED")
    need(set(names) == set(expected_availability) == set(completed_availability),
         "EVERY_MODALITY_REQUIRES_INDEPENDENT_EXPECTED_AND_COMPLETED_MASKS")
    blocks = {"baseline": baseline}
    expected = {"baseline": baseline_expected}
    completed = {"baseline": baseline_done}
    for name in names:
        block = np.asarray(feature_blocks[name])
        need(block.ndim == 2 and block.shape[0] == n and block.shape[1] > 0 and block.dtype.kind in "fi",
             "MATCHING_NUMERIC_MODALITY_MATRIX_REQUIRED")
        blocks[name] = block
        need(expected_availability[name] is not None and completed_availability[name] is not None,
             "EXPLICIT_EXPECTED_AND_COMPLETED_MODALITY_MASKS_REQUIRED")
        expected[name] = boolean(expected_availability[name], "EXPECTED_" + name)
        completed[name] = boolean(completed_availability[name], "COMPLETED_" + name)
    common = cv_scope & np.isfinite(y)
    C = np.empty((n, 0), dtype=np.float64)
    predictions = {name: np.full(n, np.nan) for name in blocks}
    arms = {}; backend = None
    for name, X in blocks.items():
        eligible = common & expected[name]
        total = int(eligible.sum()); pending = int((eligible & ~completed[name]).sum())
        if pending:
            arms[name] = dict(state="PENDING_EXPECTED_FEATURES", subjects=total, pending_subjects=pending,
                              fit_performed=False, Pearson_OOF=None)
            continue
        if total < minimum_subjects:
            arms[name] = dict(state="NOT_ASSESSED_BELOW_MINIMUM_SUBJECTS", subjects=total,
                              fit_performed=False, Pearson_OOF=None)
            continue
        if backend is None:
            backend = _backend()
        indexes = np.flatnonzero(eligible)
        try:
            pp, metric, metadata = _outer_oof(X[indexes], y[indexes], C[indexes],
                [ids[i] for i in indexes], folds[indexes],
                {fold: values[indexes] for fold, values in shared_inner.items()},
                {"wrapper_numpy_seed": 42}, backend)
        except UnscorableTarget as error:
            arms[name] = dict(state="NOT_ASSESSED_UNSCORABLE_TARGET", subjects=total,
                              reason=str(error), fit_performed=True, Pearson_OOF=None)
            continue
        predictions[name][indexes] = pp
        arms[name] = dict(state="COMPLETE_OUTER_OOF", subjects=total, fit_performed=True,
                          Pearson_OOF=metric, outer_fold_metadata=metadata)
    fused, complete, expected_count = mean_raw_oof(
        {name: predictions[name] for name in names}, {name: expected[name] for name in names}, common)
    paired = complete & np.isfinite(predictions["baseline"])
    official = bool(common.any() and complete[common].all() and paired[common].all())
    scores = None
    if official:
        scores = dict(subjects=int(paired.sum()), baseline_Pearson_OOF=pearson(y[paired], predictions["baseline"][paired]),
                      feature_mean_Pearson_OOF=pearson(y[paired], fused[paired]))
    return dict(state="PAIRED_FIXED_COHORT_COMPLETE" if official else "PENDING_OR_UNSCORABLE_FIXED_COHORT",
        method=dict(METHOD, minimum_subjects=minimum_subjects), subject_ids=ids, target=y,
        cv_scope_mask=cv_scope, common_target_mask=common, outer_fold_ids=folds, shared_inner_fold_ids=shared_inner,
        predictions_by_arm=predictions, feature_mean_predictions=fused, expected_modality_count=expected_count,
        fusion_complete_mask=complete, paired_mask=paired, paired_comparison_complete=official,
        arms=arms, paired_scores=scores,
        counts=dict(input_subjects=n, CV_subjects=int(cv_scope.sum()), target_observed_subjects=int(common.sum()),
                    target_missing_in_CV=int((cv_scope & ~np.isfinite(y)).sum()),
                    pending_fusion_subjects=int((common & ~complete).sum())))


def summary(result):
    """Return scalar metrics/counts; row-level arrays remain with the caller."""
    return dict(state=result["state"], method=result["method"], counts=result["counts"],
        paired_comparison_complete=result["paired_comparison_complete"], paired_scores=result["paired_scores"],
        arms={name: {key: value for key, value in row.items() if key != "outer_fold_metadata"}
              for name, row in result["arms"].items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Caller-owned NPZ; see docs/paired_glmnet.md")
    parser.add_argument("--output-dir", required=True, help="New local output directory")
    parser.add_argument("--minimum-subjects", type=int, default=1001)
    args = parser.parse_args(argv)
    import numpy as np
    with np.load(args.input, allow_pickle=False) as data:
        names = [name.removeprefix("feature__") for name in data.files if name.startswith("feature__")]
        optional = {name: data[name] if name in data.files else None for name in
                    ("scientific_eligible", "baseline_available", "baseline_complete", "outer_fold_ids")}
        result = fit_paired(data["subject_ids"].tolist(), data["target"], data["baseline"],
            {name: data["feature__" + name] for name in names},
            expected_availability={name: data["expected__" + name] for name in names},
            completed_availability={name: data["completed__" + name] for name in names},
            minimum_subjects=args.minimum_subjects, **optional)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    arrays = dict(subject_ids=np.asarray(result["subject_ids"], dtype=str), target=result["target"],
        outer_fold_ids=result["outer_fold_ids"], common_target_mask=result["common_target_mask"],
        feature_mean_predictions=result["feature_mean_predictions"], paired_mask=result["paired_mask"])
    for name, values in result["predictions_by_arm"].items():
        arrays["oof__" + name] = values
    np.savez(destination / "private_oof.npz", **arrays)
    (destination / "summary.json").write_text(json.dumps(summary(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary(result), allow_nan=False))


if __name__ == "__main__":
    main()

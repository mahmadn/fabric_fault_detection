"""
train.py
--------
Dataset building, model training, and evaluation utilities.
"""

import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict)
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix,
                             precision_recall_curve)
import joblib

from src.features import tile_to_features, FEATURE_COLS


def build_dataset(normal_dir, fault_dirs, extensions=('jpg', 'jpeg', 'png')):
    """
    Build feature matrix from directory structure.

    Args:
        normal_dir: path to directory of normal tiles
        fault_dirs: list of paths to fault directories

    Returns:
        df: DataFrame with features, label, fault_type, path columns
    """
    rows = []

    # Normal tiles
    normal_paths = []
    for ext in extensions:
        normal_paths += glob.glob(
            os.path.join(normal_dir, '**', f'*.{ext}'), recursive=True
        )
    print(f'Processing {len(normal_paths)} NORMAL tiles...')
    for p in tqdm(normal_paths):
        f = tile_to_features(p)
        if f:
            rows.append({**f, 'label': 0, 'fault_type': 'normal', 'path': p})

    # Fault tiles
    for fault_dir in fault_dirs:
        fault_name  = os.path.basename(fault_dir)
        fault_paths = []
        for ext in extensions:
            fault_paths += glob.glob(
                os.path.join(fault_dir, '**', f'*.{ext}'), recursive=True
            )
        print(f'Processing {len(fault_paths)} FAULTY tiles [{fault_name}]...')
        for p in tqdm(fault_paths, desc=f'  {fault_name[:15]}'):
            f = tile_to_features(p)
            if f:
                rows.append({**f, 'label': 1, 'fault_type': fault_name, 'path': p})

    df = pd.DataFrame(rows).fillna(0)

    print(f'\nDataset summary:')
    print(f'  {"NORMAL":20s}: {(df.label==0).sum()}')
    for ft in df[df.label==1].fault_type.unique():
        print(f'  {ft:20s}: {(df.fault_type==ft).sum()}')
    print(f'  {"TOTAL FAULTY":20s}: {(df.label==1).sum()}')
    print(f'  {"TOTAL":20s}: {len(df)}')

    return df


def make_classifier(random_state=42):
    """SVM pipeline with StandardScaler. class_weight='balanced' handles imbalance."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=1.0, class_weight='balanced',
                       probability=True, random_state=random_state)),
    ])


def cross_validate_model(clf, X, y, n_splits=5, random_state=42):
    """
    5-fold stratified CV. Returns results dict and OOF probabilities.

    The train/test gap (train_score - test_score) is the key diagnostic:
      gap < 0.05: good generalisation
      gap > 0.15: overfitting — reduce features or add regularisation
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = cross_validate(
        clf, X, y, cv=cv,
        scoring=['f1', 'roc_auc', 'precision', 'recall'],
        return_train_score=True
    )

    print(f'\n{n_splits}-Fold Stratified CV Results')
    print('═' * 62)
    print(f'  {"Metric":12s}  {"Test":>16s}  {"Train":>8s}  {"Gap":>7s}')
    print(f'  {"─"*56}')
    for m in ['f1', 'roc_auc', 'precision', 'recall']:
        test  = results[f'test_{m}']
        train = results[f'train_{m}']
        gap   = train.mean() - test.mean()
        flag  = ' ⚠' if gap > 0.10 else ' ✓'
        print(f'  {m:12s}  {test.mean():.3f} ± {test.std():.3f}   '
              f'{train.mean():.3f}   {gap:+.3f}{flag}')
    print('═' * 62)

    oof_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return results, oof_proba


def tune_threshold(y, oof_proba, recall_target=0.90):
    """
    Find threshold that achieves recall >= recall_target while maximising precision.
    Returns best threshold and summary dict.
    """
    pr_arr, rec_arr, thr_arr = precision_recall_curve(y, oof_proba)
    valid = rec_arr[:-1] >= recall_target

    if not valid.any():
        print(f'⚠ Cannot reach recall={recall_target}. Using default threshold=0.50.')
        best_thresh = 0.50
    else:
        best_idx    = np.where(valid)[0][np.argmax(pr_arr[:-1][valid])]
        best_thresh = float(thr_arr[best_idx])

    pred_qc  = (oof_proba >= best_thresh).astype(int)
    pred_def = (oof_proba >= 0.50).astype(int)
    cm       = confusion_matrix(y, pred_qc)

    print(f'\n{"":22s}  {"Default (0.50)":>14s}  {"QC thresh":>10s}')
    print('─' * 52)
    for label, vd, vq in [
        ('F1',        f1_score(y, pred_def),        f1_score(y, pred_qc)),
        ('Precision', precision_score(y, pred_def), precision_score(y, pred_qc)),
        ('Recall',    recall_score(y, pred_def),    recall_score(y, pred_qc)),
    ]:
        print(f'  {label:20s}  {vd:>14.3f}  {vq:>10.3f}')
    print(f'\n  QC threshold = {best_thresh:.3f}')
    print(f'  Faults missed (FN) : {cm[1, 0]}')
    print(f'  False alarms  (FP) : {cm[0, 1]}')

    return best_thresh, {
        'threshold' : best_thresh,
        'f1_qc'     : f1_score(y, pred_qc),
        'recall_qc' : recall_score(y, pred_qc),
        'precision_qc': precision_score(y, pred_qc),
        'fn'        : int(cm[1, 0]),
        'fp'        : int(cm[0, 1]),
        'cm'        : cm,
    }


def save_model(clf, threshold, feature_cols, training_df, output_dir='model'):
    """Save all model artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf,          os.path.join(output_dir, 'fault_detector.pkl'))
    joblib.dump(threshold,    os.path.join(output_dir, 'qc_threshold.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'feature_cols.pkl'))
    training_df[['path', 'label', 'fault_type']].to_csv(
        os.path.join(output_dir, 'training_manifest.csv'), index=False
    )
    print(f'\nSaved to {output_dir}/')
    print(f'  fault_detector.pkl      — SVM pipeline')
    print(f'  qc_threshold.pkl        — {threshold:.3f}')
    print(f'  feature_cols.pkl        — {len(feature_cols)} features')
    print(f'  training_manifest.csv   — {len(training_df)} tiles')


def predict_tile(path, model_dir='model'):
    """
    Classify a single tile. Loads model from disk.

    Returns:
        dict with keys: is_faulty (bool), probability (float), features (dict)
        or {'error': str} if tile cannot be read.
    """
    import cv2
    from src.pipeline import apply_clahe, compute_residual

    clf       = joblib.load(os.path.join(model_dir, 'fault_detector.pkl'))
    threshold = joblib.load(os.path.join(model_dir, 'qc_threshold.pkl'))
    feat_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))

    bgr = cv2.imread(str(path))
    if bgr is None:
        return {'error': f'Cannot read {path}'}

    import cv2 as _cv2
    gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY).astype(np.float32)

    try:
        from src.features import extract_features
        feats = extract_features(compute_residual(apply_clahe(gray)))
        x     = np.array([[feats.get(c, 0.0) for c in feat_cols]])
        prob  = clf.predict_proba(x)[0, 1]
        return {
            'is_faulty'  : bool(prob >= threshold),
            'probability': float(prob),
            'features'   : feats,
        }
    except Exception as e:
        return {'error': str(e)}

import os, json, numpy as np, pandas as pd
from .config import CFG

def make_offline_submission(cache_dir: str, data_dir: str, folds=None, out_csv="submission.csv"):
    if folds is None: folds = list(range(CFG.n_folds))
    with open(os.path.join(cache_dir, "target_cols.json"), "w") as f:
        json.dump(CFG.target_cols, f)
    ss = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    preds = [np.load(os.path.join(cache_dir, f"pred_test_fold{f}.npy")) for f in folds]
    P = np.mean(preds, axis=0)
    sub = pd.DataFrame({"id": ss["id"].values})
    for i, c in enumerate(CFG.target_cols):
        if c in ss.columns:
            sub[c] = P[:, i]
    sub = sub[ss.columns]
    sub.to_csv(out_csv, index=False)
    return out_csv

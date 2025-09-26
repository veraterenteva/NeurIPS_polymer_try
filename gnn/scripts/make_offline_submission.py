import argparse, os, numpy as np, pandas as pd, json
from polygnn.config import CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=CFG.data_dir)
    ap.add_argument("--cache_dir", default=CFG.cache_dir)
    ap.add_argument("--folds", default=None, help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    if args.folds is None:
        folds = list(range(CFG.n_folds))
    else:
        folds = [int(x) for x in args.folds.split(",")]

    ss = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    preds = [np.load(os.path.join(args.cache_dir, f"pred_test_fold{f}.npy")) for f in folds]
    P = np.mean(preds, axis=0)

    sub = pd.DataFrame({"id": ss["id"].values})
    for i, c in enumerate(CFG.target_cols):
        if c in ss.columns:
            sub[c] = P[:, i]
    sub = sub[ss.columns]
    sub.to_csv(args.out, index=False)
    print("Saved:", args.out, sub.shape)

if __name__ == "__main__":
    main()

import argparse, os
from polygnn.config import CFG
from polygnn.utils import seed_everything, ensure_dir
from polygnn.data import load_frames, add_folds
from polygnn.train import kfold_train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=CFG.data_dir)
    ap.add_argument("--cache_dir", default=CFG.cache_dir)
    args = ap.parse_args()

    CFG.data_dir = args.data_dir
    CFG.cache_dir = args.cache_dir

    ensure_dir(CFG.cache_dir)
    seed_everything(CFG.seed)

    train_df, test_df, _ = load_frames(CFG.data_dir)
    train_df = add_folds(train_df, CFG.n_folds, CFG.seed)
    kfold_train(train_df, test_df)

if __name__ == "__main__":
    main()

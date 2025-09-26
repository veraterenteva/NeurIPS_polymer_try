import os
import pandas as pd
from sklearn.model_selection import KFold

def load_frames(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    ss    = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train, test, ss

def add_folds(train_df: pd.DataFrame, n_folds: int, seed: int = 42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = -1 * pd.Series([-1]*len(train_df), index=train_df.index)
    for f,(trn, val) in enumerate(kf.split(train_df)):
        folds.iloc[val] = f
    train_df = train_df.copy()
    train_df["fold"] = folds.values
    return train_df

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class SMILESVocab:
    def __init__(self, smiles_list):
        chars = sorted(set("".join(smiles_list)))
        self.vocab = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 = padding
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}

    def tokenize(self, smiles, max_len=120):
        tokens = [self.vocab.get(ch, 0) for ch in smiles]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

class PolymerDataset(Dataset):
    def __init__(self, df, vocab: SMILESVocab, targets, max_len=120):
        self.smiles = df["SMILES"].tolist()
        self.desc = df.drop(columns=["SMILES"] + targets).values.astype(np.float32)
        self.targets = df[targets].values.astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.vocab.tokenize(self.smiles[idx], self.max_len)
        x_smiles = torch.tensor(s, dtype=torch.long)
        x_desc = torch.tensor(self.desc[idx], dtype=torch.float)

        y = torch.tensor(self.targets[idx], dtype=torch.float)
        mask = ~torch.isnan(y)  # 1, если значение есть
        y[~mask] = 0.0          # заменим NaN на 0

        return x_smiles, x_desc, y, mask.float()
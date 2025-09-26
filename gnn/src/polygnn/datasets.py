import os, torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from typing import List, Optional
from .featurizer import smiles_to_graph

class PolymerGraphDataset(InMemoryDataset):
    def __init__(self, df, targets: Optional[List[str]], cache_path: Optional[str] = None):
        self.df = df.reset_index(drop=True).copy()
        self.targets = targets
        self.cache_path = cache_path
        super().__init__(None)
        if cache_path and os.path.exists(cache_path):
            self.data, self.slices = torch.load(cache_path, weights_only=False)
        else:
            data_list = []
            for i, row in self.df.iterrows():
                d: Data = smiles_to_graph(row["SMILES"])
                if targets is not None:
                    y = torch.tensor([row[t] if pd.notna(row[t]) else 0.0 for t in targets], dtype=torch.float)
                    mask = torch.tensor([1.0 if pd.notna(row[t]) else 0.0 for t in targets], dtype=torch.float)
                    d.y = y; d.mask = mask
                data_list.append(d)
            self.data, self.slices = self.collate(data_list)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save((self.data, self.slices), cache_path)

    def get_targets_tensor(self):
        assert self.targets is not None
        Y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        M = torch.stack([self.get(i).mask for i in range(len(self))], dim=0)
        return Y, M

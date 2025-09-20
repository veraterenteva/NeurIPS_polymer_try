import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

class HybridLSTMMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden,
                 desc_dim, mlp_hidden, output_dim=4):
        super(HybridLSTMMLP, self).__init__()

        # SMILES and LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        # Descriptors go to MLP
        self.desc_mlp = nn.Sequential(
            nn.Linear(desc_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + mlp_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, smiles_seq, desc):
        x = self.embedding(smiles_seq)        # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)            # h_n: (1, batch, lstm_hidden)
        h_smiles = h_n.squeeze(0)             # (batch, lstm_hidden)

        h_desc = self.desc_mlp(desc)          # (batch, mlp_hidden)

        h = torch.cat([h_smiles, h_desc], dim=1)
        out = self.fc(h)
        return out

class MaskedMSELoss(nn.Module):
    def forward(self, y_pred, y_true, mask):
        diff = (y_pred - y_true) ** 2
        diff = diff * mask
        return diff.sum() / mask.sum().clamp(min=1)
import numpy as np
import torch

class HybridTrainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for x_smiles, x_desc, y, mask in dataloader:
            x_smiles, x_desc, y, mask = (
                x_smiles.to(self.device),
                x_desc.to(self.device),
                y.to(self.device),
                mask.to(self.device)
            )
            self.optimizer.zero_grad()
            y_pred = self.model(x_smiles, x_desc)
            loss = self.criterion(y_pred, y, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_smiles, x_desc, y, mask in dataloader:
                x_smiles, x_desc, y, mask = (
                    x_smiles.to(self.device),
                    x_desc.to(self.device),
                    y.to(self.device),
                    mask.to(self.device)
                )
                y_pred = self.model(x_smiles, x_desc)
                loss = self.criterion(y_pred, y, mask)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def predict(self, dataloader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for x_smiles, x_desc, _, _ in dataloader:
                x_smiles, x_desc = (
                    x_smiles.to(self.device),
                    x_desc.to(self.device)
                )
                y_pred = self.model(x_smiles, x_desc)
                preds.append(y_pred.cpu().numpy())
        return np.vstack(preds)
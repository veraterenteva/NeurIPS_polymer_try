from LSTM_RF_Hybrid.model import MaskedMSELoss, HybridLSTMMLP
from LSTM_RF_Hybrid.preprocessing import PolymerDataset, SMILESVocab
from LSTM_RF_Hybrid.trainer import HybridTrainer
from LSTMmol.LSTMmol import SMILESLSTMTrainer
from data_loader import DataLoader, Visualizer
from descriptors import DescriptorCalculator
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from torch_molecule import LSTMMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# What I dislike: some EDA after descriptors should be performed
# We should merge train.csv with these ones to get some additional data:
# dataset1.csv - Tc data from the host’s older simulation results
# dataset2.csv - SMILES from this Tg table. We are only able to provide the list of SMILES.
# dataset3.csv - data from the host’s older simulation results
# dataset4.csv - data from the host’s older simulation results
# or somehow mask the lack of the data, how it's recommended

# TODO: model classes and OOP design for future maintenance
# TODO: make baseline solutions to the submission ready form
# TODO: more features from RDKit
# TODO: make it run and get the result!
# TODO: requirements.txt

class PolymerAnalysis:
    # Put full pipeline here

    def __init__(self, filepath: str):
        self.loader = DataLoader(filepath)
        self.descriptor_calc = DescriptorCalculator()
        self.visualizer = Visualizer()
        self.data = None
        self.data_with_desc = None

    def run(self):
        self.data = self.loader.load()
        print("Basic statistical info", self.data.describe())

        # Correlation (even though the dataset is informative, let's try)
        if "SMILES" in self.data.columns:
            self.visualizer.plot_smiles_length(self.data)
        self.visualizer.plot_correlation_heatmap(self.data)
        self.visualizer.plot_target_availability(self.data, ["Tg", "Tc", "Rg", "FFV"])

        # Descriptors calculation
        desc_df = self.descriptor_calc.compute_for_dataframe(self.data)
        self.data_with_desc = pd.concat(
            [self.data.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1
        )

        print(self.data_with_desc.head())
        return self.data_with_desc


if __name__ == "__main__":
    pipeline = PolymerAnalysis("train.csv")
    descripted_df = pipeline.run()
    print(descripted_df)

    nan_counts = descripted_df.isna().sum().sort_values(ascending=False)
    print(nan_counts)

    targets = ["Tg", "Tc", "Rg", "FFV"]
    descriptor_cols = [
        c for c in descripted_df.columns
        if c not in ["SMILES"] + targets
    ]

    # Опция 1: RANDOM FOREST MODEL
    # rf_model = RandomForestRegressor()
    # rf_model.fit(descripted_df)
    # # # Оценка
    # results = rf_model.evaluate(descripted_df)
    # print("Результаты:")
    # for target, metrics in results.items():
    #     print(f"{target}: MSE={metrics['MSE']:.3f}, R²={metrics['R2']:.3f}, samples={metrics['n_samples']}")
    #
    # # Предсказания на всём датасете
    # preds = rf_model.predict(descripted_df)
    # print("Предсказания:")
    # print(preds.head())
    # #
    # vocab = SMILESVocab(descripted_df["SMILES"].tolist())


    # Опция 2: ГИБРИДНАЯ МОДЕЛЬ LSTM И ДЕСКРИПТОРЫ
    # dataset = PolymerDataset(descripted_df, vocab, targets)
    # dataloader = TorchDataLoader(dataset, batch_size=64, shuffle=True)
    #
    # #  Модель
    # desc_dim = dataset[0][1].shape[0]
    # model = HybridLSTMMLP(
    #      vocab_size=len(vocab.vocab)+1,
    #      embed_dim=64,
    #      lstm_hidden=128,
    #      desc_dim=desc_dim,
    #      mlp_hidden=64,
    #      output_dim=len(targets)
    #  )
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = MaskedMSELoss()
    #
    # trainer = HybridTrainer(model, optimizer, criterion, device="cpu")
    #
    # #  Обучение
    # for epoch in range(5):
    #     train_loss = trainer.train_epoch(dataloader)
    #     val_loss = trainer.evaluate(dataloader)
    #     print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    #
    # #  Предсказания
    # preds = trainer.predict(dataloader)
    # print("Предсказания (shape):", preds.shape)

    # Опция 3: SMILES LSTM molecular predictor
    csv_path = 'train.csv'
    train_df = pd.read_csv(csv_path)

    trainer = SMILESLSTMTrainer("train.csv")

    dev_train, dev_val, dev_test = trainer.prepare_data()
    trainer.train(dev_train, dev_val)

    mse_per_task, mse_overall = trainer.evaluate(dev_test)

    print("MSE per task:")
    for name, mse in mse_per_task.items():
        print(f"  {name}: {mse:.4f}")
    print(f"Overall MSE: {mse_overall:.4f}")
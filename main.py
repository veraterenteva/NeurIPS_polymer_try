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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# What I dislike: some EDA after descriptors should be performed
# We should merge train.csv with these ones to get some additional data:
# dataset1.csv - Tc data from the host’s older simulation results
# dataset2.csv - SMILES from this Tg table. We are only able to provide the list of SMILES.
# dataset3.csv - data from the host’s older simulation results
# dataset4.csv - data from the host’s older simulation results
# or somehow mask the lack of the data, how it's recommended

# TODO: merge all datasets in one, like in full_pipeline_for_kaggle.py
# TODO: achieve all models results and ready to submit state
# TODO: more features from RDKit
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
    train_path =  r"C:\Users\darte\OneDrive\Desktop\projects\NeurIPS_polymer_try\train.csv"
    pipeline = PolymerAnalysis(train_path)
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
    from sklearn.impute import SimpleImputer

    X = descripted_df[descriptor_cols]
    y = descripted_df[targets]

    imputer = SimpleImputer(strategy="mean")
    y_imputed = pd.DataFrame(imputer.fit_transform(y), columns=targets)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X, y_imputed)
     # # Оценка
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred = rf_model.predict(X)

    results = {}
    for i, target in enumerate(targets):
        y_true = y_imputed[target]
        y_hat = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        mse = mean_squared_error(y_true, y_hat)
        mae = mean_absolute_error(y_true, y_hat)
        r2 = r2_score(y_true, y_hat)
        results[target] = {"MSE": mse, "MAE": mae, "R2": r2, "n_samples": len(y_true)}

    print("Результаты:")
    for target, metrics in results.items():
        print(f"{target}: MSE={metrics['MSE']:.3f}, "
          f"MAE={metrics['MAE']:.3f}, "
          f"R2={metrics['R2']:.3f}, "
          f"samples={metrics['n_samples']}")
    
     # Предсказания на всём датасете
    preds = rf_model.predict(X)
     # если предсказаний несколько (для всех target сразу)
    if preds.ndim > 1:
        preds = pd.DataFrame(preds, columns=targets, index=descripted_df.index)
    else:
        preds = pd.Series(preds, name=targets[0], index=descripted_df.index)
    print("Предсказания:")
    print(preds.head())
     #

    vocab = SMILESVocab(descripted_df["SMILES"].tolist())


    # Опция 2: ГИБРИДНАЯ МОДЕЛЬ LSTM И ДЕСКРИПТОРЫ
    #dataset = PolymerDataset(descripted_df, vocab, targets)
    #dataloader = TorchDataLoader(dataset, batch_size=64, shuffle=True)
    
    #  Модель
    #desc_dim = dataset[0][1].shape[0]
    #model = HybridLSTMMLP(
    #     vocab_size=len(vocab.vocab)+1,
    #     embed_dim=64,
    #     lstm_hidden=128,
    #     desc_dim=desc_dim,
    #     mlp_hidden=64,
    #     output_dim=len(targets)
    # )
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #criterion = MaskedMSELoss()
    
    #trainer = HybridTrainer(model, optimizer, criterion, device="cpu")
    
    #  Обучение
    #for epoch in range(5):
    #    train_loss = trainer.train_epoch(dataloader)
    #    val_loss = trainer.evaluate(dataloader)
    #    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    #  Предсказания
    #preds = trainer.predict(dataloader)
    #print("Предсказания (shape):", preds.shape)

    # Опция 3: SMILES LSTM molecular predictor
     
    #csv_path = r"C:\Users\darte\OneDrive\Desktop\projects\NeurIPS_polymer_try\train.csv"
    #train_df = pd.read_csv(csv_path)

    #trainer = SMILESLSTMTrainer(csv_path)

    #dev_train, dev_val, dev_test = trainer.prepare_data()
    #trainer.train(dev_train, dev_val)

    #mse_per_task, mse_overall = trainer.evaluate(dev_test)

    #print("MSE per task:")
    #for name, mse in mse_per_task.items():
    #    print(f"  {name}: {mse:.4f}")
    #print(f"Overall MSE: {mse_overall:.4f}")

    # ---------- Сохранение submission (используем descripted_df / train.csv) ----------
import numpy as np
import pandas as pd

# целевые колонки, которые ожидает соревнование / твоя модель
TARGET_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]

# preds — то, что вернул rf_model.predict(descripted_df)
# может быть DataFrame или numpy array
def preds_to_submission_df(preds, n_rows=None):
    # если DataFrame — попробуем извлечь имена колонок
    if isinstance(preds, pd.DataFrame):
        # Если DataFrame уже содержит нужные имена — возьмём их
        if all(c in preds.columns for c in TARGET_COLS):
            sub = preds[TARGET_COLS].copy()
            sub = sub.reset_index(drop=True)
            return sub
        # иначе возьмём numpy-представление
        arr = preds.to_numpy()
    else:
        arr = np.asarray(preds)

    # приводим к двумерному виду
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # если n_rows задан — проверим соответствие
    if n_rows is not None and arr.shape[0] != n_rows:
        # если количество предсказаний не совпадает с числом строк, попробуем предупредить
        raise RuntimeError(f"Число предсказаний ({arr.shape[0]}) != число строк ({n_rows})")

    # привести к 5 столбцам: обрезать или дополнять NaN
    if arr.shape[1] >= len(TARGET_COLS):
        arr = arr[:, :len(TARGET_COLS)]
    else:
        extra = np.full((arr.shape[0], len(TARGET_COLS) - arr.shape[1]), np.nan)
        arr = np.hstack([arr, extra])

    return pd.DataFrame(arr, columns=TARGET_COLS)

 # Для random Forest
# если preds — pandas DataFrame, используем его индекс для согласования,
# иначе n_rows = descr_df.shape[0]
try:
    submission_df = preds_to_submission_df(preds, n_rows=descripted_df.shape[0])
except Exception as e:

# на случай ошибки — печатаем детали
   print("Ошибка при подготовке submission:", e)
   raise

 # Для LSTM_RF_Hybrid 
#preds = trainer.predict(dataloader)
#submission_df = preds_to_submission_df(preds, n_rows=descripted_df.shape[0])

# (опционально) можно добавить id или SMILES в файл, если нужно
# submission_df.insert(0, "SMILES", descripted_df["SMILES"].values)

# сохранить (train-версию для Random Forest)
submission_df.to_csv("submission_train.csv", index=False, float_format="%.6f")
print("Готово: submission_train.csv сохранён. Первые строки:")

# сохранить (train-версию для LSTM_RF_Hybrid)
#submission_df.to_csv("submission_hybrid.csv", index=False, float_format="%.6f")
#print("Готово: submission_hybrid.csv сохранён. Первые строки:")
#print(submission_df.head())

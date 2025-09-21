import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch_molecule import LSTMMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

class SMILESLSTMTrainer:
    def __init__(self, csv_path, task_names=None, test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.task_names = task_names or ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def prepare_data(self):
        # split 0.2 test
        temp_df, dev_test = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )

        # split train/val = 0.6/0.2
        dev_train, dev_val = train_test_split(
            temp_df,
            test_size=0.25,
            random_state=self.random_state,
            shuffle=True
        )
        return dev_train, dev_val, dev_test

    def build_model(self):
        search_parameters = {
            "output_dim": ParameterSpec(ParameterType.INTEGER, (8, 32)),
            "LSTMunits": ParameterSpec(ParameterType.INTEGER, (30, 120)),
            "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
        }
        model = LSTMMolecularPredictor(
            task_type="regression",
            num_task=len(self.task_names),
            batch_size=192,
            epochs=200,
            verbose=True
        )
        return model, search_parameters

    def train(self, dev_train, dev_val):
        X_train = dev_train['SMILES'].to_list()
        y_train = dev_train[self.task_names].to_numpy()
        X_val = dev_val['SMILES'].to_list()
        y_val = dev_val[self.task_names].to_numpy()

        self.model, search_parameters = self.build_model()
        print("Model initialized successfully")

        self.model.autofit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_parameters=search_parameters,
            n_trials=10
        )

    def evaluate(self, dev_test):
        X_test = dev_test['SMILES'].to_list()
        y_test = dev_test[self.task_names].to_numpy()

        y_predict = self.model.predict(X_test)['prediction']

        # метрики по задаче
        mse_per_task = {}
        for i, name in enumerate(self.task_names):
            mask = ~np.isnan(y_test[:, i])
            if mask.sum() > 0:
                mse = mean_squared_error(y_test[mask, i], y_predict[mask, i])
                mse_per_task[name] = mse
            else:
                mse_per_task[name] = np.nan

        # общие метрики
        mask_all = ~np.isnan(y_test)
        y_true_flat = y_test[mask_all]
        y_pred_flat = y_predict[mask_all]
        mse_overall = mean_squared_error(y_true_flat, y_pred_flat)

        return mse_per_task, mse_overall

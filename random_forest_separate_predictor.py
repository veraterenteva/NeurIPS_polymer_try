import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# NB: Для реального submission, конечно, не тянет, но это эксперимент
# Задним числом выясняется, что даже авторы контеста используют её для бейзлайна

class RandomForestPredictor:
    # Это обёртка для обучения отдельных инстансов RandomForest по каждому таргету
    # (Tg, Tc, Rg, FFV), с учётом пропусков
    # Нам приходится идти на этот шаг, потому что в противном случае это просто неэффективно
    # какие-то строки содержат только Tg, какие-то пары
    # В качестве бейзлайна пробуем их предсказывать по дескрипторам без привязки друг к другу

    def __init__(self, targets=None, descriptor_cols=None, n_estimators=300, random_state=42):
        self.targets = targets or ["Tg", "Tc", "Rg", "FFV"]
        self.descriptor_cols = descriptor_cols  # список признаков (задаётся снаружи)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}  # словарь {target: model}

    def fit(self, df: pd.DataFrame):
        # Отдельный фит по колонкам

        if self.descriptor_cols is None:
            # все колонки, кроме SMILES и таргетов
            self.descriptor_cols = [c for c in df.columns if c not in ["SMILES"] + self.targets]

        for target in self.targets:
            df_target = df.dropna(subset=[target])
            if df_target.empty:
                print(f"Нет данных для {target}, пропускаем этот таргет для предсказания (в train.csv такого нет, но вдруг)")
                continue

            X = df_target[self.descriptor_cols].values
            y = df_target[target].values

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )

            model.fit(X, y)
            self.models[target] = model
            print(f"Обучена модель для {target} на основе {len(df_target)} записей")

    def evaluate(self, df: pd.DataFrame, test_size=0.2):
        # Смотрим качество по MSE и R2

        results = {}
        for target in self.targets:
            if target not in self.models:
                continue

            df_target = df.dropna(subset=[target])
            if df_target.empty:
                continue

            X = df_target[self.descriptor_cols].values
            y = df_target[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[target] = {"MSE": mse, "MAE": mae, "R2": r2, "n_samples": len(df_target)}

        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Предсказания по каждому таргету

        preds = pd.DataFrame(index=df.index)
        for target, model in self.models.items():
            mask = df[self.descriptor_cols].notna().all(axis=1)
            X = df.loc[mask, self.descriptor_cols].values
            y_pred = model.predict(X)
            preds.loc[mask, target] = y_pred
        return preds

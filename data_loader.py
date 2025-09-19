import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class DataLoader:
    # Handles loading and cleaning polymer dataset

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    def clean(self):
        if self.data is None:
            raise ValueError("Data not loaded")

        # Drop records without SMILES and duplicated records
        self.data = self.data.dropna(subset=["SMILES"]).drop_duplicates()

        # Get rid of occasional
        self.data.columns = self.data.columns.str.strip()
        print(f"After cleaning: {self.data.shape}")
        return self.data


class Visualizer:
    # Basic visualization for EDA

    @staticmethod
    def plot_smiles_length(df, smiles_col="SMILES"):
        df["smiles_length"] = df[smiles_col].str.len()
        sns.histplot(df["smiles_length"], bins=50, kde=False)
        plt.title("Distribution of SMILES Lengths")
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="viridis")
        plt.title("Correlation Heatmap")
        plt.show()
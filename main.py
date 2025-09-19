from data_loader import DataLoader, Visualizer
from descriptors import DescriptorCalculator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# What I dislike: some EDA after descriptors should be performed
# We should merge train.csv with these ones to get some additional data:
# dataset1.csv - Tc data from the host’s older simulation results
# dataset2.csv - SMILES from this Tg table. We are only able to provide the list of SMILES.
# dataset3.csv - data from the host’s older simulation results
# dataset4.csv - data from the host’s older simulation results
# or somehow mask the lack of the data, how it's recommended

# TODO: model classes and OOP design for future maintenance
# TODO: baseline solution (notebook example)
# TODO: more features from RDKit
# TODO: EDA?
# TODO: How to deal with the lack of data?
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
    available_targets = [t for t in targets if t in descripted_df.columns]


    # Features - everything except SMILES and target columns
    X = descripted_df.drop(columns=["SMILES"] + available_targets, errors="ignore")
    y = descripted_df[available_targets]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


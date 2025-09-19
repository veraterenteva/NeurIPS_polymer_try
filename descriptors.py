from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
import numpy as np
import pandas as pd

# Obviously, it should be noted that I don't know anything about the descriptors and their point,
# so I collected the rdkit.Chem descriptor modules and asked GPT about the most relevant ones to our task.
# It is still not enough for a good shot, since probably even all the list of possible descriptors is not enough
# The relevant features should be chosen, the additional ones should be calculated from scratch (according to the
# physical point of the task)

class DescriptorCalculator:
    # Handles RDKit descriptor calculations for SMILES

    def __init__(self):
        # Core 2D descriptors
        self.descriptor_funcs = {
            # Physicochemical
            'MolWt': Descriptors.MolWt,
            'ExactMolWt': rdMolDescriptors.CalcExactMolWt,
            'HeavyAtomCount': Descriptors.HeavyAtomCount,
            'MolLogP': Descriptors.MolLogP,
            'MolMR': Descriptors.MolMR,
            'TPSA': Descriptors.TPSA,
            'LabuteASA': rdMolDescriptors.CalcLabuteASA,

            # Flexibility
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'FractionCSP3': Descriptors.FractionCSP3,
            'RingCount': Descriptors.RingCount,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,

            # H-bonding
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumHDonors': Descriptors.NumHDonors,

            # Graph

            'BertzCT': Descriptors.BertzCT,
            'HallKierAlpha': Descriptors.HallKierAlpha,
            'Chi0': Descriptors.Chi0,
            'Chi1': Descriptors.Chi1,
            'Chi2n': Descriptors.Chi2n,
            'Chi3n': Descriptors.Chi3n,
            'Chi4n': Descriptors.Chi4n,
            'Kappa1': Descriptors.Kappa1,
            'Kappa2': Descriptors.Kappa2,
            'Kappa3': Descriptors.Kappa3,
            'BalabanJ': GraphDescriptors.BalabanJ,
        }

    def compute_for_smiles(self, smiles: str) -> dict:
        # Compute descriptors for a single SMILES

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: np.nan for name in self.descriptor_funcs} | {"NumChiralCenters": np.nan}

            results = {}
            for name, func in self.descriptor_funcs.items():
                try:
                    results[name] = func(mol)
                except Exception:
                    results[name] = np.nan

            # Stereo - chiral centers
            try:
                chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                results['NumChiralCenters'] = len(chiral_centers)
            except Exception:
                results['NumChiralCenters'] = np.nan

            return results

        except Exception:
            return {name: np.nan for name in self.descriptor_funcs} | {"NumChiralCenters": np.nan}

    def compute_for_dataframe(self, df, smiles_col="SMILES"):
        # Compute descriptors for the whole dataset

        desc_df = df[smiles_col].apply(self.compute_for_smiles).apply(pd.Series)
        return desc_df

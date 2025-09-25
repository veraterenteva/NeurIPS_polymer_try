from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors, AllChem
import numpy as np
import pandas as pd

# Дополнительные дескрипторы
from mordred import Calculator, descriptors as mordred_descriptors

class DescriptorCalculator:
    # Handles RDKit + selected Mordred descriptor calculations for SMILES

    def __init__(self, use_mordred=True, use_3D=True):
        # RDKit базовые дескрипторы
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

            # Graph / Topological
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

            # Geometric / 3D proxies
            'Asphericity': rdMolDescriptors.CalcAsphericity,
            'Eccentricity': rdMolDescriptors.CalcEccentricity,
            'InertialShapeFactor': rdMolDescriptors.CalcInertialShapeFactor,
            'RadiusOfGyration': rdMolDescriptors.CalcRadiusOfGyration,
            'SpherocityIndex': rdMolDescriptors.CalcSpherocityIndex,

            # Charge / electronic proxies
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge,
            'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge,
            'MaxPartialCharge': Descriptors.MaxPartialCharge,
            'MinPartialCharge': Descriptors.MinPartialCharge,

            # Autocorrelation / BCUT
            'BCUT2D_MWHI': Descriptors.BCUT2D_MWHI,
            'BCUT2D_MWLOW': Descriptors.BCUT2D_MWLOW,
            'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI,
            'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,

            # E-state indices
            'EState_VSA1': Descriptors.EState_VSA1,
            'EState_VSA2': Descriptors.EState_VSA2,
            'EState_VSA3': Descriptors.EState_VSA3,
            'EState_VSA4': Descriptors.EState_VSA4,

            # Fragment counts / functional groups
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
            'NumSaturatedRings': Descriptors.NumSaturatedRings,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'NumHalogenAtoms': Descriptors.NumHalogenAtoms,
        }

        # Mordred descriptors (выборка ~40 ключевых, без дублирующих RDKit)
        self.use_mordred = use_mordred
        self.use_3D = use_3D
        if self.use_mordred:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=not self.use_3D)

            # whitelist полезных Mordred-дескрипторов (убраны дубли TPSA и т.п.)
            self.mordred_whitelist = set([
                # WHIM / GETAWAY
                "WHIM.A","WHIM.E2u","WHIM.E3u","WHIM.E1v","WHIM.E2v","WHIM.E3v",
                "GETAWAY.R1u","GETAWAY.R2u","GETAWAY.R3u",
                "GETAWAY.HATS1u","GETAWAY.HATS2u","GETAWAY.HATS3u",
                # Volume / Surface (без TopoPSA, т.к. есть TPSA из RDKit)
                "ASA.MSA","ASA.MVSA","ASA.MV",
                # Autocorrelations (mass)
                "MATS1m","MATS2m","MATS3m","GATS1m","GATS2m","GATS3m","ATS1m","ATS2m","ATS3m",
                # RDF
                "RDF010m","RDF020m","RDF030m","RDF040m",
                # Autocorrelations (electronegativity)
                "GATS1e","GATS2e","GATS3e","MATS1e","MATS2e","MATS3e","ATS1e","ATS2e","ATS3e",
                # E-State / VSA
                "VE1sign_D","VE2sign_D","VE3sign_D","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4"
            ])

    def _prepare_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol and self.use_3D:
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=0)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
        return mol

    def compute_for_smiles(self, smiles: str) -> dict:
        try:
            mol = self._prepare_mol(smiles)
            if mol is None:
                return {name: np.nan for name in self.descriptor_funcs} | {"NumChiralCenters": np.nan}

            results = {}
            # RDKit descriptors
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

            # Mordred descriptors (только whitelist)
            if self.use_mordred:
                try:
                    mordred_values = self.mordred_calc(mol)
                    for desc, val in mordred_values.items():
                        if str(desc) in self.mordred_whitelist:
                            try:
                                results[str(desc)] = float(val)
                            except Exception:
                                results[str(desc)] = np.nan
                except Exception:
                    pass

            return results

        except Exception:
            return {name: np.nan for name in self.descriptor_funcs} | {"NumChiralCenters": np.nan}

    def compute_for_dataframe(self, df, smiles_col="SMILES"):
        desc_df = df[smiles_col].apply(self.compute_for_smiles).apply(pd.Series)
        return desc_df

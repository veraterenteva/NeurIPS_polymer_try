import torch
from torch_geometric.data import Data
from rdkit import Chem

ATOM_LIST = [1,6,7,8,9,14,15,16,17,35,53]
BOND_TYPES = {Chem.BondType.SINGLE:0, Chem.BondType.DOUBLE:1, Chem.BondType.TRIPLE:2, Chem.BondType.AROMATIC:3}

def atom_features(atom):
    Z = atom.GetAtomicNum()
    z_idx = ATOM_LIST.index(Z) if Z in ATOM_LIST else len(ATOM_LIST)
    vec = torch.zeros(len(ATOM_LIST)+1)
    vec[z_idx] = 1.0
    deg = atom.GetDegree()
    arom = float(atom.GetIsAromatic())
    return torch.cat([vec, torch.tensor([deg, arom], dtype=torch.float)])

def bond_features(bond):
    bt = BOND_TYPES.get(bond.GetBondType(), len(BOND_TYPES))
    vec = torch.zeros(len(BOND_TYPES)+1); vec[bt]=1.0
    conj = float(bond.GetIsConjugated())
    ring = float(bond.IsInRing())
    return torch.cat([vec, torch.tensor([conj, ring], dtype=torch.float)])

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Bad SMILES: {smiles}"
    Chem.Kekulize(mol, clearAromaticFlags=False)
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()], dim=0)  
    ei_src, ei_dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        ei_src += [i, j]; ei_dst += [j, i]
        eattr.append(bf); eattr.append(bf)
    edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
    edge_attr  = torch.stack(eattr, dim=0) if len(eattr)>0 else torch.zeros((0, len(BOND_TYPES)+1+2))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

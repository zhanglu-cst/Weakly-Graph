import dgl
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def mol_to_smile(mol):
    smiles = Chem.MolToSmiles(mol)
    try:
        smiles = rdMolStandardize.StandardizeSmiles(smiles)
    except Exception as e:
        pass
    return smiles


def get_single_bonds(mol):
    bond_infos = []
    bonds_ids = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                bond_idx = bond.GetIdx()
                beginatom = bond.GetBeginAtomIdx()
                endatom = bond.GetEndAtomIdx()
                bond_infos.append([bond_idx, beginatom, endatom])
                bonds_ids.append(bond_idx)
    return bond_infos, bonds_ids


def get_single_bonds_number(mol):
    if (isinstance(mol, str)):
        mol = get_mol(mol)
    ans = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE and not bond.IsInRing():
            ans += 1
    return ans


def get_atom_features(atom):
    # The usage of features is along with the Attentive FP.
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        # print("atom degree larger than 5. Please check before featurizing.")
        # print(atom)
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc + 24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31 + hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[36] = 1
        try:
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R', 'S']
            loc = ChiList.index(chi)
            feature[37 + loc] = 1
            # print("Chirality resolving finished.")
        except:
            feature[37] = 0
            feature[38] = 0
    return feature


def get_bond_features(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    BondTypeList = [rdkit.Chem.rdchem.BondType.SINGLE,
                    rdkit.Chem.rdchem.BondType.DOUBLE,
                    rdkit.Chem.rdchem.BondType.TRIPLE,
                    rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in BondTypeList:
        loc = BondTypeList.index(type)
        feature[0 + loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = conj

    # stereo
    stereo = bond.GetStereo()
    StereoList = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                  rdkit.Chem.rdchem.BondStereo.STEREOANY,
                  rdkit.Chem.rdchem.BondStereo.STEREOZ,
                  rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in StereoList:
        loc = StereoList.index(stereo)
        feature[6 + loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature


def build_dgl_graph_from_smile(smile):
    if (isinstance(smile, str)):
        mol = get_mol(smile)
    else:
        mol = smile
    if (mol is None):
        return None
    src = []
    dst = []
    atom_feature = []
    bond_feature = []
    try:
        for atom in mol.GetAtoms():
            one_atom_feature = get_atom_features(atom)
            atom_feature.append(one_atom_feature)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            one_bond_feature = get_bond_features(bond)
            src.append(i)
            dst.append(j)
            bond_feature.append(one_bond_feature)
            src.append(j)
            dst.append(i)
            bond_feature.append(one_bond_feature)
    except Exception as e:
        # print(str(e))
        return None

    src = torch.tensor(src).long()
    dst = torch.tensor(dst).long()
    atom_feature = torch.tensor(atom_feature).float()
    bond_feature = torch.tensor(bond_feature).float()
    graph_cur_smile = dgl.graph((src, dst), num_nodes = len(mol.GetAtoms()))
    graph_cur_smile.ndata['x'] = atom_feature
    graph_cur_smile.edata['x'] = bond_feature
    # node feature:[x, 39], edge feature:[y, 10]
    return graph_cur_smile


if __name__ == '__main__':
    smile = '[Cl-].N[C@@H]1C[C@H]1c2ccccc2.[H+]'
    build_dgl_graph_from_smile(smile)

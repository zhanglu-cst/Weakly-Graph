import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from extract_keysubgraph import Moleculars

m = Moleculars(task_name = 'BBBP')

smile_one = r'NC(N)=O'
mol = Chem.MolFromSmiles(smile_one)


def DrawMolGraph(mol, loc, name):
    AllChem.Compute2DCoords(mol)
    image_name = loc + name + '.png'
    Draw.MolToFile(mol, image_name)


# DrawMolGraph(mol,loc_number_subgraph = '',name = 'test')


def GetBondFeatures(bond):
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

print('atom')
for atom in mol.GetAtoms():
    print(atom.GetSymbol())
    print(atom.GetIdx())
    print()

print('bond')
for bond in mol.GetBonds():
    # GetBondFeatures(bond)
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    print(i,j)


def GetNeiList(mol):
    atomlist = mol.GetAtoms()
    TotalAtom = len(atomlist)
    NeiList = {}

    for atom in atomlist:
        atomIdx = atom.GetIdx()
        neighbors = atom.GetNeighbors()
        NeiList.update({"{}".format(atomIdx) : []})
        for nei in neighbors:
            neiIdx = nei.GetIdx()
            NeiList["{}".format(atomIdx)].append(neiIdx)

    return NeiList

def GetAdjMat(mol):
    # Get the adjacency Matrix of the given molecule graph
    # If one node i is connected with another node j, then the element aij in the matrix is 1; 0 for otherwise.
    # The type of the bond is not shown in this matrix.

    NeiList = GetNeiList(mol)
    TotalAtom = len(NeiList)
    AdjMat = np.zeros([TotalAtom, TotalAtom])

    for idx in range(TotalAtom):
        neighbors = NeiList["{}".format(idx)]
        for nei in neighbors:
            AdjMat[idx, nei] = 1

    return AdjMat

print('neight')
print(GetAdjMat(mol))

for one_smile_label in m.smiles_GT_labels:
    smile_one = one_smile_label[0]
    print(smile_one)
# mol = Chem.MolFromSmiles(smile_one)

#
#
#
#     break

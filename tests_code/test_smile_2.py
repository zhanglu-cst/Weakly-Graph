from rdkit import Chem

s = '[Cl-].N[C@@H]1C[C@H]1c2ccccc2.[H+]'

mol = Chem.MolFromSmiles(s)


for atom in mol.GetAtoms():
    print(atom.GetSymbol())
    print(atom.GetIdx())

for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    print(i,j)
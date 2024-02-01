import os

from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import RDConfig

fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
fparams.GetNumFuncGroups()


fcat = FragmentCatalog.FragCatalog(fparams)
fcgen = FragmentCatalog.FragCatGenerator()
m = Chem.MolFromSmiles('OCC=CC(=O)O')
fcgen.AddFragsFromMol(m, fcat)
num_entries = fcat.GetNumEntries()
print("matched the function group ids is", list(fcat.GetEntryFuncGroupIds(num_entries - 1)))
fg1 = fparams.GetFuncGroup(1)
fg34 = fparams.GetFuncGroup(34)
print("name of group 1 ", fg1.GetProp('_Name'))
print("name of group 34 ", fg34.GetProp('_Name'))

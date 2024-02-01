from rdkit import Chem
from rdkit.Chem import Draw

# from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules

task_name = 'bbbp'
keysubgraphs_each_class = [["O=CC1=CCS[C@@H]2CC(=O)N12"], ["O=C(c1ccc(O)cc1)c1ccccc1O"]]


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


for class_index, item_class in enumerate(keysubgraphs_each_class):
    cur_subgraph = item_class[0]
    filename = '{}_{}.png'.format(task_name, class_index)
    mol = get_mol(cur_subgraph)
    Draw.MolToFile(
            mol,  # mol对象
            filename,  # 图片存储地址
            size = (1000, 1000),
            kekulize = True,
            wedgeBonds = True,
            imageType = None,
            fitImage = False,
            options = None,
    )

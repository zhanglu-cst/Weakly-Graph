import os

import mmcv
from rdkit import Chem
from rdkit.Chem import Draw

# from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules


dir = r'/remote-home/zhanglu/weakly_molecular/work_dir/sider_Vascular disorders/baseline/'
path_keysubgraph = os.path.join(dir, 'key_subgraph_1.pkl')
keysubgraph = mmcv.load(path_keysubgraph)
keysubgraph = keysubgraph.subgraph_each_classes

task_name = 'sider'


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


for class_index, item_class in enumerate(keysubgraph):
    for index in range(0, 10):
        cur_subgraph = item_class[index]
        filename = '{}_class_{}_index{}.png'.format(task_name, class_index, index)
        mol = get_mol(cur_subgraph)
        if (mol is None):
            continue
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

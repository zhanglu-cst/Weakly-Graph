from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from compent.molecular_ops import get_mol, get_single_bonds, mol_to_smile

RDLogger.DisableLog('rdApp.*')


class Enumrate_Subgraphs_Mol():
    def __init__(self, max_frag_number = 6, max_return_subgraphs_number = 1000):
        super(Enumrate_Subgraphs_Mol, self).__init__()
        self.has_tokens = {}
        self.exist_subgraph = {}
        self.link_table = []
        self.number_frags = 0
        self.max_frag_number = max_frag_number
        self.max_return_subgraphs_number = max_return_subgraphs_number

    def clear(self):
        self.exist_subgraph = {}
        self.has_tokens = {}
        self.link_table = []
        self.number_frags = 0

    def add_to_subgraph_exist(self, nodes, links):
        # print(len(self.exist_subgraph), self.exist_subgraph)
        n_path = sorted(nodes)
        str_nodes = '_'.join([str(item) for item in n_path])
        if (str_nodes not in self.exist_subgraph):
            self.exist_subgraph[str_nodes] = [nodes.copy(), links.copy()]

    def __delete_remnant__(self, mol_list):
        ans = []
        for item in mol_list:
            item_deleted = Chem.DeleteSubstructs(item, Chem.MolFromSmiles('*'))
            ans.append(item_deleted)
        return ans

    def dfs(self, cur_nodes, link_nodes):
        if (len(self.exist_subgraph) > self.max_return_subgraphs_number):
            return
        self.add_to_subgraph_exist(cur_nodes, link_nodes)
        if (len(cur_nodes) > self.max_frag_number):
            return
        for node_id in range(self.number_frags):
            if (node_id not in cur_nodes and node_id > cur_nodes[-1]):
                flag = False
                for one_neighbor in self.link_table[node_id]:
                    if (one_neighbor in cur_nodes):
                        flag = True
                        link_nodes.append(one_neighbor)
                        break
                if (flag):
                    cur_nodes.append(node_id)
                    self.dfs(cur_nodes, link_nodes)
                    cur_nodes.pop()
                    link_nodes.pop()

    def __call__(self, smile, return_fragment_list = False):
        return self.do_token(smile, return_fragment_list)

    def do_token(self, smile, return_fragment_list):
        self.clear()

        mol = get_mol(smile)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        number_atoms_origin_mol = mol.GetNumAtoms()
        bond_info, single_bond_ids = get_single_bonds(mol)
        if (len(single_bond_ids) == 0):
            if (return_fragment_list == False):
                return smile
            else:
                return [smile], [smile], [get_mol(smile)]

        cuted = Chem.FragmentOnBonds(mol, single_bond_ids)
        frag_id_each_atom = []
        cuted_frag_list = Chem.GetMolFrags(cuted, asMols = True, frags = frag_id_each_atom)
        frag_id_each_atom = frag_id_each_atom[:number_atoms_origin_mol]
        cuted_frag_list = self.__delete_remnant__(cuted_frag_list)
        number_frags = len(cuted_frag_list)

        edges_of_frags = []
        frag_id_edge_cor_atom_id_connect = {}
        for one_bond in bond_info:
            begin_atom_id = one_bond[1]
            end_atom_id = one_bond[2]
            begin_frag_id = frag_id_each_atom[begin_atom_id]
            end_frag_id = frag_id_each_atom[end_atom_id]

            key_frags_connect = '{}_{}'.format(begin_frag_id, end_frag_id)
            frag_id_edge_cor_atom_id_connect[key_frags_connect] = [begin_atom_id, end_atom_id]

            key_frags_connect = '{}_{}'.format(end_frag_id, begin_frag_id)
            frag_id_edge_cor_atom_id_connect[key_frags_connect] = [end_atom_id, begin_atom_id]
            edges_of_frags.append([begin_frag_id, end_frag_id])

        # frag graph: node: number_frags,  edges: edges_of_frags,  edge_connect_atom: frag_id_edge_cor_atom_id_connect
        # dfs
        self.number_frags = number_frags
        # print('cur number_frags:{}'.format(number_frags))

        link_table = [[] for i in range(number_frags)]
        for one_edge in edges_of_frags:
            begin_node, end_node = one_edge
            link_table[begin_node].append(end_node)
            link_table[end_node].append(begin_node)
        self.link_table = link_table

        for start_node in range(number_frags):
            self.dfs(cur_nodes = [start_node], link_nodes = [])

        # build subgraph
        # all_subgraphs_mol = []
        all_subgraphs_smiles = []
        for key_subgraph, nodes_links_one_subgraph in self.exist_subgraph.items():
            nodes, links = nodes_links_one_subgraph
            assert len(nodes) == len(links) + 1
            subgraph = cuted_frag_list[nodes[0]]
            for index in range(1, len(nodes)):
                subgraph = Chem.CombineMols(subgraph, cuted_frag_list[nodes[index]])
            connect_atom_id_origin = []
            for i in range(len(links)):
                start_frag_id = nodes[i + 1]
                end_frag_id = links[i]
                key = '{}_{}'.format(start_frag_id, end_frag_id)
                atom_begin, atom_end = frag_id_edge_cor_atom_id_connect[key]
                connect_atom_id_origin.append([atom_begin, atom_end])
            # build map from origin_atom_id to new_id
            map_origin_to_new_id_atom = {}
            for atom in subgraph.GetAtoms():
                new_id = atom.GetIdx()
                origin_id = atom.GetAtomMapNum()
                map_origin_to_new_id_atom[origin_id] = new_id
            # do connect
            edcombo = Chem.EditableMol(subgraph)
            for one_bond in connect_atom_id_origin:
                atom_begin, atom_end = one_bond
                atom_id_new_begin = map_origin_to_new_id_atom[atom_begin]
                atom_id_new_end = map_origin_to_new_id_atom[atom_end]
                edcombo.AddBond(atom_id_new_begin, atom_id_new_end, order = Chem.rdchem.BondType.SINGLE)
            subgraph_mol = edcombo.GetMol()
            for atom in subgraph_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            subgraph_smile = Chem.MolToSmiles(subgraph_mol)
            try:
                subgraph_smile = rdMolStandardize.StandardizeSmiles(subgraph_smile)
            except Exception as e:
                # print(str(e))
                pass
            # subgraph_mol = Chem.MolFromSmiles(subgraph_smile)

            all_subgraphs_smiles.append(subgraph_smile)
            # all_subgraphs_mol.append(subgraph_mol)
        all_subgraphs_smiles = list(set(all_subgraphs_smiles))

        if (return_fragment_list == False):
            return all_subgraphs_smiles
        else:
            all_frags_str = []
            all_frags_mol = []
            for item_frag in cuted_frag_list:
                for atom in item_frag.GetAtoms():
                    atom.SetAtomMapNum(0)
                str_frag = mol_to_smile(item_frag)
                mol_frag = get_mol(str_frag)
                if (mol_frag is not None):
                    all_frags_str.append(str_frag)
                    all_frags_mol.append(mol_frag)
            return all_subgraphs_smiles, all_frags_str, all_frags_mol


if __name__ == '__main__':
    token_m = Enumrate_Subgraphs_Mol()
    # token_m('C(=C)OC=C')
    all_subgraphs = token_m.do_token('C1=C(C(=CC=C1C2CNC(C2)=O)OC)OC3CCCC3')
    print(all_subgraphs)
    print(len(all_subgraphs))

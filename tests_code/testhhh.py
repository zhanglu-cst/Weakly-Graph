from dataset import Moleculars

token = ['FCCC(F)C(F)C(F)CF', 'FCC(F)CC(F)C(F)CF', 'CC(F)C(F)C(F)C(F)CF']
token = set(token)

mols = Moleculars(filename = 'tox21_NR-ER.json', split = 'train', token_func = 'func_group')

# path = r'/apdcephfs/share_1364275/xluzhang/weakly_graph/data/token_results/func_group/tox21_NR-ER.json'

# with open(path, 'r') as f:
#     smile_to_token = json.load(f)

for item_key in token:
    print('start:{}'.format(item_key))
    for smile, tokens_cur in mols.smile_to_tokens.items():
        if (item_key in tokens_cur):
            label = mols.smile_to_label[smile]
            print(smile, label)

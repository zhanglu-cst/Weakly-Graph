import mmcv


path = r'/remote-home/zhanglu/weakly_molecular/data/tox21_SR-MMP/subgraphs.json'

obj = mmcv.load(path)

for item_key, item_value in obj.items():
    print(len(item_value))
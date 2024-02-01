import json

path = r'/apdcephfs/share_1364275/xluzhang/weakly_graph/data/token_results/func_group/tox21_NR-ER.json'
# tox21_NR-ER.json  clintox_CT_TOX
with open(path, 'r') as f:
    smile_to_token = json.load(f)

print('number smiles:{}'.format(len(smile_to_token)))


def count_token_times(global_dict):
    token_to_times = {}
    for item_smile, tokens_cur_smile in global_dict.items():
        tokens_cur_smile = set(tokens_cur_smile)
        for item_token in tokens_cur_smile:
            if (item_token not in token_to_times):
                token_to_times[item_token] = 0
            token_to_times[item_token] += 1
    print('len tokens:{}'.format(len(token_to_times)))
    times_to_counttoken = {}
    for item_token, item_times in token_to_times.items():
        if (item_times not in times_to_counttoken):
            times_to_counttoken[item_times] = 0
        times_to_counttoken[item_times] += 1
    times_to_counttoken = list(times_to_counttoken.items())
    times_to_counttoken = sorted(times_to_counttoken, key = lambda x: x[0], reverse = True)
    print('token_appear_times to number tokens:{}'.format(times_to_counttoken))
    times_to_counttoken = sorted(times_to_counttoken, key = lambda x: x[1], reverse = True)
    print('token_appear_times to number tokens:{}'.format(times_to_counttoken))


count_token_times(smile_to_token)

import json
import multiprocessing
import os

from mmcv import Config

from dataset.moleculars import Moleculars
from extract_keysubgraph import build_token_mol

cfg = Config.fromfile('config/bbbp_gin.py')
dataset_filename = cfg.data_filename
save_dir_root = r'data/token_results/func_group/'
split = 'test'
number_process = 32

token_func = build_token_mol(cfg = cfg.token)


def one_process(rank, queue_input, global_dict):
    print('rank:{}, start'.format(rank))
    while not queue_input.empty():
        item_smile = queue_input.get()
        print('rank:{}, process:{}'.format(rank, item_smile))
        token_res = token_func(item_smile)
        global_dict[item_smile] = token_res
        print('rank:{}, total finished:{}, left:{}'.format(rank, len(global_dict), queue_input.qsize()))
    print('rank:{} finish'.format(rank))


def count_token_times(global_dict):
    token_to_times = {}
    for item_smile, tokens_cur_smile in global_dict.items():
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
    print('token_appear_times to number tokens:{}'.format(times_to_counttoken))


def main():
    mols = Moleculars(filename = dataset_filename, split = split)
    global_dict_results = multiprocessing.Manager().dict()
    all_process = []
    queue_input = multiprocessing.Queue()
    for item_smile in mols.smiles:
        queue_input.put(item_smile)

    for index in range(number_process):
        item_p = multiprocessing.Process(target = one_process, args = (index, queue_input, global_dict_results))
        all_process.append(item_p)

    for item_p in all_process:
        item_p.start()
    for item_p in all_process:
        item_p.join()

    global_dict_results = dict(global_dict_results)
    save_dir = os.path.join(save_dir_root, split)
    if (os.path.exists(save_dir) == False):
        os.makedirs(save_dir)

    path_saving = os.path.join(save_dir, dataset_filename)
    with open(path_saving, 'w') as f:
        json.dump(global_dict_results, f)


if __name__ == '__main__':
    main()

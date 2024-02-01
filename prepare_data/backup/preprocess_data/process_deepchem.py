import json
import os

import deepchem as dc

print(dc.__version__)

all_dataset_names = ['sider', 'clintox', 'bbbp', 'hiv', 'tox21']
dir_saving = r'/apdcephfs/share_1364275/xluzhang/weakly_graph/data'


def process_one_split(one_split_data, task_index):
    lines_cur_split = []
    for item_smile, item_label in zip(one_split_data.ids, one_split_data.y):
        assert isinstance(item_smile, str)
        item_label = item_label.tolist()
        item_label = item_label[task_index]
        if (isinstance(item_label, float)):
            assert str(item_label).split('.')[1] == '0'
            item_label = int(item_label)
        assert isinstance(item_label, int), (item_label, type(item_label))
        lines_cur_split.append((item_smile, item_label))
    return lines_cur_split


def process_one_dataset(dataset_name):
    load_func_name = 'load_{}'.format(dataset_name)
    load_func = getattr(dc.molnet, load_func_name)
    tasks, datasets, transformers = load_func(featurizer = 'raw', splitter = 'random')
    train_dataset, valid_dataset, test_dataset = datasets
    print('dataset:{}, task number:{}'.format(dataset_name, len(tasks)))
    for task_index, item_task_name in enumerate(tasks):
        print('task index:{}, task name:{}'.format(task_index, item_task_name))
        top = {}
        train_lines = process_one_split(train_dataset, task_index)
        val_lines = process_one_split(valid_dataset, task_index)
        test_lines = process_one_split(test_dataset, task_index)
        print('len train:{}, len val:{}, len test:{}'.format(len(train_lines), len(val_lines), len(test_lines)))
        print('train samples smile:{} \n label:{}'.format(train_lines[0][0], train_lines[0][1]))
        print('val samples smile:{} \n label:{}'.format(val_lines[0][0], val_lines[0][1]))
        print('test samples smile:{} \n label:{}'.format(test_lines[0][0], test_lines[0][1]))
        top['train'] = train_lines
        top['val'] = val_lines
        top['test'] = test_lines
        if (len(tasks) == 1):
            filename = '{}.json'.format(dataset_name)
        else:
            filename = '{}_{}.json'.format(dataset_name, item_task_name)
        path_saving = os.path.join(dir_saving, filename)
        with open(path_saving, 'w') as f:
            json.dump(top, f)
        print('saving to:{}'.format(path_saving))


for item_dataset_name in all_dataset_names:
    process_one_dataset(item_dataset_name)

import os

import mmcv

task_name = 'sider_Vascular'
root_dir = r'/remote-home/zhanglu/weakly_molecular/data/dataset/'
target_root = r'/remote-home/zhanglu/weakly_molecular/data/'

target_dir = os.path.join(target_root, task_name)
if (os.path.exists(target_dir) == False):
    os.makedirs(target_dir)

filename = '{}.json'.format(task_name)
path = os.path.join(root_dir, filename)
print('path:{}'.format(path))

all_split = mmcv.load(path)

for item_split in all_split:
    target_split_path = os.path.join(target_dir, '{}.json'.format(item_split))
    data_cur = all_split[item_split]
    print('len_data:{}'.format(len(data_cur)))
    mmcv.dump(obj = data_cur, file = target_split_path)
    print('finish:{}'.format(item_split))

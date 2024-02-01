import os

import mmcv

DIR_SAVE = '/remote-home/zhanglu/weakly_molecular/data/syn_version5/ab_study'

path = os.path.join(DIR_SAVE, 'train.pkl')

train = mmcv.load(path)
print(train)

train100 = train[:100]
train200 = train[:200]
train300 = train[:300]

mmcv.dump(train100,file = '/remote-home/zhanglu/weakly_molecular/data/syn_version5/ab_study/train100.pkl')
mmcv.dump(train200,file = '/remote-home/zhanglu/weakly_molecular/data/syn_version5/ab_study/train200.pkl')
mmcv.dump(train300,file = '/remote-home/zhanglu/weakly_molecular/data/syn_version5/ab_study/train300.pkl')
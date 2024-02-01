import os

str = os.popen('ps -aux | grep python').read()

cfg_name = 'hiv'


def kill_one_line(line):
    items = line.split()
    print(items)
    ID = items[1]
    res = os.popen('kill -9 {}'.format(ID)).read()
    print('kill:{}'.format(ID))


lines = str.split('\n')
for line in lines:
    if (cfg_name in line):
        print(line)
        kill_one_line(line)

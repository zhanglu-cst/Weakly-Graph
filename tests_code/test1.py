import time

from func_timeout import func_set_timeout


@func_set_timeout(3)
def sss():
    time.sleep(10)


try:
    sss()
except:
    print('ss')

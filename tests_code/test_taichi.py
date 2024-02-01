import time

import taichi as ti

ti.init(arch = ti.gpu)


@ti.func
def judge(n):
    result = True
    for i in range(2, n):
        if (n % i == 0):
            result = False
            break
    return result


@ti.kernel
def count_prime():
    count = 0
    for i in range(3000000):
        if (judge(i)):
            count += 1
    print(count)


if __name__ == '__main__':
    s = time.time()
    count_prime()
    e = time.time()
    print(e - s)

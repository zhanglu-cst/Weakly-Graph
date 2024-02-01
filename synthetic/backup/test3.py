# for i in range(100):
#     n = get_random_value(mu = 2, sigma = 1, lower_bound = 1, upper_bound = 5)
#     print(n)
# import random

# x = random.randint(0,5)

import numpy

x = numpy.array([1, 10, 100, 10000, 0, 1])
print(numpy.argmax(x))

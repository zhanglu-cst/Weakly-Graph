from compent.utils import get_random_value

while True:
    number_classes = get_random_value(mu = 1.5, sigma = 0.7, lower_bound = 1, upper_bound = 2.9)
    print(number_classes)

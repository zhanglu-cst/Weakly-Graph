import multiprocessing


def func(queue_input, global_list):
    print(queue_input)
    # time.sleep(1)
    # print('args:{}, ok'.format(args))
    while not queue_input.empty():
        item = queue_input.get()
        global_list.append(item * item)


def main():
    queue_input = multiprocessing.Queue()
    for i in range(10):
        queue_input.put(i)

    all_process = []
    global_list = multiprocessing.Manager().list()
    for i in range(5):
        p = multiprocessing.Process(target = func, args = (queue_input, global_list))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()
    print(global_list)


if __name__ == '__main__':
    main()

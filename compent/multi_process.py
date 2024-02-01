import multiprocessing


def multi_process_main(data_list, func, other_params, number_process):
    queue_input = multiprocessing.Queue()
    for item in data_list:
        queue_input.put(item)
    all_process = []
    global_ans = multiprocessing.Manager().list()
    for i in range(5):
        p = multiprocessing.Process(target = func, args = (queue_input, global_ans))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()
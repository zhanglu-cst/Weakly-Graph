import multiprocessing

from compent.networkx_ops import judge_contain_a_subgraph


def one_process(rank, queue_input, global_ans, all_subgraphs):
    # print('rank start:{}'.format(rank))
    while not queue_input.empty():
        item_graph = queue_input.get()
        # print('cur graph node number:{}, edges:{}'.format(item_graph.number_of_nodes(), item_graph.number_of_edges()))
        for item_subgraph in all_subgraphs:
            if (judge_contain_a_subgraph(big_graph = item_graph, subgraph = item_subgraph)):
                global_ans.append(item_graph)
                break
        # print(
        # 'finish graph node number:{}, edges:{}'.format(item_graph.number_of_nodes(), item_graph.number_of_edges()))
        if (queue_input.qsize() % 10 == 0):
            pass
            # print('rank:{}, global len:{}, left:{}'.format(rank, len(global_ans), queue_input.qsize()))
    # print('rank:{}, finish'.format(rank))


def cal_coverage(all_subgraphs, all_bigraphs, number_of_process = 20):
    queue_input = multiprocessing.Queue()
    for item_graph in all_bigraphs:
        queue_input.put(item_graph)
    global_ans = multiprocessing.Manager().list()
    all_process = []
    for i in range(number_of_process):
        p = multiprocessing.Process(target = one_process,
                                    args = (i, queue_input, global_ans, all_subgraphs))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()
    hit_number = len(global_ans)
    hit_rate = hit_number / len(all_bigraphs)
    print('hit number:{}, rate:{}'.format(hit_number, hit_rate))
    return hit_rate

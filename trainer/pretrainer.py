import random

import dgl
import networkx as nx
import numpy
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from compent.global_var import Global_Var
from compent.networkx_ops import get_random_graph, combine_subgraph_to_big_graph, networkx_to_DGL
from compent.utils import get_random_value, move_to_device


class Dataset_Big_Graph_Pretrain(Dataset):
    def __init__(self, key_subgraph, syn_dataset, all_label_vector, number_classes):
        super(Dataset_Big_Graph_Pretrain, self).__init__()
        self.key_subgraph = key_subgraph
        self.number_classes = number_classes
        self.syn_dataset = syn_dataset
        self.logger = Global_Var.logger()

        # ----------- build ----------
        self.logger.info('building graph of keysubgraph')
        self.graph_of_key = nx.Graph()
        for index, item_subgraph in enumerate(key_subgraph):
            class_id = key_subgraph.subgraph_to_class[item_subgraph]
            self.graph_of_key.add_node(index, class_id = class_id)

        for item_label_vector in all_label_vector:
            connected_index = numpy.where(item_label_vector != -1)
            for s_index in range(len(connected_index)):
                s = connected_index[s_index].tolist()[0]
                for e_index in range(s_index + 1, len(connected_index)):
                    e = connected_index[e_index].tolist()[0]
                    if (self.graph_of_key.has_edge(s, e) == False):
                        self.graph_of_key.add_edge(s, e, count = 1.0)
                    else:
                        self.graph_of_key[s][e]['count'] += 1.0
                self.graph_of_key.add_edge(s, s, count = 1.0)

        di_graph = nx.DiGraph(self.graph_of_key)
        # print(di_graph)
        self.graph_key_dgl = dgl.from_networkx(di_graph, node_attrs = ['class_id'], edge_attrs = ['count'])

        self.estimated_parameters(all_label_vector)
        self.logger.info('finish building graph')
        # print('loc_number_subgraph:{}, std_number_subgraph:{}'.format(self.loc_number_subgraph, self.std_number_subgraph))

    def estimated_parameters(self, all_label_vector):
        number_each_sample = []
        for item_line in all_label_vector:
            cur_number = numpy.sum(item_line != -1)
            number_each_sample.append(cur_number)
        self.loc_number_subgraph = numpy.mean(number_each_sample)
        self.std_number_subgraph = numpy.std(number_each_sample)

        number_nodes_each_graph = []
        prob_edges_each_graph = []
        for item_graph in self.syn_dataset.graphs_networkx:
            number_nodes = item_graph.number_of_nodes()
            number_edges = item_graph.number_of_edges()
            number_nodes_each_graph.append(number_nodes)
            fully_connected_number = number_nodes * (number_nodes - 1) / 2
            edge_prob = number_edges / fully_connected_number
            prob_edges_each_graph.append(edge_prob)
        self.loc_number_nodes = numpy.mean(number_nodes_each_graph)
        self.std_number_nodes = numpy.std(number_nodes_each_graph)
        self.loc_prob_edges = numpy.mean(prob_edges_each_graph)
        self.std_prob_edges = numpy.std(prob_edges_each_graph)
        self.max_number_nodes = numpy.max(number_nodes_each_graph)

    def get_node_number(self):
        x = numpy.random.normal(loc = self.loc_number_subgraph, scale = self.std_number_subgraph)
        number = int(x)
        if (number <= 0):
            number = 1
        return number

    def graph_generation(self, subgraph_ids):
        all_subgraphs = []
        for item_subgraph_id in subgraph_ids:
            all_subgraphs.append(self.key_subgraph[item_subgraph_id])
        min_nodes_number = 0
        for item_subgraph in all_subgraphs:
            min_nodes_number += item_subgraph.number_of_nodes()
        number_nodes = get_random_value(mu = self.loc_number_nodes, sigma = self.std_number_nodes,
                                        lower_bound = min_nodes_number, upper_bound = self.max_number_nodes)
        number_nodes = int(number_nodes)
        prob_edges = get_random_value(mu = self.loc_prob_edges, sigma = self.std_prob_edges,
                                      lower_bound = 0.1, upper_bound = 1.0001)
        big_graph = get_random_graph(n = number_nodes, p = prob_edges, need_connect = True)
        big_graph, picked_nodes_each_subgraph = combine_subgraph_to_big_graph(big_graph, subgraphs = all_subgraphs,
                                                                              return_pick_nodes = True)
        return big_graph, picked_nodes_each_subgraph

    def generate_node_target(self, subgraph_indexes, generated_graph, picked_nodes_each_subgraph):
        target = numpy.full(generated_graph.number_of_nodes(), fill_value = self.number_classes)
        for item_subgraph_index, item_nodes_cur_subgraph in zip(subgraph_indexes, picked_nodes_each_subgraph):
            class_index_cur_subgraph = self.key_subgraph.index_to_class[item_subgraph_index]
            for item_node in item_nodes_cur_subgraph:
                target[item_node] = class_index_cur_subgraph
        return target

    def generate_graph_target(self, subgraph_indexes):
        target = numpy.zeros(self.number_classes)
        for item_subgraph_index in subgraph_indexes:
            class_index = self.key_subgraph.index_to_class[item_subgraph_index]
            target[class_index] = 1
        return target

    def get_ssl_train_samples(self):
        cur_class = random.randint(0, self.number_classes - 1)
        # print('get pretrain sample')
        all_keysubgraphs_cur_class = self.key_subgraph.subgraph_each_classes[cur_class]
        cur_subgraph = random.sample(all_keysubgraphs_cur_class, 1)[0]
        subgraph_start_index = self.key_subgraph.subgraph_to_index[cur_subgraph]
        # print('subgraph_start_index:{}'.format(subgraph_start_index))
        length = self.get_node_number()
        # print('length:{}'.format(length))
        g_traces, g_types = dgl.sampling.random_walk(self.graph_key_dgl, subgraph_start_index, length = length,
                                                     prob = 'count')
        sampled_subgraph_ids, concat_types, lengths, offsets = dgl.sampling.pack_traces(g_traces, g_types)
        sampled_subgraph_ids = list(set(sampled_subgraph_ids.tolist()))
        generated_graph, picked_nodes_each_subgraph = self.graph_generation(sampled_subgraph_ids)
        node_target = self.generate_node_target(sampled_subgraph_ids, generated_graph, picked_nodes_each_subgraph)
        graph_target = self.generate_graph_target(subgraph_indexes = sampled_subgraph_ids)
        generated_graph = networkx_to_DGL(graph_networkX = generated_graph)
        node_target = torch.LongTensor(node_target)
        graph_target = torch.FloatTensor(graph_target)
        # print('finish')
        return generated_graph, node_target, graph_target

    def __getitem__(self, item):
        return self.get_ssl_train_samples()

    def __len__(self):
        return 10000000


def collector_pretrain_DGL(batch):
    batch_graph, batch_node_labels, batch_graph_labels = map(list, zip(*batch))
    batch_graph = dgl.batch(batch_graph)
    batch_node_labels = torch.cat(batch_node_labels, dim = 0)
    batch_graph_labels = torch.stack(batch_graph_labels, dim = 0)
    return {'batch_graph': batch_graph, 'batch_node_labels': batch_node_labels,
            'batch_graph_labels': batch_graph_labels}


class SSL_Pretrainer():
    def __init__(self, key_subgraph, syn_dataset, all_label_vector, number_classes, cfg_dataloader, cfg_optimizer):
        super(SSL_Pretrainer, self).__init__()
        self.dataset_pretrain = Dataset_Big_Graph_Pretrain(key_subgraph, syn_dataset, all_label_vector, number_classes)
        self.dataloader_pretrain = DataLoader(self.dataset_pretrain, batch_size = cfg_dataloader.batch_size,
                                              shuffle = False,
                                              collate_fn = collector_pretrain_DGL,
                                              num_workers = cfg_dataloader.num_worker)
        self.cfg_optimizer = cfg_optimizer
        self.max_itr = cfg_dataloader.max_itr
        self.logger = Global_Var.logger()

    def pretrain_annotator(self, model_pretrain):
        self.logger.info('start pretrain classifier...', key = 'state')
        model_pretrain = model_pretrain.cuda()
        model_pretrain.train()

        optimizer = AdamW(params = model_pretrain.parameters(), lr = self.cfg_optimizer.lr)

        loss_record = []
        for itr, batch in enumerate(self.dataloader_pretrain):
            if (itr == self.max_itr):
                break
            batch = move_to_device(batch)
            loss = model_pretrain(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.append(loss.data.item())

            if (itr % 100 == 0):
                mean_loss = numpy.mean(loss_record)
                self.logger.dict({'SSL_Pretrain/ssl_loss': mean_loss, 'SSL_Pretrain/ssl_itr': itr})
                loss_record = []

        return model_pretrain.classifier

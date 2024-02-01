import numpy
import torch
from torch import nn
from torch.optim import AdamW

from compent.global_var import Global_Var


class LabelModel_Theta(nn.Module):
    def __init__(self, number_keys, M = 5, lamda1 = 0.00001):
        super(LabelModel_Theta, self).__init__()
        self.M = M
        self.number_keys = number_keys
        self.theta = torch.nn.Parameter(torch.zeros(self.number_keys), requires_grad = True)
        self.lamda1 = lamda1

    def parameters_sigmoid(self, x):
        return torch.sigmoid(x - self.M)

    def forward(self, Y):
        assert Y.dim() == 2
        beta = self.parameters_sigmoid(self.theta)
        alpha = 1 - beta
        Ds = []
        for item_sample_Y in Y:
            mask = item_sample_Y != -1
            if (torch.sum(mask) == 0):
                continue
            item_sample_Y = item_sample_Y[mask]
            cur_alpah = alpha[mask]
            mu = torch.sum(item_sample_Y * cur_alpah) / len(cur_alpah)
            temp = item_sample_Y * cur_alpah - mu
            temp = temp * temp
            di = torch.sum(temp) / len(cur_alpah)
            Ds.append(di)
        obj1 = sum(Ds)
        regular = self.theta * self.theta
        obj2 = torch.sum(regular)
        return obj1 + self.lamda1 * obj2

    def output_alpha(self):
        beta = self.parameters_sigmoid(self.theta)
        alpha = 1 - beta
        alpha = alpha.tolist()
        return alpha


class Divergence_Minimization():
    def __init__(self, number_keys_subgraphs, number_itr = 10000, lamda1 = 0.0001):
        super(Divergence_Minimization, self).__init__()
        self.number_itr = number_itr
        self.lamda1 = lamda1
        self.number_keys_subgraphs = number_keys_subgraphs
        self.logger = Global_Var.logger()

    def __call__(self, label_matrix):
        label_matrix = torch.tensor(label_matrix).long()
        labelmodel = LabelModel_Theta(number_keys = self.number_keys_subgraphs, lamda1 = self.lamda1)
        labelmodel.train()
        optimizer = AdamW(labelmodel.parameters(), lr = 0.01)
        labelmodel.train()
        # label_matrix = label_matrix.cuda()
        # labelmodel = labelmodel.cuda()
        all_loss = []
        loss_record = []
        self.logger.info('start training denoise model...')
        # s_time = time.time()
        for i in range(self.number_itr):
            loss = labelmodel(label_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.data.item())
            if (i % 10 == 0):
                mean_loss = numpy.mean(loss_record)
                self.logger.info({'Divergence_Minimization/dm_loss': mean_loss, 'Divergence_Minimization/dm_itr': i})
                loss_record = []
                # e_time = time.time()
                # print('time:{}'.format(e_time - s_time))
                all_loss.append(mean_loss)
                print('all_loss:{}'.format(all_loss))
        alpha = labelmodel.output_alpha()
        self.logger.info('learn alpha:{}'.format(alpha))

        return alpha


if __name__ == '__main__':
    # Y = [[0, 0, -1, 1],
    #      [-1, 0, 0, 1],
    #      [0, 0, 0, 1],
    #      [-1, 0, 0, 1],
    #      [0, 1, 0, 1],
    #      [-1, 0, 0, 1]]
    import mmcv

    path_matrix = r'/remote-home/zhanglu/weakly_molecular/work_dir/syn_version5/full_gogogo/label_matrix_itr_1.pkl'
    matrix = mmcv.load(path_matrix)
    matrix = matrix[1]
    # print(matrix)
    labeler = Divergence_Minimization(number_keys_subgraphs = 33, number_itr = 10000, lamda1 = 0.00001)
    labeler(matrix)

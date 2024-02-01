import numpy
import torch
from torch import nn
from torch.optim import AdamW

from compent.global_var import Global_Var


class LabelModel_Theta(nn.Module):
    def __init__(self, number_keys, M = 5, lamda1 = 0.0001):
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
    def __init__(self, number_keys_subgraphs, cfg_dm):
        super(Divergence_Minimization, self).__init__()
        self.number_itr = cfg_dm.number_itr
        self.lamda1 = cfg_dm.lamda1
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
        loss_record = []
        self.logger.info('start training denoise model...', key = 'state')
        for i in range(self.number_itr):
            loss = labelmodel(label_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.data.item())
            if (i % 100 == 0):
                mean_loss = numpy.mean(loss_record)
                self.logger.dict({'Divergence_Minimization/dm_loss': mean_loss, 'Divergence_Minimization/dm_itr': i})
                loss_record = []
        alpha = labelmodel.output_alpha()
        self.logger.info('learn alpha:{}'.format(alpha), key = 'learn_alpha')
        return alpha


if __name__ == '__main__':
    Y = [[0, 0, -1, 1],
         [-1, 0, 0, 1],
         [0, 0, 0, 1],
         [-1, 0, 0, 1],
         [0, 1, 0, 1],
         [-1, 0, 0, 1]]
    labeler = Divergence_Minimization(number_keys_subgraphs = 4, number_itr = 5000)
    labeler(Y)

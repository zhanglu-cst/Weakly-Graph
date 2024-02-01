import numpy
from sklearn.metrics import recall_score, precision_score, f1_score

from dataset import Moleculars


def ana_one_key_posi(key, mol_dataset: Moleculars):
    all_assigned_smile = []
    all_assigned_label = []
    for item_smile in mol_dataset.smiles:
        tokens_cur = mol_dataset.smile_to_tokens[item_smile]
        label_cur = mol_dataset.smile_to_label[item_smile]
        if (key in tokens_cur):
            all_assigned_smile.append(item_smile)
            all_assigned_label.append(label_cur)
    # print(all_assigned_smile)
    # print(all_assigned_label)
    all_assigned_label = numpy.array(all_assigned_label)
    num_positive = numpy.sum(all_assigned_label == 1)
    num_negative = numpy.sum(all_assigned_label == 0)
    # print('number 0:{}, 1:{}'.format(num_negative, num_positive))
    positive_rate = num_positive / (num_positive + num_negative)
    # print('positive rate:{}'.format(positive_rate))
    result = {'key': key, 'positive_rate': positive_rate, 'num_positive': num_positive,
              'all_assigned_smile': all_assigned_smile, 'all_assigned_label': all_assigned_label}
    return result


class LF_Analyze_for_One():
    def __init__(self, class_index, mol_dataset: Moleculars):
        self.class_index = class_index
        self.mol_dataset = mol_dataset
        labels = []
        for item_label in mol_dataset.labels:
            if (item_label == class_index):
                labels.append(1)
            else:
                labels.append(0)
        self.labels = numpy.array(labels)

    def __call__(self, key_token):
        pred_label = []
        find_GT_true_samples = []
        find_all_samples = []
        for index, item_smile in enumerate(self.mol_dataset.smiles):
            tokens_cur_smile = self.mol_dataset.smile_to_tokens[item_smile]
            if (key_token in tokens_cur_smile):
                pred_label.append(1)
                if (self.labels[index] == 1):
                    find_GT_true_samples.append(item_smile)
                find_all_samples.append(item_smile)
            else:
                pred_label.append(0)
        pred_label = numpy.array(pred_label)
        recall = recall_score(y_true = self.labels, y_pred = pred_label)
        precision = precision_score(y_true = self.labels, y_pred = pred_label)
        f1 = f1_score(y_true = self.labels, y_pred = pred_label)
        results = {'key_token': key_token, 'recall': recall, 'precision': precision, 'f1': f1,
                   'find_GT_true_samples': find_GT_true_samples, 'find_all_samples': find_all_samples}
        return results

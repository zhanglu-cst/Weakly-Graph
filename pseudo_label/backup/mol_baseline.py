import numpy
from sklearn.metrics import classification_report, f1_score

from dataset.moleculars import Moleculars


class Assign_Label_Contain_Number_Only_Positive():
    def __init__(self, positive_index = 1, score_thr = 0):
        self.positive_index = positive_index
        self.score_thr = score_thr

    def get_score_with_all_tokens(self, tokens_one_smile, key_token_to_score):
        sum_score = 0
        for item_token in tokens_one_smile:
            if (item_token in key_token_to_score):
                sum_score += key_token_to_score[item_token]
        return sum_score

    def __call__(self, key_tokens_each_class, mol_dataset: Moleculars):
        assert mol_dataset.number_classes == 2
        print('dataset len:{}'.format(mol_dataset))
        print('label count:{}'.format(mol_dataset.label_count_each_class))

        postive_key_tokens = key_tokens_each_class[self.positive_index]
        key_token_to_score = {}
        for item_token, item_score in postive_key_tokens:
            key_token_to_score[item_token] = item_score

        postive_scores = []
        for item_smile in mol_dataset.smiles:
            tokens_cur_smile = mol_dataset.smile_to_tokens[item_smile]
            score_cur_smile = self.get_score_with_all_tokens(tokens_cur_smile, key_token_to_score)
            postive_scores.append(score_cur_smile)
        postive_scores = numpy.array(postive_scores)
        print('pred score > 0 :{}'.format(numpy.sum(postive_scores > 0)))
        positive_pred = postive_scores > self.score_thr
        print('after thr, positive:{}'.format(numpy.sum(positive_pred > 0)))
        print(classification_report(y_true = mol_dataset.labels, y_pred = positive_pred))
        f1_micro = f1_score(y_true = mol_dataset.labels, y_pred = positive_pred)
        print('f1_micro:{}'.format(f1_micro))
        return f1_micro

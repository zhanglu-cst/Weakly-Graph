import numpy

from .builder import UPDETAR


@UPDETAR.register_module()
class Updater_TF_IDF_Mol():
    def __init__(self, IDF_power = 3, number_classes = 2, keep_tokens = 200):
        super(Updater_TF_IDF_Mol, self).__init__()
        self.IDF_power = IDF_power
        self.number_classes = number_classes
        self.keep_tokens = keep_tokens

    def build_all_tokens_list(self, smile_to_tokens):
        all_tokens = set()
        for smile, tokens_cur_smile in smile_to_tokens.items():
            all_tokens.update(tokens_cur_smile)
        all_tokens = list(all_tokens)
        token_to_index = {}
        for index, token_item in enumerate(all_tokens):
            token_to_index[token_item] = index
        return all_tokens, token_to_index

    def cal_IDF(self, smile_to_tokens, all_tokens, token_to_index):
        number_mol = len(smile_to_tokens)
        doc_frequency = numpy.zeros(len(all_tokens))  # shape[num_word]
        for one_mol in smile_to_tokens:
            tokens_cur_mol = smile_to_tokens[one_mol]
            tokens_cur_mol = set(tokens_cur_mol)
            for one_token in tokens_cur_mol:
                index_token = token_to_index[one_token]
                doc_frequency[index_token] += 1
        tmp = number_mol / doc_frequency
        IDF = numpy.log(tmp)
        return IDF

    def get_token_number_each_class(self, smile_to_tokens, smiles, labels):
        token_number_each_class = numpy.zeros(self.number_classes)
        for item_smile, item_label in zip(smiles, labels):
            tokens_cur_smile = smile_to_tokens[item_smile]
            token_number_each_class[item_label] += len(tokens_cur_smile)
        return token_number_each_class

    def cal_TF(self, smile_to_tokens, smiles, labels, tokens, token_to_index):
        frequency_each_class_each_token = numpy.zeros((self.number_classes, len(tokens)))
        token_number_each_class = self.get_token_number_each_class(smile_to_tokens, smiles, labels)
        token_number_each_class = token_number_each_class.reshape(-1, 1)
        token_number_each_class = token_number_each_class + 1  # TODO: can be changed to number of smile number each class
        for item_smile, item_label in zip(smiles, labels):
            tokens_cur_smile = smile_to_tokens[item_smile]
            for item_token in tokens_cur_smile:
                token_index = token_to_index[item_token]
                frequency_each_class_each_token[item_label][token_index] += 1
        TF = frequency_each_class_each_token / token_number_each_class
        # print('TF:{}'.format(TF))
        # TF = numpy.tanh(TF) # TODO: rethinking if need tanh
        return TF

    def __call__(self, smiles, labels, smile_to_tokens):
        return self.get_key_tokens(smiles, labels, smile_to_tokens)

    def get_key_tokens(self, smiles, labels, smile_to_tokens):

        all_tokens, token_to_index = self.build_all_tokens_list(smile_to_tokens)
        IDF_vector = self.cal_IDF(smile_to_tokens, all_tokens, token_to_index)
        TF = self.cal_TF(smile_to_tokens, smiles, labels, all_tokens, token_to_index)
        indicator = TF * numpy.power(IDF_vector, self.IDF_power)

        classes_per_keywords = numpy.argmax(indicator, axis = 0)
        cor_max_scores = numpy.max(indicator, axis = 0)
        class_to_words = {}
        for class_index in range(self.number_classes):
            class_to_words[class_index] = []
        for key_token, score, class_index in zip(all_tokens, cor_max_scores, classes_per_keywords):
            class_to_words[class_index].append([key_token, score])
        filtered_class_to_words = []
        for class_index in range(self.number_classes):
            tokens_cur_class = class_to_words[class_index]
            sort_pair = sorted(tokens_cur_class, key = lambda x: x[1], reverse = True)
            sort_pair = sort_pair[:self.keep_tokens]
            filtered_class_to_words.append(sort_pair)

        # print(filtered_class_to_words)
        # filtered_class_to_words: list, item[0]-> class_0 list: item:[key_token, score]
        return filtered_class_to_words

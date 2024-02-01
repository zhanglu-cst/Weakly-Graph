import numpy


def conbina_score_label_matrix(final_score_each_subgraph, label_matric, number_classes):
    final_labels = []
    for label_vector_one_sample in label_matric:
        class_distribution = numpy.zeros(number_classes)
        for item_score, item_class_index in zip(final_score_each_subgraph, label_vector_one_sample):
            if (item_class_index == -1):
                continue
            class_distribution[item_class_index] += item_score
        class_pred = numpy.argmax(class_distribution)
        final_labels.append(class_pred)
    return final_labels

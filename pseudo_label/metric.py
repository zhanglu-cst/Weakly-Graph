import numpy
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score, \
    roc_auc_score


def cal_metric_binary_for_pseudo(all_pseudo_labels, all_gt_labels, mol_dataset, pre_key = ''):
    acc = accuracy_score(y_true = all_gt_labels, y_pred = all_pseudo_labels)
    recall = recall_score(y_true = all_gt_labels, y_pred = all_pseudo_labels)
    precision = precision_score(y_true = all_gt_labels, y_pred = all_pseudo_labels)
    f1 = f1_score(y_true = all_gt_labels, y_pred = all_pseudo_labels)
    print(classification_report(y_true = all_gt_labels, y_pred = all_pseudo_labels))
    hit_number = len(all_pseudo_labels)
    hit_rate = hit_number / len(mol_dataset)
    positive_pseudo_number = numpy.sum(numpy.array(all_pseudo_labels) == 1)
    gt_positve_in_pseudo = 0
    for item_pseudo, item_gt in zip(all_pseudo_labels, all_gt_labels):
        if (item_pseudo == 1 and item_gt == 1):
            gt_positve_in_pseudo += 1
    result = {'{}/acc'.format(pre_key): acc,
              '{}/recall'.format(pre_key): recall,
              '{}/precision'.format(pre_key): precision,
              '{}/f1'.format(pre_key): f1,
              '{}/hit_rate_total'.format(pre_key): hit_rate,
              '{}/hit_number_total'.format(pre_key): hit_number,
              '{}/positive_number'.format(pre_key): positive_pseudo_number,
              '{}/gt_positve_in'.format(pre_key): gt_positve_in_pseudo}
    return result


def cal_metric_binary_class(all_pred_scores, all_gt_labels, thr = 0.5):
    all_pred = all_pred_scores > thr
    acc = accuracy_score(y_true = all_gt_labels, y_pred = all_pred)
    recall = recall_score(y_true = all_gt_labels, y_pred = all_pred)
    precision = precision_score(y_true = all_gt_labels, y_pred = all_pred)
    f1 = f1_score(y_true = all_gt_labels, y_pred = all_pred)
    auc = roc_auc_score(y_true = all_gt_labels, y_score = all_pred_scores)
    print(classification_report(y_true = all_gt_labels, y_pred = all_pred))
    result = {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'auc': auc}
    return result


def cal_metric_multi_class(all_pred_class, all_gt_labels):
    acc = accuracy_score(y_true = all_gt_labels, y_pred = all_pred_class)
    f1_macro = f1_score(y_true = all_gt_labels, y_pred = all_pred_class, average = 'macro')
    f1_micro = f1_score(y_true = all_gt_labels, y_pred = all_pred_class, average = 'micro')
    result = {'acc': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
    return result

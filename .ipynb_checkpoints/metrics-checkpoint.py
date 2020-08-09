#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:38:24 2019

@author: zhouhonglu
"""
import numpy as np
import pdb

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def compute_AP(actual, predicted):
    k = len(actual)

    if len(predicted) > k:
        predicted = predicted[:k]

    actual = actual[1:]  # the first is knwon source
    predicted = predicted[1:]  # the first is knwon source

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def compute_MAP(actual, predicted):
    return np.mean([compute_AP(a, p) for a, p in zip(actual, predicted)])


def compute_Patk_each_sample(actual, predicted, k):

    predicted = predicted[1:k+1]  # the first is knwon source

    num_hits = 0.0
    score = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            # print(num_hits, i + 1, k)
            if i + 1 >= k:
                score = num_hits / k
                break

    if not actual:
        return 0.0

    return score


def compute_Patk(actual, predicted, k):
    return np.mean([compute_Patk_each_sample(a, p, k) for a, p in zip(actual, predicted)])


def compute_precision_recall_f1_auc(actual, predicted, uid_uname, uid_uname_reverse):
    total_numusers = len(uid_uname)

    avg_f1 = 0
    avg_auc = 0
    avg_precision = 0
    avg_recall = 0
    avg_acc = 0

    for i in range(len(actual)):
        actual_i = actual[i][1:]
        predicted_i = predicted[i][1:]

        y_true = np.zeros(total_numusers)
        y_pred = np.zeros(total_numusers)

        for j in range(len(actual_i)):
            y_true[uid_uname_reverse[actual_i[j]]] = 1
            y_pred[uid_uname_reverse[predicted_i[j]]] = 1

        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)
        # pdb.set_trace()

        avg_f1 += f1
        avg_auc += auc
        avg_precision += precision
        avg_recall += recall
        avg_acc += acc

    avg_f1 = avg_f1/len(actual)
    avg_auc = avg_auc/len(actual)
    avg_precision = avg_precision/len(actual)
    avg_recall = avg_recall/len(actual)
    avg_acc = avg_acc/len(actual)

    return (avg_precision, avg_recall, avg_f1, avg_auc, avg_acc)


if __name__ == "__main__":
    actual = [185, 905, 1335, 1336, 1334, 187, 908, 906, 909, 907, 904]
    predicted = [185, 905, 187, 1335, 906, 1334, 909, 1333, 907, 908, 910]
    print(compute_Patk(actual, predicted, k=2))

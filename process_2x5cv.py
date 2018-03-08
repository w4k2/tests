#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import csv

np.random.seed(1337)

from sklearn import model_selection, metrics

from tqdm import tqdm

repetitions = 10
datasets = h.datasets()
clfs = h.classifiers()

for dataset in datasets:
    print dataset
    # Gather dataset
    ds = pd.read_csv(dataset[0], header=None).as_matrix()
    X, y = ds[:,:-1], ds[:,-1].astype('int')

    # CV
    for repetition in xrange(repetitions):
        cv = model_selection.RepeatedStratifiedKFold(n_splits=2,n_repeats=5, random_state=np.random.randint(9999))
        fold = 0
        k_accuracies = []
        for train, test in cv.split(X, y):
            fold_X_train, fold_y_train = X[train], y[train]
            fold_X_test, fold_y_test = X[test], y[test]

            clf_accuracies = []
            for clf_n in clfs:
                clf = clfs[clf_n]
                clf.fit(fold_X_train, fold_y_train)
                probas = clf.predict_proba(fold_X_test)
                prediction = np.argmax(probas,axis = 1)
                accuracy = metrics.accuracy_score(fold_y_test, prediction)
                clf_accuracies.append(accuracy)
            k_accuracies.append(clf_accuracies)

            fold += 1
        filename = "results/%s_r%i_k2x5.csv" % (dataset[1], repetition)
        print filename
        k_accuracies = np.array(k_accuracies)
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(clfs.keys())
            for row in k_accuracies:
                spamwriter.writerow(row)

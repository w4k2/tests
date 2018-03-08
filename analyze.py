#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import csv
import scipy

repetitions = 10
datasets = h.datasets()
clfs = h.classifiers().keys()
results = h.results()

for result in results:
    dbname = result[2]
    repetition = int(result[3][1:])
    cv_method = result[4]
    filename = result[0]
    df = pd.read_csv(filename)

    #print "Repetition %i with %s on %s" % (repetition, cv_method, dbname)
    #print df

    #print clfs
    for i, clf_a in enumerate(clfs):
        if i == len(clfs) - 1:
            break
        #print i
        #print clf_a
        acc_vector_a = df[clf_a].values
        #print acc_vector_a
        #print "vs"
        for j in xrange(i+1,len(clfs)):
            clf_b = clfs[j]
            #print clf_b
            acc_vector_b = df[clf_b].values
            #print acc_vector_b

            _, p_w = scipy.stats.wilcoxon(acc_vector_a,acc_vector_b)
            _, p_t = scipy.stats.ttest_ind(acc_vector_a,acc_vector_b)
            #print p_w
            #print p_t

    #break

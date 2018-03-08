#!/usr/bin/env python
import helper as h
import numpy as np
import pandas as pd
import csv

repetitions = 10
datasets = h.datasets()
clfs = h.classifiers()
results = h.results()

for result in results:
    dbname = result[2]
    repetition = int(result[3][1:])
    cv_method = result[4]
    filename = result[0]
    df = pd.read_csv(filename)
    print "Repetition %i with %s on %s" % (repetition, cv_method, dbname)
    print df

    break

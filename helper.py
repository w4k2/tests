import os  # to list files
import re  # to use regex
import numpy as np
from sklearn import neural_network, neighbors, svm, gaussian_process, tree, ensemble, naive_bayes, discriminant_analysis, model_selection
from sklearn import datasets

def datasets():
    directory = 'datasets/'
    files = np.array([(directory + x, x[:-4]) for \
        x in os.listdir(directory) \
        if re.match('^([a-zA-Z0-9])+\.csv$', x)])
    return files

def classifiers():
    return {
        "Nearest Neighbors": neighbors.KNeighborsClassifier(3),
        #"RBF SVM": svm.SVC(gamma=2, C=1, probability=True),
        #"Decision Tree": tree.DecisionTreeClassifier(max_depth=5),
        #"Random Forest": ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #"Neural Net": neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        "Naive Bayes": naive_bayes.GaussianNB(),

        #"Gaussian Process": gaussian_process.GaussianProcessClassifier(1.0 * gaussian_process.kernels.RBF(1.0)),
        #"Linear SVM": svm.SVC(kernel="linear", C=0.025, probability=True),
        #"AdaBoost": ensemble.AdaBoostClassifier(),
    }

def ks():
    return [10, 20, 30]

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from result_collection.helper_func import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import graphviz
from sklearn import tree




# Given data in numpy array format reduce dim of data
# using principle components
def pca(X, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(X)


# Given data in numpy array format reduce dim of data
# and visualise it on a 2D plot, one can add labels for the data
def visualise_2D_with_pca(X, labels=None):
    X = pca(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.4)
    plt.show()


def visualise_2D_with_lda(X, labels=None):
    X = lda(X, [0 if 'r' == l else 1 for l in labels])
    plt.scatter(np.zeros(len(X)), X, c=labels, alpha=0.4)
    plt.show()


def lda(X, y):
    clf = LinearDiscriminantAnalysis()
    return clf.fit_transform(X, y)


def visualise_2D_with_qda(X, labels=None):
    X = lda(X, [0 if 'r' == l else 1 for l in labels])
    plt.scatter(np.zeros(len(X)), X, c=labels, alpha=0.4)
    plt.show()


def qda(X, y):
    clf = QuadraticDiscriminantAnalysis()
    return clf.fit(X, y).predict(X)


def visualise_decision(X, labels):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, [0 if 'r' == l else 1 for l in labels])

    feature_names = ["P1(C|s0)", "P1(C|CC)", "P1(C|CD)", "P1(C|DC)", "P1(C|DD)"]
    if len(X[0, 0, :]) == 10:
        feature_names = ["P1(C|s0)", "P1(C|CC)", "P1(C|CD)", "P1(C|DC)", "P1(C|DD)",
                         "P2(C|s0)", "P2(C|CC)", "P2(C|DC)", "P2(C|CD)", "P2(C|DD)"]

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=["Other", "TFT"],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph
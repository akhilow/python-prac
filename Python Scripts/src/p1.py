# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:43:46 2019

@author: LowR2
"""
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

seed = 7
scoring = 'accuracy'


def default_value_test(a, L = none):
    '''
    '''
    if len(L) == 0:
        L = []
    L.append(a)
    return L


def map_test(default=[]):
    '''Itertools test function.
    '''
    a = iter(default.copy())
    # c,d=[('a', 'b'), ('c','d')]

    b = iter([('a', 'b'), ('c', 'd')])
    return list(a)

    def inner_mapTest(a):
        return None  # a*2

    def inner_filterTest(a):
        return None  # (a%2==0)
    # print(list(filter(inner_filterTest, a)))


def get_dataset(filePath):
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(filePath, names=names)
    return dataset;


def show_example_plots(dataset):
    '''Damn, I don't need to declare void functions pog
    '''
    # dataset size
    # print(dataset.shape)

    # first 20 rows
    print(dataset.head(20))

    # summary() from R lol
    # print(dataset.describe())

    # class distribution
    # print(dataset.groupby('class').size())

    # dataset.plot(kind='box', layout=(2,2), subplots=True, sharex=False, sharey=False)
    # dataset.hist()
    # scatter_matrix(dataset)
    # plt.show()


def training_validation(dataset):
    array = dataset.values
    X = array[:, :-1]
    Y = array[:, -1]
    validation_size = 0.20
    # fix the randomness
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# setup various models
def model_setup(modelData):
    X_train, X_validation, Y_train, Y_validation = modelData
    models = []
    results = []
    names = []

    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return results, names


def compare_algorithms(resultsTuple):
    results, names = resultsTuple
    fig = plt.figure()
    fig.suptitle('Algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def main():
    #    show_example_plots(get_dataset("C:\\Users\\LowR2\\Desktop\\iris.csv"))
    # compare_algorithms(model_setup(training_validation(get_dataset("C:\\Users\\LowR2\\Desktop\\iris.csv"))))
    #    mapTest()
    print(default_value_test(1))
    print(default_value_test(2))
    print(default_value_test(3))


main()

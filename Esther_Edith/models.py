'''
These are the models that we are going to go through
'''
#Imports
#Pandas, numpy and marplot
import pandas as pd
import numpy as np

#sklearn models
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, HuberRegressor,\
    BayesianRidge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,\
    IsolationForest, RandomTreesEmbedding

#Defined constants for this assignment
#models we will use
REGRESSION = "Logistic Regression"
KNN = "K Nearest Neighbors"
TREE = "Decision Trees"
SVM = "Support Vector Machines"
FOREST = "Random Forests"
EXTRA = "Extra Trees"
ADA_BOOSTING = "Ada Boosting"
BAGGING = "Bagging"
HUBER = "Huber Regressor"
BAY = "Bayesian Ridge"
ISOLATION = "Isolation Forest"
EMBED = "Random Trees Embedding"

C_VALS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NEIGHBORS = [1, 2, 5, 10, 20, 50, 100, 200]
MAX_DEPTH = [1, 5, 10, 20, 50, 100, 200, 300, 500]
NUM_TREES = [1000, 5000, 10000, 15000, 20000]
MAX_FEATURES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
MAX_LEAVES = [10, 20, 50, 100, 200, 300, 500, 1000, None]
N_ESTIMATORS = [10, 30, 50, 100, 150, 200, 300]
LEARNING_RATE = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
MAX_SAMPLES = MAX_FEATURES[:]
EPSILON = [1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
N_ITER = [50, 100, 300, 500, 700, 1000]

def training_models():
    '''
    '''    
    return regression_svm_modeling() + knn_modeling() + forest_modeling() +\
        ada_boost_modeling() + bagging_modeling() + huber_modeling() +\
        bays_modeling()

def bays_modeling():
    '''
    '''
    models_lst = []
    for n in N_ITER:
        param = "Max Number of Iterations: " + str(n)
        model = BayesianRidge(n_iter=n)
        models_lst.append(tuple([BAY, param, model]))
    return models_lst

def huber_modeling():
    '''
    '''
    models_lst = []
    for e in EPSILON:
        param = "Epsilon: " + str(e)
        models_lst.append(tuple([HUBER, param, HuberRegressor(epsilon=e)]))
    return models_lst

def regression_svm_modeling():
    '''
    Creates multiple regression models
    '''
    models_lst = []
    for c in C_VALS:
        param = "C value: " + str(c)
        model_SVM = LogisticRegression(C=c, solver='sag')
        model_linear = LinearSVC(C=c)
        models_lst.append(tuple([SVM, param, model_SVM]))
        models_lst.append(tuple([REGRESSION, param, model_linear]))
    return models_lst

def knn_modeling(train_variable, train_features, test_variable,\
    test_features):
    '''
    Creates multiple nearest neighbors models
    '''
    models_lst = []
    for k in NEIGHBORS:
        param = "K Neighbors: " + str(k)
        models_lst.append(tuple([KNN, param,
            KNeighborsClassifier(n_neighbors=k)]))
    return models_lst

def forest_modeling():
    '''
    Creates multiple decision tree models, random forest models and extra trees
        models
    '''
    models_lst = []
    for depth in MAX_DEPTH:
        tree_param = "Max Depth of Trees: " + str(depth)
        model_tree = DecisionTreeClassifier(max_depth=depth)
        models_lst.append(tuple([TREE, tree_param, model_tree]))
        for trees in NUM_TREES:
            for feat in MAX_FEATURES:
                for leaf in MAX_LEAVES:
                    param = "Number of Trees: " + str(trees) +\
                        ", Max Depth of Trees: " + str(depth) + \
                        ", Max Fraction of Features: " + str(feat) +\
                        ", Max Number of Leaves: " + str(leaf)
                    model_forest = RandomForestClassifier(n_estimators=trees,
                        max_depth=depth, max_features=feat, max_leaf_nodes=leaf)
                    model_extra = ExtraTreesClassifier(n_estimators=trees,
                        max_depth=depth, max_features=feat, max_leaf_nodes=leaf)
                    model_embed = RandomTreesEmbedding(n_estimators=trees,
                        max_depth=depth, max_features=feat, max_leaf_nodes=leaf)
                    models_lst.append(tuple([FOREST, param, model_forest]))
                    models_lst.append(tuple([EXTRA, param, model_extra]))
                    models_lst.append(tuple([EMBED, param, model_extra]))
    return models_lst

def ada_boost_modeling():
    '''
    Creates multiple AdaBoost models
    '''
    models_lst = []
    for n in N_ESTIMATORS:
        for rate in LEARNING_RATE:
            param = "Estimators: " + str(n) + ", Learning Rate: " + str(rate)
            model = AdaBoostClassifier(n_estimators=n, learning_rate=rate)
            models_lst.append(tuple([ADA_BOOSTING, param, model]))
    return models_lst

def bagging_modeling():
    '''
    Creates multiple bagging models

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        bag_dict: a dictionary with all information about all bagging models
    '''
    models_lst = []
    for n in N_ESTIMATORS:
        for sample in MAX_SAMPLES:
            for feat in MAX_FEATURES:
                param = "Estimators: " + str(n) +\
                    ", Samples: " + str(sample) +\
                    ", Features: " + str(feat)
                model_bag = BaggingClassifier(n_estimators=n,
                    max_samples=sample, max_features=feat,
                    bootstrap_features=True)
                model_iso = IsolationForest(n_estimators=n,
                    max_samples=sample, max_features=feat, bootstrap=True)
                models_lst.append(tuple([BAGGING, param, model_bag]))
                models_lst.append(tuple([ISOLATION, param, model_iso]))
    return models_lst
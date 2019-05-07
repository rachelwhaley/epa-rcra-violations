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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

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

def training_models(train_variable, train_features, test_variable,\
    test_features):
    '''
    Trains all models on training data

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        models_dict: a dictionary with all information about all models
    '''
    models_dict = {}
    
    #Set the value for all model types
    models_dict[REGRESSION], models_dict[SVM] =\
        regression_svm_modeling(train_variable, train_features, test_variable,\
        test_features)
    models_dict[KNN] = knn_modeling(train_variable, train_features,\
        test_variable, test_features)
    models_dict[FOREST], models_dict[EXTRA], models_dict[TREE] =\
        forest_modeling(train_variable, train_features, test_variable,\
        test_features)
    models_dict[ADA_BOOSTING] = ada_boost_modeling(train_variable,\
        train_features, test_variable, test_features)
    models_dict[BAGGING] = bagging_modeling(train_variable, train_features,\
        test_variable, test_features)

    return models_dict

def regression_svm_modeling(train_variable, train_features, test_variable,\
    test_features):
    '''
    Creates multiple regression models

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        reg_dict: a dictionary with all information about all regression models
        svm_dict: a dictionary with all information about all svm models
    '''
    reg_dict = {}
    svm_dict = {}
    C_VALS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for c in C_VALS:
        param = "C value: " + str(c)
        model_unfit = LogisticRegression(C=c, solver='sag')
        reg_dict[param] = test_models(model_unfit, False, train_variable,\
            train_features, test_variable, test_features)
        model_unfit = LinearSVC(C=c)
        svm_dict[param] = test_models(model_unfit, True, train_variable,\
            train_features, test_variable, test_features)
    return reg_dict, svm_dict

def knn_modeling(train_variable, train_features, test_variable,\
    test_features):
    '''
    Creates multiple nearest neighbors models

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        knn_dict: a dictionary with all information about all nearest neighbors
            models
    '''
    knn_dict = {}
    NEIGHBORS = [1, 2, 5, 10, 20, 50, 100, 200]
    for k in NEIGHBORS:
        param = "K Neighbors: " + str(k)
        model_unfit = KNeighborsClassifier(n_neighbors=k)
        knn_dict[param] = test_models(model_unfit, False, train_variable,\
            train_features, test_variable, test_features)
    return knn_dict

def forest_modeling(train_variable, train_features, test_variable,\
    test_features):
    '''
    Creates multiple decision tree models, random forest models and extra trees
        models
    (Random forests and extra trees take the same parameters, which is why
    we are putting them together. We will use the same depth for decision
    trees which is why this is with them)

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        forest_dict: a dictionary with all information about all random forest
            models
        extra_dict: a dictionary with all information about all extra tree
            models
        tree_dict: a dictionary with all information about all decision tree
            models
    '''
    forest_dict = {}
    extra_dict = {}
    tree_dict = {}
    MAX_DEPTH = [1, 5, 10, 20, 50, 100, 200, 300, 500]
    NUM_TREES = [1000, 5000, 10000, 15000, 20000]
    MAX_FEATURES = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    MAX_LEAVES = [10, 20, 50, 100, 200, 300, 500, 1000, None]
    for depth in MAX_DEPTH:
        tree_param = "Max Depth of Trees: " + str(depth)
        model_unfit = DecisionTreeClassifier(max_depth=depth)
        tree_dict[tree_param] = test_models(model_unfit, False, train_variable,\
            train_features, test_variable, test_features)
        for trees in NUM_TREES:
            for feat in MAX_FEATURES:
                for leaf in MAX_LEAVES:
                    param = "Number of Trees: " + str(trees) +\
                        ", Max Depth of Trees: " + str(depth) + \
                        ", Max Fraction of Features: " + str(feat) +\
                        ", Max Number of Leaves: " + str(leaf)
                    model_unfit = RandomForestClassifier(n_estimators=trees,
                        max_depth=depth, max_features=feat, max_leaf_nodes=leaf)
                    forest_dict[param] = test_models(model_unfit, False,
                        train_variable, train_features, test_variable,
                        test_features)
                    model_unfit = ExtraTreesClassifier(n_estimators=trees,
                        max_depth=depth, max_features=feat, max_leaf_nodes=leaf)
                    extra_dict[param] = test_models(model_unfit, False,
                    	train_variable, train_features, test_variable,
                    	test_features)
    return forest_dict, extra_dict, tree_dict

def ada_boost_modeling(train_variable, train_features, test_variable,\
    test_features):
    '''
    Creates multiple AdaBoost models

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        ada_dict: a dictionary with all information about all ada boost models
    '''
    ada_dict = {}
    N_ESTIMATORS = [10, 30, 50, 100, 150, 200, 300]
    LEARNING_RATE = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    for n in N_ESTIMATORS:
        for rate in LEARNING_RATE:
            param = "Estimators: " + str(n) + ", Learning Rate: " + str(rate)
            model_unfit = AdaBoostClassifier(n_estimators=n,\
                learning_rate=rate)
            ada_dict[param] = test_models(model_unfit, False, train_variable,\
                train_features, test_variable, test_features)
    return ada_dict

def bagging_modeling(train_variable, train_features, test_variable,\
    test_features):
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
    bag_dict = {}
    N_ESTIMATORS = [5, 10, 30, 50]
    MAX_SAMPLES = [10, 50, 100, 500]

    for n in N_ESTIMATORS:
        for sample in MAX_SAMPLES:
            param = "Estimators: " + str(n) + ", Samples: " + str(sample)
            model_unfit = BaggingClassifier(n_estimators=n,\
                max_samples=sample)
            bag_dict[param] = test_models(model_unfit, False, train_variable,\
                train_features, test_variable, test_features)
    return bag_dict
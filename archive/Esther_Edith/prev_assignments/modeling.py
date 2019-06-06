'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #3: creating and testing models
'''
#Imports
#Pandas, numpy and marplot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta
import matplotlib.pyplot as plt

#sklearn models
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

#sklearn metrics
from sklearn.metrics import accuracy_score as accuracy,\
    precision_score, recall_score, roc_auc_score, f1_score,\
    precision_recall_curve

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

#evaluation metrics we will use
ACCURACY = "Accuracy"
PRECISION = "Precision"
RECALL = "Recall"
ROC_AUC = "ROC_AUC"
F1 = "F1"

def split_by_date(df_all_data, split, variable, features):
    '''
    Splits the data by date listed in the split column

    Inputs:
        df_all_data: a pandas dataframe
        split: the name of the column we are splitting on
        variable: the name of the variable column
        features: list of the names of the feature columns

    Outputs:
        models_dict: a dictionary with all information about all models for all
            dates 
    '''
    models_dict = {}
    time_series = df_all_data[split]
    final_date = time_series.max()

    #Initialize test and train dates
    end_train = time_series.min() - timedelta(days=1)
    begin_train = 0
    begin_test = 0
    end_test = end_train

    while end_test < final_date:
        #The training data ends 180 days after the beginning of the train
        #the training data begins the day after the ending of train data
        begin_train = end_train + timedelta(days=1)
        end_train = begin_train + timedelta(days=180)
        #Testing data begins the day after training data ends
        #Testing data ends 180 days after it begins
        begin_test = end_train + timedelta(days=1)
        end_test = begin_test + timedelta(days=180)
        #Prevents there being a set that is just a few days
        if (final_date - end_test).days <= 30:
            end_test = final_date
        dates = str(begin_test) + " - " + str(end_test)
        
        #Now we create the training and testing data
        train_filter =\
            (df_all_data[split] <= end_train) &\
            (df_all_data[split] >= begin_train)
        train_data = df_all_data[train_filter]
        test_filter =\
            (df_all_data[split] <= end_test) &\
            (df_all_data[split] >= begin_test)
        test_data = df_all_data[test_filter]

        #Now we have to create the variable and features data
        train_variable = train_data[variable]
        train_features = train_data[features]
        test_variable = test_data[variable]
        test_features = test_data[features]

        #Now we create the models dictionary
        #By the end of this assignent, I suspect you will tell me I rely too
        #much on dictionaries
        print(dates)
        models_dict[dates] = training_models(train_variable, train_features,\
            test_variable, test_features)

    return models_dict

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
    C_VALS = [1.0, 1.2, 1.5, 2.0, 2.5]
    for c in C_VALS:
        param = "C value: " + str(c)
        model_unfit = LogisticRegression(C=c)
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
    NEIGHBORS = [1, 5, 10, 20, 50, 100]
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
    NUM_TREES = [5, 25, 75]
    MAX_DEPTH = [1, 5, 20, 50, 100, 200]
    for depth in MAX_DEPTH:
        tree_param = "Max Depth of Trees: " + str(depth)
        model_unfit = DecisionTreeClassifier(max_depth=depth)
        tree_dict[tree_param] = test_models(model_unfit, False, train_variable,\
            train_features, test_variable, test_features)
        for trees in NUM_TREES:
            param = "Number of Trees: " + str(trees) +\
                ", Max Depth of Trees: " + str(depth)
            model_unfit = RandomForestClassifier(n_estimators=trees,\
                max_depth=depth)
            forest_dict[param] = test_models(model_unfit, False,
                train_variable, train_features, test_variable, test_features)
            model_unfit = ExtraTreesClassifier(n_estimators=trees,\
                max_depth=depth)
            extra_dict[param] = test_models(model_unfit, False, train_variable,\
                train_features, test_variable, test_features)
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
    N_ESTIMATORS = [10, 30, 50, 100, 200]
    LEARNING_RATE = [0.5, 1.0, 2.0]
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

def test_models(model_unfit, is_svm, train_variable, train_features,\
    test_var, test_features):
    '''
    Fits a model to the data, tests the model, and then evaluates the model

    Inputs:
        model_unfit: a model that has not been fitted to the data
        is_svm: a boolean that is True is the model is an SVM model and False
            for all other models
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set

    Outputs:
        eval_dict: a dictionary with all evaluation metrics for all thresholds
    '''
    eval_dict = {}
    THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    #First we fit the model to the training data
    model = model_unfit.fit(train_features, train_variable)
    
    #Next, we predict the probabilities for the testing data
    if is_svm:
        probabilities = model.decision_function(test_features)
    else:
        probabilities = model.predict_proba(test_features)[:,1]
    
    #Now we evaluate
    #First evaluations only need the probabilities
    key = "No Threshold"
    roc_auc = roc_auc_score(y_true=test_var, y_score=probabilities)
    eval_dict[key] = {ROC_AUC: roc_auc}
    
    #All other evaluations need to loop through the thresholds
    for thresh in THRESHOLDS:    
        calc_threshold = lambda x,y: 0 if x < y else 1
        predicted = np.array([calc_threshold(score, thresh) for score in
            probabilities])
        key = "Threshold: " + str(thresh)
        eval_dict[key] = evaluate_models(test_var, predicted)
    return eval_dict

def evaluate_models(true, predicted):
    '''
    Evaluates models on multiple evaluations metrics

    Inputs:
        true: a pandas series of the true outcome for testing data
        predicted: the model's prediction of the outcome

    Outputs:
        eval_dict: a dictionary mapping an evaluation metric to its
            corresponding score
    '''
    eval_dict = {}
    eval_dict[ACCURACY] = accuracy(y_true=true, y_pred=predicted)
    eval_dict[PRECISION] = precision_score(y_true=true, y_pred=predicted)
    eval_dict[RECALL] = recall_score(y_true=true, y_pred=predicted)
    eval_dict[F1] = f1_score(y_true=true, y_pred=predicted)
    return eval_dict

def plot_pre_rec(train_variable, train_features,\
    test_var, test_features, is_svm, model, name):
    '''
    Plots the precision recall score and saves it for select models

    First note: this code borrows heavily from sklearn documentation:

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision
        _recall_curve.html

    Second Note: Because there are so many models, I am only going to call this
        function for select models I want to see the full curve for

    Inputs:
        train_variable: pandas series of variable column for training set
        train_features: pandas dataframe of features for training set
        test_variable: pandas series of variable column for testing set
        test_features: pandas dataframe of features for testing set
        is_svm: boolean determining if the model is an svm model
        model: the model we want to get the precision-recall curve for
        name: the name of the graph we will save
    '''
    #First we fit the model to the data
    model = model.fit(train_features, train_variable)
    #Now we find the probabilities
    if is_svm:
        probabilities = model.decision_function(test_features)
    else:
        probabilities = model.predict_proba(test_features)[:,1]
    precision, recall, thresholds = precision_recall_curve(test_var,\
        probabilities)

    #Now we graph the data
    plt.plot(thresholds, precision, color='b')
    plt.plot(thresholds, recall, color='orange')    
    plt.ylabel('Precision / Recall Scores')
    plt.xlabel('Threshold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precison-Recall Curve')

    #Now we save the figure
    plt.savefig(name)


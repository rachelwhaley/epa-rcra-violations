'''
Evaluation we are going to use

Since we said we wanted to maximize precision, I'm only evaluating precision and
recall
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta
import matplotlib.pyplot as plt

#sklearn metrics
from sklearn.metrics import precision_score, recall_score

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
SVM = "Support Vector Machines"

def fit_all(dates_lst, models_lst, variable, features, data_df, date_col):
    '''
    '''
    col_lst = ['Date', 'Model Name', 'Parameters', 'Threshold', 'Precision', 
        'Recall']
    df_lst = []

    for dates_breakdown in date_lst:
        train_variable, train_features, test_variable, test_features, dates = \
            create_train_test(dates_breakdown, variable, features, data_df,
                date_col)
        for tup in models_lst:
            name, param, model = tup
            model = model.fit(train_features, train_variable)
            if name == SVM:
                is_svm = True
            else:
                is_svm = False
            eval_lst = testing(model, is_svm, test_variable, test_features)
            for tup2 in eval_lst:
                threshold, precision, recall = tup2
                this_lst = [dates, name, param, threshold, precision, recall]
                df_lst.append(this_lst)

    df_evaluated_models = pd.DataFrame(np.array(df_lst), columns=col_lst)
    df_evaluated_models.to_csv("Modeling_RCRA_Data.csv")
    
    return df_evaluated_models

def create_train_test(dates_breakdown, variable, features, data_df, date_col):
    '''
    '''
    train, test = dates_breakdown
    begin_train, end_train = train
    begin_test, end_test = test

    dates = str(begin_test) + " - " + str(end_test)

    train_filter =\
        (data_df[date_col] <= end_train) &\
        (data_df[date_col] >= begin_train)
    train_data = df_all_data[train_filter]

    test_filter =\
        (data_df[date_col] <= end_test) &\
        (data_df[date_col] >= begin_test)
    test_data = df_all_data[test_filter]

    train_variable = train_data[variable]
    train_features = train_data[features]
    test_variable = test_data[variable]
    test_features = test_data[features]

    return train_variable, train_features, test_variable, test_features, dates

def testing(model, is_svm, test_variable, test_features):
    '''
    '''
    eval_lst = []

    if is_svm:
        probabilities = model.decision_function(test_features)
    else:
        probabilities = model.predict_proba(test_features)[:,1]

    for thresh in THRESHOLDS:    
        calc_threshold = lambda x,y: 0 if x < y else 1
        predicted = np.array([calc_threshold(score, thresh) for score in
            probabilities])
        precision = precision_score(y_true=true, y_pred=predicted)
        recall = recall_score(y_true=true, y_pred=predicted)
        eval_lst.append(tuple([thresh, precision, recall]))

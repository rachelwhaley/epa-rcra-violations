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
            dates_breakdown
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

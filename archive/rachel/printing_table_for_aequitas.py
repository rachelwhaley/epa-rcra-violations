import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from scipy import stats
from plotnine import *
import graphviz
import pylab as pl
import create_features as cf
import dave_pipelib as dp
import datetime

'''
Mini-Pipe with One Model, Limited Features, like a prototype
'''
evals = pd.read_csv('RCRA_EVALUATIONS.csv')
facs = pd.read_csv('RCRA_FACILITIES.csv')
to_drop = ['ACTIVITY_LOCATION','FULL_ENFORCEMENT','HREPORT_UNIVERSE_RECORD', 'STREET_ADDRESS', 'CITY_NAME', 'STATE_CODE', 'ZIP_CODE', 'LATITUDE83', 'LONGITUDE83','FED_WASTE_GENERATOR', 'TRANSPORTER', 'ACTIVE_SITE', 'OPERATING_TSDF']
features = ['Sum_Violations', 'PCT_EVALS_FOUND_VIOLATION', 'PCT_OF_ALL_EVALS']
classes = ['DT']
pars = {'DT': ['gini', 'entropy']}

def simp_run_through(evals, facs, features, year_col, start, split, end,
                     classes, parameters, thresholds):
    rv = {'class': [], 'DT': [], 'threshold': [], 'precision': [], 'recall': [],
         'preds': [], 'top_features': []}
    df = make_df(evals, facs)
    train, test = simp_windows(df, year_col, start, split, end, features)
    trx, tr_y, tex, te_y = simp_x_y_split(train, test)
    train_dates = trx['EVALUATION_START_DATE']
    test_dates = tex['EVALUATION_START_DATE']
    trx.drop(['EVALUATION_START_DATE', 'FACILITY_NAME', 'MostRecentEval'], inplace=True, axis=1)
    tex.drop(['EVALUATION_START_DATE', 'FACILITY_NAME', 'MostRecentEval'], inplace=True, axis=1)
    for c in classes:
        for p in pars[c]:
            if c == 'DT':
                scores, imps = dectree_classifier(trx, tr_y, tex, p)
            for t in thresholds:
                preds = [compare_to_threshold(x, t) for x in list(scores)]
                rv[c].append(p)
                rv['class'].append(c)
                rv['threshold'].append(t)
                rv['precision'].append(precision(te_y, preds))
                rv['recall'].append(recall(te_y, preds))
                rv['preds'].append(preds)
                rv['top_features'].append(list(zip(list(tex.columns), list(imps))))

    final = pd.DataFrame(rv)
    final.to_csv('results.csv')

    return print(final)

def dectree_classifier(x_train, y_train, x_test, crit):
    '''
    takes x-train, y-train, and x-test and predicts probable class
    for test data. will not allow split with less than 4% of data in node
    to avoid overfitting
    '''
    lim = int(len(list(y_train))/25)
    dec_tree = DecisionTreeClassifier(criterion=crit)
    dec_tree.fit(x_train, y_train)
    importances = dec_tree.feature_importances_


    return dec_tree.predict_proba(x_test)[:,0], importances
    

def make_df(evals, facs, dropcols=to_drop):
    dfbeta = cf.create_features(facs, evals)
    evals.drop([ 'ACTIVITY_LOCATION', 'EVALUATION_IDENTIFIER',
        'EVALUATION_TYPE', 'EVALUATION_DESC', 'EVALUATION_AGENCY',
        'FOUND_VIOLATION'], inplace=True, axis=1)
    evals[['month', 'day', 'year']] = evals['EVALUATION_START_DATE'].str.split(
    '/', expand=True)
    dfbeta.drop(dropcols, inplace=True, axis=1)
    return pd.merge(evals, dfbeta, on='ID_NUMBER', how='left')

def simp_windows(df, year_col, start, split, end, features=features):
    '''
    simple hard coded window splitter for purpose of simple pipe
    '''
    train = df[(df[year_col].astype('int') >= start) & (df[year_col].astype(
    'int') < split)]
    test = df[(df[year_col].astype('int') >= split) & (df[year_col].astype(
    'int') <= end)]
    
    return train, test

def simp_x_y_split(train, test):
    train_x, train_y = x_y_split(train)
    test_x, test_y = x_y_split(test)
    
    train_x.drop(['month', 'day', 'year', 'ID_NUMBER'], inplace=True, axis=1)
    test_x.drop(['month', 'day', 'year', 'ID_NUMBER'], inplace=True, axis=1)
    
    return train_x, train_y, test_x, test_y

def rolling_window_splitter(df, date_col, window, features=features):
    '''
    splits df into 6 month periods based on a column
    window is in months
    '''
    features.append('Y')
    features.append('EVALUATION_START_DATE')
    df = df.sort_values('EVALUATION_START_DATE')
    df = df.loc[:,features]
    start = pd.Timestamp(df.iloc[0][date_col])
    next_edge = pd.Timestamp(add_months(start, window))
    end = pd.Timestamp(df.iloc[-1][date_col])
    rv = []
    
    while next_edge <= end:
        rv.append(df.loc[(df[date_col] < next_edge) & (df[date_col] > start)])
        start = next_edge
        next_edge = pd.Timestamp(add_months(start, window))
        
    rv.append(df.loc[df[date_col] > start])
    features.pop()
    features.pop()
        
    return rv

def add_months(start, months):
    '''
    sourced from stack overflow:
    https://stackoverflow.com/questions/4130922/how-to-increment-datetime-by-custom-months-in-python-without-using-library
    '''
    month = start.month - 1 + months
    year = start.year + month // 12
    month = month % 12 + 1
    day = min(start.day, calendar.monthrange(year,month)[1])
    
    return datetime.date(year, month, day)

def x_y_split(data):
    y = data.Y
    x = data.drop('Y', axis=1)
    return x, y

def convert_with_format(df, column_name):
    return pd.to_datetime(df[column_name], format='%m/%d/%y', errors='coerce')

def YN_to_binary(data_series):
    return data_series.map(dict(Y=1, N=0))

def compare_to_threshold(score, threshold):
    '''
    takes threshold, temporarily aggregates data and comes up with score
    that represents threshold% of population, then compares each score to that
    adjusted threshold
    '''
    if score > threshold:
        return 1
    else:
        return 0


def main():
    simp_run_through(evals, facs, features, 'year', 2012, 2013, 2013,
classes, pars, [.05, .5, .9])

if __name__ == '__main__':
    main()


import model_analyzer as ma
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                             RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid

#read data
def read_csv_handledates(filepath, date_cols):
    '''
    input: filepath to a csv, list of date columns to parse
    output: a pandas dataframe with correctly typed date columns
    '''
    return pd.read_csv(filepath, parse_dates=date_cols)

#explore the data


#for creating an outcome variable and features
def calculate_duration(df, beginning_col, ending_col, new_col):
    '''
    input: a dataframe with two date columns you're interested in the time between
    output: that dataframe now has a column representing the time between them
    '''
    df[new_col] = df[ending_col] - df[beginning_col]

def compare_tdelta_generate_binary(df, col_to_compare, threshold, new_col):
    '''
    inputs: a dataseries with time deltas, a number (threshold) to compare
    members of that data series to and a name for the new column with the
    binary values
    output: that dataseries now has a column for the binary values
    '''
    df[new_col] = np.where(df[col_to_compare] >= pd.Timedelta(threshold), 1, 0)

def id_potential_features(df):
    '''
    input: dataframe
    output: list of string columns and float columns that can become features
    '''
    str_cols = [column for column in df.columns if (df[column].dtype=='O')
                   and (len(df[column].unique())<=20)]
    flt_cols = [column for column in df.columns if
                (df[column].dtype=='float64')]

    return str_cols, flt_cols

def select_features(df):
    """
    Returns a list of column names of features to use in the model.
    """
    # should select only columns that are numeric or dummy
    features_list = []

    for column in df:
        if df[column].dtype == 'float64':
            features_list.append(column)
        elif df[column].dtype == 'int64':
            features_list.append(column)
        elif df[column].dtype == 'uint8':
            features_list.append(column)

    return features_list

def strings_to_dummies(df, strcols):
    '''
    turn the string columns we identified into dummies and generate a new
    df with these dummies to act as features
    '''
    temp = pd.get_dummies(df, dummy_na=True, columns=strcols)
    df[list(temp.columns)] = temp


def add_discretized_float_cols(og_df, feature_df, fltcols):
    '''
    make bins 3 or 5 please
    adds discretized float cols to a features column
    fltcols are columns from og_df to be discretized and added to feature_df
    '''
    l = ['low', 'medium low', 'medium high', 'high']
    for column in fltcols:
        new_col = column + '_d'
        feature_df[column] = og_df[column]
        feature_df[new_col] = pd.qcut(og_df[column], q=[0, .25, .5, .75, 1.],
                                    labels=l)

#clean data
def fillna_modal(df, col_list):
    '''
    fills nans in col_list with the column's mode
    '''
    for x in col_list:
        df[x].fillna(df[x].value_counts().index[0], inplace=True)

def fillzero_mean(df, col):
    '''
    needs a int or float dataseries where nans have already been turned to 0's
    '''
    imputation = df[col].notnull().mean()
    df[col].fillna(imputation, inplace=True)
    df[col].replace(0, imputation, inplace = True)

#split data (temporal and x/y)
def temp_holdout(df, date_col, period, holdout):
    '''
    rolling window in months, holdout in months
    returns list of training and testing sets
    '''
    period = pd.DateOffset(months=period)
    holdout = pd.DateOffset(months=holdout)

    first = df[date_col].min()
    training_ends = first + period
    testing_begins = training_ends + holdout
    last = df[date_col].max()
    trains = []
    tests = []
    train_ends = []
    test_ends = []

    while (testing_begins + period) <= last:
        trains.append(df[(df[date_col] >= first) & (df[date_col] < training_ends)])
        tests.append(df[(df[date_col] >= testing_begins) & (df[date_col] <
                                                            (testing_begins +
                                                            period))])
        train_ends.append(training_ends)
        test_ends.append((testing_begins + period))
        first += period
        training_ends += period
        testing_begins += period

    return trains, tests, train_ends, test_ends

def seperate_ids_feats_ys(periods, id_cols, y_col, feature_columns):
    '''
    takes a list of periods from temp_holdout() and returns a list of lists of
    each period's ids, features, and ys
    '''
    rv = []

    for x in periods:
        rv.append([x[id_cols], x[feature_columns], x[y_col]])
        
    for x in rv:
        for df in x:
            print(len(df.index))

    return rv

#select your grid   
def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l2', C=1e5, solver='lbfgs'),
        'SVM': SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="log", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BAG': BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, n_estimators = 20)
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'solver': ['liblinear']},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.01,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10, 20], 'max_samples' : [.25, .5, .75]}
       }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.1,1,10], 'solver': ['liblinear']},
    'SGD': { 'loss': ['log','perceptron'], 'penalty': ['l2','l1'], 'max_iter':[1000], 'tol':[1]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,500]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.5],'subsample' : [0.5,1.0], 'max_depth': [5]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, None],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [10,50],'weights': ['uniform','distance'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] }
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l2'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5], 'max_samples' : [.25] } 

           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

#run and analyze models
def model_analyzer(clfs, grid, plots, thresholds, x_train, y_train, x_test, y_test):
    '''
    inputs: clfs dict of default models
            selected grid
            plots ('show' to see all plots, 'save' to see save all plots)
            prec_limit - set a precision limit models must surpass to be graphed
            thresholds - list of thresholds to iterate through
            split training and testing data
    outputs: df of all models and their predictions/metrics
             df of all predictions with model id as column name for later use
    filter placed on plotting to prevent plotting of excessive plots
    '''

    predictions = pd.DataFrame()
    stats_df = pd.DataFrame()
    models = []

    for klass, model in clfs.items():
        parameter_values = grid[klass]
        for p in ParameterGrid(parameter_values):
            try:
                name = klass + str(p)
                m = ma.ClassifierAnalyzer(model, p, name, thresholds,
                                            plots, x_train, y_train, x_test,
                                            y_test)
                pd.concat([predictions, m.predictions], axis=1)
                pd.concat([stats_df, m.metrics_matrix], axis=0)
                models.append(m)
                if plots == 'show':
                    print(m.name)
                    m.plot_precision_recall(False, True, None)
                    m.plot_roc(False, True, None)
                elif plots == 'save':
                    m.plot_precision_recall(True, False, name + 'pr.png')
                    m.plot_roc(True, False, name + 'roc.png')
                elif plots == 'both':
                    m.plot_precision_recall(True, True, name + 'pr.png')
                    m.plot_roc(True, True, name + 'roc.png')

            except IndexError as e:
                    print('Error:',e)
                    continue

    predictions['truth'] = y_test

    return predictions, stats_df, models

def model_analyzer_over_time(clfs, grid, plots, thresholds, list_of_x_train,
                             list_of_y_train, list_of_x_test, list_of_y_test,
                             feat_list):
    '''
    iterate through a list of x train dataframes and make an aggregate list of
    stats dics and models for each model in each timeframe. this list will be
    used to determine the best possible model.
    '''

    predictions = []
    stats = []
    models = []

    for i, x in enumerate(list_of_x_train):
        temp_preds, temp_stats, temp_models = model_analyzer(clfs, grid, plots,
                                                             thresholds,
                                                             x.loc[:,feat_list],
                                                             list_of_y_train[i],
                                                             list_of_x_test[i].loc[:, feat_list],
                                                             list_of_y_test[i])
        predictions.append(temp_preds)
        models.extend(temp_models)
        stats.append(temp_stats)
        

    return predictions, models, stats
        
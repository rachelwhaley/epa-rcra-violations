'''
Splits training and testing data by date

I think once we have the data cleaned up, we can run this once and then just
keep a list of the dates we are going to use.
'''

import pandas as pd
from datetime import datetime, timedelta

def split_by_date(variable, data_df, date_col):
    '''
    Splits date range by dates

    Inputs:
    Outputs:
    ''' 
    time_series = data_df[date_col]
    max_date = time_series.max()
    #Even though we have more data, we are only going back 7 years
    begin_train = max_date - timedelta(days=2555)
    end_train = begin_train
    all_dates = []

    while max_date > end_test:
        end_train = end_train + timedelta(days=365)
        begin_test = end_train + timedelta(days=1)
        end_test = begin_test + timedelta(days=365)
        all_dates.append(create_train_test(variable, features, data_df,
            date_col, begin_train, end_train, begin_test, end_test))
    return all_dates

def create_train_test(variable, data_df, date_col, begin_train,
    end_train, begin_test, end_test):
    '''
    '''
    dates = str(begin_test) + " - " + str(end_test)

    train_filter =\
        (data_df[date_col] <= end_train) &\
        (data_df[date_col] >= begin_train)
    train_data = df_all_data[train_filter]

    test_filter =\
        (data_df[date_col] <= end_test) &\
        (data_df[date_col] >= begin_test)
    test_data = df_all_data[test_filter]
    
    #We need to make the function that will get us our features
    train_variable = train_data[variable]
    train_features = get_features(train_data)
    test_variable = test_data[variable]
    test_features = get_features(test_data)

    return tuple([train_variable, train_features, test_variable, test_features,
        dates])

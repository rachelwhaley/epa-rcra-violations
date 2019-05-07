'''
Splits training and testing data by date

I think once we have the data cleaned up, we can run this once and then just
keep a list of the dates we are going to use.
'''

import pandas as pd
from datetime import datetime, timedelta

def split_by_date(dates):
    '''
    Splits date range by dates

    Inputs:
        dates: a tuple with the max and min values of the dates
    Outputs:
        date_lst: a list of date tuples with the beginning and ending
            dates for all of the training and testing splits
    '''
    max_date, min_date = dates
    #Train on 6 months, 1 year, 18 monhts, 2 years, 3 years, 4 years, 5 years
    training = [180, 365, 545, 730, 1095, 1460, 1825]
    #Test on 6 months, 1 year, 18 months, 2 years
    testing = [180, 365, 545, 730]
    all_dates = []

    for x in training:
        for y in testing:
            begin_train = max_date
            begin_test = max_date + timedelta(days=1)
            end_train = max_date
            end_test = max_date
            while min_date + timedelta(days=x) < begin_train:
                end_test = begin_test - timedelta(days=1)
                begin_test = end_test - timedelta(days=y)
                end_train = begin_test - timedelta(days=1)
                begin_train = end_train - timedelta(days=x)
                all_dates.append([tuple([begin_train, end_train]),
                    tuple([begin_test, end_test])])
    return all_dates

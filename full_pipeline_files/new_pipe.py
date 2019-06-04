'''
A new pipeline
'''

import pandas as pd
import has_violation

def pipeline():
    '''
    Goes from the beginning to the end of the pipeline
    '''
    df = has_violation.go()
    train_features, train_variable, test_features, test_variable = \
        temporal_split(df)
    return True

def temporal_split(df, year_col='YEAR_EVALUATED', period=1, holdout=1,\
    to_ignore=['ID_NUMBER'], variable='HasViolation'):
    '''
    Splits time by year

    df: a dataframe with all of the facilities for all of the years and the features
    year_col: the name of the column with the year
    period: the number of years we want in our training/testing sets
    holdout: the number of years we want to hold out between our training and testing
    to_ignore: a list of column names we don't need
    variable: the name of the column we want to predict
    '''
    all_cols = list(df.columns)
    to_ignore += [year_col, variable]
    features = list(set(all_cols) - set(to_ignore))
    first = df[year_col].min()
    training_ends = first + period
    testing_begins = training_ends + holdout
    last = df[year_col].max()
    train_features = []
    train_variable = []
    test_features = []
    test_variable = []

    while (testing_begins + period) <= last:
        trains = df[(df[year_col] >= first) & (df[year_col] < training_ends)]
        tests = df[(df[year_col] >= testing_begins) & (df[year_col] <
            (testing_begins + period))]
        train_features.append(trains[features])
        train_variable.append(trains[variable])
        test_features.append(tests[features])
        test_variable.append(tests[variable])

        first += period
        training_ends += period
        testing_begins += period

    return train_features, train_variable, test_features, test_variable

if __name__ == "__main__":
    pipeline() 
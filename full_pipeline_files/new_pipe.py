'''
A new pipeline
'''

import pandas as pd
import has_violation
import epa_pipeline
import ml_pipe as ml

def pipeline():
    '''
    Goes from the beginning to the end of the pipeline
    '''
    print("Creating dataframe")
    df = has_violation.go()
    print("Dataframe created")
    print("Creating temporal split")
    list_of_trainx, list_of_trainy, list_of_testx, list_of_testy, features = \
        temporal_split(df)
    return epa_pipeline.run_models('small', 'show', list_of_trainx, list_of_trainy,
               list_of_testx, list_of_testy, features)

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

    return train_features, train_variable, test_features, test_variable, features

def run_models(grid_size, plots, list_of_trainx, list_of_trainy,
               list_of_testx, list_of_testy, features):
    '''
    takes features and y data for all train and test periods and fits/runs all
    models on grid on all
    '''
    clfs, grid = ml.define_clfs_params(grid_size)

    predictions, models, metrics = ml.model_analyzer_over_time(clfs, grid,\
        plots, thresholds, list_of_trainx, list_of_trainy,\
        list_of_testx, list_of_testy, features)

    return predictions, models, metrics

if __name__ == "__main__":
    pipeline() 
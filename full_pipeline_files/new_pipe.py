'''
A new pipeline
'''

import pandas as pd
import numpy as np
import has_violation
import epa_pipeline as ep
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
    return ep.run_models('small', 'show', list_of_trainx, list_of_trainy,
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

def rank_classifiers(df,feat,criteria):
    aa = df.groupby([feat])
    lst_to_keep = []
    output_df = None
    for key, gr in aa:
        add_ord = gr[[criteria]]
        add_ord.drop_duplicates(inplace = True)
        add_ord.sort_values(by =criteria,ascending=False,inplace=True)
        add_ord[str(key)] = np.arange(len(add_ord))
        gr.sort_values(by=criteria, ascending=False,inplace=True)
        total_df = pd.merge(gr,add_ord,on = criteria)
        total_df = total_df[['model_type','clf','parameters','str_param',str(key)]]
        if output_df is None:
            output_df = total_df
        else:
            output_df = pd.merge(output_df,total_df,on = ['model_type', 'str_param'])
        lst_to_keep.append(str(key))
    lst_to_keep = ['model_type','clf','parameters','str_param'] + lst_to_keep 
        
    return output_df[lst_to_keep]

def rank(df, model_names_col, criteria):
    rv = df.groupby(model_names_col)[
        list(df.columns[:-1])].mean().sort_values(criteria, ascending=False)

    return rv
        

def merge_acs(df):
    acs = pd.read_csv('all_acs_data.csv')
    df['acs_data'] = np.where(df['year'] in acs.year.unique(), df['year'], 2016)
    df.merge(acs, on=['ID_NUMBER', 'year'])

if __name__ == "__main__":
    pipeline() 
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import acs_features
from census_area import Census
import explore as exp
pd.options.mode.chained_assignment = None
import numpy as np


def smaller_data_time(data, temp_feat, year):
    '''
    Reduce the dataframe according starting on the year specified
    Input:
        data(DataFrame): data we want to reduce
        temp_feat: feature that contains date
        year: year we want to separate the data on
    Output:
        Returns smaller dataframe
    '''
    data['temp'] = pd.to_datetime(data[temp_feat], errors = 'coerce')
    data_temp = data[data['temp'].dt.year >= year]
    return data_temp

def keep_miss_nonmiss(data, lst_feat):
    '''
    separates data depending on missing value 
    Input:
        data(DataFrame): dataframe we want to split
        lst_feat(list): list of missing features we want to separate on
    Output:
        Returns:
        dataframes divided by missing values and non missing values
    '''
    
    non_miss_df = data.dropna(subset = ['LONGITUDE83','LATITUDE83'])
    miss_df = data[pd.isnull(data[lst_feat[0]])]
    return miss_df, non_miss_df


def get_acs_data(years, tuple_val):
    '''
    Gets ACS data 
    Input:
        years(list): list of years we want to get
        tuple_val(tuple): tuple with all the feature names we want to obtain
    Output:
        returns dataframe with acs data
    '''
    c = Census("782594777378b4dae32651de41285b8430095ed4")
    df_acs = None
    for i,yr in enumerate(years):    
        data = c.acs5.zipcode(tuple_val, Census.ALL, year = yr)
        df_data = pd.DataFrame(data)
        df_data['year'] = int(yr)
        if df_acs is None:
            df_acs = df_data
        else:
            lst = [df_acs, df_data]
            df_acs = pd.concat(lst)
    return df_acs

def drop_features(data, features_lst):
    '''
    Drops list of features specified
    Input:
        data(dataframe): Dataframe 
        features_lst: Features to eliminate
    Output:
        Nothing, it just modifies the dataframe
    '''
    data.drop(features_lst,axis =1, inplace = True)
    return True
    
def convert_percent(df, pref):
    '''
    Converts a continuous variable into a percent variable, for acs
    Input:
        df(DataFrame): dataframe to be divided
        pref(str): prefix of the variable. 
    Output:
        dataframe with convertd percentage
    '''
    filter_col = [col for col in df if col.startswith(pref)]
    small_df = df[filter_col]
    small_df['Total'] = small_df[filter_col].sum(axis = 1)
    small_df.loc[small_df[filter_col[0]].isnull(),'Total'] = np.nan
    for col in filter_col:
        small_df[col] = np.where(small_df['Total']==0, 0, small_df[col]/small_df['Total'])
    del small_df['Total']
    return small_df

def change_cat(df,new_cat):
    '''
    Changes the name of the variables
    Input:
        df(Dataframe): affected dataframe
        new_cat(dictionary): dictionary with new categories
    Ouput:
        Nothing, it just changes the dataframe
    '''
    df.rename(columns = new_cat,inplace = True)
    return True

def unite_the_perc(pref,df, cat):
    '''
    Merges the percentage dataframe with the biggest dataset
    
    Input:
        pref(str): prefix i want to affect
        df(DataFrame): Dataframe i want to change
        cat(dictionary): dictionary i want to use to change
    Output:
        Nothing, it just changes the dataframe
    '''
    in_pct = convert_percent(df,pref)
    change_cat(in_pct,cat)
    in_pct['idx'] = np.arange(len(in_pct))
    df['idx'] = np.arange(len(df))
    df = df.merge(in_pct, on ='idx')
    return df


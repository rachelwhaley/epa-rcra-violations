'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #2: prepares the data for testing and training
'''

#Imports
import pandas as pd
import os.path
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Defined constants for the column names
PROJ_ID = 'projectid'
TEACH_ID = 'teacher_acctid'
SCHOOL_ID1 = 'schoolid'
SCHOOL_ID2 = 'school_ncesid'
LAT = 'school_latitude'
LONG = 'school_longitude'
CITY = 'school_city'
STATE = 'school_state'
METRO = 'school_metro'
DISTRICT = 'school_district'
COUNTY = 'school_county'
CHARTER = 'school_charter'
MAGNET = 'school_magnet'
PREFIX = 'teacher_prefix'
SUBJECT = 'primary_focus_subject'
AREA = 'primary_focus_area'
SUBJECT_2 = 'secondary_focus_subject'
AREA_2 = 'secondary_focus_area'
RESOURCE = 'resource_type'
POVERTY = 'poverty_level'
GRADE = 'grade_level'
PRICE = 'total_price_including_optional_support'
STUDENTS = 'students_reached'
DOUBLE = 'eligible_double_your_impact_match'
POSTED = 'date_posted'
FUNDED = 'datefullyfunded'
VAR = 'funded_in_60_days'

def import_data(csv_name):
    '''
    Imports data from a CSV file

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want

    Outputs:
        df_all_data: a pandas dataframe with all of the data unchanged
    '''
    if os.path.exists(csv_name):
        df_all_data = pd.read_csv(csv_name, parse_dates=[POSTED, FUNDED],
            infer_datetime_format=True)
    else:
        print("Pathway to the CSV does not exist")
        return None
    return df_all_data

def explore_data(df_all_data, all_cols):
    '''
    Explores the data in the CSV

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        description_dict: a dictionary describing the data in each column
    '''
    #At this point in the class, I'm really not sure what I'm supposed
    #to do with this

    description_dict = {}
    for col in all_cols:
        print(col)
        curr_series = df_all_data[col]
        #Describes the data in the column
        description_dict[col] = curr_series.describe()
        #Plots the data in the column and saves the plot
        #plt.hist(col, data=df_all_data)
        #plt.savefig(col)
    return description_dict

def clean_data(df_all_data, all_cols):
    '''
    Cleans the data

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_all_data: a cleaned pandas dataframe
    '''
    for col in all_cols:
        df_all_data[col] = df_all_data[col].fillna('None')

    all_rows = df_all_data.shape[0]
    for col in [LAT, LONG, SUBJECT, AREA, SUBJECT_2, AREA_2, RESOURCE, GRADE,
        PRICE, STUDENTS, DOUBLE]:
        num_entries = df_all_data[col].value_counts().size
        ratio = num_entries / all_rows
        if ratio > 0.60 and ratio != 1.0:
            curr_series = df_all_data[col]
            df_all_data[col] = pd.cut(curr_series, bins=10, labels=False,
                include_lowest=True)
    return df_all_data

def generate_var_feat(df_all_data, all_cols):
    '''
    Generates the variable and features for the dataset

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_all_data: a pandas dataframe with only the columns we will need
        variable: name of the variable column
        features: a list of the feature columns
        split: name of the column we will use to cordon off the training and 
            testing data
    '''
    variable = VAR
    split = POSTED
    
    #First, we create the variable: 0 of finded within 60 days and 1 if not
    df_all_data[VAR] = df_all_data[FUNDED] - df_all_data[POSTED]
    df_all_data[VAR] = df_all_data[VAR]\
        .apply(lambda x: 1 if x.days <= 60 else 0)
    
    #Now we need to find the features    
    all_cols = df_all_data.columns
    features = []
    var_series = df_all_data[variable]
    possible_features = [TEACH_ID, SCHOOL_ID1, SCHOOL_ID2, LAT, LONG, CITY,
        STATE, DISTRICT, COUNTY, CHARTER, MAGNET, PREFIX, SUBJECT, SUBJECT_2,
        AREA_2, RESOURCE, POVERTY, GRADE, PRICE, STUDENTS, DOUBLE]

    for col in possible_features:
        ser = df_all_data[col].astype(dtype='float64', errors='ignore')
        if ser.dtype == 'float64':
            col_mean = ser.mean()
            ser = ser.fillna(col_mean)
            correlation = var_series.corr(ser, method='pearson')
            if abs(correlation) > 0.01:
                df_all_data[col] = ser
                features.append(col)
    
    used_cols = features + [variable, split]
    df_all_data = drop_extra_columns(df_all_data, used_cols, all_cols)

    return df_all_data, variable, features, split

def drop_extra_columns(df_all_data, col_list, all_cols):
    '''
    Drops columns from the dataframe we are not going to use in analysis

    I am looking ahead with this function. I do not expect to use it for this
    assignment.

    Inputs:
        df_all_data: a pandas dataframe
        col_list: list of columns we are going to use
        all_cols: list of all columns in the dataframe

    Outputs:
        df_all_data: a pandas dataframe
    '''
    to_drop = []
    for col in all_cols:
        if col not in col_list:
            to_drop.append(col)
    if to_drop != []:
        df_all_data = df_all_data.drop(to_drop, axis=1)
    return df_all_data

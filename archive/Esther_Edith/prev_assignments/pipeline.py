'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 2: Machine Learning Pipeline
'''

#Imports
import pandas as pd
from sklearn.cross_validation import train_test_split
import os.path
import numpy as np
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy

#Constant for this assignment
csv_file = 'credit-data.csv'

def pipeline(csv_name=csv_file):
    '''
    Goes from the beginning to the end of the machine learning pipeline

    Inputs:
        csv_name: the pathway to a csv file that we will download data from
            It is set to the name of the csv_file that I will use for this
            assignment but can be anything
    '''
    df_all_data, var, features = import_data(csv_name)
    if var is not None:
        accuracy_dict = train_data(df_all_data, var, features)
        return accuracy_dict
    else:
        print('Data is not consistent with the assumptions of this code')

def import_data(csv_name):
    '''
    Loads data from a CSV file into a pandas datafram, processes the data,
    explores the data, and generates features and the variable

    Inputs:
        csv_name: the pathway to a csv file that we will download data from

    Outputs:
        df_all_data: a pandas data frame with the cleaned data
        var: a column name that we will want to predict
        features: a list of column names that we will use to predict the 
            variable
    '''
    if os.path.exists(csv_name):
        df_all_data = pd.read_csv(csv_name)
    else:
        print("Pathway to the CSV does not exist")
        return None, None, None, None
    all_cols = df_all_data.columns
    corr_dict, description_dict = explore_data(df_all_data, all_cols)
    df_all_data = process_data(df_all_data, all_cols)
    var, features = generate_var_feat(df_all_data, all_cols, corr_dict)
    if len(features) != len(all_cols):
        df_all_data = drop_extra_columns(df_all_data, features + [var],
            all_cols)
    #Need to update the all_cols variable
    all_cols = df_all_data.columns
    df_all_data = continuous_to_discrete(df_all_data, all_cols)
    #Splits the data into training and testing data
    return df_all_data, var, features

def explore_data(df_all_data, all_cols):
    '''
    Explores the raw data
    Note to grader/professor: I wasn't 100% certain what you wanted from this
    I have added some of the output from this into the later code and the
    writeup. I hope that is enough for you!

    Inputs:
        df_all_data: a pandas dataframe
        all_cols: list of the column names in df

    Outputs:
        corr_dict: a dictionary showing the correlation coefficient of all
           columns to each other
        description_dict: a dictinary describing the data in each column
    '''
    corr_dict = {}
    description_dict = {}
    for col in all_cols:
        curr_series = df_all_data[col]
        #Describes the data in the column
        description_dict[col] = curr_series.describe()
        corr_dict[col] = {}
        for col2 in all_cols:
            comp_series = df_all_data[col2]
            #Finds correlation between two columns
            curr_corr = curr_series.corr(comp_series, method='pearson')
            corr_dict[col][col2] = curr_corr
    return corr_dict, description_dict

def process_data(df_all_data, all_cols):
    '''
    Cleans and processes data

    The more in-depth cleaning is in anticipation of receiving data 
    I haven't seen

    Inputs:
        df_all_data: a pandas dataframe
        all_cols: list of all column names in the data frame

    Outputs:
        df_all_data: a pandas dataframe (cleaned)
    '''
    row_count, col_count = df_all_data.shape
    for col in all_cols:
        na_vals = df_all_data[col].isna().sum()
        if na_vals > (row_count / 3):
            #Deletes all cols with more than a third of entries listed as na
            df_all_data = df_all_data.drop([col], axis=1)
        else:
            for col_compare in all_cols:
                if col_compare != col:
                    if df_all_data[col].equals(df_all_data[col_compare]):
                        #Deletes cols that are duplicates
                        df_all_data = df_all_data.drop([col_compare], axis=1)
            if na_vals > 0:
                #Fills NA values wil column mean
                col_mean = df_all_data[col].mean()
                curr_series = df_all_data[col]
                df_all_data[col] = curr_series.fillna(col_mean)

    return df_all_data

def generate_var_feat(df_all_data, all_cols, corr_dict):
    '''
    Identifies which column will be the variable we want to predict and which
    columns will be the features we want to use to predict the variable

    Inputs:
       df_all_data: a pandas dataframe
       all_cols: column names in the df
       corr_dict: a dictionary of correlation between all columns

    Outputs:
        var: the column name we want to predict
        features: a list of columns we will use to predict the variable
    '''
    #First, we find the variable
    potential_var = []
    for col in all_cols:
        num_entries = df_all_data[col].value_counts().size
        if num_entries == 2:
            potential_var.append(col)
    if potential_var == []:
        print("No binary variable")
        var = None
        return var, []
    elif len(potential_var) == 1:
        var = potential_var[0]
    else:
        #Pick the variable with the strongest correlation
        var = find_strongest_corr(potential_var, corr_dict)
    
    #Now we find the features
    features = []
    for col in all_cols:
        var_corr = corr_dict[var][col]
        if abs(var_corr) > 0.01 and col != var:
            #Only adds column to features if its linear correlation with var
            #is within 0.01 of 0
            features.append(col)
    return var, features

def find_strongest_corr(potential_var, corr_dict):
    '''
    Determines which potential variable is most strongly correlated with
    the other columns in the dataframe

    Inputs:
        potential_var: list of potential variables in the df
        corr_dict: dictionary of correlation coefficients between columns
    Outputs:
        var: the variable for our database
    '''
    max_corr = 0
    best_var = None
    for col in potential_var:
        curr_corr = 0
        curr_dict = corr_dict[col]
        for key, val in curr_dict.items():
            curr_corr += abs(val)
        if curr_corr > max_corr:
            #If there is a tie, we choose the one that comes first
            max_corr = curr_corr
            best_var = col
    return best_var

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

def continuous_to_discrete(df_all_data, all_cols):
    '''
    Determines which columns can be classified as continuous and turns them
    into discrete columns

    Inputs:
        df_all_data: a pandas dataframe
        all_cols: list of all column names

    Outputs:
        df_all_data: a pandas dataframe
    '''
    all_rows = df_all_data.shape[0]
    for col in all_cols:
        num_entries = df_all_data[col].value_counts().size
        ratio = num_entries / all_rows
        if ratio > 0.80 and ratio != 1.0:
            curr_series = df_all_data[col]
            df_all_data[col] = pd.cut(curr_series, bins=10, labels=False,
                include_lowest=True)
    return df_all_data

'''
I wrote much of the followng code using help from lab2
'''

def train_data(df_all_data, var, features):
    '''
    Takes the training data and creates a model to predict futre data

    We will use a decision tree for the model and we will use different depths
    to determine the most accurate depth

    Inputs:
        df_all_data: pandas dataframe
        var: column name of variable
        features: list of column names of features

    Outpts:
        accuracy_dict: a dictionary mapping the decision tree depths to the
            accuracy of the model
    '''
    #First, we split data into testing and training
    var_data = df_all_data[var]
    feat_data = df_all_data[features]
    feat_train, feat_test, var_train, var_test = train_test_split(feat_data,
        var_data, test_size=0.1)

    #Now we create our model and test it
    accuracy_dict = {}
    model_depths = [1, 5, 10, 20, 50, 100, 200]
    for dep in model_depths:
        model = DecisionTreeClassifier(max_depth=dep)
        model.fit(feat_train, var_train)
        accuracy = test_data(model, var_test, feat_test)
        accuracy_dict[dep] = accuracy

    return accuracy_dict

def test_data(model, var_test, feat_test):
    '''
    Tests the model for accuracy

    Inputs:
        model: the machine learning model we are testing
        var_test: the variable column of the testing data
        feat_test: the feature columns of the testing data
    
    Outputs:
        thresh_acc_dict: a dictionary mapping threshold to accuracy
    '''
    test_predictions = model.predict_proba(feat_test)[:,1]

    thresh_acc_dict = {}
    test_tresholds = [0.01, 0.1, 0.4, 0.6, 0.8, 1.0]
    for threshold in test_tresholds:
        calc_threshold = lambda x,y: 0 if x < y else 1
        test = np.array([calc_threshold(score, threshold) for score in
            test_predictions])
        test_acc = accuracy(test, var_test)
        thresh_acc_dict[threshold] = test_acc

    return thresh_acc_dict

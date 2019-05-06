'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #1: putting the pipeline together
'''

#Imports
import pandas as pd
import numpy as np
import prep_data
import modeling

#Constant for this assignment
csv_file = 'projects_2012_2013.csv'

def pipeline(csv_name=csv_file):
    '''
    Goes from the beginning to the end of the machine learning pipeline

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want
            (this is initialized to the CSV file we were given for this
            assignment)

    Outputs:
        models_eval: a pandas dataframe of the different models we have tested,
            the different parameters we have tried on them and the evaluation
            metrics we have used
    '''

    print('Importing')
    df_all_data = prep_data.import_data(csv_name)
    if df_all_data is None:
        return None
    all_cols = df_all_data.columns

    print('Exploring')
    descriptions = prep_data.explore_data(df_all_data, all_cols)
    print('Cleaning')
    df_all_data = prep_data.clean_data(df_all_data, all_cols)

    print('Generating Var and Feat')
    df_all_data, variable, features, split = prep_data.generate_var_feat(
        df_all_data, all_cols)
    df_all_data.to_csv("Data_For_Eval.csv")

    print('Modeling')
    models_dict = modeling.split_by_date(df_all_data, split, variable, features)
    
    print('Creating final table')
    return table_models_eval(models_dict)

def table_models_eval(models_eval):
    '''
    Loops through the dictionary of models we have created 
    and puts those results into a pandas dataframe

    Inputs:
        models_eval: all the models we have created and the evaluation for them
    Output:
        df_evaluated_models: a dataframe listing the models, their evaluation
            metric and how well those models did on that metric
    '''
    col_lst = ['Date', 'Model Name', 'Parameters', 'Evaluation Name',
        'Threshold', 'Result']
    df_lst = []

    for dates, model_dict in models_eval.items():
        for model, param_dict in model_dict.items():
            for param, eval_dict in param_dict.items():
                for threshold, eval_outcome_dict in eval_dict.items():
                    for eval_name, outcome in eval_outcome_dict.items():
                        this_lst = [dates, model, param, eval_name, threshold,\
                            outcome]
                        df_lst.append(this_lst)

    df_evaluated_models = pd.DataFrame(np.array(df_lst), columns=col_lst)

    df_evaluated_models.to_csv("Modeling_Projects_2012_2013.csv")
    return df_evaluated_models

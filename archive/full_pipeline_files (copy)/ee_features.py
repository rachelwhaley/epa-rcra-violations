'''
The Features Esther Edith is creating
'''

import pandas as pd

def num_inspections(yr1, yr2=None):
    '''
    Generates features based on the number of inspections
    yr1: we want evaluations before this
    yr2: if we want to filter dates between two dates
    '''
    csv_name = 'RCRA_EVALUATIONS.csv'
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    m = 'M'
    d = 'D'
    y = 'Y'

    df_ins = pd.read_csv(csv_name, usecols=[0,6])
    df_ins[[m,d,y]] = df_ins[date].str.split('/',expand=True)
    df_ins[y] = pd.to_numeric(df_ins['Y'], downcast='integer')
    if yr2 is not None:
        filt = \
            (df_ins[y] <= yr1)&\
            (df_ins[y] >= yr2)
    else:
        filt = df_ins[y] <= yr1
    df_ins = df_ins[filt]
    return df_ins.groupby(by=ids).size()

def corrective_event():
    '''
    Generates features based on a corrective action event
    
    I NEED TO RE-DOWNLOAD THIS DATA
    '''
    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0,5])
    return df_ca

def type_waste():
    '''
    Generates features based on the type of waste created
    
    I NEED TO FIGURE OUT HOW TO HANDLE THE SIZE OF THIS FILE
    '''
    csv_name = 'Biennial_Report_GM_Waste_Code.csv'
    df_wc = pd.read_csv(csv_name, header=[0,6])
    return df_wc

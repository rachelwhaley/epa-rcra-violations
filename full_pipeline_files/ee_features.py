'''
The Features Esther Edith is creating
'''

import pandas as pd

def time_late(date1, date2, df_all_data):
    '''
    Filter by zip codes

    Filter by all facility ids
    '''
    #You can change these if need be
    #I am assuming all of these will be in the df passed here
    ids = 'ID_NUMBER'
    date_to_split = 'ACTUAL_RTC_DATE'
    actual = date_to_split
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early'
    late = 'late'

    #So this dictionary is going to give you wat to merge on, what the name
    #of the feature column should be and the groupby object you can merge with
    features = {ids: {}, zips: {}, states: {}}

    df_all_data[diff] = df_all_data[actual] - df_all_data[scheduled]
    df_all_data[early] = df_all_data[diff]\
        .apply(lambda x: 0 if x.days > 0 else 1)
    df_all_data[late] = df_all_data[diff]\
        .apply(lambda x: 0 if x.days < 0 else 1)

    filt_between =\
        (df_all_data[date_to_split] <= date1) &\
        (df_all_data[date_to_split] >= date2)
    filt_before = (df_all_data[date_to_split] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    for col in [early, late]:
        for group in [ids, zips, states]:
            for db_label in [(filt_between, " between"), (filt_before, " before")]:
                db, label = db_label
                name = "average time " + col + " for " + group + label
                filt = db[col] == 1
                our_db = db[filt]
                features[group][name + ": avg"] = our_db.groupby([group])[diff].mean()
                features[group][name + ": sum"] = our_db.groupby([group])[diff].sum()

    return features

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

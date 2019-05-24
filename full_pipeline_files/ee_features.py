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
    loc = 'ACTIVITY_LOCATION'
    diff = 'difference'
    early = 'early '
    late = 'late '
    between = ' between: ' + str(date2) + " - " + str(date1)
    before = ' before: '+ str(date1)

    df_all_data[diff] = df_all_data[actual] - df_all_data[scheduled]
    df_all_data[diff] = df_all_data[diff]\
        .apply(lambda x: x.days)
    df_all_data[early] = df_all_data[diff]\
        .apply(lambda x: 0 if x >= 0 else 1)
    df_all_data[late] = df_all_data[diff]\
        .apply(lambda x: 0 if x <= 0 else 1)

    filt_between =\
        (df_all_data[date_to_split] <= date1) &\
        (df_all_data[date_to_split] >= date2)
    filt_before = (df_all_data[date_to_split] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    for col in [early, late]:
        for db_label in [(df_between, between), (df_before, before)]:
            db, label = db_label
            label = col + " " + label
            filt = (db[col] == 1)
            our_db = db[filt]
            for group in [ids, zips, states, loc]:
                label += " " + group
                avg = our_db.groupby(group)\
                    [diff].mean().reset_index().rename(\
                    columns={diff:label+" avg"})
                sums = our_db.groupby(group)\
                    [diff].sum().reset_index().rename(\
                    columns={diff:label+" sum"})
                count = our_db.groupby(group)\
                    [col].sum().reset_index().rename(\
                    columns={col:label+" count"})
                for gb in [avg, sums, count]:
                    df_all_data = pd.merge(df_all_data, gb,\
                        on=group, how='left')

    return df_all_data

def num_inspections(date1, date2, df_all_data):
    '''
    Generates features based on the number of inspections
    yr1: we want evaluations before this
    yr2: if we want to filter dates between two dates
    '''
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    loc = 'ACTIVITY_LOCATION'

    filt_between =\
        (df_all_data[date_to_split] <= date1) &\
        (df_all_data[date_to_split] >= date2)
    filt_before = (df_all_data[date_to_split] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    for group in [ids, zips, states, loc]:
        for db in [filt_between, filt_before]:
            sums = db.groupby(group)\
                .size().reset_index()
            df_all_data = pd.merge(df_all_data, sums,\
                on=group, how='left')

    return df_all_data

def corrective_event():
    '''
    Generates features based on a corrective action event
    
    I NEED TO RE-DOWNLOAD THIS DATA
    '''
    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0,5])
    return df_ca

def type_waste(df_all_data):
    '''
    Generates features based on the type of waste created
    
    I NEED TO FIGURE OUT HOW TO HANDLE THE SIZE OF THIS FILE
    '''
    #csv_name = 'Biennial_Report_GM_Waste_Code.csv'
    #df_wc = pd.read_csv(csv_name, header=[0,6])
    #I'm assuming the the three below columns will all be merged into the db
    waste_codes = 'Hazardous Waste Code'
    code_owner = 'Hazardous Waste Code Owner'
    naics = 'NAICS_CODE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    #loc = 'ACTIVITY_LOCATION'
    loc = 'Activity Location'

    #for col in [waste_codes, code_owner, naics]:
    for col in [waste_codes, code_owner]:
        ser = df_all_data[col]
        val_unique = ser.unique()
        for val in val_unique:
            new_col = col + str(val)
            df_all_data[new_col] = df_all_data[waste_codes]\
                .apply(lambda x: 1 if x == val else 0)
            #for group in [zips, states, loc]:
            for group in [loc]:
                to_merge = df_all_data.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                df_all_data = pd.merge(df_all_data, to_merge,\
                    on=group,how='left')
        
    return df_all_data

def num_facilities(df_all_data):
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    loc = 'ACTIVITY_LOCATION'

    for group in [zips, states, loc]:
        sums = db.groupby(group).size().reset_index()
        df_all_data = pd.merge(df_all_data, sums, on=group, how='left')

    return df_all_data  

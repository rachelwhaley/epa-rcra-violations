'''
The Features Esther Edith is creating
'''

import pandas as pd

def time_late(date1, date2, df_all_data):
    '''
    !!!DONE!!!

    Calculates:
        Number of times early/late
        Average time early/late
        Total time early/late
        Number of days since early/late

        for a sinlg location
        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

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
            for group in [ids, zips, states]:
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
                last = our_db.groupby(group)\
                    [actual].max()\
                    .apply(lambda x: (date1 - x).days)\
                    .reset_index().rename(\
                    columns={actual:"last " + label})
                for gb in [avg, sums, count, last]:
                    df_all_data = pd.merge(df_all_data, gb,\
                        on=group, how='left')

    return df_all_data

def num_inspections(date1, date2, df_all_data):
    '''
    !!!DONE!!!
    Calculates:
        Number of inspections
        Days since last inspection
        
        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

    Generates features based on the number of inspections
    date1: we want evaluations before this
    date2: if we want to filter dates between two dates
    '''
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    filt_between =\
        (df_all_data[date] <= date1) &\
        (df_all_data[date] >= date2)
    filt_before = (df_all_data[date] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    #for group in [ids, zips, states]:
    for group in [ids, 'ACTIVITY_LOCATION']:
        for db in [df_between, df_before]:
            sums = db.groupby(group)\
                .size().reset_index()
            last = db.groupby(group)\
                [date].max()\
                .apply(lambda x: (date1 - x).days)\
                .reset_index()
            for gb in [sums, last]:
                df_all_data = pd.merge(df_all_data, gb,\
                    on=group, how='left')

    return df_all_data

def corrective_event(date1, date2, df_all_data):
    '''
    Generates features based on a corrective action event
    '''
    date = 'Actual Date of Event'
    event = 'Corrective Action Event Code'

    filt_between =\
        (df_all_data[date] <= date1) &\
        (df_all_data[date] >= date2)
    filt_before = (df_all_data[date] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    val_unique = df_all_data[event].unique()
    for val in val_unique:
        for df in [df_between, df_before]:
            new_col = str(val)
            df_all_data[new_col] = df_all_data[events]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = df_all_data.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                df_all_data = pd.merge(df_all_data, to_merge,\
                    on=group,how='left')


    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0,5])
    return df_ca

def type_waste(df_all_data):
    '''
    !!!DONE!!!

    calculates:
        dummy variable for waste code/ code owner/ naics code
            for a single facility
        calculates all facilities with waste code/ code owner/ naics code
            in a zip/state

    Generates features based on the type of waste created
    '''
    #csv_name = 'Biennial_Report_GM_Waste_Code.csv'
    #df_wc = pd.read_csv(csv_name, header=[0,6])
    #I'm assuming the the three below columns will all be merged into the db
    waste_codes = 'Hazardous Waste Code'
    code_owner = 'Hazardous Waste Code Owner'
    naics = 'NAICS_CODE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    for col in [waste_codes, code_owner, naics]:
        ser = df_all_data[col]
        val_unique = ser.unique()
        for val in val_unique:
            new_col = col + str(val)
            df_all_data[new_col] = df_all_data[waste_codes]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = df_all_data.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                df_all_data = pd.merge(df_all_data, to_merge,\
                    on=group,how='left')
        
    return df_all_data

def num_facilities(df_all_data):
    '''
    !!!DONE!!!

    Calculates:
        Number of facilities in a zip code and state
    '''
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    for group in [zips, states]:
        sums = df_all_data.groupby(group).size().reset_index()
        df_all_data = pd.merge(df_all_data, sums, on=group, how='left')
    return df_all_data

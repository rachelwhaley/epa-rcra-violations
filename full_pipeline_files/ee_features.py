'''
The Features Esther Edith is creating
'''

import pandas as pd

def create_final(df_all_data):
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    #al = 'ACTIVITY_LOCATION'
    gb = df_all_data.groupby([ids, zips, states])\
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    #gb = df_all_data.groupby([ids, al])\
        #.size().reset_index()
    #d = {ids: gb[ids], al: gb[al]}
    final_df = pd.DataFrame(data=d)
    return final_df

def time_late(df_all_data, max_date):
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
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early '
    late = 'late '
    #al = 'ACTIVITY_LOCATION'

    final_df = create_final(df_all_data)

    df_all_data[diff] = df_all_data[actual] - df_all_data[scheduled]
    df_all_data[diff] = df_all_data[diff]\
        .apply(lambda x: x.days)
    df_all_data[early] = df_all_data[diff]\
        .apply(lambda x: 0 if x >= 0 else 1)
    df_all_data[late] = df_all_data[diff]\
        .apply(lambda x: 0 if x <= 0 else 1)

    
    gb = df_all_data.groupby([ids, zips, states]).ffill()\
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    #gb = df_all_data.groupby([ids, al])\
        #.size().reset_index()
    #d = {ids: gb[ids], al: gb[al]}
    final_df = pd.DataFrame(data=d)

    for col in [early, late]:
        filt = (df_all_data[col] == 1)
        our_db = df_all_data[filt]
        #for group in [ids, al]:
        for group in [ids, zips, states]:
            label = col + " " + group
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
                .apply(lambda x: (max_date - x).days)\
                .reset_index().rename(\
                columns={actual:"last " + label})
            for gb_bool in [(avg, False), (sums, False), (count, False),\
                (last, True)]:
                gb, bool_val = gb_bool
                final_df = pd.merge(final_df, gb,\
                    on=group, how='left')
                if bool_val:
                    to_fill = "last " + label
                    final_df[to_fill] = final_df[to_fill]\
                        .fillna(value=float('Inf'))

    return final_df

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

    final_df = create_final(df_all_data)

    for group in [ids, zips, states]:
        sums = db.groupby(group)\
            .size().reset_index()
        last = db.groupby(group)\
            [date].max()\
            .apply(lambda x: (date1 - x).days)\
            .reset_index().rename{columns={date:group+" last"}}
        for gb_bool in [(sums, False), (last, True)]:
        	gb, bool_val = gb_bool
            final_df = pd.merge(final_df, gb,\
                on=group, how='left')
            if bool_val:
            	to_fill = group+" last"
                final_df[to_fill] = final_df[to_fill]\
                    .fillna(value=float('Inf'))

    return final_df

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

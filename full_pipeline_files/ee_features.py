'''
The Features Esther Edith is creating
'''

import pandas as pd

def create_final(facilities_df):
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    al = 'ACTIVITY_LOCATION'
    gb = facilities_df.groupby([ids, zips, states])\
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    #gb = facilities_df.groupby([ids, al])\
        #.size().reset_index()
    #d = {ids: gb[ids], al: gb[al]}
    factlities_with_features_df = pd.DataFrame(data=d)
    return factlities_with_features_df

def time_late(facilities_df, max_date):
    '''
    Inputs:
        facilities_df: a dataframe with all violations information
        max_date: the maximum date in our training/testing set
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

    factlities_with_features_df = create_final(facilities_df)

    facilities_df[diff] = facilities_df[actual] - facilities_df[scheduled]
    facilities_df[diff] = facilities_df[diff]\
        .apply(lambda x: x.days)
    facilities_df[early] = facilities_df[diff]\
        .apply(lambda x: 0 if x >= 0 else 1)
    facilities_df[late] = facilities_df[diff]\
        .apply(lambda x: 0 if x <= 0 else 1)

    
    gb = facilities_df.groupby([ids, zips, states]).ffill()\
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    #gb = facilities_df.groupby([ids, al])\
        #.size().reset_index()
    #d = {ids: gb[ids], al: gb[al]}
    factlities_with_features_df = pd.DataFrame(data=d)

    for col in [early, late]:
        filt = (facilities_df[col] == 1)
        our_db = facilities_df[filt]
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
                factlities_with_features_df = pd.merge(factlities_with_features_df, gb,\
                    on=group, how='left')
                if bool_val:
                    to_fill = "last " + label
                    factlities_with_features_df[to_fill] = factlities_with_features_df[to_fill]\
                        .fillna(value=float('Inf'))

    return factlities_with_features_df

def num_inspections(facilities_df, max_date):
    '''
    Inputs:
        facilities_df: a dataframe with all evaluations information
        max_date: the maximum date in our training/testing set

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
    al = 'ACTIVITY_LOCATION'

    factlities_with_features_df = create_final(facilities_df)
    
    #for group in [ids, al]:
    for group in [ids, zips, states]:
        sums = facilities_df.groupby(group)\
            .size().reset_index()
        last = facilities_df.groupby(group)\
            [date].max()\
            .apply(lambda x: (max_date - x).days)\
            .reset_index().rename(columns={date:group+" last"})
        for gb_bool in [(sums, False), (last, True)]:
            gb, bool_val = gb_bool
            factlities_with_features_df = pd.merge(\
            	factlities_with_features_df, gb,\
                on=group, how='left')
            if bool_val:
                to_fill = group+" last"
                factlities_with_features_df[to_fill] = factlities_with_features_df[to_fill]\
                    .fillna(value=float('Inf'))

    return factlities_with_features_df.rename(\
        columns={'0_x':"eval_sum_zips", '0_y':"eval_sum_states"})

def corrective_event(date1, date2, facilities_df):
    '''
    Generates features based on a corrective action event
    '''
    '''
    date = 'Actual Date of Event'
    event = 'Corrective Action Event Code'

    filt_between =\
        (facilities_df[date] <= date1) &\
        (facilities_df[date] >= date2)
    filt_before = (facilities_df[date] <= date1)
    
    df_between = facilities_df[filt_between]
    df_before = facilities_df[filt_before]

    val_unique = facilities_df[event].unique()
    for val in val_unique:
        for df in [df_between, df_before]:
            new_col = str(val)
            facilities_df[new_col] = facilities_df[events]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = facilities_df.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                facilities_df = pd.merge(facilities_df, to_merge,\
                    on=group,how='left')


    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0,5])
    return df_ca
    '''

def type_waste(facilities_df):
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
        ser = facilities_df[col]
        val_unique = ser.unique()
        for val in val_unique:
            new_col = col + str(val)
            facilities_df[new_col] = facilities_df[waste_codes]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = facilities_df.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                facilities_df = pd.merge(facilities_df, to_merge,\
                    on=group,how='left')
        
    return facilities_df

def num_facilities(facilities_df):
    '''
    !!!DONE!!!

    Calculates:
        Number of facilities in a zip code and state
    '''
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    for group in [zips, states]:
        sums = facilities_df.groupby(group).size().reset_index()
        facilities_df = pd.merge(facilities_df, sums, on=group, how='left')
    return facilities_df.rename(\
        columns={'0_x':"facilities_in_zip", '0_y':"facilities_in_state"})

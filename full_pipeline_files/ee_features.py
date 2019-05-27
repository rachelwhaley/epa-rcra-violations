'''
The Features Esther Edith is creating
'''

import pandas as pd

def create_final(more_info_df,facilities_df):
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    al = 'ACTIVITY_LOCATION'
    more_info_df = pd.merge(more_info_df,\
        facilities_df[[ids, zips, states]], on=ids,\
        how='left')
    gb = more_info_df.groupby([ids, zips, states])\
        .size().reset_index()
    data = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    facilities_with_features_df = pd.DataFrame(data=data)
    return facilities_with_features_df

def time_late(violations_df, max_date, facilities_df):
    '''
    Inputs:
        violations_df: a dataframe with all violations information
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

    facilities_with_features_df = create_final(violations_df, facilities_df)

    violations_df[diff] = violations_df[actual] - violations_df[scheduled]
    violations_df[diff] = violations_df[diff]\
        .apply(lambda x: x.days)
    violations_df[early] = violations_df[diff]\
        .apply(lambda x: 0 if x >= 0 else 1)
    violations_df[late] = violations_df[diff]\
        .apply(lambda x: 0 if x <= 0 else 1)

    for col in [early, late]:
        filt = (violations_df[col] == 1)
        our_db = violations_df[filt]
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
                facilities_with_features_df = pd.merge(facilities_with_features_df, gb,\
                    on=group, how='left')
                if bool_val:
                    to_fill = "last " + label
                    facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill]\
                        .fillna(value=float('Inf'))

    return facilities_with_features_df

def num_inspections(evals_df, max_date, facilities_df):
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

    facilities_with_features_df = create_final(evals_df, facilities_df)
    
    #for group in [ids, al]:
    for group in [ids, zips, states]:
        sums = evals_df.groupby(group)\
            .size().reset_index()
        last = evals_df.groupby(group)\
            [date].max()\
            .apply(lambda x: (max_date - x).days)\
            .reset_index().rename(columns={date:group+" last"})
        for gb_bool in [(sums, False), (last, True)]:
            gb, bool_val = gb_bool
            facilities_with_features_df = pd.merge(\
            	facilities_with_features_df, gb,\
                on=group, how='left')
            if bool_val:
                to_fill = group+" last"
                facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill]\
                    .fillna(value=float('Inf'))
    return facilities_with_features_df.rename(\
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

def type_waste(waste_codes_df, naics_df, facilities_df):
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
    ids_waste = 'EPA Handler ID'
    ids_naics = 'ID_NUMBER'
    waste_codes = 'Hazardous Waste Code'
    code_owner = 'Hazardous Waste Code Owner'
    naics = 'NAICS_CODE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    naics_df = pd.merge(\
        naics_df[[ids_naics, naics]],\
        facilities_df[[ids_naics, zips, states]],
        on=ids_naics, how='left')

    facilities_with_features_df = pd.merge(\
        naics_df[[ids_naics, naics, zips, states]],\
        waste_codes_df[[ids_waste, waste_codes, code_owner]],\
        left_on=ids_naics, right_on=ids_waste,\
        how='left')

    for col in [waste_codes, code_owner, naics]:
        ser = facilities_with_features_df[col]
        val_unique = ser.unique()
        for val in val_unique:
            new_col = col + str(val)
            facilities_with_features_df[new_col] = facilities_with_features_df[waste_codes]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = facilities_with_features_df.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                facilities_with_features_df = pd.merge(facilities_with_features_df, to_merge,\
                    on=group,how='left')
        
    return facilities_with_features_df

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

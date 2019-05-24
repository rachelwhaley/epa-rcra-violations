from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import acs_features as af
from census_area import Census

#%whos DataFrame

##OUT FOR THE MOMENT
#,'B15003_002E','B15003_003E',
#                             'B15003_004E','B15003_005E','B15003_006E','B15003_007E','B15003_008E',
#                             'B15003_009E','B15003_010E','B15003_011E','B15003_012E','B15003_013E',
#                             'B15003_014E','B15003_015E','B15003_016E','B15003_017E','B15003_018E',
#                             'B15003_019E','B15003_020E','B15003_021E','B15003_022E',
#                             'B15003_023E','B15003_024E'

#'/Users/csolisu/Downloads/zip_to_zcta_2017.xlsx'
#'/Users/csolisu/Downloads/tl_2017_us_zcta510/tl_2017_us_zcta510.shp'
def go(csv_path_facilities,csv_path_evalu, zip_to_zta_csv,shp_file):
    fac = exp.read_data(csv_path_facilities)
    evalu = exp.read_data(csv_path_evalu)
    years = [2011, 2012, 2013, 2014, 2015, 2016]

    
    race_cat = {'B02001_002E': 'white alone', 
                'B02001_003E': 'black alone',
                'B02001_004E': 'native alone',
                'B02001_005E': 'asian alone',
                'B02001_006E': 'pacific alone',
                'B02001_007E': 'other alone',
                'B02001_008E': 'two or more',
                'B02001_009E': 'two or more some other'}

    inc_cat = { 'B19001_002E': 'less10k', 'B19001_003E': '10kto15k',
                'B19001_004E': '15kto20k', 'B19001_005E': '20kto25k',
                'B19001_006E': '25kto30k', 'B19001_007E': '30kto35k',
                'B19001_008E': '35kto40k', 'B19001_009E': '40kto45k',
                'B19001_010E': '45kto50k', 'B19001_011E': '50kto55k',
                'B19001_012E': '60kto75k', 'B19001_013E': '75kto100k',
                'B19001_014E': '100kto125k', 'B19001_015E': '125kto145k',
                'B19001_016E': '150kto200k', 'B19001_017E': '200kmore'}

    family_cat = {'B11016_003E': '2 person',
                  'B11016_004E': '3 person',
                  'B11016_005E': '4 person',
                  'B11016_006E': '5 person',
                  'B11016_007E': '6 person',
                  'B11016_008E': '7plusperson'} 

    ratio_pov_cat_fam = {
                'B17026_002E': 'under_p5', 
                'B17026_003E': 'p5top74',
                'B17026_004E': 'p75top99',
                'B17026_005E': '1to1p24',
                'B17026_006E': '1p25to1p49',
                'B17026_007E': '1p50to1p74',
                'B17026_008E': '1p75to1p84',
                'B17026_009E': '1p85to1p99',
                'B17026_010E': '2to2p99',
                'B17026_011E': '3to3p99',
                'B17026_012E': '4to4p99',
                'B17026_013E': '5andover'
    }

    ratio_pov_cat_peop = {
                'C17002_002E': 'under_p5', 
                'C17002_003E': 'p5top99',
                'C17002_004E': '1to1p24',
                'C17002_005E': '1p25to1p49',
                'C17002_006E': '1p50to1p84',
                'C17002_007E': '1p85to1p99',
                'C17002_007E': '2andver' }

    pop_cat = {'B01003_001E': 'population'}
    med_inc_cat = {'B19013_001E': 'median income'}

    educ_cat = { 'B15003_002E':'no school',
                 'B15003_003E':'nursery'  ,
                 'B15003_004E':'kindergarten'  ,
                 'B15003_005E':'1stgrade' ,
                 'B15003_006E':'2ndgrade',
                 'B15003_007E':'3rdgrade' ,
                 'B15003_008E':'4thgrade',
                 'B15003_009E':'5thgrade',
                 'B15003_010E':'6thgrade',
                 'B15003_011E':'7thgrade',
                 'B15003_012E':'8thgrade',
                 'B15003_013E':'9thgrade',
                 'B15003_014E':'10thgrade',
                 'B15003_015E':'11thgrade',
                 'B15003_016E':'12thgrade',
                 'B15003_017E':'regular_hsd',
                 'B15003_018E':'ged',
                 'B15003_019E':'some college',
                 'B15003_020E':'some college no degree',
                 'B15003_02E1':'associate degree',
                 'B15003_022E':'bachelor',
                 'B15003_023E':'master',
                 'B15003_023E':'professional school' ,
                 'B15003_024E':'doctorate'           
               }

    dcat = {'B02001_00':race_cat,'B19001_00':inc_cat, 'B11016_00':family_cat, 'B17026_00':ratio_pov_cat_fam,
            'C17002_00':ratio_pov_cat_peop,'C17002_00':ratio_pov_cat_peop}

    unique_cat = {'B01003_00': pop_cat,'B19013_00':med_inc_cat}


    df_acs= af.get_acs_data(years, ('NAME','B19013_001E','B01003_001E','B19001_002E','B19001_003E', 'B19001_004E',
                                 'B19001_005E', 'B19001_006E','B19001_007E', 'B19001_008E',
                                 'B19001_009E', 'B19001_010E','B19001_011E', 'B19001_012E',
                                 'B19001_013E', 'B19001_014E','B19001_015E', 'B19001_016E','B19001_017E',
                                 'B02001_002E','B02001_003E', 'B02001_004E',
                                 'B02001_005E', 'B02001_006E','B02001_007E', 'B02001_008E',
                                 'B02001_009E', 'B02001_010E','B11016_002E',
                                 'B11016_003E', 'B11016_004E','B11016_005E', 'B11016_006E',
                                 'B11016_007E', 'B11016_008E','B17026_002E',
                                 'B17026_003E','B17026_004E', 'B17026_005E','B17026_006E','B17026_007E',
                                 'B17026_008E','B17026_009E','B17026_010E','B17026_011E','B17026_012E','B17026_013E',
                                 'C17002_002E','C17002_003E','C17002_004E','C17002_005E',
                                 'C17002_006E','C17002_007E','C17002_008E'))


    geo_zcta = gpd.read_file(shp_file)


    drop_eval = ['ACTIVITY_LOCATION_y', 'ALAND10','AWATER10','INTPTLAT10',
    'INTPTLON10', 'LATITUDE83', 'LONGITUDE83', 'MTFCC10','PO_NAME', 'STATE',
    'ZIP_TYPE', 'Zip_join_type', 'index_right']

    fac = fac.astype({"ZIP_CODE": str})
    fac['ZIP_CODE'] = fac['ZIP_CODE'].str.split('-').str[0]

    #pref = {'Race':'B02001_00', 'Income':'B19001_00', 'Family Size':'B11016_00','Income Poverty':'B17026_00'}
    evalu_11 = af.smaller_data_time(evalu, 'EVALUATION_START_DATE', 2011) 
    evalu_11['year'] = evalu_11['temp'].dt.year
    evalu_11['year2'] = evalu_11['year']

    evalu_11.loc[evalu_11['year'] == 2018, 'year2'] = 2016
    evalu_11.loc[evalu_11['year'] == 2017, 'year2'] = 2016

    df = pd.merge(evalu_11, fac, how='left', on='ID_NUMBER')

    evalu_miss, evalu_nonmiss = af.keep_miss_nonmiss(df, ['LONGITUDE83','LATITUDE83'])
     
    evalu_nonmiss_geom = evalu_nonmiss.apply(lambda x : Point([x['LONGITUDE83'],x['LATITUDE83']]),axis = 1)

    geo_evalu_nonmiss = gpd.GeoDataFrame(evalu_nonmiss, geometry = evalu_nonmiss_geom)
        
    geo_evalu_nonmiss_zcta = gpd.sjoin(geo_evalu_nonmiss, geo_zcta, how = "left", op='intersects')


    geo_evalu_nonmiss_acs = pd.merge(geo_evalu_nonmiss_zcta,df_acs, how= 'left', 
                       right_on= ['zip code tabulation area', 'year'],
                               left_on = ['ZCTA5CE10', 'year2'])

    zip_to_zta = pd.read_excel(zip_to_zta_csv,
                                converters={'ZIP_CODE': '{:0>5}'.format,'ZCTA': '{:0>5}'.format })

    evalu_miss_zta = pd.merge(evalu_miss, zip_to_zta, how='left', on = "ZIP_CODE")

    geo_evalu_miss_acs = pd.merge(evalu_miss_zta,df_acs, how= 'left', 
                       right_on= ['zip code tabulation area', 'year'],
                               left_on = ['ZCTA', 'year2'])

    entire_df = pd.concat([geo_evalu_nonmiss_acs, geo_evalu_miss_acs])
    af.drop_features(entire_df, drop_eval)

    for key, val in dcat.items():
        entire_df = af.unite_the_perc(key,entire_df,val)

    for _, val in unique_cat.items():
        af.change_cat(entire_df,val)



    drop_fac = ['ALAND10','AWATER10','INTPTLAT10',
    'INTPTLON10', 'LATITUDE83', 'LONGITUDE83', 'MTFCC10','PO_NAME', 'STATE',
    'ZIP_TYPE', 'Zip_join_type', 'index_right']


    df_acs_2016 = df_acs[df_acs['year'] == 2016]
    fac_miss, fac_nonmiss = af.keep_miss_nonmiss(fac, ['LONGITUDE83','LATITUDE83'])
    fac_nonmiss_geom = fac_nonmiss.apply(lambda x : Point([x['LONGITUDE83'],x['LATITUDE83']]),axis = 1)
    geo_fac_nonmiss = gpd.GeoDataFrame(fac_nonmiss, geometry = fac_nonmiss_geom)    
    geo_fac_nonmiss_zcta = gpd.sjoin(geo_fac_nonmiss, geo_zcta, how = "left", op='intersects')
    geo_fac_nonmiss_acs = pd.merge(geo_fac_nonmiss_zcta,df_acs_2016, how= 'left', 
                       right_on= ['zip code tabulation area'],
                               left_on = ['ZCTA5CE10'])

    fac_miss_zta = pd.merge(fac_miss, zip_to_zta, how='left', on = "ZIP_CODE")
    geo_fac_miss_acs = pd.merge(fac_miss_zta,df_acs_2016, how= 'left', 
                       right_on= ['zip code tabulation area'],
                               left_on = ['ZCTA'])
    entire_fac = pd.concat([geo_fac_nonmiss_acs, geo_fac_miss_acs])
    af.drop_features(entire_fac, drop_fac)

    for key, val in dcat.items():
        entire_fac = af.unite_the_perc(key,entire_fac,val)

    for _, val in unique_cat.items():
        af.change_cat(entire_fac,val)
    
    return entire_df, entire_fac

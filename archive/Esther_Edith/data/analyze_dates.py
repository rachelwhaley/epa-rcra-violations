'''
Esther Edith Spurlock

Group Project

Analyzing dates for compliance
'''
#Imports
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Column Constants
csv_name = "RCRA_VIOLATIONS.csv"
ID = "ID_NUMBER"
LOC = "ACTIVITY_LOCATION"
TYPE = "VIOLATION_TYPE"
DESC = "VIOLATION_TYPE_DESC"
AGEN = "VIOL_DETERMINED_BY_AGENCY"
FOUND = "DATE_VIOLATION_DETERMINED"
COMPLIANCE = "ACTUAL_RTC_DATE"
SCHEDULED = "SCHEDULED_COMPLIANCE_DATE"
COMP_TIME = "FOUND_TO_COMPLIANCE"
SHCE_COMP_TIME = "FOUND_TO_SCHEDULED_COMPLIANCE"
DIFF = "SCHEDULED_COMPLIANCE_TO_ACTUAL_COMPLIANCE"
HAS_BAD = "CONTAINS BAD VALUE"

def find_dates():
    data_df = pd.read_csv(csv_name)
    data_df = data_df.dropna(subset=[FOUND, COMPLIANCE, SCHEDULED])
    data_df[HAS_BAD] = data_df[FOUND].str.endswith('0001').sort_values()
    date_filter = data_df[HAS_BAD] == False
    data_df = data_df[date_filter]

    return data_df


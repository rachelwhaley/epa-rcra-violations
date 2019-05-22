import sys
from datetime import datetime
from pipeline_library import *


def create_features(facilities_df, evals_df):
    # create features
    evals_counts = evals_df.groupby("ID_NUMBER").size().reset_index().rename(columns={
        0: 'EvalCount'}).sort_values("EvalCount",ascending=False)

    facilities_w_counts = pd.merge(evals_counts, facilities_df, left_on='ID_NUMBER', right_on='ID_NUMBER', how='left')

    evals_df["Y"] = pd.Series(np.where(evals_df.FOUND_VIOLATION.str.contains("Y"), 1, 0),
                                               evals_df.index)

    violation_sums = evals_df.groupby("ID_NUMBER")['Y'].sum().reset_index().rename(
        columns={'Y': 'Sum_Violations'}).sort_values("Sum_Violations", ascending=False)

    facilities_w_violations = pd.merge(facilities_w_counts, violation_sums, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')

    facilities_w_violations["PCT_EVALS_FOUND_VIOLATION"] = facilities_w_violations["Sum_Violations"] / facilities_w_violations["EvalCount"]

    facilities_w_violations["PCT_OF_ALL_EVALS"] = facilities_w_violations["EvalCount"] / facilities_w_violations["EvalCount"].sum()

    facilities_w_violations["PCT_OF_ALL_VIOLATIONS"] = facilities_w_violations["Sum_Violations"] / facilities_w_violations["Sum_Violations"].sum()

    max_date = evals_df.groupby("ID_NUMBER").agg({'EVALUATION_START_DATE': 'max'})[
        ['EVALUATION_START_DATE']].reset_index().rename(columns={'EVALUATION_START_DATE': 'MostRecentEval'})

    facilities_w_violations = pd.merge(facilities_w_violations, max_date, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')

    facilities_w_violations["MostRecentEval"] = pd.to_datetime(facilities_w_violations["MostRecentEval"],
                                                                 format='%m/%d/%Y', errors='coerce')

    facilities_w_violations["NumMonthsSinceEval"] = (datetime.today() - facilities_w_violations["MostRecentEval"]) / (
        np.timedelta64(1, 'M'))

    return facilities_w_violations


def main():
    if len(sys.argv) != 3:
        print("Usage: analyze_projects.py <facilities_filename> <evals_filename>", file=sys.stderr)
        sys.exit(1)

    # read in, process, and explore data
    facilities_df = read_data(sys.argv[1])
    evals_df = read_data(sys.argv[2])

    facilities_w_violations = create_features(facilities_df, evals_df)

    print(facilities_w_violations.info())


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as accuracy
import sys
from sklearn.metrics import precision_recall_curve
from scipy import stats

# References include: documentation from labs


def read_data(csv_filename):
    return pd.read_csv(csv_filename)


# takes in the dataframe and output variable; prints exploratory info
def explore_data(df, output_var):
    # variable distributions
    print("----- VARIABLE DISTRIBUTIONS -----")
    print("For all data:")
    df.hist(figsize=(20, 10))
    plt.tight_layout()
    plt.show()
    print("Grouped by output variable:")
    df.groupby(output_var).hist(figsize=(20, 10))
    plt.tight_layout()
    plt.show()

    # data summaries
    print("----- DATA SUMMARY -----")
    print(df.describe())
    return


# replace any NaN values with zero and remove any outliers
def pre_process(df):

    for column in df:
        replace_val = 0
        df[column].fillna(replace_val, inplace=True)

    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df


# takes a continuous variable and turns it into a categorical variable
def continuous_to_discrete(df, var, num):
    buckets = np.linspace(df[var].min(), df[var].max(), num=num)
    df[var] = pd.cut(df[var], buckets)
    return df


# takes a categorical variable and turns it into a binary/dummy variable
def discrete_to_binary(df, var):
    df = continuous_to_discrete(df, var, 3)
    df = pd.get_dummies(df, columns=[var])
    return df


# runs the classifiers specified by the user
# options are all, log-reg, knn, tree, svm, rand-forest, boosting, bagging
def run_classifiers(df, selected_features, output_var, classifier_choice, test_size):
    results_df = pd.DataFrame([], columns=('modelthresh', 'which_temporal', 'model', 'threshold',
                                           'accuracy', 'precision', 'recall'))
    tests = temporal_train_test(df, selected_features, output_var)
    for (which_temporal_set, x_train, x_test, y_train, y_test) in tests:

        classifiers = [LogisticRegression(), DecisionTreeClassifier(), LinearSVC(), KNeighborsClassifier(),
                       BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier()]
        if classifier_choice == 'all':
            for c in classifiers:
                print(str(c).split('(')[0] + " RESULTS")
                predictions = classify(x_train, y_train, x_test, c)
                result = evaluate_classifier(y_test, predictions, str(c).split('(')[0], which_temporal_set)
                results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'log-reg':
            print("LOGISTIC REGRESSION RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, LogisticRegression())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'knn':
            print("KNN RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, KNeighborsClassifier())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'tree':
            print("DECISION TREE RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, DecisionTreeClassifier())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'svm':
            print("SVM RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, LinearSVC())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'rand-forest':
            print("RANDOM FOREST RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, RandomForestClassifier())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'boosting':
            print("BOOSTING RESULTS")
            predicted_scores =  classify(x_train, y_train, x_test, AdaBoostClassifier())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        elif classifier_choice == 'bagging':
            print("BAGGING RESULTS")
            predicted_scores = classify(x_train, y_train, x_test, BaggingClassifier())
            result = evaluate_classifier(y_test, predicted_scores, classifier_choice, which_temporal_set)
            results_df = results_df.append(result, ignore_index=True)

        else:
            print("Invalid argument", file=sys.stderr)
            sys.exit(1)

    print(results_df)
    return results_df


# creates a decision tree and returns the predicted scores for selected features on the output variables
def classify(x_train, y_train, x_test, classifier):
    model = classifier
    model.fit(x_train, y_train)
    if str(classifier).split('(')[0] == 'LinearSVC':
        predicted_scores = model.decision_function(x_test)
    else:
        predicted_scores = model.predict_proba(x_test)[:, 1]
    plt.hist(predicted_scores)
    plt.show()
    return predicted_scores


def plot_precision_recall_k(predicted_scores, true_labels):
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)
    plt.plot(recall, precision, marker='.')
    plt.show()


# evaluates the accuracy of classifier and plots precision and recall; takes in model name as a string
def evaluate_classifier(y_test, predicted_scores, model_name, which_temporal_set):
    thresholds = {0.01: [], 0.02: [], 0.05: [], 0.10: [], 0.20: [], 0.30: [], 0.50: []}
    # threshold = 0.4
    results_df = pd.DataFrame([], columns=('modelthresh', 'which_temporal', 'model', 'threshold',
                                            'accuracy', 'precision', 'recall'))
    for threshold in thresholds.keys():
        calc_threshold = lambda x, y: 0 if x < y else 1
        predicted_test = np.array([calc_threshold(score, threshold) for score in predicted_scores])
        test_acc = accuracy(predicted_test, y_test)
        precision, recall, thresholds = precision_recall_curve(y_test, predicted_test)
        this_result = pd.DataFrame([[model_name+str(threshold), which_temporal_set,  model_name, threshold, test_acc,
                                     np.mean(precision), np.mean(recall)]],
                                   columns=('modelthresh', 'which_temporal', 'model', 'threshold',
                                            'accuracy', 'precision', 'recall'))
        results_df = results_df.append(this_result, ignore_index=True)

    return results_df


# divides data into 6 month test sets
def temporal_train_test(df, selected_features, output_var):
    first = df.loc[(df['date_posted'] > '2012-1-1') & (df['date_posted'] <= '2012-6-30')]
    second = df.loc[(df['date_posted'] > '2012-7-1') & (df['date_posted'] <= '2012-12-31')]
    third = df.loc[(df['date_posted'] > '2013-1-1') & (df['date_posted'] <= '2013-7-1')]
    fourth = df.loc[(df['date_posted'] > '2013-7-1') & (df['date_posted'] <= '2013-12-31')]
    # return ordered tuples of (x_train, x_test, y_train, y_test)
    return [("first", first[selected_features], second[selected_features], first[output_var], second[output_var]),
            ("second", second[selected_features], third[selected_features], second[output_var], third[output_var]),
            ("third", third[selected_features], fourth[selected_features], third[output_var], fourth[output_var])]


# defines and charts which classifer works best, how they do over time, and which does best at each threshold
def chart_results(results_df):

    # how classifiers do over time
    fig, ax = plt.subplots()

    for modelthresh in results_df.nlargest(8, 'precision')['modelthresh']:
        ax.plot(results_df[results_df.modelthresh == modelthresh].which_temporal,
                results_df[results_df.modelthresh == modelthresh].precision,
                label=modelthresh)

    plt.title("Model Performance Over Time")
    plt.ylabel("Precision")
    plt.legend(loc='right')
    plt.show()


    # which classifier works best at the 5% threshold?
    fig2, ax2 = plt.subplots()

    for model in results_df.nlargest(8, 'precision')['model']:
        ax2.plot(results_df[results_df.model == model].threshold,
                 results_df[results_df.model == model].precision,
                 label=model)


    plt.title("Model Performance at Various Thresholds")
    plt.xlabel("Percent of population")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


    return

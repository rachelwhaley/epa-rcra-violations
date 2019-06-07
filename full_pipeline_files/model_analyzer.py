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
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy import stats
import sys

#todo: make metrics methods that take thresholds as an argument

class ClassifierAnalyzer:
    '''
    CA takes a model and a set of parameters from
    class is intended to store all metrics of model applied to data in one place
    '''
    def __init__(self, model, parameters, name, thresholds, plots, x_train, y_train,
                 x_test, y_test):
        self.name = name
        self.params = parameters
        self.model = model.set_params(**parameters)
        self.scores = classify(x_train, y_train, x_test, self.model)
        self.truth = y_test
        self.t = thresholds
        self.metrics_matrix, self.predictions = self.make_prediction_matrix()
        self.roc_auc = None

    def __repr__(self):
        return str(self.name)

    def make_prediction_matrix(self):
        rv_dic = {}
        predictions_df = pd.DataFrame()
        for thresh in self.t:
            x = round((1 - thresh), 2)
            preds = 'predictions_{}pct'.format(x)
            a = 'precision_{}pct'.format(x)
            b = 'recall_{}pct'.format(x)
            c = 'f1_{}pct'.format(x)
            predictions = predict(self.scores, thresh)
            predictions = [int(x) for x in predictions]
            d = '{}_at_{}pct'.format(self.name, x)
            predictions_df[d] = predictions
            prec = precision(self.truth, predictions)
            rec = recall(self.truth, predictions)
            rv_dic[a] = [prec]
            rv_dic[b] = [rec]
            rv_dic[c] = [(prec * rec * 2) / (prec + rec)]
            rv_dic['model'] = [self.name]

        return pd.DataFrame(rv_dic), predictions_df

    def plot_precision_recall(self, save, show, name):
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        self.truth, self.scores)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(self.scores)
        for value in pr_thresholds:
            num_above_thresh = len(self.scores[self.scores>=value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'g')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='g')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0,1])
        ax2.set_xlim([0,1])
        if show == True:
            plt.show()
            plt.close()
        if save == True:
            plt.savefig("precision_recall_" + str(name) + '.png')
            plt.close()

    def plot_roc(self, save, show, name):
        fpr, tpr, thresholds = roc_curve(self.truth, self.scores)
        self.roc_auc = auc(fpr, tpr)
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(name)
        plt.legend(loc="lower right")
        if show == True:
            plt.show()
            plt.close()
        if save == True:
            plt.savefig("roc_auc_" + str(name)+'.png')
            plt.close()

def classify(x_train, y_train, x_test, classifier):
    '''
    from rachel's pipeline_library
    '''
    model = classifier
    model.fit(x_train, y_train)
    if str(classifier).split('(')[0] == 'LinearSVC':
        predicted_scores = model.decision_function(x_test)
    else:
        predicted_scores = model.predict_proba(x_test)[:, 1]

    return predicted_scores

def compare_to_threshold(score, threshold):
    '''
    takes threshold, temporarily aggregates data and comes up with score
    that represents threshold% of population, then compares each score to that
    adjusted threshold CORRECTED TO PREDICT 1 FOR THRESHOLD% CORRECTLY
    '''
    if score > threshold:
        return 1
    else:
        return 0

def predict(scores, threshold):
    l = list(stats.rankdata(scores, 'average')/len(scores))

    return [compare_to_threshold(x, threshold) for x in l]
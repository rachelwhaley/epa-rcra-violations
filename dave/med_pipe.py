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
    plt.hist(predicted_scores)
    plt.save_fig()
    return predicted_scores

def compare_to_threshold(score, threshold):
    '''
    takes threshold, temporarily aggregates data and comes up with score
    that represents threshold% of population, then compares each score to that
    adjusted threshold
    '''
    if score > threshold:
        return 1
    else:
        return 0

def predict(scores, threshold):
    l = list(stats.rankdata(scores, 'average')/len(self.scores))
    
    return [compare_to_threshold(x, threshold) for x in l]

class Classifier(model_type, parameters, threshold, x_train, y_train, x_test, y_test):
    '''
    parameters is list of tuples with (parameter name, value)
    order of arguments:
    decision tree: criterion, splitter(keep at best), max_depth
    knn: k, weights
    bagging: base_estimator, n_estimators, n_features
    
    '''
    self.classifier = model_type(*parameters)
    self.scores = classify(x_train, y_train, x_test, self.classifier)
    self.predictions = list(stats.rankdata(self.scores, 'average')/len(self.scores))
    self.accuracy = 
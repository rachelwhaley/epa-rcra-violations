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

class Classifier:
    '''
    model needs parameters passed to it before being loaded to class
    class is intended to store all metrics of model applied to data in one place
    
    '''
    def __init__(self, model, parameters, threshold, x_train, y_train,
                 x_test, y_test):
        self.classifier = model_type(*parameters)
        self.scores = classify(x_train, y_train, x_test, self.classifier)
        self.truth = y_test
        self.predictions = predict(self.scores, threshold)
        self.accuracy = accuracy(self.truth, self.predictions)
        self.precision = precision(self.truth, self.predictions)
        self.recall = recall(self.truth, self.predictions)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def to_dict(self):
        return vars(self)
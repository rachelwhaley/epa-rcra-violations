from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                             RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

test0 = {'DT': {'criterion':['gini'], 'max_depth':[1], 'min_samples_split':[25]}}
grid0 = { 
    'RF':{'n_estimators': [10], 'max_depth': [5], 'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, None],'min_samples_split': [2]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [10]}
    }
clfstest = {'DT': DecisionTreeClassifier()}
clfs0 =  {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'DT': DecisionTreeClassifier()
            }


grid1 = {
    'SGD': { 'loss': ['log','perceptron'], 'penalty': ['l2','l1'], 'max_iter':[1000], 'tol':[1]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]}
    }


grid2 = {
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,500]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.5],'subsample' : [0.5,1.0], 'max_depth': [5]},
    'NB' : {},
    }
grid3 = {
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, None],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] }
    }
    

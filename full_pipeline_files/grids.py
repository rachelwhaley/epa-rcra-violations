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


clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'LR': LogisticRegression(penalty='l2', C=1e5, solver='lbfgs'),
    'SVM': SVC(kernel='linear', probability=True, random_state=0),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'SGD': SGDClassifier(loss="log", penalty="l2"),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'BAG': BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, n_estimators = 20)
        }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'solver': ['liblinear']},
    'SGD': { 'loss': ['hinge','log'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.01,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10, 20], 'max_samples' : [.25, .5, .75]}
       }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.1,1,10], 'solver': ['liblinear']},
    'SGD': { 'loss': ['log'], 'penalty': ['l2','l1'], 'max_iter':[1000], 'tol':[1]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,500]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.5],'subsample' : [0.5,1.0], 'max_depth': [5]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, None],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [10,50],'weights': ['uniform','distance'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] }
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l2'], 'C': [0.01]},
    'SGD': { 'loss': ['log'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5], 'max_samples' : [.25] } 

           }

    

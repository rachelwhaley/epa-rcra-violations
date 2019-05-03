'''
General set of Pipeline Tools
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from scipy import stats
from plotnine import *
import graphviz
import donation_analysis as don
import pylab as pl

'''
Data Cleaning
'''
def fillna_pro(df, fillobs=False):
    '''
    fills cells with their mean. if cell dtype is int64 function will round mean
    '''
    nan_dict = {}
    cols = df.columns[df.isna().any()].tolist()
    
    for col in cols:
        if df[col].dtype == 'float64':
            nan_dict[col] = df[col].mean()
        if df[col].dtype == 'int64':
            nan_dict[col] = int(df[col].mean())
        if fillobs == True:
            if df[col].dtype == 'object' or df[col].dtype == 'str':
                nan_dict[col] = df[col].mode()
    
    return df.fillna(nan_dict)

def yesno_to_binary(data_series):
    '''
    converts yes/no to 1/0
    '''
    data_series = data_series.str.strip()
    data_series = data_series.str.lower()
    
    return data_series.map(dict(yes=1, no=0))

def tf_to_binary(data_series):
    return data_series.map(dict(t=1, f=0))

def int_to_binary(data_series, marker, valifyes, valifno):
    '''
    take a series of ints and return 1 if > marker, 0 if not
    '''
    return np.where(data_series > marker, valifyes, valifno)

def encode_column(col):
    '''
    wrapper for labelencoder so i dont forget it
    encodes categorical data without making 
    '''
    le = preprocessing.LabelEncoder()
    
    return le.transform(col)

def x_y_split(df):
    return df.drop('date_posted'), df.iloc[:,-1]

'''
Classifiers
'''
classifiers = ['logreg', 'knn', 'dectree', 'svm', 'RandomForest', 'Boosting',
              'Bagging']

def svm_classifier(x_train, y_train, x_test):
    '''
    fit an svm and return confidence scores
    '''
    svm = LinearSVC(random_state=0)
    svm.fit(x_train, y_train)
    confidence = svm.decision_function(x_test)
    
    return list(confidence)

def logreg_classifier(x_train, y_train, x_test, c, p, solver='lbfgs'):
    '''
    fit a logistic regression model and return predicted probabilities
    '''
    logreg = LogisticRegression(penalty=p, C=c)
    logreg.fit(x_train, y_train)
    
    return list(logreg.predict_proba(x_test)[:,1])

def knn_classifier(x_train, y_train, x_test, k):
    '''
    uses knn classification to predict on x_test
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    
    return knn.predict_proba(x_test)

def dectree_classifier(x_train, y_train, x_test, crit):
    '''
    takes x-train, y-train, and x-test and predicts probable class
    for test data. will not allow split with less than 4% of data in node
    to avoid overfitting
    '''
    lim = int(len(list(y_train))/25)
    dec_tree = DecisionTreeClassifier(criterion=crit,
                                      min_samples_split=lim)
    dec_tree.fit(x_train, y_train)
    
    return dec_tree.predict_proba(x_test)[:,0]

def random_forest_classifier(n_trees, max_features, x_train, y_train,
                            x_test):
    '''
    generates a random forest and applies predicted class probabilities
    to x_test data.
    '''
    rfc = RandomForestClassifier(n_estimators=n_trees,
                                 max_features=max_features, random_state=0)
    rfc.fit(x_train, y_train)
    
    return list(rfc.predict_proba(x_test)[:,1])

def gradient_boosting_classifier(x_train, y_train, x_test):
    '''
    creates a gradient boosting classifier
    
    returns predicted class probabilities to x_test
    '''
    sgbc = GradientBoostingClassifier()
    sgbc.fit(x_train, y_train)
    
    return sgbc.predict_proba(x_test)[:,1]

def bagging_classifier(x_train, y_train, x_test):
    '''
    creates a bagging classifier
    
    returns the predicted class probabilities on x_test from that classsifier
    '''
    bc = BaggingClassifier(max_features=.75)
    bc.fit(x_train, y_train)
    return bc.predict_proba(x_test)[:,0]
    
'''
Eval Main Functions
'''
classifiers = ['logreg', 'knn', 'dectree', 'svm', 'RandomForest', 'Boosting',
              'Bagging']
thresh = [.01, .02, .05, .1, .2, .3, .5]
#included .01 and 1.0 to gather precision/recall scores on full inclusion/
#exclusion
def big_board(windows, classifiers=classifiers,thresh=thresh):
    '''
    runs all of the models you tell it to and creates master stats board for
    all models
    '''
    rv = []
    rdf = pd.DataFrame()
    
    for i, period in enumerate(windows[:-1]):
        if i > 0:
            a, b = don.x_y_split(period)
            x = pd.concat([x, a], ignore_index=True, sort=False)
            y = pd.concat([y, b], ignore_index=True, sort=False)
        else:
            x, y = don.x_y_split(period)
        x = x.drop('date_posted', axis=1)
        xt, yt = don.x_y_split(windows[i+1])
        xt = xt.drop('date_posted', axis=1)
        if 'logreg' in classifiers:
            rv.append(evaluate_logreg(x,y,xt,yt))
        if 'knn' in classifiers:
            rv.append(evaluate_knn(x,y,xt,yt))
        if 'dectree' in classifiers:
            rv.append(evaluate_dectree(x,y,xt,yt))
        if 'svm' in classifiers:
            rv.append(evaluate_svm(x,y,xt,yt))
        if 'RandomForest' in classifiers:
            rv.append(evaluate_rf(x,y,xt,yt))
        if 'Boosting' in classifiers:
            rv.append(evaluate_gb(x,y,xt,yt))
        if 'Bagging' in classifiers:
            rv.append(evaluate_bagging(x,y,xt,yt))
            
    for df in rv:
        rdf = pd.concat([rdf, df], ignore_index=True, sort=False)
        
    rdf.to_csv('results.csv')
            
        
        

def evaluate_logreg(x_train, y_train, x_test, y_test,
                    c_values=[.01,.1,1,10,100], thresh=thresh):
    '''
    generates df of predictions, penalties, c_values, thresholds, precision, recall, and
    accuracy of logistic regression
    '''
    penalties = ['l2']
    rd = {'predicted': [], 'penalty': [], 'C': [], 'threshold': [],
          'precision': [], 'recall': [], 'accuracy':[], 'class': []}
    
    for p in penalties:
        for c in c_values:
            scores = logreg_classifier(x_train, y_train, x_test, c, p)
            for t in thresh:
                scores = list(stats.rankdata(scores, 'average')/len(scores))
                preds = [compare_to_threshold(x, t)for x in scores]
                rd['predicted'].append(preds)
                rd['penalty'].append(p)
                rd['C'].append(c)
                rd['threshold'].append(t)
                rd['precision'].append(precision(y_test, preds))
                rd['recall'].append(recall(y_test, preds))
                rd['accuracy'].append(accuracy(y_test, preds))
                rd['class'].append('logreg')

    return pd.DataFrame(rd)

def evaluate_knn(x_train, y_train, x_test, y_test, kays=[3,5,7,9,11],
                thresh=thresh):
    '''
    generates df of predictions, penalties, k values, thresholds, precision,
    recall, and accuracy to help find best model
    '''
    rd = {'predicted': [], 'k':[], 'threshold': [],
          'precision': [], 'recall': [], 'accuracy':[], 'class': []}
    for k in kays:
        scores = knn_classifier(x_train, y_train, x_test, k)[:,1]
        for t in thresh:
            scores = list(stats.rankdata(scores, 'average')/len(scores))
            preds = [compare_to_threshold(x, t) for x in scores]
            rd['predicted'].append(preds)
            rd['k'].append(k)
            rd['threshold'].append(t)
            rd['precision'].append(precision(y_test, preds))
            rd['recall'].append(recall(y_test, preds))
            rd['accuracy'].append(accuracy(y_test, preds))
            rd['class'].append('knn')

    return pd.DataFrame(rd)

def evaluate_dectree(x_train, y_train, x_test, y_test, thresh=thresh):
    '''
    you get it
    '''
    criterion = ['entropy', 'gini']
    rd = {'predicted': [], 'crit': [], 'threshold': [],
          'precision': [], 'recall': [], 'accuracy':[], 'class': []}
    
    for c in criterion:
        scores = dectree_classifier(x_train, y_train, x_test, c)
        for t in thresh:
            scores = list(stats.rankdata(scores, 'average')/len(scores))
            preds = [compare_to_threshold(x, t) for x in list(scores)]
            rd['predicted'].append(preds)
            rd['crit'].append(c)
            rd['threshold'].append(t)
            rd['precision'].append(precision(y_test, preds))
            rd['recall'].append(recall(y_test, preds))
            rd['accuracy'].append(accuracy(y_test, preds))
            rd['class'].append('dectree')

    return pd.DataFrame(rd)

def evaluate_rf(x_train, y_train, x_test, y_test, thresh=thresh, ntrees=[25,100,500],
                maxfeats=[1, .5, 4]):
    rd = {'predicted': [], 'ntrees':[], 'nfeats': [], 'threshold': [],
          'precision': [], 'recall': [], 'accuracy':[], 'class': []}
    for size in ntrees:
        for f in maxfeats:
            scores = random_forest_classifier(size, f, x_train, y_train, x_test)
            for t in thresh:
                scores = list(stats.rankdata(scores, 'average')/len(scores))
                preds = [compare_to_threshold(x, t) for x in scores]
                rd['predicted'].append(preds)
                rd['ntrees'].append(size)
                rd['nfeats'].append(f)
                rd['threshold'].append(t)
                rd['precision'].append(precision(y_test, preds))
                rd['recall'].append(recall(y_test, preds))
                rd['accuracy'].append(accuracy(y_test, preds))
                rd['class'].append('rf')
                
    return pd.DataFrame(rd)

def evaluate_gb(x_train, y_train, x_test, y_test, thresh=thresh):
    rd = {'predicted': [], 'threshold': [], 'precision': [], 'recall': [],
          'accuracy':[], 'class': []}
    scores = list(gradient_boosting_classifier(x_train, y_train, x_test))
    for t in thresh:
        scores = list(stats.rankdata(scores, 'average')/len(scores))
        preds = [compare_to_threshold(x, t) for x in scores]
        rd['predicted'].append(preds)
        rd['threshold'].append(t)
        rd['precision'].append(precision(y_test, preds))
        rd['recall'].append(recall(y_test, preds))
        rd['accuracy'].append(accuracy(y_test, preds))
        rd['class'].append('gb')
        
    return pd.DataFrame(rd)

def evaluate_svm(x_train, y_train, x_test, y_test, thresh=thresh):
    rd = {'predicted': [],'threshold': [], 'precision': [], 'recall': [],
          'accuracy':[], 'class': []}
    scores = svm_classifier(x_train, y_train, x_test)
    for t in thresh:
        scores = list(stats.rankdata(scores, 'average')/len(scores))
        preds = [compare_to_threshold(x, t) for x in scores]
        rd['predicted'].append(preds)
        rd['threshold'].append(t)
        rd['precision'].append(precision(y_test, preds))
        rd['recall'].append(recall(y_test, preds))
        rd['accuracy'].append(accuracy(y_test, preds))
        rd['class'].append('svm')
        
    return pd.DataFrame(rd)

def evaluate_bagging(x_train, y_train, x_test, y_test, thresh=thresh):
    rd = {'predicted': [], 'threshold': [], 'precision': [], 'recall': [],
          'accuracy':[], 'class': []}
    scores = bagging_classifier(x_train, y_train, x_test)
    for t in thresh:
        scores = list(stats.rankdata(scores, 'average')/len(scores))
        preds = [compare_to_threshold(x, t) for x in list(scores)]
        rd['predicted'].append(preds)
        rd['threshold'].append(t)
        rd['precision'].append(precision(y_test, preds))
        rd['recall'].append(recall(y_test, preds))
        rd['accuracy'].append(accuracy(y_test, preds))
        rd['class'].append('bagging')
        
    return pd.DataFrame(rd)


    


'''
Eval Helper Functions
'''

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
    
    
'''
Visualize Data
'''
def plot_roc(name, probs, true, output_type):
    '''
    professor's plot roc function from github magicloops
    output_types: 'show', 'save'
    '''
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def plot_precision_recall_curve(y_test, pred_probas):
    '''
    sklearn documentation says y_true goes first but this disagrees with our
    lab
    
    ONLY FOR USE WITHIN JUPYTER NOTEBOOK
    
    anyways, this function plots precision-recall pairs for different
    probablitity thresholds (restricted to binary classification)
    '''
    precision, recall, thresholds = precision_recall_curve(y_test,
                                                           pred_probas)
    plt.plot(recall, precision, marker='.')
    
    return plt.show()

def plot_rocc(y_test, predictions):
    '''
    ONLY FOR USE WITHIN JUPYTER NOTEBOOK
    
    plots the Reciever Operating Characteristic Curve of predictions
    '''
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    
    plt.plot(fpr, tpr, marker='.')
    
    return plt.show()

def plot_binary_as_mean(df, dep, ind):
    '''
    Takes a binary variable as argument dep and groups by categorical ind, then
    shows mean values of dep over values of ind. 
    
    requirements: dep and ind must be col names from df.
    '''
    temp = df.groupby(ind)[dep].mean()
    temp = pd.DataFrame({ind:temp.index, dep:temp.values})
    
    return (ggplot(temp, aes(ind, dep)) + geom_point())

def plot_scatter(df, dep, ind):
    
    return (ggplot(df, aes(ind, dep)) + geom_point())

def plot_bar(df, dep, ind):
    '''
    this function groups by the independent variable and shows
    the median score from the dependent variable column for each group
    '''
    temp = df.groupby(ind)[dep].median()
    temp = pd.DataFrame({ind:temp.index, dep:temp.values})
    
    return (ggplot(temp, aes(x=ind, y=dep)) + geom_bar(stat='identity',
                                                       position='dodge'))

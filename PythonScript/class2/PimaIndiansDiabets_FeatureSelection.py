# -*- coding: utf-8 -*-
"""
Pima Indians Diabets Data Set 
"""


import pandas

pandas.set_option('display.width', 100)  # change the preferred width of the output
pandas.set_option('precision', 3)  #change the precision of the numbers

#Read data set 
# read the data into a Pandas DataFrame
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

# View first 5 rows
print(pima.head(5))

#Dimensions of Your Data
print(pima.shape)

#Question: Can we predict the diabetes status of a patient given their health measurements?

# define X and y
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
X = pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)#, random_state=0)

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


print(logreg.intercept_)
print(logreg.coef_)


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

# print the first 25 true and predicted responses
print('True:', y_test[0:25])
print('Pred:', y_pred_class[0:25])
print('Prob:', y_pred_prob[0:25])

# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# Metrics computed from a confusion matrix
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))

# Calculate TP rate or recall
print(TP / float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))

# Precision: When a positive value is predicted, how often is the prediction correct?
# How "precise" is the classifier when predicting positive instances?
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]


'''
ROC Curves and Area Under the Curve (AUC)
'''

# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier\n AUC={auc}'.format(auc=metrics.roc_auc_score(y_test, y_pred_prob)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('True Positive Rate:', tpr[thresholds > threshold][-1])
    print('False Positive Rate:', fpr[thresholds > threshold][-1])
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
evaluate_threshold(0.5)

'''
AUC is the percentage of the ROC plot that is underneath the curve:
'''

# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))

# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()



from sklearn.linear_model import Lasso, RidgeClassifier,LogisticRegression
clf = Lasso(alpha=0.1)

clf.fit(X_train, y_train)

cv_scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
cv_scores.mean()
cv_scores.std()


# try lambda=0.0001 through lambda=2 and record testing accuracy
import numpy as np
lambda_range = np.arange(0.0001, 10, 0.1)
scores = []
for lam in lambda_range:
    lm_ridge = RidgeClassifier(alpha=1/lam)
    scores.append(cross_val_score(lm_ridge, X, y, cv=5, scoring='roc_auc').mean())

import matplotlib.pyplot as plt

plt.plot(lambda_range, scores)
plt.xlim([0.0, 10])
#plt.ylim([0.0, 1.0])
plt.title('L2 Regularization')
plt.xlabel('Lambda')
plt.ylabel('AUC')
plt.grid(True)
plt.show()
    

# Step 1: Select the first variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    X = pima[f]
    X = X.values.reshape(-1, 1)
    average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
    print(f, average_auc)
    
# Step 2: Select the second variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose']:
        X = pima[['glucose', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)

# Step 3: Select the third variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose', 'bmi']:
        X = pima[['glucose', 'bmi', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)

# Step 4: Select the 4th variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose', 'bmi', 'pregnant']:
        X = pima[['glucose', 'bmi', 'pregnant', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)

# Step 5: Select the 5th variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose', 'bmi', 'pregnant', 'pedigree']:
        X = pima[['glucose', 'bmi', 'pregnant', 'pedigree', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)

# Step 6: Select the 6th variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin']:
        X = pima[['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)

# Step 7: Select the 7th variable
for f in feature_cols:# = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    if f not in ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', 'bp']:
        X = pima[['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', 'bp', f]]
        average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
        print(f, average_auc)


ForwardSelection_CV = {'M1':['glucose'],
                       'M2': ['glucose', 'bmi'],
                       'M3': ['glucose', 'bmi', 'pregnant'],
                       'M4': ['glucose', 'bmi', 'pregnant', 'pedigree', ],
                       'M5': ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin'],
                       'M6': ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', 'bp'],
                       'M7': ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', 'bp', 'age'],
                       'M8': ['glucose', 'bmi', 'pregnant', 'pedigree', 'insulin', 'bp', 'age', 'skin']
                       }

for model in ForwardSelection_CV.keys():
    X = pima[ForwardSelection_CV[model]]
    average_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
    print(model, average_auc)







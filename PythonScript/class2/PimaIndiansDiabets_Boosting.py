# -*- coding: utf-8 -*-
"""
Pima Indians Diabets Data Set 
"""


import pandas
import numpy as np
import matplotlib.pyplot as plt

#Read data set 
# read the data into a Pandas DataFrame
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)


#Question: Can we predict the diabetes status of a patient given their health measurements?

# define X and y
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
X = pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)#, random_state=0)

# train a logistic regression model on the training set
from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0)
boost.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = boost.predict(X_test)

# store the predicted probabilities for class 1
y_pred_prob = boost.predict_proba(X_test)[:, 1]


''' feature_importances_ : array of shape = [n_features]
The feature importances (the higher, the more important the feature).'''

print(boost.feature_importances_)

importances = boost.feature_importances_
std = np.std([tree.feature_importances_ for tree in boost.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

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


'''
ROC Curves and Area Under the Curve (AUC)
'''

# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier\n AUC={auc}'.format(auc=metrics.roc_auc_score(y_test, y_pred_prob)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


'''
AUC is the percentage of the ROC plot that is underneath the curve:
'''

# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))

# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(boost, X, y, cv=10, scoring='roc_auc').mean()


# Tuning Parameters for Boosting
# try B=1 through B=100 and record testing accuracy
B_range = np.arange(1, 100, 1)
scores = []
for b in B_range:
    boost = AdaBoostClassifier(n_estimators = b, learning_rate = 1.0)
    scores.append(cross_val_score(boost, X, y, cv=5, scoring='roc_auc').mean())

# Plot AUC for different values of B
plt.plot(B_range, scores)
plt.xlim([0.0, 100])
#plt.ylim([0.0, 1.0])
plt.title('Boosting')
plt.xlabel('B')
plt.ylabel('AUC')
plt.grid(True)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, max_features="auto", min_samples_split=2)
cross_val_score(rf, X, y, cv=10, scoring='roc_auc').mean()

# Tuning Parameters for Random Forest
# try max_features=1 through max_features=8 and record testing accuracy
max_features_range = np.arange(1, 8, 1)
scores = []
for num_features in max_features_range:
    rf = RandomForestClassifier(n_estimators = 100, max_features=num_features, min_samples_split=2)
    scores.append(cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean())

# Plot AUC for different values of B
plt.plot(max_features_range, scores)
plt.xlim([0.0, 8])
#plt.ylim([0.0, 1.0])
plt.title('Ranfom Forest')
plt.xlabel('max_features')
plt.ylabel('AUC')
plt.grid(True)
    
# Tuning Parameters for Random Forest
# try B=1 through B=100 and record testing accuracy
B_range = np.arange(1, 100, 1)
scores = []
for b in B_range:
    rf = RandomForestClassifier(n_estimators = b, max_features=num_features, min_samples_split=2)
    scores.append(cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean())

# Plot AUC for different values of B
plt.plot(B_range, scores)
plt.xlim([0.0, 100])
#plt.ylim([0.0, 1.0])
plt.title('Boosting')
plt.xlabel('B')
plt.ylabel('AUC')
plt.grid(True)


''' feature_importances_ : array of shape = [n_features]
The feature importances (the higher, the more important the feature).'''
rf = RandomForestClassifier(n_estimators = 100, max_features="auto", min_samples_split=2)
rf.fit(X_train, y_train)

print(rf.feature_importances_)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


## Missing Values
# Count missing values
print((pima[['glucose','bp','skin','insulin','bmi']] == 0).sum())


# mark zero values as missing or NaN
pima[['glucose','bp','skin','insulin','bmi']] = pima[['glucose','bp','skin','insulin','bmi']].replace(0, numpy.NaN)
# count the number of NaN values in each column
print(pima.isnull().sum())
# drop rows with missing values
pima.dropna(inplace=True)


# fill missing values with mean column values
pima.fillna(pima.mean(), inplace=True)




X = pima[feature_cols]
y = pima.label
boost = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0)
cross_val_score(boost, X, y, cv=10, scoring='roc_auc').mean()

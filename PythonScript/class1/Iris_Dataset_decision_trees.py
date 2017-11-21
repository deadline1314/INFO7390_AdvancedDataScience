from sklearn.datasets import load_iris
iris = load_iris()


# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# Fit a decision tree
# import the class

from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier()
clf = decisionTree.fit(X, y)

import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# STEP 2: fit into decision tree model
decisionTree = DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred_class = decisionTree.predict(X_test)
y_pred_prob = decisionTree.predict_proba(X_test)




# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

# print the first 25 true and predicted responses
print('True:', y_test[0:25])
print('Pred:', y_pred_class[0:25])
print('Prob:', y_pred_prob[0:25])

# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))

# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(decisionTree, X, y, cv=10, scoring='accuracy').mean()

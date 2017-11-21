
# An introduction to scikit-learn

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

# print the iris data
print(iris.data)

# print the names of the four features
print(iris.feature_names)


# print integers representing the species of each observation
print(iris.target)


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


'''
Requirements for working with data in scikit-learn
* Features and response are separate objects
* Features and response should be numeric
* Features and response should be NumPy arrays
'''
# check the types of the features and response
print(type(iris.data))
print(type(iris.target))

# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)


# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# Fit a logistic regression
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
logreg.predict(X_new)


'''Validation Set Approach
Step 1: Split the dataset into two pieces: a training set and a testing set.
Step 2: Train the model on the training set.
Step 3: Test the model on the testing set, and evaluate how well we did.
'''

# STEP 1: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))



# calculate accuracy
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))

y_pred_prob = logreg.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)


'''
K Nearest Neighbor (KNN)
'''

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
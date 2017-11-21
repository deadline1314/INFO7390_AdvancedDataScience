
'''
Linear Regression Using scikit-learn for Advertising Problem
'''

#use the pandas library to read data into Python
# conventional way to import pandas
import pandas as pd

# read CSV file directly from a URL and save the results
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# display the first 5 rows
data.head()


# display the last 5 rows
data.tail()

# check the shape of the DataFrame (rows, columns)
data.shape


'''
What are the features?
* TV: advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
* Radio: advertising dollars spent on Radio
* Newspaper: advertising dollars spent on Newspaper

What is the response?
* Sales: sales of a single product in a given market (in thousands of items)

What else do we know?
* Because the response variable is quantitative, this is a regression problem.
* There are 200 observations (represented by the rows), and each observation is a single market.
'''

# Visualizing data using seaborn
# Seaborn: Python library for statistical data visualization built on top of Matplotlib

# conventional way to import seaborn
import seaborn as sns

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=7, aspect=0.7)


# Fit a linear model 
'''Preparing X and y using pandas
* scikit-learn expects X (feature matrix) and y (response vector) to be NumPy arrays.
* However, pandas is built on top of NumPy.
* Thus, X can be a pandas DataFrame and y can be a pandas Series!
'''

# create a Python list of feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['TV', 'radio', 'newspaper']]

# print the first 5 rows
X.head()

# select a Series from the DataFrame
y = data['sales']

# equivalent command that works if there are no spaces in the column name
y = data.sales

# print the first 5 values
y.head()

# check the type and shape of y
print(type(y))
print(y.shape)

#Splitting X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)


# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


#Interpreting model coefficients

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

'''
How do we interpret the TV coefficient (0.0466)?
* For a given amount of Radio and Newspaper ad spending, a "unit" increase in TV ad spending is associated with a 0.0466 "unit" increase in Sales.
* Or more clearly: For a given amount of Radio and Newspaper ad spending, an additional $1,000 spent on TV ads is associated with an increase in sales of 46.6 items.
'''

# Making predictions
# make predictions on the testing set
y_pred = linreg.predict(X_test)

#Model evaluation metrics for regression

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))

#Mean Squared Error (MSE) is the mean of the squared errors:
    
# Feature selection

#Does Newspaper "belong" in our model? In other words, does it improve the quality of our predictions?
# Let's remove it from the model and check the MSE!

# create a Python list of feature names
feature_cols = ['TV', 'radio']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the DataFrame
y = data.sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the MSE of our predictions
print(metrics.mean_squared_error(y_test, y_pred))


'''
Cross-validation example: feature selection
Advantages of cross-validation:
* More accurate estimate of test error
* More "efficient" use of data (every observation is used for both training and testing)

Goal: Select whether the Newspaper feature should be included in the linear regression model on the advertising dataset
'''
from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with all three features
lm = LinearRegression()
X = data[['TV', 'radio', 'newspaper']]
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
print(-scores)

# calculate the average MSE
print(-scores.mean())

# 10-fold cross-validation with two features

X = data[['TV', 'radio']]
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')

# calculate the average MSE
print(-scores.mean())


# -*- coding: utf-8 -*-
"""
Pima Indians Diabetes Data Set 
"""


import pandas
import numpy

pandas.set_option('display.width', 100)  # change the preferred width of the output
pandas.set_option('precision', 3)  #change the precision of the numbers

#Read data set 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

# View first 20 rows
peek = data.head(20)
print(peek)

#Dimensions of Your Data
shape = data.shape
print(shape)


# Data Types for Each Attribute
types = data.dtypes
print(types)

#You can see that most of the attributes are integers and that mass and pedi are floating point values.

"""
Descriptive Statistics: can give you great insight into the shape of each attribute.


Often you can create more summaries than you have time to review. 
The describe() function on the Pandas DataFrame lists 8 statistical properties of each attribute:

Count
Mean
Standard Devaition
Minimum Value
25th Percentile
50th Percentile (Median)
75th Percentile
Maximum Value
"""

description = data.describe()
print(description)


"""
Class Distribution (Classification Only): 
On classification problems you need to know how balanced the class values are.
"""
class_counts = data.groupby('class').size()
print(class_counts)


"""
Correlation Between Attributes

Correlation refers to the relationship between two variables and 
how they may or may not change together.

The most common method for calculating correlation is Pearson’s Correlation Coefficient, 
that assumes a normal distribution of the attributes involved. 
A correlation of -1 or 1 shows a full negative or positive correlation respectively. 
Whereas a value of 0 shows no correlation at all.

Some machine learning algorithms like linear and logistic regression 
can suffer poor performance if there are highly correlated attributes in your dataset. 
As such, it is a good idea to review all of the pair-wise correlations of 
the attributes in your dataset. 

We can use the corr() function on the Pandas DataFrame to calculate a correlation matrix.
"""

correlations = data.corr(method='pearson')
print(correlations)


"""
Skew of Univariate Distributions

Skew refers to a distribution that is assumed Gaussian (normal or bell curve) 
that is shifted or squashed in one direction or another.

Many machine learning algorithms assume a Gaussian distribution. 
Knowing that an attribute has a skew may allow you to perform data preparation 
to correct the skew and later improve the accuracy of your models.

We can calculate the skew of each attribute using the skew() function on the Pandas DataFrame.
"""


skew = data.skew()
print(skew)


"""
The skew result show a positive (right) or negative (left) skew. 
Values closer to zero show less skew.

"""



import matplotlib
matplotlib.style.use('ggplot')

data['class'].plot(kind='density')


data['preg'].plot.box()


# Count missing values
print((data[['plas','pres','skin','test','mass']] == 0).sum())


# mark zero values as missing or NaN
data[['plas','pres','skin','test','mass']] = data[['plas','pres','skin','test','mass']].replace(0, numpy.NaN)
# count the number of NaN values in each column
print(data.isnull().sum())



print(data.describe())

# drop rows with missing values
data.dropna(inplace=True)

# summarize the number of rows and columns in the dataset
print(data.shape)

# fill missing values with mean column values
data.fillna(data.mean(), inplace=True)
# count the number of NaN values in each column
print(data.isnull().sum())


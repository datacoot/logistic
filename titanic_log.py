# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:33:55 2019

@author: Subhashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# importing the metrics class
from sklearn import metrics

#Using a magic function to print all the plots inline
%matplotlib inline

#reading the data through a csv file

titanic_train = pd.read_csv('C:/Users/Anupriya/OneDrive/Work/Self Study/Kaggle/Titanic/train.csv')
titanic_test = pd.read_csv('C:/Users/Anupriya/OneDrive/Work/Self Study/Kaggle/Titanic/test.csv')

# Data Exploration and dataset understanding
# First we will expand the output display to see more columns

pd.set_option('display.expand_frame_repr', False)

summarystat = titanic_train.describe(include = 'all')
print(summarystat)

# Data type of different variable in the dataframe

titanic_train.info()

titanic_train.head(5)
titanic_train.count()

#Cabin (204), age (714), and embarked (889) have missing data

# Plotting the correlation matrix to identify the inbuilt relationships between variables

plt.matshow(titanic_train.corr())
plt.show()

"""The correlation matrix doesn't perform well here as some of the explanatory variables are categorical
Instead our aim should be to understand the data from the categorical perspective as the response variable 
is categorical
"""
sns.set(style="darkgrid")
sns.countplot(x='Survived', hue='Sex', data=titanic_train)
sns.countplot(x='Survived', hue='Pclass', data=titanic_train)

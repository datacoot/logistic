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

# It looks like maximum deaths were in the 3rd class. But what was the ratio of survived vs deaths in each class
# For that we will calculate the mean of the value and store it in seperate variable

Pclass_mean = titanic_train[["Survived", "Pclass"]].groupby(["Pclass"], as_index=False).mean()
print(Pclass_mean)
sns.barplot(x = "Pclass", y = "Survived", data = Pclass_mean)
sns.countplot(x='Survived', hue='Embarked', data=titanic_train)

#simiar analysis for Embarked

Embarked_mean = titanic_train[["Survived", "Embarked"]].groupby(["Embarked"], as_index=False).mean()
print(Embarked_mean)
sns.barplot(x = "Embarked", y = "Survived", data = Embarked_mean)
sns.countplot(x='Embarked', hue='Pclass', data=titanic_train)

# lOoking at gender and chance of survival

Sex_mean = titanic_train[["Survived", "Sex"]].groupby(["Sex"], as_index=False).mean()
print(Sex_mean)
sns.barplot(x = "Sex", y = "Survived", data = Sex_mean)

# This clearly shows the female passengers have a higher survival rate

titanic_train.SibSp.describe()
titanic_train.SibSp.head(10)

titanic_train.Parch.describe()
titanic_train.Parch.head(10)

# Since SibSp and Parch both variables provide a count of number of family members
# We can combine to create a familyCount variable

titanic_train['FamilyCount'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_test['FamilyCount'] = titanic_test['SibSp'] + titanic_test['Parch']

titanic_train.FamilyCount.head(10)
FamilyCount_mean = titanic_train[["Survived", "FamilyCount"]].groupby(["FamilyCount"], as_index=False).mean()
print(FamilyCount_mean)
sns.barplot(x = "FamilyCount", y = "Survived", data = FamilyCount_mean)


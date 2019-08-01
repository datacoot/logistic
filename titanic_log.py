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

# Dropping SibSp, Parch, Name, PassengerId, Fare, and Ticket from the data
# Fare and Ticket price would indicate the same thing as Pclass and therefore may introduce multicolinearity

titanic_train.Pclass.head()

titanic_train = titanic_train.drop(['SibSp','Parch', 'PassengerId', 'Name', 'Fare', 'Ticket'], axis=1)
titanic_test = titanic_test.drop(['SibSp','Parch', 'PassengerId', 'Name', 'Fare', 'Ticket'], axis=1)

sns.countplot(x='Embarked', data=titanic_train)
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

# Completing the null values in the age coloumn based on mean ages of Pclass

age_mean = titanic_train.groupby(['Pclass'])['Age'].mean()
print(age_mean)

# Adding the mean age by Pclass for each null value in the Age column

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age

titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(impute_age,axis=1)
titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)

titanic_train = titanic_train.drop(['Cabin'], axis=1)
titanic_test = titanic_test.drop(['Cabin'], axis=1)

titanic_train.info()
titanic_test.info()

# Hot encoding categorical variables

Pasclass = {1:'Class1', 2:'Class2', 3:'Class3'}
titanic_train.Pclass = [Pasclass[item] for item in titanic_train.Pclass]
titanic_test.Pclass = [Pasclass[item] for item in titanic_test.Pclass]

titanic_train.info()
titanic_train.head(15)

titanic_test.info()
titanic_test.head(15)

dSex = pd.get_dummies(titanic_train['Sex'])
dEmbarked = pd.get_dummies(titanic_train['Embarked'])
dPclass = pd.get_dummies(titanic_train['Pclass'])

titanic_train1 = titanic_train.drop(['Sex', 'Embarked', 'Pclass'], axis = 1)
titanic_train1 = pd.concat([titanic_train1, dSex, dEmbarked, dPclass], axis=1)

titanic_train1.info()
titanic_train1.head(15)

dtSex = pd.get_dummies(titanic_test['Sex'])
dtEmbarked = pd.get_dummies(titanic_test['Embarked'])
dtPclass = pd.get_dummies(titanic_test['Pclass'])

titanic_test1 = titanic_test.drop(['Sex', 'Embarked', 'Pclass'], axis = 1)
titanic_test1 = pd.concat([titanic_test1, dtSex, dtEmbarked, dtPclass], axis=1)

titanic_test1.info()
titanic_test1.head(15)

# Our data is ready for model building 
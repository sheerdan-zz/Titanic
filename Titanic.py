import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Import the data
train = pd.read_csv('train.csv')
sur = train.Survived == 1
boy = train.Sex == 'male'
noAge = np.isnan(train.Age)

# Descriptive Statistics
print(train.describe())

## Data Visualization
# Age
plt.figure(0)
ax = plt.subplot(2, 1, 1)
plt.hist(train.Age[~noAge][sur], 30)
ax.plot([np.nanmean(train.Age[sur]),   np.nanmean(train.Age[sur])],   [0, 50], 'r', label='Mean')
ax.plot([np.nanmedian(train.Age[sur]), np.nanmedian(train.Age[sur])], [0, 50], 'b', label='Median')
plt.ylabel('Survivors Frequency')
ax.legend()
ax.axis([0, 80, 0, 50])

ax = plt.subplot(2, 1, 2)
plt.hist(train.Age[~noAge][~sur], 30)
ax.plot([np.nanmean(train.Age[~sur]),   np.nanmean(train.Age[~sur])],   [0, 50], 'r', label='Mean')
ax.plot([np.nanmedian(train.Age[~sur]), np.nanmedian(train.Age[~sur])], [0, 50], 'b', label='Median')
plt.xlabel('Age')
plt.ylabel('~Survivors Frequency')
ax.legend()
ax.axis([0, 80, 0, 50])

# No Age
print(sum(train.Survived[noAge])/sum(noAge))

# Gender
sb.catplot(x="Sex", y="Survived", kind="bar", data=train)

# Age and Gender
# histogram
plt.figure()
ax = plt.subplot(2, 2, 1)
plt.hist(train.Age[sur & boy], 30)
plt.title('Men who survived')
ax.axis([0, 80, 0, 40])

ax = plt.subplot(2, 2, 2)
plt.hist(train.Age[sur & ~boy], 30)
plt.title('Women who survived')
ax.axis([0, 80, 0, 40])

ax = plt.subplot(2, 2, 3)
plt.hist(train.Age[~sur & boy], 30)
plt.title('Men who didn''t survived')
ax.axis([0, 80, 0, 40])

ax = plt.subplot(2, 2, 4)
plt.hist(train.Age[~sur & ~boy], 30)
plt.title('Women who didn''t survived')
ax.axis([0, 80, 0, 40])

# scatter plot
plt.figure()
sb.swarmplot(x="Survived", y="Age", hue="Sex", palette=["c", "m"], data=train)

# Class
sb.catplot(x="Pclass", y="Survived", kind="bar", data=train)

# Gender and Class
sb.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train)

# Embarked
sb.catplot(x="Embarked", y="Survived", kind="bar", data=train)

# Embarked and Class
sb.catplot('Pclass', col='Embarked', kind='count', data=train)

# Embarked and Sex
sb.catplot('Sex', col='Embarked', kind='count', data=train)

# Fare and Pclass
ax = plt.figure()
for iClass in np.unique(train.Pclass):
    plt.hist(train.Fare[train.Pclass == iClass], 30, label=str(iClass))
plt.xlabel('Fare')
plt.ylabel('Frequency')
ax.legend()
sb.catplot(x='Pclass', y='Fare', data=train, kind='point')

# Family with you
sb.catplot(x="SibSp", y="Survived", kind="bar", data=train)
sb.catplot(x="Parch", y="Survived", kind="bar", data=train)

## Arrange the Data
print(train.info())
validArray = train.drop(columns=['PassengerId', 'Name'])
validArray.Age[np.isnan(validArray.Age)]  = np.nanmean(train.Age)
validArray.Sex[boy]  = 0
validArray.Sex[~boy] = 1
validArray.Embarked[validArray.Embarked == 'C']  = 1
validArray.Embarked[validArray.Embarked == 'Q']  = 2
validArray.Embarked[validArray.Embarked == 'S']  = 3
validArray['Family'] = validArray.SibSp+validArray.Parch
validArray['FamilySur']  = validArray.Survived+101
validArray['SameTicket'] = validArray.Survived+101

for iPassenger in range(0, len(validArray.Ticket)):
    if type(validArray.Embarked[iPassenger]) != int:
        validArray.Embarked[iPassenger] = 0
    if type(validArray.Cabin[iPassenger]) != str:
            validArray.Cabin[iPassenger] = 0
    if type(validArray.Cabin[iPassenger]) == str:
            validArray.Cabin[iPassenger] = 1
    if len(re.split(' ', validArray.Ticket[iPassenger]))> 1:
        validArray.Ticket[iPassenger] = int(re.split(' ', validArray.Ticket[iPassenger])[-1])
    elif type(re.search('[0-9]+', validArray.Ticket[iPassenger])) != re.match:
        validArray.Ticket[iPassenger] = 0

for iTicket in range(0, len(validArray.Ticket)):
    if validArray.FamilySur[iTicket] > 100:
        sameIdx = validArray.Ticket[iTicket] == validArray.Ticket
        validArray.SameTicket[sameIdx] = sum(sameIdx)
        if sum(sameIdx) == 1:
            validArray.FamilySur[sameIdx] = -1
        else:
            validArray.FamilySur[sameIdx] = sum(validArray.Survived[sameIdx])/sum(sameIdx)

plt.figure()
sb.heatmap(validArray.corr(), annot=True, fmt=".2f", cmap="coolwarm")

## Find a Model
# Create and Split validation data set
validFeatures = list(validArray)
validFeatures = validFeatures[1:]
validArray = validArray.astype('float')
validArray = validArray.values
X = validArray[:, 1:]
Y = validArray[:, 0]
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

# Spot Check Algorithms
models = []
models.append(('LR',   LogisticRegression()))
models.append(('LDA',  LinearDiscriminantAnalysis()))
models.append(('KNN',  KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF',   RandomForestClassifier()))
models.append(('GB',   GradientBoostingClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('LDA')
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('CART')
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('Random Forest')
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
predictions = RF.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

gini = RF.feature_importances_
giniIdx = sorted(range(len(gini)), key=gini.__getitem__)
for iFeature in giniIdx:
    print('%s - %f' % (validFeatures[iFeature], gini[iFeature]))

print('Gradient Boosting Classifier')
GB = GradientBoostingClassifier()
GB.fit(X_train, Y_train)
predictions = GB.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

plt.show()

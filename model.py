#!/usr/bin/env python
#import pdb; pdb.set_trace()

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# Importing and cleaning data
df = pd.read_csv(r'train.csv')
columns_names = ['Pclass', 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare']

X = df.loc[:,columns_names]
X['Sex'] = pd.factorize(df.Sex)[0]
X['Age'] = X['Age'].fillna(X['Age'].mean())
X = X.fillna(0) # TODO: find a better solution
y = df['Survived']

# Cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.4, random_state=0)

# We fit the classifier to the train data and then we test it
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
results = rfc.predict(X_test)

def count_mistakes(results, test):
	"""Given two list-like objects it returns a count of mistakes."""
	mistakes = 0
	r = results.tolist()
	t = test.tolist()
	for index in range(len(t)):
		if t[index] != r[index]:
			mistakes +=1
	return mistakes

mistakes = count_mistakes(results, y_test)
print "The classifier made",mistakes,"mistakes."
print "There were a total of",len(y_test.tolist()),"to identify."

import pdb; pdb.set_trace()

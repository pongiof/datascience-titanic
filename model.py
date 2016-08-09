#!/usr/bin/env python
#import pdb; pdb.set_trace()

# Importing libraries
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

import matplotlib.pyplot as plt

def calculate_stats(results, test):
	"""Given two list-like objects calculates model stats."""
	mistakes = 0
	r = results.tolist()
	t = test.tolist()
	fp = 0
	fn = 0
	tp = 0
	tn = 0
	positive_values = sum(t)
	negative_values = len(t) - positive_values
	for index in range(len(t)):
		if r[index] == 1:
			if r[index] == t[index]:
				tp+=1
			else:
				fp+=1
		else:
			if r[index] == t[index]:
				tn+=1
			else:
				fn+=1
	accuracy = 1 - float(fn + fp) / len(t)
	recall = float(tp) / positive_values
	precision = float(tp) / (tp + fp)
	return {'accuracy': accuracy, 'recall': recall, 'precision': precision}

# Importing and cleaning data
df = pd.read_csv(r'train.csv')
columns_names = ['Pclass', 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare']

X = df.loc[:,columns_names]
X['Sex'] = pd.factorize(df.Sex)[0]
X['Age'] = X['Age'].fillna(X['Age'].mean())
X = X.fillna(0) # TODO: find a better solution
y = df['Survived']

accs = []
x = range(1,10) 
for giuseppe in x:

	# We fit the classifier to the train data and then we test it
	rfc = RandomForestClassifier(class_weight='balanced', random_state=1, max_depth = giuseppe)

	runs = []
	for index in range(5):
		# Cross validation
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.4)
		rfc.fit(X_train, y_train)
		runs.append(calculate_stats(rfc.predict(X_test), y_test))

	final_accuracy = reduce(lambda x, y: x + y, [run['accuracy'] for run in runs]) / len(runs)
	final_recall = reduce(lambda x, y: x + y, [run['recall'] for run in runs]) / len(runs)
	final_precision = reduce(lambda x, y: x + y, [run['precision'] for run in runs]) / len(runs)

	accs.append(final_accuracy)

# print "stats: \n - accuracy ", final_accuracy, "\n - recall", final_recall, "\n - precision", final_precision

print(accs)

plt.plot(x, accs)
plt.show()

import pdb; pdb.set_trace()

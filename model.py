#!/usr/bin/env frameworkpython
#import pdb; pdb.set_trace()

# Importing libraries
import time
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.tree import export_graphviz

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

if __name__ == "__main__":
	# Importing and cleaning data
	df = pd.read_csv(r'train.csv')
	columns_names = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'SibSp']

	X = df.loc[:,columns_names]
	X['Sex'] = pd.factorize(df.Sex)[0]
	X['Age'] = X['Age'].fillna(X['Age'].mean())
	X = X.fillna(0) # TODO: find a better solution
	y = df['Survived']

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.4)

	# Fit the classifier
	rfc = RandomForestClassifier(class_weight='balanced')
	rfc.fit(X_train, y_train)
	results = rfc.predict(X_test)
	stats = calculate_stats(results, y_test)

	# Print the statistics
	print("Statistics:")

	for s, value in stats.iteritems():
		print(s+":",value)

	# Check feature importance
	importances = rfc.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
				 axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the classifier
	if len(sys.argv) > 1 and sys.argv[1] == "--plot":
		import matplotlib.pyplot as plt
		plt.figure()
		plt.title("Feature importances")
		plt.bar(range(X.shape[1]), importances[indices],
		       color="r", yerr=std[indices], align="center")
		plt.xticks(range(X.shape[1]), [columns_names[x] for x in indices])
		plt.xlim([-1, X.shape[1]])
		plt.show()

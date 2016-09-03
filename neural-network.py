#!/usr/bin/env frameworkpython
#import pdb; pdb.set_trace()

# Importing libraries
import pandas as pd
import numpy as np
from sklearn import cross_validation
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

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

    # Create the sample
	ds = SupervisedDataSet(6,1)

	for index, row in X_train.iterrows():
		x_value = list(row)
		y_value = [y_train[index]]
    		ds.addSample(x_value, y_value)

    # Build the neural network
	n = buildNetwork(ds.indim,8,8,ds.outdim,recurrent=True)
	t = BackpropTrainer(n,learningrate=0.01,momentum=0.5,verbose=True)
	t.trainOnDataset(ds,1000)
	t.testOnData(verbose=True)

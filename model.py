#!/usr/bin/env python

#import pdb; pdb.set_trace()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv(r'train.csv')
columns_names = ['Pclass', 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare']
X = df.loc[:,columns_names]

X['Sex'] = pd.factorize(df.Sex)[0]
X['Age'] = X['Age'].fillna(X['Age'].mean())
X = X.fillna(0) # TODO: find a better solution
y = df['Survived']

rfc = RandomForestClassifier()
rfc.fit(X,y)
result = rfc.predict(X)

mistakes = 0
for x in range(len(y.index)):
	if y[x] != result[x]:
		mistakes +=1


import pdb; pdb.set_trace()

# TODO Cross validation instead X

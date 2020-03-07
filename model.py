import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import pickle

iris = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

# Missing values if any
'''for column in iris.columns:
    print(column, ', n(missing_values) = ', iris[column].isnull().values.sum()*100/len(iris), ', unique values= ', iris[column].nunique())
    print('')
'''
iris_setosa = iris.loc[iris['species'] == 'setosa']
iris_virginica = iris.loc[iris['species'] == 'virginica']
iris_versicolor = iris.loc[iris['species'] == 'versicolor']

#### Label encoding using .replace() method 

iris['species'].replace(['setosa', 'virginica', 'versicolor'], [0, 1, 2], inplace=True)

# Converting Pandas DataFrame into a Numpy array
X = iris.iloc[:, 0:4].values # from column(sepal_length) to column(petal_width)

# Converting each species name into digits
y = pd.factorize(iris['species'])[0]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)
clf = RandomForestClassifier(random_state=18)
clf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.1, 3.0, 1.4, 0.4]]))
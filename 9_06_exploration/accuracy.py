from sklearn.datasets import load_iris

iris = load_iris()

print(dir(iris))
iris.keys()

iris_data = iris.data

print(iris_data.shape) 
print(iris_data[0])
iris_label = iris.target
print(iris_label.shape)
iris_label
print(iris.DESCR)
print(iris.feature_names)
print(iris.filename)

import pandas as pd
print(pd.__version__)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
print(iris_df)
iris_df["label"] = iris.target
print(iris_df)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_train, y_test)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
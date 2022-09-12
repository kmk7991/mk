from sklearn.datasets import load_iris

iris = load_iris()

print(dir(iris))
iris.keys()

iris_data = iris.data

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
pritn(y_pred)
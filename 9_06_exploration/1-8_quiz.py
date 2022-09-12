from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

digits = load_digits()
digits.keys()
digits_data = digits.data
digits_data.shape
digits_label = digits.target
print(digits_label.shape)
digits_label[:20]
new_label = [3 if i == 3 else 0 for i in digits_label]
new_label[:20]


X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    new_label,
                                                    test_size=0.2,
                                                    random_state=15)

decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
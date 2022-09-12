import sklearn

print(sklearn.__version__)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
digits_label = digits.target
digits_data = digits.data

X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    digits_label,
                                                    test_size=0.3,
                                                    random_state=15)

random_forest = RandomForestClassifier(random_state=15)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
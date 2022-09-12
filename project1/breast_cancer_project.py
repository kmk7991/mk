#(1) 필요한 모듈 import 해오기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd



#(2) 데이터 준비
breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target

#(3) 데이터 이해하기
features = breast_cancer['data']
labels = breast_cancer['target']
#print(breast_cancer.target_names)
#print(breast_cancer.DESCR)


#(4) train, test 데이터 분리하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state= 32)

#print(X_train.shape, y_train.shape) 
#print(y_train, y_test)  


#(5) 다양한 모델로 학습시켜보자!!
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
#print(decision_tree._estimator_type)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
#print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['malignant','benign']))


#Randomforest로 해볼까?
from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2,  
                                                    random_state=32) 

random_forest = RandomForestClassifier(random_state=32) 
random_forest.fit(X_train, y_train) 
y_pred = random_forest.predict(X_test) 
#print(y_pred)
#print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['malignant','benign']))



# sgd 모델
from sklearn.linear_model import SGDClassifier 
sgd_model = SGDClassifier() 
sgd_model.fit(X_train, y_train) 
y_pred = sgd_model.predict(X_test)
#print(y_pred3)
#print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['malignant','benign']))


#logistic regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)



from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['malignant','benign']))

'''전체적인 accuracy 테스트를 보면 decision tree가 가장 
높게 나왔으나, 유방암 진단은 예측의 정확성이 중요하므로 
confusion matrix에서 precision 비율이 높은것으로 결정한다. 그런데 
confusion matrix의 precision 결과값도 decision tree 가 0.94로 가장
높으므로 decision tree를 선택하는 것이 바람직하다.'''


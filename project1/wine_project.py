#(1) 필요한 모듈 import 해오기
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd



#(2) 데이터 준비
wine = load_wine()
wine_data = wine.data
wine_label = wine.target

#(3) 데이터 이해하기
features = wine['data']
labels = wine['target']
#print(wine.target_names)
#print(wine.DESCR)

#(4) train, test 데이터 분리하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine_data, wine_label, test_size=0.3, random_state= 32)

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



#Randomforest로 해볼까?
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트라는 분류기를 사용하기 위해 import

X_train, X_test, y_train, y_test = train_test_split(wine_data, # iris 데이터의 data 컬럼
                                                    wine_label, # iris 데이터의 target 컬럼
                                                    test_size=0.2, # test_size : train data와 test data를 몇대몇으로 나눌지 정하는 옵션
                                                    random_state=32) # random_state : 랜덤 패턴의 값을 지정

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
print(classification_report(y_test, y_pred, target_names=['class_0','class_1','class_2']))




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


#logistic regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)

#print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

'''digits 데이터와 유사하게, wine 데이터 또한 accuracy 테스트를
해본 결과 random forest가 가장 높게 나왔다. 따라서 예측모델은
random forest로 하고, confusion matrix를 사용하여 좀 더 
세부적으로 평가해보고자 했다. confusion matrix를 사용하여 보니
class_0은 16, class_01은 9, class_02는 10개로 비교적 0을 더 잘
맞추는 경향이 있었다. 따라서 기본적인 accuracy 보다는 f1 score를 
평가모델로 사용하는 것이 더 바람직해 보인다.'''
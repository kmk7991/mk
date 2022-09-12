

#(1) 필요한 모듈 import하기
import sklearn
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#(2) 데이터 준비
digits = load_digits()
digits_data = digits.data
digits_label = digits.target

#(3) 데이터 이해하기
features = digits['data']
feature_names = digits['feature_names']
#print(features[:1])
#print(feature_names)


labels = digits['target']
#print(labels)

target_names = digits['target_names']
print(target_names)

#print(digits.DESCR)


digits_df = pd.DataFrame(data=digits_data, columns=digits.feature_names)
digits_df['label'] = digits.target
#print(digits_df)




#(4) train, test 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    digits_label,
                                                    test_size=0.2,
                                                    random_state=15) 
#train과 test의 형상정보 확인
#print(X_train.shape, y_train.shape) 
#label이 잘 분리되었는지 확인
#print(y_train, y_test)  

#(5) 다양한 모델로 학습시켜보기
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
#print(decision_tree._estimator_type)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
#print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)


#Randomforest로 해볼까?
from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
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
print(classification_report(y_test, y_pred, target_names=['0', '1','2','3','4','5','6','7','8','9']))



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



'''load_digit의 여러개의 예측모델 가운데 가장 accuracy가 높은
 random forest가 가장 좋을 것 같다.confusion matrix표를 보면 
 평균적으로 균등하게 맞히는 것을 볼 수있었다. 그러나
 이를 좀더 정확하게 평가하기위해서는 f1 score로 평가하는 것이 좋을 것 같다'''
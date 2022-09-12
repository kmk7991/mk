from sklearn.datasets import load_wine
data = load_wine()
print(type(data))
print(data.target)
print(data.feature_names)

print(data.target_names)
print(print(data.DESCR))

#특성행렬은 통상 변수명 x에 저장하고, 타겟 벡터는 y에 저장한다.
X = data.data
y = data.target

#모델생성! 분류이므로 RandomForestClassifier를 사용하겟다
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#훈련시키자!
model.fit(X, y)

#예측을 하자!
y_pred = model.predict(X)

#이제 성능을 평가해보자! 성능은 sklearn.metrics 모듈을 사용한다고 했고, 
# 분류이므로 classification_report 와 accuracy_score를 이용하자!
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#타겟 벡터 즉 라벨인 변수명 y와 예측값 y_pred을 각각 인자로 넣습니다. 
print(classification_report(y, y_pred))
#정확도를 출력합니다. 
print("accuracy = ", accuracy_score(y, y_pred))

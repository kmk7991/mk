#(1) 데이터 가져오기
#(2) 모델에 입력할 데이터 X 준비하기
from pydoc import describe
from pyexpat import features
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


diabetes=load_diabetes()


x = diabetes.data
y = diabetes.target

#arrayname.isalnum()
# #arrayname.isalpha()



#(3) 모델에 예측할 데이터 y준비하기
y = diabetes.target #이미 어레이임!




#(4) train 데이터와 test데이터로 분리하기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#(5)모델 준비하기
W = np.random.rand(10)
b = np.random.rand()


def model(x, W, b):
    predictions = 0
    for i in range(10):
        predictions += x[:, i] * W[i]
    predictions += b
    return predictions
    

#(6) 손실 loss 정의하기
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse

def loss(x, W, b, y):
    predictions = model(x, W, b)
    L = MSE(predictions, y)
    return L

#(7)기울기를 구하는 gradient 함수 구현하기
def gradient(x, W, b, y):
    # N은 가중치의 개수
    N = len(W)
    
    # y_pred 준비
    y_pred = model(x, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * x.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db

dW, db = gradient(x, W, b, y)
print("dW:", dW)
print("db:", db)


#(8) 하이퍼 파라미터인 학습률 설정하기
LEARNING_RATE = 0.1

#(9) 모델 학습하기
losses = []

for i in range(1, 50000):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))


#(10) test 데이터에 대한 성능 확인하기
prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
print(mse)

#(11) 정답 데이터와 예측한 데이터 시각화하기
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()



















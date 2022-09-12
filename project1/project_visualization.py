import sklearn

print(sklearn.__version__)

#(1) 필요한 모듈 import하기
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#(2) 데이터 준비
digits = load_digits()
digits_data = digits.data
digits_label = digits.target

#(3) 데이터 이해하기
features = digits['data']
feature_names = digits['feature_names']
labels = digits['target']
target_names = digits['target_names']

df = pd.DataFrame(digits)
print(df.head())

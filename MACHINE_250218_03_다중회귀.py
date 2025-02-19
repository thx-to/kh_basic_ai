# 다중 회귀(Multiple Linear Regression)
# 여러 개의 입력 특성(독립 변수)를 사용해 하나의 타겟 값(종속 변수)를 예측하는 선형 회귀 모델

# 데이터 로드 및 변환
# pd.read_csv()를 사용해 데이터를 불러옴
# to_numpy()를 통해 NumPy 배열로 변환
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

# 무게 데이터 넣기
# perch_weight는 타겟 변수(종속 변수)
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 / 테스트 데이터 분할
# train_test_split()을 사용해 데이터를 훈련 세트와 테스트 세트로 나눔
# random_state=42를 설정하여 재현 가능한 결과 보장
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# 다항 특징 변환
# PolynomialFeatures(degree=5, include_bias=False)를 사용해 다항식 특성을 생성
# fit_transform()을 통해 훈련 데이터를 변환하고, transform()으로 테스트 데이터 변환
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5, include_bias=False)
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

# 선형 회귀 모델 훈련
# R^2 점수로 모델의 설명력 평가
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))


# 규제 (Regularization)
# 머신러닝 모델이 훈련 데이터에 과도하게 맞춰지는 현상(과대적합)을 방지하게 위해 모델의 복잡도를 줄이는 기법
# 릿지 규제 : 계수의 제곱합을 사용하여 큰 계수를 억제
# 라쏘 규제 : 계수의 절댓값 합을 사용
from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # 데이터를 표준화하기 위해 사용
ss.fit(train_poly) # 데이터를 기반으로 평균과 표준편차를 계산
train_scaled = ss.transform(train_poly) # 훈련 데이터의 표준화된 결과가 저장
test_scaled = ss.transform(test_poly)   # 테스트 데이터의 표준화된 결과가 저장

# 릿지 회귀
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘 회귀
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
print(np.sum(lasso.coef_ == 0))
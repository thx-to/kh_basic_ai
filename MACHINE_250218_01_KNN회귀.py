# 회귀 : 연속적인 수치값 예측
# k-최근접 이웃 회귀 : 가장 가ㅏㄲ운 k개의 이웃 값의 평균으로 예측

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 훈련세트와 테스트세트 분리
from sklearn.neighbors import KNeighborsRegressor # KNN 회귀모델
from sklearn.metrics import mean_absolute_error # 평균 절댓값 오차

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

plt.scatter(perch_length, perch_weight)
plt.xlabel('length(cm)')
plt.ylabel('weight(g)')
plt.show()

# 훈련 세트와 테스트 세트 분리
# train_test_split() : 데이터를 훈련 세트와 테스트 세트로 나누는 함수 (from sklearn-model_selection)
# perch_length: 입력(feature) 데이터
# perch_weight : 타겟(label) 데이터
# random_state=42 : 데이터 분할 시 무작위성을 조정하여 실행할 때마다 동일한 결과가 나오도록 고정
# train_test_split()은 기본적으로 데이털르 75%:25%(훈련:테스트) 비율로 나눔
# 만약 비율을 조정하고 싶다면 test_size 매개변수 설정
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# 2차원 배열로 변환
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

# k-최근접 이웃 회귀 모델 훈련
knr = KNeighborsRegressor() # 모델 생성
knr.fit(train_input, train_target) # 모델 훈련

# 모델 평가 : 1에 가까울수록 좋음
test_score = knr.score(test_input, test_target) # 테스트 세트 점수
print("👉 테스트 세트 결정계수(R^2) : ", test_score )

# 평균 절댓값 오차 (MAE, Mean Absolute Error) : 값이 작을수록 예측이 실제 값에 가까움
test_prediction = knr.predict(test_input) # 테스트 세트에 대한 예측 생성
mae = mean_absolute_error(test_target, test_prediction) # 테스트 세트에 대한 평균 절댓값 오차 계산
print("👉 평균 절댓값 오차(MAE) : ", mae)

# 과대 적합 vs 과소 적합
# 과대 적합 : 모델이 훈련 세트에 너무 맞춰져 새로운 데이터에 대한 예측력이 떨어지는 것
# 훈련 세트 점수는 높지만 테스트 세트 점수가 낮게 나옴
# 과소 적합 : 모델이 충분히 훈련되지 ㅇ낳아 데이터 패턴을 잘 학습하지 못한 경우
# 훈련 세트와 테스트 세트 점수가 모두 낮은 경우, 테스트 세트 점수가 훈련 세트보다 높은 경우

# 훈련 세트 점수
train_score = knr.score(train_input, train_target)
print(f"👉 훈련 세트 결정 계수(R^2) : {train_score}")

# 테스트 세트 점수
test_score = knr.score(test_input, test_target)
print(f"👉 테스트 세트 결정 계수(R^2) : {test_score}")

# 모델 개선 : 이웃 수 변경
# 이웃의 개수를 3으로 설정
knr.n_neighbors = 3

# 모델 재훈련
knr.fit(train_input, train_target)

# 새로운 훈련 세트 점수
new_train_score = knr.score(train_input, train_target)
print("👉 새로운 훈련 세트 결정계수(R^2):", new_train_score)

# 새로운 테스트 세트 점수
new_test_score = knr.score(test_input, test_target)
print("👉 새로운 테스트 세트 결정계수(R^2):", new_test_score)
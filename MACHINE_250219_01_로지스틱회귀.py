# 로지스틱 회귀 : 입력값을 0과 1 사이의 확률로 변환하는 시그모이드 함수를 사용
# 머신러닝과 통계학에서 자주 사용되는 분류 알고리즘 (주로 이진 분류에 많이 사용)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 데이터를 표준화하는 도구(평균을 0, 표준편차 1)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1️⃣ 데이터 불러오기
fish = pd.read_csv('https://bit.ly/fish_csv_data')

# 2️⃣ 입력(x)과 타겟(y) 분리
# 물고기의 무게, 길이, 대각선 길이, 높이, 너비 정보를 선택
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()  # 타깃인 물고기의 종류

# 3️⃣ 학습 데이터와 테스트 데이터 분할
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 4️⃣ 데이터 표준화 (평균 0, 표준편차 1로 변환)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 5️⃣ K-최근접 이웃 (K-NN) 분류기
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

# K-NN 모델 성능 확인
print("========================= 🤖 K-NN 모델 성능 =========================")
print(f"📍 훈련 세트 정확도\n{kn.score(train_scaled, train_target)}")
print(f"📍 테스트 세트 정확도\n{kn.score(test_scaled, test_target)}")
print(f"📍 테스트 데이터 예측 결과\n{kn.predict(test_scaled[:5])}")
print(f"📍 각 클래스의 확률 예측\n{kn.predict_proba(test_scaled[:5])}")

# 6️⃣ 로지스틱 회귀 (이진 분류 : Bream vs Smelt)
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
lr_binary = LogisticRegression()
lr_binary.fit(train_bream_smelt, target_bream_smelt)

# 로지스틱 회귀 (이진 분류) 성능 확인
print("\n================ 👾 로지스틱 회귀 이진 분류 모델 성능 ================")
print(f"📍 이진 분류 예측 결과\n{lr_binary.predict(train_bream_smelt[:5])}")
print(f"📍 이진 분류 확률 예측\n{lr_binary.predict_proba(train_bream_smelt[:5])}")
print(f"📍 이진 분류 계수\n{lr_binary.coef_}")
print(f"📍 이진 분류 절편\n{lr_binary.intercept_}")

# 7️⃣ 로지스틱 회귀 (다중 분류: 전체 7개 물고기 분류)
lr_multi = LogisticRegression(C=20, max_iter=1000)
lr_multi.fit(train_scaled, train_target)

# 로지스틱 회귀 (다중 분류) 성능 확인
print("\n================ 👽 로지스틱 회귀 다중 분류 모델 성능 ================")
print(f"📍 훈련 세트 정확도\n{lr_multi.score(train_scaled, train_target)}")
print(f"📍 테스트 세트 정확도\n{lr_multi.score(test_scaled, test_target)}")
print(f"📍 테스트 데이터 예측 결과\n{lr_multi.predict(test_scaled[:5])}")
print(f"📍 다중 클래스 확률 예측 \n{lr_multi.predict_proba(test_scaled[:5])}")

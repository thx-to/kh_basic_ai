import matplotlib.pyplot as plt # 시각화를 위함

# 🐠도미와 🐟빙어의 길이와 무게 데이터 입력
bream_length = [25.4, 26.3, 26.5, 29.0, 29.7, 30.0, 31.5, 32.0, 33.0, 33.5, 34.0, 35.0, 36.0, 37.0, 38.5]
bream_weight = [242.0, 290.0, 340.0, 363.0, 450.0, 500.0, 340.0, 600.0, 700.0, 610.0, 685.0, 725.0, 850.0, 920.0, 1000.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 12.0, 12.2, 12.4, 13.0, 13.5, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.8, 9.9, 10.5, 11.0, 12.0, 19.7, 19.9]

# 데이터 시각화 📊
plt.scatter(bream_length, bream_weight, label='Bream', color='orange')
plt.scatter(smelt_length, smelt_weight, label='Smelt', color='skyblue')
plt.xlabel('Length (cm)')
plt.ylabel('⚖Weight (g)')
plt.title('Bream & Smelt Data Visualization')
plt.legend()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# K-인접회귀(K-최근접 이웃, KNN, K-Nearest Neighbor) 알고리즘
# 가장 가까운 데이터를 참고하여 다수결의 원칙으로 예측
# 주변(인접 환경)에 도미가 많으면 🐠도미, 빙어가 많으면 🐟빙어가 됨

# 데이터 준비 📊
length = bream_length + smelt_length # 🐠도미와 🐟빙어의 길이를 합친 길이 데이터
weight = bream_weight + smelt_weight # 🐠도미와 🐟빙어의 무게를 합친 무게 데이터
fish_data = [[l, w] for l, w in zip(length, weight)] # 위 두개의 리스트를 쌍으로, 하나의 데이터로 묶어줌
fish_target = [1] * len(bream_length) + [0] * len(smelt_length) # 1과 0으로 라벨링을 하기 위한 부분 (🐠도미는 1, 🐟빙어는 0)

# 모델 훈련 🤖
kn = KNeighborsClassifier() # 모델에 대한 객체를 만듦, 훈련 모델 선정
kn.fit(fish_data, fish_target) # fit() : 훈련 진행

# 모델 평가 📊
score = kn.score(fish_data, fish_target) # 점수화하기 위해 데이터로 스코어를 만듦
print(f'Model accuracy: {score:.2f}') # 소수점 2자리까지 출력

# 새로운 데이터 예측 🤖
prediction = kn.predict([[30, 600]]) # fit()을 통해 훈련된 모델에 예측치 넣기
if prediction[0] == 1: # kn.predict()의 결과를 저장하는 prediction 리스트(변수)에 결과 저장 / 도미는 1, 빙어는 0
    print('🐠Bream!')
else:
    print('🐟Smelt!')
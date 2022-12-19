---
title : "[알고리즘/지도학습] 회귀문제 - 선형회귀, 의사결정회귀나무, 의사결정회귀나무 앙상블(그래디언트 부스트)"
excerpt : "공부한 내용을 정리한 글"

categories : 
- computer science
- algorithm
- study 

tags : 
- [computer science, algorithm, study]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-08-13
last_modified_at : 2022-08-13

---

'프로그래머가 알아야 할 알고리즘 40'(임란 아마드 지음, 길벗 출판사) 을 통해 선형회귀, 의사결정회귀나무, 의사결정회귀나무 앙상블을 공부하고 나서, 그 내용을 내 언어로 바꾸어 기록한다. 

---

# 회귀문제 

분류문제는 타겟값이 카테고리 확률변수였다. 

$\Rightarrow$ 회귀문제는 타겟값이 연속확률변수다. 

아래는 회귀문제 해결하는 데 사용할 수 있는, 회귀 알고리즘 들이다. 

---

# 선형회귀 (Linear regression) 알고리즘

## 정의 

여러 독립변수들과, 종속변수 사이 관계 선형으로 나타낸 것. 

## 목표 

현실에서 얻은 표본값들과 기댓값 예측치 사이 오차 가장 작게 하는, 독립변수 $i$ 의 가중치 $\beta_{i}$ 찾기. 

## 종류 

단순선형회귀

- 독립변수 1개와 종속변수 사이 관계 나타낸 것. 
- $\hat{y} = \beta_{0} + \beta_{1}x_{1}$ or $y = \beta_{0} + \beta_{1}x_{1} + \epsilon$, $\epsilon$ 은 오차. 
- $\beta_{0}$ 은 독립변수 가중치 $0$ 일 때 $\hat{y}$ 값

다중선형회귀

- 독립변수 $n$개와 종속변수 사이 관계 나타낸 것. 
- $\hat{y} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$ or $y = \beta_{0} + \beta_{1}x_{1} + ... \beta_{n}x_{n} + \epsilon$, $\epsilon$ 은 오차. 
- $\beta_{0}$ 은 독립변수 가중치 $0$ 일 때 $\hat{y}$ 값

## 선형회귀모델 예측값 $\hat{y}$ 의미

기댓값 예측치. 

## 선형회귀모델 손실함수(RMSE)

Root Mean Squared Error. 

오차제곱합에 루트 씌운 것. $\Rightarrow$ 모델의 전반적 오차. 

$loss = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{i} - y^{i})^{2}}$

## 선형회귀모형으로 회귀문제 해결하기 

데이터셋 로드 

```python 
import pandas as pd 

df = pd.read_csv('/Users/kibeomkim/Desktop/auto.csv') ; df 
```
<img width="717" alt="Screen Shot 2022-08-13 at 17 15 53" src="https://user-images.githubusercontent.com/83487073/184475386-5db07ad0-abf8-49ab-8844-003056b159cf.png">

```python 
df.describe()
```
<img width="528" alt="Screen Shot 2022-08-13 at 17 16 24" src="https://user-images.githubusercontent.com/83487073/184475407-2c8876ea-22a6-4ccc-95a4-f016b62bbb80.png">

```python 
df.info() 
```
<img width="377" alt="Screen Shot 2022-08-13 at 17 16 54" src="https://user-images.githubusercontent.com/83487073/184475422-73c841e0-c402-44a2-9fff-a84e0f01f0ce.png">

```python 
df.isnull().sum() 
```
<img width="160" alt="Screen Shot 2022-08-13 at 17 17 24" src="https://user-images.githubusercontent.com/83487073/184475446-3b6227d8-c41d-4186-8d81-93958b7bdef4.png">

데이터셋 전처리 

```python 
# 데이터 파이프라인 정의 
def pipeline(df) : 
    df.drop('NAME', axis=1, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    return df 
df = pipeline(df)
```

훈련용 데이터셋과 테스트용 데이터셋으로, 전체 데이터셋 분리하기 

```python 
# 전처리 후, 훈련셋과 검증셋으로 데이터셋 나누기 
print(df.shape)
from sklearn.model_selection import train_test_split 

# 타겟
y = df['MPG'].values
# 특성변수들 
X = df.values 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
```

선형회귀모델 훈련시키기 

이 경우 여러 독립변수와 종속변수 1개 사용했으므로, 모델이 다중선형회귀모델이 된다. 

```python 
# 선형회귀 모델 훈련시키기 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()

# 회귀모델 훈련 
regressor.fit(x_train, y_train)
```

모델 예측 

```python 
y_pred = regressor.predict(x_test)

# Regressor 성능 평가 ; RMSE
from sklearn.metrics import mean_squared_error 
from math import sqrt
print(sqrt(mean_squared_error(y_test, y_pred))) 
```
RMSE value: 2.7248410669138417e-14

---

# 의사결정회귀나무 

타겟값이 연속형이라는 것 제외하면, 의사결정나무(분류기)와 거의 같다. 

- 기댓값 예측치 뽑아낸다. 

## 의사결정회귀나무 모형으로 회귀문제 해결하기 

```python 
from sklearn.tree import DecisionTreeRegressor

# 최대 깊이=3 인 의사결정 회귀나무 정의 
regressor = DecisionTreeRegressor(max_depth=3)

# 모델 훈련 
regressor.fit(x_train, y_train)

# 테스트 데이터에 대해 기댓값 근사치 예측 수행 
y_pred = regressor.predict(x_test)

print(sqrt(mean_squared_error(y_test, y_pred)))
```
RMSE value: 1.4008722335107024

---

# 그레디언트 부스팅 알고리즘

경사하강으로 손실 계속 줄여가며 모델 (성능) 업데이트 하기. 

앙상블에 계속 추가하는 개별 약 분류기로, 의사결정회귀나무 사용한다. 

## 회귀문제 해결하기 

```python 
from sklearn.ensemble import GradientBoostingRegressor 

# 그레디언트 부스팅 회귀모델 정의 
regressor = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01, loss='ls')

# 회귀모델 훈련 
regressor.fit(x_train, y_train)
```

GradientBoostingRegressor(learning_rate=0.01, loss='ls', max_depth=4,
                          n_estimators=500)

```python 
# 테스트셋에 대해 예측 
y_pred = regressor.predict(x_test)

# 회귀모델 성능 
print(sqrt(mean_squared_error(y_test, y_pred)))
print('단일 의사결정 회귀나무 보다, 의사결정회귀나무 앙상블이 성능 더 좋았다 (1.4 > 0.265)')
```
RMSE value: 0.265994740897516

단일 의사결정 회귀나무 보다, 의사결정회귀나무 앙상블이 성능 더 좋았다 (1.4 > 0.265)

아래는 선형회귀모델, 단일 의사결정회귀나무, 의사결정회귀나무 앙상블(그래디언트 부스팅) 성능 비교 한 것이다. 

```python 
result = pd.DataFrame({
    'Algorithm' : ['Linear regression', 'Regression tree', 'Gradient Boosting'], 
    'RMSE' : [2.7404298600226993e-15, 1.4008722335107024, 0.265994740897516] 
})
result
```
<img width="239" alt="Screen Shot 2022-08-13 at 17 31 53" src="https://user-images.githubusercontent.com/83487073/184475915-82a75b09-0b1b-45a6-a8cb-97cebed84995.png">

데이터프레임 시각화 

```python 
plt.bar(range(len(result['RMSE'].values )), result['RMSE'].values )
plt.xticks(range(len(result['RMSE'].values)), result['Algorithm'].values)
plt.title('RMSE result')
plt.show() 
```
<img width="1109" alt="Screen Shot 2022-08-13 at 17 32 57" src="https://user-images.githubusercontent.com/83487073/184475962-82a00f91-8600-48d8-9cf1-01ee5c645beb.png">

이 경우엔 선형회귀모델, 그레디언트 부스팅, 의사결정 회귀나무 순으로 성능 잘 나왔다.







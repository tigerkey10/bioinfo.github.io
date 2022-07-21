---
title : "[논문 읽기] Distilling the Knowledge in a Neural Network"
excerpt : "관심있는 딥러닝 분야 논문 읽고 정리하기"

categories : 
- ML
- Deep learning

tags : 
- [ML, Deep learning]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-04-25
last_modified_at : 2022-04-25

---

# Distilling the Knowledge in a Neural Network 

Authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean

https://arxiv.org/abs/1503.02531

---

# Knowledge Distillation 지식 증류

## 정의: Teacher 모델에서 정보를 '증류'해서, Student 모델에 '전수' 시키는 작업

덩치 큰 딥러닝 네트워크(히든 층 갯수가 많다거나, 히든유닛 수가 많다거나...)는 분류 정확도는 높을 수 있지만, 알고리듬이 '무거워서' 일반 사용자 대상으로 배포하기에는 적합하지 않다.

99% 정확도 자랑하지만 분류 수행에 3시간 걸리는 모델 vs 90% 정확도지만 분류하는 데 3분이면 되는 모델 

$\Rightarrow$ 오른쪽이 일반 사용자에겐 더 적합할 수 있다. 기업 관점에서 생각해도, 왼쪽 모델은 KPI 충족하지 못할 수 있다. 

$\Rightarrow$ 배포에 적합하도록 알고리듬 경량화 하되, 성능은 최대한 원래 무거운 모델처럼 확보하기 위한 방법으로 Knowledge Distillation 이 등장했다. 

논문에 따르면, 실제로 

### 큰 모델 훈련 후 지식 증류 해서 작은 모델 훈련시키기 $>$ 큰 모델 훈련시킨 데이터로 작은 모델 훈련시키기 

왼쪽이 더 효과적인 모델 훈련법이다. 

---

# Knowledge Distillation 수행 

## Soft target: Teacher 모델 소프트맥스 함수에 온도 항($T$) 추가해서 식 변형 후, $T$ 값 증가시키면서 얻어낸, $T=1$ 일 때 보다 엔트로피 커진 결과값 분포

$Adjusted$ $softmax$: $q_{i} = \frac{\exp{(z_{i}/T)}}{\sum_{j}\exp{(z_{j}/T)}}$

- T를 온도(Temperature) 라고 한다. 
- T 증가할 수록 소프트맥스 출력 분포 엔트로피는 커진다. 
- T 기본값은 $1$ 이다. 

거대 모델 증류한 온도 T 값을 small model 훈련할 때도 똑같이 소프트맥스 함수에 적용한다. 

small model은 위 상태에서, soft target을 잘 맞추도록 훈련된다. 

## $\Rightarrow$ 지식 증류 과정 정리 

1. 거대 모델을 Original dataset 으로 훈련시킬 때, 소프트맥스 층에 온도 항 $T$ 추가한다. 온도 계속 높여서 출력 분포를 보다 soft 하게 만든다. 
2. 이렇게 얻어낸 soft target을 small model 훈련 타겟으로 설정한다. 
3. small model 을 transfer set이나 original dataset으로 훈련시킨다. 
4. 매 훈련 case 마다, soft target을 target으로 설정하고, small 모델 softmax 층 출력과 비교한다. 이때, small 모델 에도 거대 모델과 같은 T 값 적용한다. 
5. small 모델 소프트맥스 출력과 soft target 사이 크로스 엔트로피, 그러니까 훈련과정 전체의 로그손실을 목적함수로 삼아 모델 최적화 하고, 훈련 다 끝나면 T값은 1로 되돌린다. 

---

# Small 모델 성능 더 향상시키는 방법

$\Rightarrow$ 만약 transfer set 정답 값 알 경우, 목적함수 바꿔서 small 모델 성능 더 향상시킬 수 있다. 

- small 모델 소프트맥스 출력 결과와 soft target 사이 로그 손실 
- small 모델 소프트맥스 출력 결과와(온도 항 적용 안함) transfer set 정답 값 사이 로그 손실 

위 두 로그 손실 가중평균으로 새 목적함수 찾고, 최적화 한다. 

논문에 따르면, 최적 결과는 두번째 로그 손실에 첫번째 보다 훨씬 낮은 가중치 줬을 때 얻을 수 있었다. 

---

# MNIST 숫자 이미지 데이터로 예비 실험 수행 결과 

## 지식 증류 & 전수 전

거대 모델
- 2개 은닉 층, 총 1,200개 히든 유닛.으로 구성했다. 총 60,000개 데이터로 구성된 훈련 셋으로 훈련시켰다. 
- 드롭아웃 층 추가하는 등, 과대적합 방지하기 위해 강력히 규제했다. 
- 이미지 증식 사용, 훈련 데이터 부풀림으로써 모델 학습이 충분히 이루어질 수 있도록 했다. 

$\Rightarrow$ 거대 모델은 테스트 셋에서 전체 중 67개 분류 오류만 기록했다. 

small model
- 2개 은닉 층, 총 800개 히든유닛, 규제 없음. 

$\Rightarrow$ 지식 증류해서 전수하기 전, 테스트셋에서 총 146개 틀렸다. 

## 지식 증류 & 전수 후 

- 거대 모델 증류해서 얻어낸 soft target 으로 small model 훈련시켰을 때: 테스트셋에서 74개만 틀렸다. 오답 갯수 절반수준으로 감소했다!
- small model 각 층 히든 유닛 수를 300개 정도로 줄였을 때, 온도가 8 이상이기만 하면 히든유닛 총 800개 일 때랑 꽤 비슷한 결과 나왔다. 
- 만약 히든유닛 수를 극단적으로 줄일 경우(1층 당 30개 히든유닛), T 값을 $2.5~4$ 범위로 유지하는 게 $2.5$ 미만 또는 $4$ 초과보다 더 나은 분류결과 가져왔다. 

## 한편 

$\Rightarrow$ 지식 증류 & 전수 과정은 그대로 유지하면서, small model 훈련시키는 transfer set 에서 3을 아예 빼고 훈련시켜봤다. 

따라서 모델은 한번도 3을 본적 없다. 

그럼에도 모델은 3이 1010개 포함된 test set에서 단지 206개만 틀렸다! 이 중 133개만 3을 틀린 것이다. 

### $\Rightarrow$ 거대 모델이 갖고 있는 지식이 small model 로 성공적 전수 되었다고 볼 수 있다. 

심지어 transfer set에 오직 7과 8만 넣어서 small model 훈련했을 때도 오답률은 $47.3%$ 로, 절반을 넘기지 않았다!

---
























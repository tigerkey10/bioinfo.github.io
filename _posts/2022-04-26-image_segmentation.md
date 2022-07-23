---
title : "[딥러닝 연구주제 탐색] Image segmentation"
excerpt : "관심있는 딥러닝-컴퓨터 비전 분야 탐색"

categories : 
- Research Topic
- Machine Learning
- Deep Learning

tags : 
- [Research Topic, Machine Learning, Deep Learning]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-04-26
last_modified_at : 2022-04-26

---

*아래는 관심있는 딥러닝-컴퓨터비전 분야인 Image segmentation 에 대해 알아보고, 내용을 간략히 기록한 글이다. 완성된 문서가 아니며, 계속 add-up 해 나가기 위해 작성했다. 

---

# Image segmentation

## 정의 

이미지 구성하는 단일 픽셀 하나하나를 특정 클래스로 분류하는 작업.

## 종류 

### 1. Semantic segmentation

단일 픽셀 하나하나를 특정 클래스로 분류한다. 대신, 동일한 객체 끼리는 구분 못한다. (이미지에 서로 다른 고양이 2마리 있으면 둘 다 똑같은 고양이로 분류한다.)

### 2. Instance segmentation

Semantic segmentation 처럼 단일 픽셀 하나하나를 특정 클래스로 분류하되, 이미지에 나타난 모든 객체를 구분해서 분류한다. (고양이 1, 고양이 2, 고양이 3)

# Image segmentation 적용 가능 분야 

- Handwriting Recognition: 손글씨 인식 할 수 있다. 
- Portrait mode photo: 인물모드 촬영 
- YouTube stories: 유튜브 스토리 촬영 중 배경 마음대로 바꿀 수 있음
- Virtual make-up
- Virtual try-on: 이미지에 가상으로 옷 착용 해볼 수 있다. 
- Visual Image Search: 사용자가 업로드한 이미지에서 특정 객체만 뽑아내서, 비슷한 객체 가진 이미지들 웹에서 찾아준다. 
- Self-driving cars: 자율주행 기술은 주변 사물에 대해 단일 픽셀 레벨에서 완벽한 이해 필요하다. Image segmentation 이 이미지(프레임)에서 각 객체 완벽하게 구분.분할 해 내기 위해 사용된다. 

https://nanonets.com/blog/semantic-image-segmentation-2020/

---

예전에는 SVM, Random Forest, K-means Clustering 등이 image segmentation 하기 위해 사용됬다. 

하지만 이제 딥러닝이 그 자리 완전히 대체했다. 

# Fully Convolutional Network 

## 정의

완전연결 분류기 없이 전체가 합성곱 층으로 구성된 네트워크.

- 1*1 합성곱 층 역할 = 완전연결 분류기 역할 로 봤다. 

네트워크가 input down sampling과 up sampling 부분으로 이루어져 있다. 

Down sampling 파트를 인코더,

Up sampling 파트를 디코더 라고 한다. 

인코더는 input 받아서 size 줄이고, 

디코더는 인코더 output 받아서 size 늘린다. 

참고: https://nanonets.com/blog/semantic-image-segmentation-2020/



# Up sampling

## 정의 

저해상도 출력 결과에 픽셀 추가해서 고해상도 출력으로 변환하는 과정. 

- 모델에서 Up sampling 파트를 디코더 라고도 부른다. (*Down sampling 파트는 인코더 라고 부른다.)


## 방법

- Bilinear interpolation: 2차원 이미지에 픽셀 (추정해서) 삽입
- Backwards Convolution: 원래 down sampling 하는 convolution 과정을 역으로 수행하는 방법. 
FCN 에서는 두 가지 함께 사용한다.

# FCN이 Image segmentation 수행하는 과정 

1. FCN은 사전학습된 이미지 분류 모델(예:VGG16)에서 완전연결 분류기를 떼고, 1*1 합성곱 층을 넣어서 사용한다. 이는 벡터 꼴 dense 층 출력이 픽셀 위치 정보를 잃어버리기 때문이다. 
2. 1*1 합성곱 층 출력 결과는 rough 한 heatmap 이다. 픽셀 별 위치 정보는 담고 있지만, segmentation 하기엔 이미지가 coarse 하다. 따라서 이 출력결과에 픽셀 추가할 필요 있다(해상도 높여야 한다). FCN은 픽셀 추정해서 삽입하는 Bilinear Interpolation 과 backwards convolution 을 함께 사용한다. 이 과정을 up sampling 이라 하고, 디코더 라고도 부른다. 
3. 디코딩 결과도 segmentation에는 부적합하다. 여전히 segmentation 결과가 rough 하기 때문이다. 이를 보완하기 위해 Skip architecture 를 사용한다. Skip architecture는 곡선, 직선, 엣지 등 local 정보를 담고 있는 신경망 얕은 층 정보와, 추상화. 의미단위 정보 담고 있는 깊은 층 정보를 결합해서 출력하는 신경망 구조를 말한다. 
4. Local 정보와 global 정보 결합해서 출력하는 Skip architecture는 보다 정교한 segmentation 결과 출력한다. 

참고: https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204







---

# Semantic Image Segmentation with DeepLab in TensorFlow

### Cutting-edge semantic image segmentation model: DeepLab-v3+

(2018 기준)

- 텐서플로 기반 모델이다. 
- CNN backbone 모델 위에 DeepLab-v3+ 얹은 형태다. 
- 구글은 PascalVOC 2012 & Cityscapes 데이터에 대해 사전훈련된 모델 제공한다.  
- 방법론, 하드웨어, 데이터셋 발전으로 최근 semantic image segmentation system 들 성능 비약적 향상 되었다. 

‘딥 라벨링’은 신경망 사용해 각 픽셀마다 분류 예측값을 할당하는 방식으로 동작한다. 

https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html




---
title : "[네이버블로그이관/데이터분석/PCA] 주성분분석 복습 및 기록 -1"
excerpt : "04.27.2021. 주성분분석 학습 및 복습 내용 옮겨와 기록"

categories : 
- Data Science
- python
- mathematics

tags : 
- [datascience, python, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-04-27
last_modified_at : 2021-08-31

---

---

글 원본 : https://blog.naver.com/tigerkey10/222324768520

이 글은 내가 지킬 블로그를 만들기 전, 

네이버 블로그에 공부 내용을 기록할 때 PCA 주성분분석을 처음 학습하고 복습하면서 기록했던 첫번째 글을 옮겨온 것이다. 

네이버 블로그에서 글을 옮겨와 지킬 블로그에도 기록해둔다. 

모든 공부 과정을 기록으로 남겨 치열했던 시간들을 기록으로 남기고, 

미래에 전달해 스스로 동기부여할 것이다. 또한 반복해서 학습. 복습하는 과정에서 

어떤 개념에 대한 내 이해가 어떻게 변화.발전 했는지 관찰함으로써 스스로의 학습 정도를 돌아보는 거울로 삼고자 한다. 

---

아래 글은 '김도형의 데이터사이언스 스쿨' 을 통해 PCA를 공부하고, 이해한 내용을 기록해두고 나중에 다시 보기 위해 내 언어로 정리한 글이다. 

---
# PCA 

- PCA는 [고차원 데이터 차원축소 시켜] 주성분을 찾는 분석이다

- 주성분 : 투영벡터 만들어주는 공간의 기저벡터들

- pca1.components_ 결과가 주성분들이다

- 주성분의 의미 : 데이터 간 차이, 변이 만들어내는 주요 원인

```python
from sklearn.datasets import load_iris
X = load_iris()
X = X.data
iris = X[:10, :2]


pca2 = PCA(n_components=1)
X_low = pca2.fit_transform(iris)
X_low 
```
<img width="601" alt="Screen Shot 2021-08-31 at 9 54 42" src="https://user-images.githubusercontent.com/83487073/131424370-bf586e91-ae24-4a0d-874a-5bde627d5602.png">

- 위 X_low 각 값은 주성분이 저차원 근사데이터에서 차지하는 비중이다. (주성분이 얼마나 들었나 보여준다)

- '비중' 이기 때문에 저차원 데이터 구성할 때 주성분의 가중치로 쓰인다.(평균 + 가중치*주성분)

- 주성분 값 이라고도 한다(=잠재변수, 기저에서 각 측정치 결정짓는 데이터)

- 주성분 * 측정치 = 주성분 값(잠재변수)

- 차원축소한 저차원데이터 각 원소는 '주성분 값'으로 이뤄져 있다(평균을 베이스로, 주성분만으로 구성되어 있다는 뜻)

```python
pca2.inverse_transform(X_low)
```
<img width="697" alt="Screen Shot 2021-08-31 at 9 55 37" src="https://user-images.githubusercontent.com/83487073/131424448-37e18387-a3dc-40e4-ba3f-f1c69abce094.png">

- 위 '근삿값'들은 평균과 주성분만으로 이루어진 근삿값들이다.

- (평균 +) 주성분1비중 * 주성분1 이런 식이다

- 바로 위에 X_low랑 실질적으로 같은 데이터다.

```python
from sklearn.datasets import fetch_olivetti_faces
faces_all = fetch_olivetti_faces()
K = 20 
faces = faces_all.data[faces_all.target == K]

from sklearn.decomposition import PCA
pca3 = PCA(n_components=2) #주성분 2개로 쪼개겠다(분석하겠다)
W3 = pca3.fit_transform(faces)
X32 = pca3.inverse_transform(W3)
```
<img width="884" alt="Screen Shot 2021-08-31 at 9 57 07" src="https://user-images.githubusercontent.com/83487073/131424578-53a88673-324d-4c91-b2c2-9efe9d6e0ca2.png">

- 2차원 근사된 투영벡터들이다. 

- 원래 데이터와 가장 비슷한 벡터들이다.

- 차원축소시켜 주성분들로만 구성한 형태다.

- [-2.21367 , 4.2135177] 예컨대 이 첫번째 행벡터에서 각 성분은 주성분 값(=주성분 비중)들이다. 1번과 2번 주성분이 각각 얼마나 들었는가를 나타낸다

<img width="892" alt="Screen Shot 2021-08-31 at 9 57 49" src="https://user-images.githubusercontent.com/83487073/131424621-6afb5141-76d6-4508-96c0-483b9d8d089d.png">

- 위 데이터들은 그 위에 2차원 축소된 데이터들이랑 같은거다. 

- 주성분으로만 이루어진 데이터다(평균은 베이스로 두고)

- 구성 : (평균+) <주성분1 비중 * 주성분 1> + <주성분2 비중 * 주성분 2 >

- 각 주성분 1, 2가 나타내는 얼굴 이미지를 '아이겐페이스'라고 한다

# 고차원 데이터 차원축소 시키는 이유는? 

- '압축'해서 주성분만으로 구성하기 위해서이다.

- 그 후 압축된 저차원 데이터 구성하는 주성분 찾아내기 위해서이다. 

## -->  PCA 목표는 '주성분 찾기'

---





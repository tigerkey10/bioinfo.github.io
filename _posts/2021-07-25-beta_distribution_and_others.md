---
title : "[수학/확률과 통계] 베타, 감마, 디리클레분포"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python
- mathematics

tags : 
- [mathematics, study, data science]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-07-26
last_modified_at : 2021-07-26

---


# 베타, 감마, 디리클레분포 : 분포 형상을 마음대로 조작할 수 있는 분포
- 분포 모숫값 조작이 가능하다. 
- 베이지안 추정에 사용한다.

---

# 1. 베타분포 : 베르누이분포 모수 $\mu$ 의 신뢰도 분포 
- 표본공간 : 0과 1사이 모든 실수 (0 < $x$ < 1)
- 모수 : a, b
- 특징 : 모숫값을 마음대로 조작해서, 분포 형상 조작이 가능하다. 
- 사용 : 베르누이분포 모수 $\mu$ 의 베이지안 추정에 사용한다. 

```python
xx = np.linspace(0,1,1000)
plt.subplot(221)
plt.fill_between(xx, sp.stats.beta(1.001, 1.001).pdf(xx))
plt.ylim(0, 6)
plt.title('(A) a = 1, b = 1')

plt.subplot(222)
plt.fill_between(xx, sp.stats.beta(4,2).pdf(xx))
plt.ylim(0, 6)
plt.title('(B) a = 4, b = 2, 최빈값 = {}'.format(3/(4)))

plt.subplot(223)
plt.fill_between(xx, sp.stats.beta(8, 4).pdf(xx))
plt.ylim(0, 6)
plt.title('(C) a = 8, b = 4, 최빈값 = {}'.format(7/(10)))

plt.subplot(224)
plt.fill_between(xx, sp.stats.beta(30,12).pdf(xx))
plt.ylim(0, 6)
plt.title('(D) a = 30, b = 12, 최빈값 = {}'.format(29/(40)))

plt.suptitle('모수 a,b 값에 따른 베타분포 형상 변화')
plt.tight_layout()
plt.show()

```
<img width="931" alt="Screen Shot 2021-07-26 at 10 41 31" src="https://user-images.githubusercontent.com/83487073/126922020-7433ad99-4e13-469d-a331-b21900803009.png">

## 베타분포 베이지안 추정
- (A) : 베르누이 분포 모수 $\mu$ 값 추정 불가 
- (B) : 베르누이 분포 모수 $\mu$ 값 0.75일 가능성 가장 높다 (정확도 낮음 = 분산이 크다)
- (C) : 베르누이 분포 모수 $\mu$ 값 0.7일 가능성 가장 높다 (정확도 중간)
- (D) : 베르누이 분포 모수 $\mu$ 값 0.725일 가능성 가장 높다 (정확도 높다 = 분산이 작다)

## 베타분포 모멘트 
- $E[X] = \frac {a}{(a+b)} $ 
- $mode = \frac {a-1}{a+b-2}$
- $V[X] = \frac {ab}{(a+b)^{2}(a+b+1)}$

$a,b$ 모숫값이 커질 수록 분산이 작아진다. 결과적으로 추정 정확도가 증가한다. 

---

# 감마분포 : 0과 $+\infty$ 사이 모수 $\mu$의 신뢰도 분포  
- 표본공간 : $(0, +\infty)$ 사이 모든 양수
- 모수 : a,b 

Scipy gamma 클래스에는 초깃값 b=1로 고정되어 있다.

- 특징 : 베타분포와 개념상 다 같은데, 표본공간만 다르다. 

모수 a가 작아질 수록, 추정 정확도가 증가한다 (분산이 작아진다). (단, b=1 고정일 때)

- 사용 : 베이지안 추정 (모수 $\mu$ 신뢰도 분포 나타내는 데 쓴다)

```python
xx = np.linspace(0,16,100)
plt.subplot(221)
plt.fill_between(xx, sp.stats.gamma(9).pdf(xx))
plt.ylim(0, 0.4)
plt.title('(A) a=9, b=1, 최빈값={}'.format(8))

plt.subplot(222)
plt.fill_between(xx, sp.stats.gamma(6).pdf(xx))
plt.ylim(0, 0.4)
plt.title('(B) a=6, b=1, 최빈값={}'.format(5))

plt.subplot(223)
plt.fill_between(xx, sp.stats.gamma(3).pdf(xx))
plt.ylim(0, 0.4)
plt.title('(C) a=3, b=1, 최빈값={}'.format(2))

plt.subplot(224)
plt.fill_between(xx, sp.stats.gamma(2).pdf(xx))
plt.ylim(0, 0.4)
plt.title('(D) a=2, b=1, 최빈값={}'.format(1))

plt.suptitle('a,b=1 모숫값 변화에 따른 감마분포 형상 변화')
plt.tight_layout()
plt.show()
```

<img width="625" alt="Screen Shot 2021-07-26 at 11 10 13" src="https://user-images.githubusercontent.com/83487073/126923783-96a25002-cc5f-4e4b-922e-eb4f4e74710c.png">

## 감마분포 베이지안 추정 
- (A) : 임의의 모수 $\mu$ 가 8일 가능성이 가장 높다(정확도 매우 낮다)
- (B) : 임의의 모수 $\mu$ 가 5일 가능성이 가장 높다 (정확도 낮다)
- (C) : 임의의 모수 $\mu$ 가 2일 가능성이 가장 높다 (정확도 높다)
- (D) : 임의의 모수 $\mu$ 가 1일 가능성이 가장 높다 (정확도 매우 높다)

## 감마분포 모멘트
- $E[X] = \frac {a}{b}$
- $mode = \frac {a-1}{b}$
- $V[X] = \frac {a}{b^{2}}$

---

# 디리클레분포 : 0과 1사이 값만 갖는 $K$ 차원 벡터들의 신뢰도 분포 
## $Dir(x;\alpha)$

- 표본공간 : $x$ 는 $K$ 차원 확률변수 벡터다. 

확률변수 벡터 $x$ 의 각 원소 $x_{i}$는 조건이 있다. 

1.  $0 <= x <= 1$
2. $\sum_{1}^{K}x_{i} = 1$

$K=2$ 일 때 디리클레분포는 베타분포가 된다

- 모수 : $\alpha$ 는 $K$ 차원 모수벡터다. 

- 특징 : 모수벡터 $\alpha$ 를 조정해서 디리클레분포 형상을 조작할 수 있다.

- 사용 : 베이지안 추정에 사용한다. 
예) 카테고리분포 모수벡터 $\mu$ 의 신뢰도 분포




---

## 디리클레분포의 모멘트 
- $E[x_{k}] = \frac {\alpha_{k}}{\sum \alpha}$
- $mode = \frac {\alpha_{k}-1}{(\sum \alpha)-K}$
- $V[x_{k}] = \frac {\alpha_{k}((\sum\alpha)-\alpha_{k})}{(\sum\alpha)^{2}((\sum\alpha)+1)}$

모수벡터 $[\alpha_{1}, \alpha_{2}, ... \alpha_{k}]$ 원소 하나의 절댓값이 커지면 $x_{k}$ 는 기댓값 근처의 비슷비슷한 값이 나온다. 

모수벡터 $[\alpha_{1}, \alpha_{2}, ... \alpha_{k}]$ 의 원소 하나하나 전반적인 절댓값이 커지면 결과적으로 표본벡터 $[x_{1}, x_{2}, ... x_{k}]$ 가 특정 값 근처에 몰리게 된다. 

---

## 디리클레분포의 응용

#### 문) "x,y,z가 양의 난수일 때, 항상 x+y+z=1 이 되게 하려면 어떻게 해야 하는가? 모든 경우는 균등하게 나와야 한다"

- 디리클레분포 모수벡터 원소들 $\alpha_{1}=\alpha_{2}=\alpha_{3}$ 이면 디리클레분포는 균등분포가 된다.(주어진 문제 조건을 모두 충족한다)
- 하지만 반대로, 같은 균등분포에서 합이 1되는 표본 3개 뽑아서 구성한 벡터들의 분포는 균등하지 못하다. 
- $\alpha_{1}=\alpha_{2}=\alpha_{3}$ 이면 x,y,z가 모두 같은 분포 따른다는 의미지만, 그 반대는 안 된다는 얘기다. 

---

### 1. 균등분포에서 합 1되는 3개 표본 추출, 3개 표본이 이루는 벡터들의 분포

<img width="484" alt="Screen Shot 2021-07-26 at 14 51 02" src="https://user-images.githubusercontent.com/83487073/126939396-00055f62-410a-4259-a69a-57530b055be3.png">

- 데이터들이 균일하게 분포하지 못하고, 중앙에 몰려 있다. 

### 2. $\alpha_{1}=\alpha_{2}=\alpha_{3}$ 인 디리클레분포

$\alpha_{1} = \alpha_{2} = \alpha_{3}$ 이고 $\alpha_{i}$ 가 모두 1일 때, 디리클레분포는 균등분포를 이루었다. 

<img width="477" alt="Screen Shot 2021-07-26 at 14 53 18" src="https://user-images.githubusercontent.com/83487073/126939603-9626f018-cbaf-40c1-837a-0b7399f822a7.png">

### 한편
1. 디리클레분포 $\alpha_{i}$가 모두 같아도 $\alpha_{i}$ 절댓값이 크면, 디리클레분포 분산이 작아져서 특정 점 근처에 표본 분포가 몰리게 된다. 

```python
x2 = sp.stats.dirichlet([50]*3) # 모수벡터 원소가 모두 같은 디리클레분포 객체 
rs = x2.rvs(10000) # k는 모수벡터 차원 따라 알아서 지정된다. 

plot_triangle(rs, kind='scatter')
```

<img width="489" alt="Screen Shot 2021-07-26 at 14 57 48" src="https://user-images.githubusercontent.com/83487073/126939982-12043c29-8e9a-4c8c-8308-5dcf8526cbf0.png">

2. $\alpha_{i}$ 가 모두 같아도, $\alpha_{i}$ 절댓값이 작으면 디리클레분포 분산이 작아져서 표본분포가 삼각형 세 꼭짓점 $(1,0,0), (0,1,0), (0,0,1)$ 에 몰리게 된다. 

```python
x2 = sp.stats.dirichlet([0.001]*3) # 모수벡터 원소가 모두 같은 디리클레분포 객체 
rs = x2.rvs(10000) # k는 모수벡터 차원 따라 알아서 지정된다. 

plot_triangle(rs, kind='scatter')
```

<img width="488" alt="Screen Shot 2021-07-26 at 14 59 53" src="https://user-images.githubusercontent.com/83487073/126940192-ac00e327-e848-4b5e-bddb-3f6a081644a8.png">

---

# 디리클레분포를 이용한 베이지안 추정

## 핵심 아이디어
- 모수벡터 $\alpha$ 를 조정해서 디리클레분포 분포 형상을 조작할 수 있다. 
- $K$차원 표본 벡터들의 신뢰도 분포를 나타낼 수 있다. 
- 신뢰도(가능성) 이 가장 높은 값이 '모수추정값'이 된다. 
- 예) 카테고리분포의 모수벡터 $\mu$ 의 신뢰도 분포를 나타낼 수 있다. 


# if $\alpha = [1,1,1]$
- 앞에 2차원 삼각형으로 나타냈던 것처럼, 표본 분포가 균등분포가 된다. 
- 삼각형 전체 면적에 데이터들이 골고루 퍼진다. 
- 따라서 어떤 벡터값이 신뢰도가 높은지 알 수 없고, 모수추정이 불가능하다. 

<img width="493" alt="Screen Shot 2021-07-26 at 16 01 30" src="https://user-images.githubusercontent.com/83487073/126946464-05b228bf-1fd2-478f-8d43-186933c32bf6.png">

# if $\alpha = [2,3,4]$
<img width="488" alt="Screen Shot 2021-07-26 at 16 02 29" src="https://user-images.githubusercontent.com/83487073/126946563-81e42479-e029-4997-870a-a0c1474821b1.png">

- 검정색 부분(신뢰도 가장 높은 지점 값)이 모수추정값이 될 수 있다. 
- 하지만 전체 분포의 분산 정도가 커서, 추정 정확도는 '낮다'

# if $\alpha = [20,30,40]$
<img width="484" alt="Screen Shot 2021-07-26 at 16 05 39" src="https://user-images.githubusercontent.com/83487073/126946935-93c977c4-1734-448c-a4ff-9cf7c63eb43d.png">

- $\alpha = [2,3,4]$ 일 때와 같은 지점이지만 분포의 분산이 훨씬 작다. 

분산이 작아진 이유는 모수벡터 $\alpha$ 의 절댓값이 커졌기 때문이다. 

- 이 경우 $\alpha = [2,3,4]$ 일 때와 같은 값을 '모수추정값'으로 제시할 수 있지만, 추정 정확도는 더 높다.

---







---
title : "[수학/확률과 통계] 추정_기본개념, 모수추정- 모멘트방법, 최대가능도추정법"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python
- mathematics

tags : 
- [mathematics, datascience, study]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-08-04
last_modified_at : 2021-08-05

---

# 추정과 검정

# 추정 : 내가 들고 있는 데이터가 나온 원래 확률분포(확률변수)를 찾는 작업

# 검정 : 추정결과가 믿을만 한가 아닌가 알아보는 작업
- 추정결과 신뢰도 올리는 방법 : 더 많은 데이터를 모으면 된다. 

---

# 확률분포의 추정
- 데이터분석 기본 가정 : 분석할 데이터는 가상의 확률변수에서 떨어져 현실세계에 실현된 '표본', '일부분', '파편'이다. 

- 데이터분석 최종목표 : 데이터가 떨어져 나온 원래 확률분포(확률변수)를 찾아내는 것. 

# 확률분포 추정 과정
# 1. 확률분포의 종류 알아내기 

### 핵심은 '추측'

- 데이터 기본특성으로 '추측'
예) 데이터가 0과 1 둘만 나온다 --> 베르누이분포 아닐까?

- 히스토그램 그려서 데이터 분포 모양 가지고 확률분포 '추측'
예) 데이터 분포 모양이 정규분포와 비슷하다. --> 정규분포 아닐까?

- 보스턴 집값 데이터로 확률분포 종류 추측해보기 

```python
from sklearn.datasets import load_boston
data = load_boston().data
df = pd.DataFrame(data, columns=load_boston().feature_names)
df['MEDV'] = load_boston().target
df
```

<img width="776" alt="Screen Shot 2021-08-04 at 15 47 12" src="https://user-images.githubusercontent.com/83487073/128134642-0312ed5a-7a16-4c02-a515-637f7b78dcba.png">


예) 
```python
plt.subplot(211)
sns.distplot(df['RM'], kde=False, bins=60)
plt.title('정규분포일 것이다')
```
<img width="750" alt="Screen Shot 2021-08-04 at 16 58 11" src="https://user-images.githubusercontent.com/83487073/128144079-b17b54f8-630a-465d-a6ba-354a302dc58e.png">

예) 
```python
plt.subplot(211)
sns.distplot(df['DIS'], kde=False, bins=60)
plt.title('0보다 큰 어떤 값이 자주 발생하는 카이제곱분포?')

plt.subplot(212)
sns.distplot(np.log(df['DIS']), kde=False, bins=50)
plt.title('로그정규분포?')

plt.tight_layout()
```
<img width="750" alt="Screen Shot 2021-08-04 at 16 59 05" src="https://user-images.githubusercontent.com/83487073/128144242-b451a287-d489-45ee-b62b-4cd7017e35d7.png">

예) 
```python
plt.subplot(211)
sns.distplot(df['AGE'], kde=False, bins=60)
plt.subplot(212)
sns.distplot(df['AGE']/100, kde=False, bins=60)
plt.title('베타분포로 바꿀 수 있지 않을까?')
plt.tight_layout()
```
<img width="748" alt="Screen Shot 2021-08-04 at 17 00 36" src="https://user-images.githubusercontent.com/83487073/128144463-2c9d8026-1175-46ea-99c7-d37908a4c491.png">


---

# 2. 확률분포의 모수 알아내기
- 모멘트방법, 최대가능도추정법, 베이즈추정법 이용

## 1) 모멘트 방법
- 전제 : 표본분포 모멘트와 이론적 확률분포 모멘트가 같다고 본다. 

큰수의 법칙에 따라 표본평균은 모평균 근삿값이다. 

비편향표본분산은 모분산의 근삿값이다. 


- 표본분포 모멘트 구하고, 이론적 분포 모멘트 구하고, 이론적 분포 모멘트 사용해서 모수 구한다. 

예)

1. 모수와 이론적 모멘트가 같은 경우 : 정규분포의 예

$\bar{x} = \mu $

$\bar{s} = \sigma^{2}$

정규분포 모수는 $\mu, \sigma^{2}$ 이다. 따라서 표본 모멘트 구하면 바로 정규분포 모수 구할 수 있다. 

2. 모수와 이론적 모멘트 다른 경우 : 베타분포 모수 추정

$E[X] = \frac{a}{a+b} = \bar{x}$

$E[(X-\mu)^{2}] = \frac{ab}{(a+b)^{2}(a+b+1)} = s^{2}$

베타분포는 정규분포와 달리 모수와 모멘트값이 다르다. 따라서 이론적 모멘트-모수 사이 관계를 이용해서 베타분포 모수 a,b를 찾아야 한다. 

---

## 2) 최대가능도 추정법

### 핵심 아이디어 : 
- 전제 : 실현된 표본값은 매우 잘 나오는(흔한) 값이다. 
- '주어진 데이터를 가장 높은 확률로 뱉어내는 분포(모수) 찾기'

---
예) 주어진 데이터가 x=1 일 때 

모수 $\mu$ 후보군 : -1,0,1

-1,0,1 세 $\mu$ 값 중 주어진 데이터의 확률밀도를 가장 크게 만드는 모수 $\mu$ 값은?

```python
# 최대가능도추정법 원리 
L = [sp.stats.norm(loc=mu).pdf(1) for mu in [-1,0,1]]
plt.scatter(1, L[0])
plt.scatter(1, L[1])
plt.scatter(1, L[2])
plt.vlines(1.00, ymin=0, ymax=0.5, ls=':')
plt.title('데이터 $x=1$ 일 때 데이터의 $\mu$값 별 확률밀도')

black = {'facecolor' : 'black'}
xx = np.linspace(-5,5,1000)
plt.plot(xx, sp.stats.norm(loc=-1, scale=1).pdf(xx), c='r', label='$\mu=-1$')
plt.plot(xx, sp.stats.norm(loc=0, scale=1).pdf(xx), c='b', ls=':' , label='$\mu=0$')
plt.plot(xx, sp.stats.norm(loc=1, scale=1).pdf(xx), c='g', ls='-.', label='$\mu=1$')
plt.text(0, 0.45, '최대 가능도값(x=1의 확률밀도)')
plt.annotate('', xy=[1, 0.4], xytext=[2,0.43], arrowprops=black)
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(0.5, -0.04, '$x=1$')
plt.scatter(1,0,30, marker='^')
plt.legend()
plt.show()
```

<img width="734" alt="Screen Shot 2021-08-05 at 15 04 28" src="https://user-images.githubusercontent.com/83487073/128299522-c89fdb4d-d9db-420b-a085-4310c09175f5.png">

모수가 $\mu=1$ 일 때 내가 들고 있는 데이터 x=1의 확률밀도가 가장 커지는 것을 알 수 있다. 

이처럼 x=1의 확률밀도(가능도)를 최대로 만드는 모수 $\mu$ 값을 모수 추정 결과값으로 삼고, $\hat{\theta}_{MLE}$ 로 표기한다. 

= 최대가능도 추정법의 해

---



#### 가능도함수
- 가능도 : 분포 모수가 특정한 $\mu$ 값일 때 주어진 데이터(표본)에 할당되는 확률밀도값

- 가능도함수 특징 : 수식은 확률분포함수 수식과 완전히 똑같다. 

가능도함수는 확률분포함수 수식에서 모수 $\theta$ 를 변수로 바꾸고 데이터 $x$ 를 상수로 바꾼 것이다.

- 하지만 가능도함수는 확률분포함수가 아니다. 두 함수는 엄연히 다른 함수다. 

가능도함수 $\ne$ 확률분포함수

- 가능도함수 예 

1. 내가 손에 들고 있는 데이터가 0이고, 정규분포 모수 중 분산값을 아는 경우 

```python
# 데이터=0, 정규분포 분산을 아는 경우 \sigma2 = 1
def likelihood(mu) : 
    return sp.stats.norm(loc=mu).pdf(0) # 가능도 값
mus = np.linspace(-5,5,100)
likelihood_values = [likelihood(mu) for mu in mus]
plt.subplot(211)
plt.plot(mus, likelihood_values)
plt.title('가능도함수 $L(\mu, \sigma^{2} = 1 | x=0)$')
plt.xlabel('$\mu$')
```

<img width="727" alt="Screen Shot 2021-08-05 at 13 47 45" src="https://user-images.githubusercontent.com/83487073/128292424-5c072242-f460-4a03-bb7c-2420fc2e8853.png">

2. 내가 손에 들고 있는 데이터가 0이고, 정규분포 기댓값 모수 $\mu$ 를 아는 경우 

```python
# 데이터=0, 정규분포 기댓값을 아는 경우 \mu = 0
def likelihood2(sigma2) : 
    return sp.stats.norm(scale=np.sqrt(sigma2)).pdf(0)

sigma2s = np.linspace(0.1, 10,1000)
likelihood_values = [likelihood2(sigma2) for sigma2 in sigma2s]
plt.subplot(212)
plt.plot(sigma2s, likelihood_values)
plt.title('가능도함수 $L(\mu=0, \sigma^{2}|x=0)$')
plt.xlabel('$\sigma^{2}$')

plt.tight_layout()
plt.show()
```
<img width="729" alt="Screen Shot 2021-08-05 at 13 49 01" src="https://user-images.githubusercontent.com/83487073/128292527-61ce4e5a-aea4-4401-83da-2d7ab4620e02.png">


3. 만약 정규분포 모수 $\mu$ , $\sigma^{2}$ 둘 다 모르는 경우 (=둘 다 입력변수인 경우) 가능도함수

내가 들고 있는 데이터는 마찬가지로 $x = 0$ 이다. 

입력변수가 2차원 벡터다. 
함수는 2차원 벡터를 받아 스칼라를 출력하므로 2차원 다변수함수다. 

따라서 3차원 surface plot으로 가능도함수를 그릴 수 있다. 

```python
MU, SIGMA2 = np.meshgrid(mus, sigma2s)
L = np.exp(-MU**2/(2*SIGMA2))/np.sqrt(2*np.pi*SIGMA2)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title('다변수 가능도함수 $L(\mu, \sigma^{2}|x=0)$')
ax.plot_surface(MU, SIGMA2, L, linewidth=0.3)
plt.xlabel('$\mu$')
plt.ylabel('$\sigma^{2}$')
ax.view_init(10,-70)
plt.show()
```
<img width="714" alt="Screen Shot 2021-08-05 at 14 14 22" src="https://user-images.githubusercontent.com/83487073/128294629-ddf097dc-f894-431f-9a09-3a12d5f914c4.png">

---

# 복수의 표본 데이터가 있는 경우 가능도함수 

- 복수 표본데이터를 들고 있다면, 벡터꼴일 것이다. 
- 이는 벡터 하나를 얻었다고 가정할 수 있다. 
- 벡터를 내뱉는 분포는 결합확률분포다. 
- 따라서 어떤 결합확률분포로부터 벡터표본 1개를 얻었다고 생각하고, 가능도함수를 구한다. 

예) 
정규분포에서 복수 표본데이터를 얻은 경우 

표본데이터 : $[-1,0,3]$

이는 어떤 결합확률분포에서 벡터꼴 데이터를 1개 얻은 것과 같다. 

이 때 가능도함수는 다음과 같다. 

$L(\theta ; x_{1}, x_{2}, x_{3}) = p(x_{1}, x_{2}, x_{3} ; \theta)$

$x_{1}, x_{2}, x_{3}$ 은 모두 똑같은 정규분포에서 독립적으로 얻은 표본이다. 

따라서 결합확률밀도함수는 개별 확률밀도함수 곱으로 나타낼 수 있다. 

$p(x_{1}, x_{2}, x_{3}) = p(x_{1})p(x_{2})p(x_{3})$

---
참고) 결합확률과 조건부확률 사이에 성립하는 연쇄법칙을 쓰면 된다. 

$p(x_{1}, x_{2}, x_{3}) = p(x_{1} \vert x_{2}, x_{3})p(x_{2} \vert x_{3})p(x_{3})$

위 식에서 $x_{1}, x_{2}, x_{3}$ 이 모두 서로 독립이라면, 조건부확률에서 조건의 영향을 안 받는다. 

따라서 $p(x_{1} \vert x_{2}, x_{3})p(x_{2} \vert x_{3})p(x_{3})$ 은 $p(x_{1})p(x_{2})p(x_{3})$ 과 같다. 

---

결국 $L(\theta) = p(-1)p(0)p(3)$ 이 된다. 

$p$ 는 정규분포의 확률밀도함수 이므로, -1,0,3을 각각 대입해서 가능도함수 $L$ 을 구할 수 있다. 

이 $L$을 최적화 해서 최대해 $\hat{\theta}_{MLE}$ 를 찾으면 그게 곧 최대가능도추정법의 해다. 

# 로그가능도함수 
- 최적화 할 때 가능도함수에 로그 씌워서(로그가능도함수) 하면 계산이 편하다!
- 로그 씌워도 최대점, 최소점 위치가 안 바뀐다. 

---

# 최대가능도추정법 사용헤서 추정한 베르누이분포의 모수 

- $\frac{N_{1}}{N}$

표본 중 1 나온 횟수와 전체 시뮬레이션 횟수의 비율과 같다. 

# 최대가증도추정법 사용해서 추정한 카테고리분포의 모수
- $\mu_{k} = \frac{N_{k}}{N}$

최대가능도 추정법에 의한 카테고리분포의 모수는 각 카테고리값(범줏값) 나온 횟수와 전체 시행 횟수의 비율이다. 

카테고리분포의 모수 벡터 $\mu = [\mu_{1}, \mu_{2}, ... \mu_{k}]$

# 최대가능도추정법 사용해서 추정한 정규분포의 모수

- $\mu = \bar{x}$
- $\sigma^{2} = s^{2}$

# 최대가능도추정법 사용해서 추정한 다변수정규분포의 모수

- $\mu = \bar{x} 표본평균벡터$
- $\Sigma = s 표본공분산행렬$

# 결론 : 모멘트방법은 일리있었다. (=결과가 모멘트방법으로 구한 모수와 같았다)

---











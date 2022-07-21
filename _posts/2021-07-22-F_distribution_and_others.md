---
title : "[수학/확률과 통계] F분포, 다변수정규분포(조건부분포, 주변분포 포함)"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python
- mathematics

tags : 
- [mathematics, study, data science]
use_math : true

toc : true 
toc_sticky : true 

date : 2021-07-25
last_modified_at : 2021-07-25

---

# F분포 : 카이제곱분포 표본/각각의 자유도 의 비율들이 이루는 분포 

- 모수 : 자유도 쌍 (N1, N2)
- 자유도 : (N1, N2)
- 카이제곱분포 표본 = 같은 정규분포에서 얻은 표본들의 제곱합
- t통계량 제곱의 분포도 자유도가 (1,N)인 F분포 따른다. 
---

```python
# t 통계량 제곱의 분포는 F(1, N)를 따른다. 
# N = 2

t = sp.stats.t(2)
t = t.rvs(100, random_state=0)**2
plt.subplot(1,2,1)
sns.distplot(t, kde=False)
plt.xlim(0, 100)
plt.title('t통계량 제곱의 분포')

f = sp.stats.f(1,2)
xx = np.linspace(0,100,100)
plt.subplot(1,2,2)
plt.plot(xx, f.pdf(xx))
plt.title('f분포 : 자유도 = (1,2)')

plt.tight_layout()
plt.show()
```
<img width="1080" alt="Screen Shot 2021-07-22 at 11 53 18" src="https://user-images.githubusercontent.com/83487073/126583832-fc5322f9-f0a9-4f5b-8c9c-12500fb375f0.png">

```python
# 2. N=30
t = sp.stats.t(N[1])
t = t.rvs(100)**2
plt.subplot(1,2,1)
sns.distplot(t, kde=False)
plt.xlim(0, 10)
plt.title('t통계량 제곱의 분포')

f = sp.stats.f(1,N[1])
xx = np.linspace(0,10)
plt.subplot(1,2,2)
plt.plot(xx, f.pdf(xx))
plt.ylim(0, 0.8)
plt.title('f 분포 : 자유도 = (1,30)')
plt.tight_layout()

plt.show()
```

<img width="1078" alt="Screen Shot 2021-07-22 at 11 53 57" src="https://user-images.githubusercontent.com/83487073/126583876-f8f0f27c-6854-4170-b2e9-11029b8c833a.png">

---

혹은

---

```python
# 다시해보기 
def plot(N) : 
    np.random.seed(0)
    t = sp.stats.t(N).rvs(1000)**2
    f = sp.stats.f(1, N).rvs(1000)

    plt.hist(t, label=f't통계량 제곱 분포 | 자유도 = {N}', bins=50, range=(0, 10), rwidth=0.5, align='left', color='r')
    plt.hist(f, label=f'f분포 표본 | 자유도 = (1, {N})', bins=50, range=(0,10), rwidth=0.5, align='mid', color='b')
    plt.xlim(0, 5)
    plt.legend()
    plt.title(f'N = {N} 인 경우')
    plt.show()


plt.subplot(211)
plot(2)
plt.subplot(212)
plot(30)
```
<img width="1072" alt="Screen Shot 2021-07-22 at 12 10 15" src="https://user-images.githubusercontent.com/83487073/126584915-be9db744-b05c-457b-8df1-25fd3c781801.png">

---

# F분포 특징
- 직관적으로 생각했을 때, N1 = N2일 경우 1근처 어떤 값이 가장 자주 나올 거라 생각된다. 

그 이유는 다음과 같다. 

카이제곱분포 표본은 같은 정규분포에서 나온다. 

표본 갯수인 N1, N2가 같다면, 아무래도 정규분포 상에서 확률밀도 높은쪽에서 N개 표본이 자주 발생할 것이다. 

예) 기댓값, 최빈값 근처

이 경우 카이제곱분포 표본값(정규분포 표본 제곱합)도 2개가 서로 비슷비슷한 값일 거다. 

N1, N2가 서로 같기 때문에 약분하면 결국 분자 분모에 서로 비슷비슷한 값이 남는다. 

이 비율값은 대략 1 근처 값이라 생각할 수 있다. 

이 경우 '자주 발생하는 F분포 표본값'은 '1근처 값 이다' 라고 직관적으로 생각할 수 있다. 

하지만 실제로는 1이 아닌 다른 어떤 값이 더 자주 나온다. 

- 이 현상은 N1=N2 값이 커질수록 사라진다. 

- 결국 1 근처 값이 가장 자주 발생하게 된다. 

##### F분포 특징 : N1=N2가 커질 수록 1근처 값이 가장 자주 발생한다. 

```python
N = [10,100,1000,10000]
M = 10000
np.random.seed(0)
rv = sp.stats.norm()

for i,n in enumerate(N) :
    plt.subplot(1,4,i+1)
    x1 = (rv.rvs(size=(n, M))**2).sum(axis=0)
    x2 = (rv.rvs(size=(n, M))**2).sum(axis=0)
    t = x1/x2
    sns.histplot(t, bins=200)
    plt.axvline(1, ls=':')
    plt.xlabel('$x$')
    plt.title(f'자유도 : {n}')
plt.suptitle('F분포 자유도 (N1,N2)가 커질 수록 1근처 값이 가장 자주 나온다')
plt.tight_layout()
plt.show()
```
<img width="1075" alt="Screen Shot 2021-07-22 at 11 08 44" src="https://user-images.githubusercontent.com/83487073/126581367-341491b0-d652-402f-97bf-4d96b0664804.png">

---

# 다변수 정규분포 
- 다변수 정규분포 차원은 확률변수벡터 차원 따라간다
- 


---
# 메모 

# 고윳값 lambda1, lambda2는 곧 확률변수 X1의 분산, X2의 분산값과 같다. 


1. 다변수정규분포 확률밀도함수 식을 보면 이 확률밀도함수는 타원이다. (반지름 다양한 여러 개 타원을 이룬다)
2. 타원 식을 보면 타원 반지름은 ld1, ld2 값에 비례한다.


3. 곧, ld1, ld2값은 '사실상' 타원 반지름이라 볼 수 있다. 
4. 타원 반지름 = 타원 사이즈 다. 
5. 타원 사이즈 = 값들의 분포 정도 (X1쪽, X2쪽으로 각각 값들이 얼마나 퍼져 있는가(평균을 중심으로))
6. 값들의 분포정도 = X1, X2의 분산 (분산 : 평균을 중심으로 퍼져있는 정도)


7. 타원 반지름 = X1, X2의 분산 
8. ld1, ld2 = V[X1 ], V[X2 ]


## --> 공분산 행렬에 w = np.diag([ld1, ld2]) 를 넣는 이유 

---
# 메모 2

# 다변수정규분포함수 확률밀도함수 모양, 방향, 사이즈의 의미 

# 표준기저벡터 직교좌표계에서
- 모양 : 타원
- 중심 : mu
- 분포사이즈(타원 반지름) 는 공분산행렬 고윳값에 비례한다 
- 분포방향은 공분산행렬 고유벡터 방향이다 

```python
plt.subplot(1,2,1)
mu = np.array([1,2])
cov = np.array([[4, -3],[-3, 4]])

xx = np.linspace(-6, 6, 100)
yy = np.linspace(-6,6, 120)
X, Y = np.meshgrid(xx, yy)

rv = sp.stats.multivariate_normal(mu, cov)
plt.contour(X, Y, rv.pdf(np.dstack([X,Y])))
plt.axis('equal')
plt.scatter(mu[0], mu[1])
plt.annotate('', xy= mu+0.35*w[0]*v[:,0], xytext=mu, arrowprops=d)
plt.annotate('', xy= mu+2*v[:,1], xytext=mu, arrowprops=d)
plt.title('좌표변환 전')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')



plt.subplot(1,2,2)
rv2 = sp.stats.multivariate_normal((0,0), np.diag(w))
plt.contour(X,Y, rv2.pdf(np.dstack([X,Y])))
plt.annotate('', xy= 0.5*w[0]*np.array([1,0]), xytext=[0,0], arrowprops=d)
plt.annotate('', xy= 1.5*w[1]*np.array([0,1]), xytext=[0,0], arrowprops=d)
plt.scatter(0,0, c='k',s=1.2)
plt.axis('equal')
plt.title('좌표변환 후')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.suptitle('다변수정규분포 결합확률밀도함수')
plt.tight_layout()
plt.show()
```
<img width="1078" alt="Screen Shot 2021-07-24 at 16 20 41" src="https://user-images.githubusercontent.com/83487073/126860836-86da6f39-e1ca-46fb-8a13-108d03fbbbec.png">


# 한편
- 확률변수들의 공분산 값도 대략적인 분포 방향 알게 해준다
- 단변수 확률변수 분산값도 대략적인 분포 사이즈 알게 해준다

```python
plt.subplot(1,2,1)
mu = np.array([1,2])
cov = np.array([[4, 0],[0, 4]])

xx = np.linspace(-6, 6, 100)
yy = np.linspace(-6,6, 120)
X, Y = np.meshgrid(xx, yy)

rv = sp.stats.multivariate_normal(mu, cov)
plt.contour(X, Y, rv.pdf(np.dstack([X,Y])))
plt.axis('equal')
plt.scatter(mu[0], mu[1])
plt.title('X1, X2 공분산이 0이고 단변수확률변수 분산 4인 경우')

plt.subplot(1,2,2)
mu = np.array([1,2])
cov = np.array([[4, -3],[-3, 4]])
plt.title('다 같은데 X1, X2 공분산만 -3으로 바뀌면')

xx = np.linspace(-6, 6, 100)
yy = np.linspace(-6,6, 120)
X, Y = np.meshgrid(xx, yy)

rv = sp.stats.multivariate_normal(mu, cov)
plt.contour(X, Y, rv.pdf(np.dstack([X,Y])))
plt.axis('equal')
plt.scatter(mu[0], mu[1])
plt.annotate('', xy= mu+0.35*w[0]*v[:,0], xytext=mu, arrowprops=d)
plt.annotate('', xy= mu+2*v[:,1], xytext=mu, arrowprops=d)
plt.axvline(mu[0], ls=':')
plt.axhline(mu[1], ls='--')
plt.tight_layout()
```
<img width="1074" alt="Screen Shot 2021-07-24 at 16 25 04" src="https://user-images.githubusercontent.com/83487073/126860962-174cedb2-976b-45d5-ba8d-3e13a1bc3d9e.png">


---

# 메모 3 

# matplotlib로 벡터 화살표 표현하기 

# plt.annotate()
주요 arguments 

- 텍스트 내용 입력 / 예) ' '
- xy = 가리킬 끝점 / 예) v[:,0 ]
- xytext = 화살표 시작점 / 예) mu
- arrowprops = d

화살표로서 벡터 표현에 요긴하게 쓸 수 있었다. 

---
# 다변수정규분포-조건부확률분포

# 다변수정규분포 따르는 결합확률분포의 '조건부확률분포'는 다변수정규분포 따른다. 

- D차원 입력벡터 만들어내는 확률변수 x가 있다. 이 확률변수 x는 $N(X \vert \mu, \Sigma)$ 다변수 정규분포를 따른다고 하자. 

- 입력벡터 X의 표본인 D차원 벡터를 열벡터로 쌓아 만든 행렬을 X라 하자. 

- 이 행렬 X를 분할해서 두 개 행렬로 나누자. 다음과 같이 표기할 수 있다. 

$X$ = [[$X_{a}$ ], [$X_{b}$ ]] (열로 생각하자)

이거는 
<img width="867" alt="Screen Shot 2021-07-25 at 18 03 55" src="https://user-images.githubusercontent.com/83487073/126893726-1671b154-de2b-46f0-9f53-d4a74f637040.png">

와 같다. 



- $X_{a}$ 의 벡터 1개는 M차원을 갖고, $X_{b}$의 벡터 1개는 D-M 차원을 갖는다. 

- 행렬 단위에서는 $X_{a}$ 는 (M*N), 

- $X_{b}$는 ((D-M)*N) 행렬일 것이다. 

- 이 때 대응되는 평균은 다음과 같다. 

$\mu$ = [[$\mu_{a}$ ], [$\mu_{b}$ ]]

$\mu_{a}$ 는 행렬 $X_{a}$에 벡터들의 평균이다. 

$\mu_{b}$ 는 행렬 $X_{b}$에 벡터들의 평균이다. 


- 공분산행렬은 다음과 같다. 

$\Sigma$ =  [[$\Sigma_{aa}$, $\Sigma_{ab}$ ],[$\Sigma_{ba}$, $\Sigma_{bb}$ ]]

예를들어 $\Sigma_{aa}$는 다음과 같다. 

<img width="1067" alt="Screen Shot 2021-07-25 at 18 05 22" src="https://user-images.githubusercontent.com/83487073/126893781-134be4a0-53a4-4155-894a-79827df4a50a.png">



- $X_{a}$ 와 $X_{b}$ 간 공분산은 

<img width="1027" alt="Screen Shot 2021-07-25 at 18 06 37" src="https://user-images.githubusercontent.com/83487073/126893833-68e06e8c-c5ae-4b6f-87da-851c625de37a.png">

- $\Sigma_{ab}$ 도 마찬가지다. 

<img width="949" alt="Screen Shot 2021-07-25 at 18 07 29" src="https://user-images.githubusercontent.com/83487073/126893856-54befed7-f835-4501-8500-411ecf997a37.png">


- 이렇게 해서 $\Sigma$ = [[$\Sigma_{aa}$, $\Sigma_{ab}$ ],[$\Sigma_{ba}$, $\Sigma_{bb}$ ]] 가 나온다. 

- 공분산 행렬 $\Sigma$ 는 대칭행렬이다. 따라서 $\Sigma^{T}$ = $\Sigma$ 다. 

- 따라서 $\Sigma_{ba}$ = $\Sigma_{ab}$ 다. 

- 정밀도행렬(공분산행렬의 역행렬)은 다음과 같이 정의할 수 있다. 

$\Lambda$ = $\Sigma^{-1}$

참고로 $\Sigma$ 는 풀랭크(모든 열이 서로 독립) 이므로 항상 역행렬 존재한다. 

- $\Lambda$ 는 다음처럼 정의할 수 있다. 

$\Lambda$ = [[$\Lambda_{aa}$ , $\Lambda_{ab}$ ],[$\Lambda_{ba}$ , $\Lambda_{bb}$]]

- $\Lambda$ 도 대칭행렬이다. (대칭행렬 $\Sigma$ 의 역행렬은 대칭행렬이다)

$\Lambda_{ab}$ = $\Lambda_{ba}$ 성립한다. 

- 단, $\Lambda_{aa}$ 는 $\Sigma_{aa}$ 역행렬 구한다고 얻을 수 있는 게 아니다. 

공분산의 부분행렬의 역행렬이 부분행렬의 정밀도 행렬이라 정의할 수 없다. 

- 우리가 찾는 건 $p(X_{a} \vert X_{b})$ 의 확률분포다. 

- 여기서 현재의 결합확률분포 $p(X)$ = $p(X_{a} \vert X_{b})$ 로 나타낼 수 있다. 

X는 맨 처음 언급했던, D차원 벡터 만들어내는 확률변수 X다. 

- 만약 여기서 $X_{b}$ 가 고정된 값이라면 $X_{a}$ 의 조건부 확률분포 얻을 수 있을 것이다. 

---
## 정규분포 '지수의 이차형식' 꼴에 X, $\mu$ 등을 넣어서 지수의 이차형식 꼴로 정리된다면, X가 정규분포 따르게 될 것이라 볼 수 있다. 

또 정리된 식에서 평균, 공분산 구할 수 있어야 한다. 

만약 지수의 이차형식 꼴로 정리되면서, 평균, 공분산 구할 수 있으면 X의 분포가 정규분포 따른다' 고 할 수 있다. 

---

- 지수의 이차형식 

$\Delta^{2}$ = $-(1/2)*$ $(X-\mu)^{T}$ $\Sigma^{-1}$ $(X-\mu)$

---

- 위 식에 $X$ 와 $\mu$ 를 넣어서 식을 전개해보자. 

<img width="1090" alt="Screen Shot 2021-07-25 at 18 10 42" src="https://user-images.githubusercontent.com/83487073/126893920-614aa3d2-d75d-4511-80fb-15342acaf1ee.png">


- 앞에서 $X_{b}$ 는 특정한 값에 고정되어 있다고 생각하기로 했다. 

- 따라서 위 식은 그냥 $X_{a}$ 에 대한 식이라 볼 수 있다. 

- 또, 정리하면 이차형식 꼴이 된다. 

- 이차형식에서 평균과 공분산 구할 수 있으면 '정규분포 따른다'고 할 수 있다. 

---

## 정규분포의 이차형식 구조를 좀 더 살펴보자. 

$-(1/2)*$ $(X-\mu)^{T}$ $\Sigma^{-1}$ $(X-\mu)$ 

$=$ $-(1/2)*X^{T} \Sigma^{-1}X$ $+$ $X^{T} \Sigma^{-1} \mu$ + $const$

이 전개 방식을 'completing the square'라 부른다. 

- $const$ 영역은 $X$ 와는 독립적인 영역이다. 

- 이차형식을 두번 미분한 값은(즉 2차식 계수에 주목해야 한다) $\Sigma^{-1}$ 이 된다. 

- 이를 통해 공분산의 역행렬을 구할 수 있다. ($\Sigma^{-1}$)

- 다음으로 1차식 계수는 $\Sigma^{-1} \mu$ 가 된다. 

- 따라서 공분산행렬 구한 다음, 1차항 계수에서 평균벡터 $\mu$ 를 구할 수 있다. 

---
- 이제 조건부 분포의 평균과 분산을 구해보자. 
- $X_{a}$ 와 $X_{b}$ 로 표현된 이차형식 영역을 $X_{a}$ 에 대해 두번 미분한다고 하면, 2차식 부분은 다음과 같다. 

$-(1/2)*X_{a}^{T} \Lambda_{aa} X_{a}$ 

- $X_{b}$ 와 관련된 영역은 모두 상수취급하면 된다. 

- 두번 미분한 값(2차 항의 계수) 은 공분산의 역행렬과 같았다. 따라서, 조건부분포 $p(X_{a} \vert X_{b})$ 의 공분산은 다음과 같다. 

- $\Sigma_{a\vert b}$ $=$ $\Lambda_{aa}^{-1}$

- 이제 평균을 구하기 위해 $X_{a}$ 의 1차식 계수를 확인해 보자. 

1차식 계수는 

$X_{a}^{T}${$\Lambda_{aa}$ $\mu$ - $\Lambda_{bb} (X_{b} - \mu_{b})$}

이다. 

- $\Lambda_{ba}^{T}$ $=$ $\Lambda_{ab}$ 였다. 

(앞에서 대칭행렬 설명할 때)

- 위 식은 앞에서 계산한 $\Sigma_{a \vert b}^{-1} \mu_{a \vert b}$ 와 같아져야 한다. 

- 식을 전개하면 다음과 같다. 

$\Sigma_{a \vert b}^{-1}\mu_{a \vert b} = [\Lambda_{aa}\mu_{a} - \Lambda_{ab}(X_{b}-\mu_{b})]$

--> $\mu_{a \vert b} = \Sigma_{a \vert b} [ \Lambda_{aa} \mu_{a} - \Lambda_{ab} (X_{b} - \mu_{b}) ]$

--> 앞에서 $(\Sigma_{a \vert b} = \Lambda_{aa}^{-1})$ 이었다. 적용하면

--> $\mu_{a\vert b} = \Lambda_{aa}^{-1}$ $[\Lambda_{aa}\mu_{a} - \Lambda_{ab} (X_{b} - \mu_{b})]$

- $\mu_{a \vert b}$ $=$ $\mu_{a} - \Lambda_{aa}^{-1}\Lambda_{ab}(X_{b}-\mu_{b})$ 이 된다. 

---

- 평균과 분산을 정밀도 행렬 $\Lambda$ 말고, $\Sigma$ 로 표현해보자. 

<img width="720" alt="Screen Shot 2021-07-25 at 18 11 39" src="https://user-images.githubusercontent.com/83487073/126893956-ae39a9ed-1cdc-413b-8bfb-25274f6e3cf8.png">

- $M^{-1}$ 은 schur completement라 불리는 행렬을 의미한다. 

- 이 식을 아래 행렬에 대입해서 풀어본다. 

<img width="725" alt="Screen Shot 2021-07-25 at 18 12 30" src="https://user-images.githubusercontent.com/83487073/126893979-05f69c16-b672-481c-b248-60e1dcb04bf6.png">


- $\Lambda_{aa} = (\Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1}\Sigma_{ba})^{-1}$
- $\Lambda_{ab} = -(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1}$ $\Sigma_{ab}\Sigma_{bb}^{-1}$


여기서 얻은 결과를 $p(X_{a} \vert X_{b})$ 의 평균. 분산에 대입하면

- $\mu_{a \vert b} = \mu_{a} + \Sigma_{aa}\Sigma_{bb}^{-1}(X_{b}-\mu_{b})$
- $\Sigma_{a \vert b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$

가 나온다. 

---
## 결론

- $\mu_{a \vert b}$ 는 $X_{b}$ 에 대한 선형함수 꼴이다. 
- $\Sigma_{a \vert b}$ 는 $X_{b}$ 에 대해 독립적인 식이다. 

- $X_{2}$ 가 어떤 값으로 주어지면 $X_{1}$의 조건부확률분포는 조건부 기댓값 $\mu_{a \vert b}$, 조건부공분산 행렬 $\Sigma_{a \vert b}$ 갖는 다변수 정규분포를 따르게 된다. 


---

# 다변수정규분포 따르는 결합확률분포의 주변확률분포도 다변수정규분포 따른다. 

- 앞에서는 $p(X_{a}, X_{b})$ 분포가 다변수정규분포이면 $p(X_{a} \vert X_{b})$ 분포도 다변수 정규분포가 된다는 걸 확인했다. 

- 그렇다면 주변확률분포는 어떨까? 

$p(X_{a}) = \int p(X_{a}, X_{b})dX_{b}$ <주변확률분포 식>

위 주변확률분포가 '지수의 이차형식' 꼴로 정리되고,
- 평균 
- 분산 

구할 수 있으면 정규분포 따른다고 할 수 있다. 

- 지수의 이차형식은 앞에서 다음과 같았다. 

$-(1/2)*(X-\mu)^{T}\Sigma^{-1}(X-\mu)$

$= -(1/2)*X^{T}\Sigma^{-1}X + X^{T}\Sigma^{-1}\mu + const$


- $X$ 에 $[[X_{a}],[X_{b}]]$ 넣고 $\mu$ 에 $[[\mu_{a}],[\mu_{b}]]$ 넣어서 전개하면 다음과 같다. 

$-(1/2)*X_{b}^{T}\Lambda_{bb}X_{b} + X^{T}_{b}m $

$= -(1/2)(X_{b}-\Lambda_{bb}^{-1}m)^{T}\Lambda_{bb}(X_{b}-\Lambda_{bb}^{-1}m)+(1/2)m^{T}\Lambda_{bb}^{-1}m$

- 위 식은 $X_{b}$ 에 대한 2차식으로 부분 변형하고, $(1/2)*m^{T}\Lambda_{bb}^{-1}m$ 을 보충해준 식이다. 

- $m = \Lambda_{aa}\mu_{b}-\Lambda_{ba}(X_{a}-\mu_{a})$

- $X_{b}$ 로 적분해보자. 

첫번째 항 $-(1/2)(X_{b}-m^{T})\Lambda_{bb}(X_{b}-\Lambda_{bb}^{-1}m)$ 은 $X_{b}$ 에 대한 식이다. 

두번째 항 $(1/2)m^{T}\Lambda_{aa}^{-1}m$ 은 $X_{a}$ 에만 종속되는 식이다. 

---

- 적분한 식을 $X_{a}$ 에 대해 정리한다. 

$(1/2)[\Lambda_{bb}\mu_{b}-\Lambda_{ba}(X_{a}-\mu_{a})]^{T}\Lambda_{bb}^{-1}[\Lambda_{bb}\mu_{b}-\Lambda_{ba}(X_{a}-\mu_{a})]-(1/2)X_{a}^{T}\Lambda_{aa}X_{a}+X_{a}^{T}(\Lambda_{aa}\mu_{a}+\Lambda_{ab}\mu_{b})+const$

$= -(1/2)X_{a}^{T}(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})X_{a} + X_{a}^{T}(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_{a}+const$

- 이것 역시 이차형식 꼴이다. 따라서 $p(X_{a})$ 가 정규분포 따를 것이라 예상할 수 있다. 

- 이제 위 식에서 평균. 분산만 구할 수 있으면 '$p(X_{a})$ 는 정규분포 따른다'고 말할 수 있다. 


복기해보면, 공분산은 이차형식 내 2차식의 계수, 평균은 이차형식 내 1차식의 계수와 같았다. 

- 이차식 계수 = $\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}$ 의 역행렬

--> $(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1} = \Sigma_{a}$

- 일차식 계수 = $\Sigma^{-1}\mu$ 와 같았다. 

위에 식에서 일차식 계수는 $(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_{a}$ 이었다. 

$\Sigma^{-1}\mu = [(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_{a}]$ 라고 두고 풀면

$\mu = \mu_{a}$ 가 된다. 

- 결국 
- $\Sigma_{a} = (\Lambda_{aa}- \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}$ 
- $\mu_{a} = \mu_{a}$ 

가 된다. 

---

# 한편
- $[[\Lambda_{aa}, \Lambda_{ab}],[\Lambda_{ba},\Lambda_{bb}]]^{-1} = [[\Sigma_{aa},\Sigma_{ab}],[\Sigma_{ba},\Sigma_{bb}]]$ 였다. 

- $[[A,B],[C,D]]^{-1} = [[M, -MBD^{-1}],[-D^{-1}CM, D^{-1}CMBD^{-1]}]$ 을 이용해서 식을 전개해 보자. 

- $(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1} = \Sigma_{aa}$ 가 나온다. 

- 정리하면 다음과 같다. 
- $E[X_{a}] = \mu_{a}$
- $COV[X_{a}] = \Sigma_{aa}$

직관적으로 매우 타당한 식이 결과로 나왔다. 

---

# 정리 및 요약 

## 다변수정규분포 따르는 어떤 결합분포가 다음과 같을 때 

- $X = [X_{a}, X_{b}]$
- $\mu = [\mu_{a}, \mu_{b}]$
- $\Sigma = [[\Sigma_{aa},\Sigma_{ab}],[\Sigma_{ba},\Sigma_{bb}]]$
- $\Lambda = [[\Lambda_{aa},\Lambda_{ab}],[\Lambda_{ba}, \Lambda_{bb}]]$

## - 조건부 확률분포 
- $p(X_{a} \vert X_{b}) = N(X_{a} \vert \mu_{a \vert b}, \Lambda_{aa}^{-1})$

## - 주변확률분포
- $p(X_{a}) = N(X_{a} \vert \mu_{a}, \Sigma_{aa})$

## 모두 다변수정규분포 따른다. 

---

# 8.6.2 연습문제 

문제 

2차원 다변수정규분포가 다음과 같은 모수를 가진다고 하자. 

$\mu = [\mu_{1}, \mu_{2}]$, 

$\Sigma = [[\sigma_{1}^{2}, p\sigma_{1}\sigma_{2}],[p\sigma_{1}\sigma_{2}, \sigma_{2}^{2}]]$

$x_{2}$가 주어졌을 때 $x_{1}$ 의 조건부확률분포함수가 다음과 같음을 보여라. 

- $N(x_{1} \vert \mu_{1}+(p\sigma_{1}\sigma_{2}/\sigma_{2}^{2})(x_{2}-\mu_{2}), \sigma_{1}^{2} - ((p\sigma_{1}\sigma_{2})^{2}/\sigma_{2}^{2}))$

---
# 내 풀이 

- $\mu = [[\mu_{1}], [\mu_{2}]]$

- $\Sigma = [[\Sigma_{11},\Sigma_{12}],[\Sigma_{21},\Sigma_{22}]]$
- $\Lambda = [[\Lambda_{11},\Lambda_{12}],[\Lambda_{21}, \Lambda_{22}]] = (\Sigma^{-1})$

- $p(x_{1} \vert x_{2})$ 는 $N(x_{1} \vert \mu_{1 \vert 2}, \Sigma_{1 \vert 2})$ 따른다


--> $-(1/2)(x-\mu)^{T}\Sigma^{-1}(x-\mu) =$ 

$-(1/2)x^{T}\Sigma^{-1}x+x^{T}\Sigma^{-1}\mu + const$

지수의 이차형식에 $x$, $\mu$ 넣어서 전개한 다음 이차형식 형태로 정리되는지 보자. 

정리하면 다음과 같다. (계산과정 생략)

$-(1/2)x_{1}^{T}\Lambda_{11}x_{1} $

$+ (-(1/2)\Lambda_{21}x_{2} - (1/2)\Lambda_{12}x_{2}+\Lambda_{11}\mu_{11}+\Lambda_{12}\mu_{2})x_{1} $

$+ (-(1/2)\Lambda_{22}x_{2}^{2} + \Lambda_{21}x_{2}\mu_{1} + \Lambda_{22}x_{2}\mu_{2}+ const)$


- $\Lambda_{11} = \Sigma^{-1}$
- $\Lambda_{11}^{-1} = \Sigma_{a \vert b}$

- $\Sigma^{-1}\mu = (-(1/2)\Lambda_{21}x_{2}-(1/2)\Lambda_{12}x_{2}+\Lambda_{11}\mu_{1} + \Lambda_{12}\mu_{12})$
- $\Sigma*\Sigma^{-1}\mu = \Sigma(-\Lambda_{21}x_{2} + \Lambda_{11}\mu_{1}+\Lambda_{12}\mu_{2})$

- $\mu_{a \vert b} = \Lambda_{11}^{-1}(-\Lambda_{21}x_{2}+\Lambda_{11}\mu_{1}+\Lambda_{12}\mu_{2})$

---
$[[\Sigma_{11}, \Sigma_{12}],[\Sigma_{21},\Sigma_{22}]]^{-1} = [[\Lambda_{11},\Lambda_{12}],[\Lambda_{21}, \Lambda_{22}]]$

를 이용하자. (구체적인 식은 앞에 참고)


<img width="1222" alt="Screen Shot 2021-07-25 at 18 14 05" src="https://user-images.githubusercontent.com/83487073/126894026-f4b023fc-7597-49c1-a45e-218484b5d90d.png">


- $\Lambda_{11} = (\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})^{-1}$
- $\Lambda_{12} = -(\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})^{-1}\Sigma_{12}\Sigma_{22}^{-1}$


- $\Sigma_{a \vert b} = \Lambda_{11}^{-1}$
- $\mu_{a \vert b} = \Lambda_{11}^{-1}(-\Lambda_{21}x_{2}+\Lambda_{11}\mu_{1}+\Lambda_{12}\mu_{2})$

= $-\Lambda_{11}^{-1}\Lambda_{21}x_{2}+\mu_{1}+\Lambda_{11}^{-1}\Lambda_{12}\mu_{2}$

= $\mu_{a \vert b} = \mu_{1} + (\Lambda_{11}^{-1}\Lambda_{12})(\mu_{2}-x_{2})$

- 앞에서 구한 $\Lambda_{11}$ 과 $\Lambda_{12}$ 을 이용해서 $\Sigma_{a \vert b}$, $\mu_{a \vert b}$ 식을 바꾸면 다음과 같다. 

- $\mu_{a \vert b} = \mu_{1}+\Sigma_{12}\Sigma_{22}^{-1}(x_{2}, \mu_{2})$
- $\Sigma_{a \vert b} = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$

---
앞에서 


$\Sigma $

$= [[\sigma_{1}^{2}, p\sigma_{1}\sigma_{2}],[p\sigma_{1}\sigma_{2}, \sigma_{2}^{2}]]$ 

$= [[\Sigma_{11},\Sigma_{12}],[\Sigma_{21}, \Sigma_{22}]]$

였다. 

$\mu_{a \vert b}$, $\Sigma_{a \vert b}$ 에 대입해서 식 정리하면 

- $\mu_{a \vert b} = \mu_{1} + ((p\sigma_{1}\sigma_{2})/\sigma_{2}^{2})$ $(x_{2}-\mu_{2})$

- $\Sigma_{a \vert b} = \sigma_{1}^{2}-(p\sigma_{1}\sigma_{2})^{2}/\sigma_{2}^{2}$

로, 증명이 완료된다. 

---

위 내용을 공부 & 정리 하면서 참조한 곳 : http://norman3.github.io/prml/docs/chapter02/3_1.html

---


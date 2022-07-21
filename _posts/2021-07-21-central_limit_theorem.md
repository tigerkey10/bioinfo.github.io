---
title : "[수학/확률과 통계] 중심극한정리 "
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python

tags : 
- [study, data science, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-07-21
last_modified_at : 2021-07-21

---

# 중심극한정리 
- 정의 

기댓값이 mu 이고, 분산이 sigma^{2} 이며 서로 독립인 분포 N개가 있다. 

각 분포에서 나온 표본 N개의 평균(또는 합)의 분포는 N이 커질수록 기댓값이 mu이고 표준편차가 sigma/np.sqrt(n) 인 정규분포에 근사한다. 

- 혹은

기댓값이 mu이고, 분산이 sigma^{2} 인 분포가 있다. 

이 분포에서 서로 독립인 표본 N개를 추출했다. 

N이 커질수록, 이 표본들의 평균(또는 합)이 이루는 분포가 기댓값이 mu이고 표준편차가 sigma/np.sqrt(n) 인 정규분포에 근사한다. 

---

- 위 표본평균(합) 값들도 그 자체로 하나의 확률변수가 된다. 
- 표본평균 확률변수에서 얻은 데이터들을 정규화하면, N이 증가할 수록 표준정규분포에 수렴한다. 

---

# 표본 시뮬레이션으로 중심극한 정리 성립하는지 보자. 

```python
# 균일분포 표본 얻어서 중심극한정리 검증해보기 
np.random.seed(0)
xx = np.linspace(-2,2,100)

plt.figure(figsize=(6,9))

for i, N in enumerate([1,2,5]) : 
    x = np.random.sample([5000, N])
    x_bar = (np.mean(x, axis=1)-0.5)/(1/12)
    ax = plt.subplot(3,2,2*i+1)
    sns.distplot(x_bar, bins=10, kde=False, norm_hist=True)
    plt.xlim(-5,5)
    plt.yticks()
    ax.set_title(f'N={N}')
    plt.subplot(3,2,2*i+2)
    sp.stats.probplot(x_bar, plot=plt)

plt.tight_layout()
plt.show()
```

<img width="502" alt="Screen Shot 2021-07-21 at 17 45 46" src="https://user-images.githubusercontent.com/83487073/126459665-621ebd65-d8a1-421d-8bf5-3ad2ee699b59.png">

- N이 커질수록 분포가 정규분포에 수렴하는 것을 볼 수 있다. 

---

# 그렇다면 N개 정규분포에서 얻은 표본들 평균(합)은 어떻게 될까? 

- N개 정규분포에서 얻은 표본들 평균(합)은 N에 관계 없이 언제나 정확한 정규분포를 이룬다. 
- 기댓값 : N *mu, 분산 : N *sigma^{2}
- N=1이면 원래 정규분포에서 나온 표본들이니 표본분포도 정규분포 그대로 따라갈 것이다. 

# 정규분포 따르는 표본들을 정규화 하면 어떻게 될까? 
- 정규분포 따르는 표본들을 정규화 한 값을 'Z 통계량' 이라 한다. 
- 앞에서는 표본평균들을 정규화 하면 N이 커질 때, 표준정규분포에 수렴했다. 
##### 이와 달리, 정규분포 따르는 표본들의 평균값들을 정규화 하면 정확하게 표준정규분포를 이룬다. 

# 시뮬레이션으로 직접 알아보자 

```python
rv = sp.stats.norm(0, 1)

plt.figure(figsize=(6,9))
for i, v in enumerate([1,10,100]) : 
    fig = plt.subplot(3,2,2*i+1)
    data = rv.rvs(size=(5000, v), random_state=0)
    x_bar = np.mean(data, axis=1)
    sns.distplot(x_bar, kde=False, bins=10)
    plt.xlim(-5,5)
    plt.yticks([])
    fig.set_title(f'N={v}')

    ax = plt.subplot(3,2,2*i+2)
    sp.stats.probplot(x_bar, plot=plt)
plt.suptitle('정규분포에서 나온 표본들 평균은 N상관없이 항상 정규분포 따른다', y=1)
plt.tight_layout()

plt.show()
```

<img width="442" alt="Screen Shot 2021-07-21 at 17 52 34" src="https://user-images.githubusercontent.com/83487073/126460676-a509e037-4586-466f-b104-a8015f17e1a4.png">

왼쪽 분포 형상도 정규분포와 같고, 오른쪽 Q-Q 플롯을 보면 스캐터 플롯이 모두 일직선으로, 

N에 상관없이 표본데이터들 분포가 정확하게 정규분포를 이룬다는 걸 알 수 있다. 

---

# 선형회귀 모형과 정규분포 
- 정규분포는 선형회귀 모형에서 잡음(disturbance) 모형화에 사용된다. 

# 선형회귀모형 

y = (c1 * x1) + (c2 * x2) ... (cn * xn) + epsilon

# epsilon이 '잡음' 이다. 
- '잡음'의 정의 : y 형성하는 영향력 중, 측정 불가한 '나머지' 영향력 하나로 퉁 친 것. 

# 책 567p 메모 )

원래 y값은 무한한 독립변수의 영향을 받는다. 

각각의 (cn * xn) 값은 독립변수 값에 따라 달라지는 '확률변수 값' 이다. 확률적 데이터가 나오기 때문에, 다음 번에 뭐가 나올지 알 수 없다. 

각 확률변숫값을 내놓는 공통의 확률변수가 있을 것이다. 이를 'y에 대한 영향' 확률변수라고 명명하겠다. 

epsilon은 확률변숫값들 중 가장 '작고', '미미해서' y에 거의 영향조차 못 미치는 값들을 더한 것이다. 

epsilon을 구성하고 있는 확률변숫값들은 무한개다. 

같은 확률변수(확률분포) 에서 나온 표본값들 갯수(N) 이 무한대이므로, 중심극한정리에 따라 epsilon의 분포는 정규분포로 가정할 수 있다. 

---

한편, epsilon의 기댓값은 매우 작은 0 근처 값일 것이다. 

이 값은 매우 작아서 y에 영향 주지 못한다. 이를 이용해 

y = E[epsilon ] + c1x1+ c2x2...+epsilon 으로 선형회귀식을 변형할 수 있다. 

이때 앞에서 y는 c1x2+c2x2+...+epsilon과 등호가 성립했으므로, 

c1x2+c2x2+...+epsilon  =  E[epsilon ] + c1x1+ c2x2...+epsilon 으로 놓을 수 있다. 

양변에서 같은 항을 제거하면 E[epsilon ] = 0 이 된다. 

이런 방법으로, y = E[epsilon ] + c1x1+ c2x2...+epsilon 를 쓰면서 E[epsilon ] = 0 이라고 할 수 있다. 

따라서 

##### epsilon의 분포는 기댓값이 0인 정규분포로 가정할 수 있다. 

##### epsilon ~ N(0, sigma^{2})

- 한편 y도 N개(무한대) 확률변숫값의 합이므로, y의 분포도 정규분포를 이룬다. 

---






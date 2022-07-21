---
title : "[수학/확률과 통계] 스튜던트 t분포, 카이제곱분포"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python
- mathematics

tags : 
- [study, data science, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-07-21
last_modified_at : 2021-07-21

---

# t분포 (스튜던트 t분포) : t통계량 분포
- 팻테일 현상 보이는 데이터에 적용하기 좋은 확률분포다. (예: 주식 일간수익률 데이터)
- 기댓값, 정밀도, 자유도를 모수로 갖는다. 
- 자유도는 2이상의 자연수를 쓴다. 
- 형상은 정규분포와 비슷하다. 

자유도 = 1 인 t분포는 '코시분포'라고 부른다. 

- t통계량의 분포다. 

<img width="707" alt="Screen Shot 2021-07-21 at 21 18 58" src="https://user-images.githubusercontent.com/83487073/126487263-d894d187-2f0d-4b7d-9ad5-ff872e354ddc.png">

# t통계량
- 정규분포에서 나온 표본들 평균을 정규화 해서 구한 값이다. 
- 정규화 할 때 Z 통계량 처럼 모분포 표준편차 안 쓰고, 비편향 표본표준편차 s 써서 정규화 한 값이다. 
- 비편향 표본표준편차 s는 정규분포 표본 N개의 비편향 표본표준편차를 말한다. 
- t통계량들의 분포를 t분포라고 한다. t분포 모수 기댓값과 정밀도는 각각 0, 1이 기본이다. 
- t분포 자유도는 N-1이다. N은 표본평균 계산한 정규분포 표본 수다. 

# t분포와 자유도 간 관계 

자유도가 커지면 정규분포에 가까워진다. 

자유도가 작아지면 정규분포에서 멀어지고, 팻테일 현상이 나타난다. 

<img width="788" alt="Screen Shot 2021-07-21 at 21 48 24" src="https://user-images.githubusercontent.com/83487073/126490953-db29f707-4fbb-4bcb-84d9-0d8e04a4a5fb.png">

---

# 카이제곱 분포 : 정규분포 표본 N개 제곱합의 분포 
- t분포 처럼 자유도 모수를 갖는다. 자유도 = N
- 모든 표본값이 양수인 분포다. 

```python
# 카이제곱분포 확률밀도함수 
xx = np.linspace(0.01, 10, 100)
dfs = np.arange(1,5)
ls = ['-', ':', '--', '-.']

for df, ls in zip(dfs, ls) : 
    rv = sp.stats.chi2(df = df)
    plt.plot(xx, rv.pdf(xx), ls=ls, label=f'자유도 : {df}')
plt.xlim(0, 10.01)
plt.ylim(0, 0.6)
plt.legend()
plt.title('자유도 변화에 따른 카이제곱분포 확률밀도함수 변화')
plt.xlabel('표본값')
plt.ylabel('$p(x)$')
plt.show()
```

<img width="701" alt="Screen Shot 2021-07-21 at 22 11 42" src="https://user-images.githubusercontent.com/83487073/126494152-844b3c6e-ae2b-4c14-aaec-07377adfdcd8.png">

---
- 자유도가 2까지는 0 언저리 값이 가장 많이 나온다. 
- 자유도가 3 이상일 때(표본 갯수 N이 3 이상 일 때) 0보다 큰 어떤 수가 가장 많이 나온다. 
- 이는 중심극한정리 때문이다. 카이제곱 분포는 결국 '정규분포 표본 제곱' 이라는 확률변수 N개 합의 분포와 같다. 

정규분포 표본 제곱 값들은 같은 정규분포 따르는 확률변수에 의해 결정된다. 곧, 제곱값 확률변수들은 같은 분포를 따른다. 

결국 중심극한정리에 따라, 제곱값 확률변수 갯수 N이 커질 수록 정규분포에 근사해간다. 

```python
N = [6,30,100,150]
M = 2000
np.random.seed(0)
rv = sp.stats.norm()

for i, n in enumerate(N) : 
    plt.subplot(1,4,i+1)
    x = rv.rvs(size=(n, M))
    data = (x**2).sum(axis=0)
    sns.distplot(data, kde=False)
    plt.title(f'자유도 : {n}')
plt.tight_layout()
plt.suptitle('중심극한정리에 따라 N 커질 수록 정규분포 형상에 가까워진다', y=1.1)
plt.show()
```

<img width="765" alt="Screen Shot 2021-07-21 at 23 10 29" src="https://user-images.githubusercontent.com/83487073/126502993-f3953403-2743-484f-8fff-f2babb21da0e.png">

---











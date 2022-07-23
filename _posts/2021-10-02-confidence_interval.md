---
title : "[수학/확률과 통계] 신뢰구간, 신뢰수준, 표본오차, 표준오차"
excerpt : "헷갈리던 내용을 확실하게 정리"

categories : 
- Data Science
- mathematics

tags : 
- [mathematics, datascience]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-10-02
last_modified_at : 2021-10-02

---

## 다음 그림이 신뢰구간, 신뢰수준이 무엇인지 압축적으로 설명해준다. 

```python
rv = sp.stats.norm()
for i in range(1, 101) :
    x_mean = np.mean(rv.rvs(100))
    if x_mean-1.96*(1/10) <= 0 and 0 <= x_mean+1.96*(1/10) : 
        plt.vlines(i, ymin=x_mean-1.96*(1/10), ymax=x_mean+1.96*(1/10), colors='b')
    else : 
        plt.vlines(i, ymin=x_mean-1.96*(1/10), ymax=x_mean+1.96*(1/10), colors='r')
    
plt.axhline(0, ls=':', c='r')
plt.suptitle('신뢰수준의 의미 : 100개 신뢰구간 중 95개 정도가 모수 $\mu$ 를 포함한다', y=1.02)
plt.title('파란 선 : 모수 $\mu$ 를 포함하는 신뢰구간들, 빨간 선 :모수 $\mu$ 를 포함하지 않는 신뢰구간들')
plt.show()
```
<img width="846" alt="Screen Shot 2021-10-02 at 22 54 17" src="https://user-images.githubusercontent.com/83487073/135719402-5e8abc5f-6f16-4625-b3a5-5320489f5927.png">

---

# 신뢰구간

## 정의:

모수 $\mu$ 가 있을 만 한 구간 

## 유념해야 할 점: 
- 신뢰구간은 하나의 '공식'이다. 
- 신뢰구간은 하나로 정해진 값이 아니다. 

## 신뢰구간 식:

$\bar{X} \pm Z\frac{\sigma}{\sqrt{n}}$

## $\Rightarrow \bar{X} - Z\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + Z\frac{\sigma}{\sqrt{n}}$

- $Z$ 는 Z-score 를 말한다. 
- $Z$ 값은 몇 % 신뢰수준에서 신뢰구간 구하느냐에 따라 달라진다. 
- n이 30 이상이고, $\sigma$ 를 모를 경우 $\sigma$ 대신 표본표준편차 s를 쓴다. 

---

# 신뢰수준

## 정의: 

'측정 방법의 정확도'

또는 

'신뢰구간이 모수를 포함할 확률(빈도주의 정의에 충실해야 한다)'

## 설명:

크기 n인 표본을 100번 추출해서 표본평균을 100개 구한다. 

표본평균 100개를 신뢰구간 식에 대입하면 신뢰구간도 100개가 생긴다.

(예컨대) 95% 신뢰수준은 이 100개 신뢰구간 중 약 95개가 모수를 포함한다는 의미다.

$\Rightarrow$ '이 방법대로 하면' 100개 신뢰구간 중 95개가 모수를 포함한다는 뜻이므로, '측정 방법의 정확도'라고 보는 게 맞다. 

## 또는 

위의 100개 신뢰구간은 사실 모두 하나의 $\bar{X} - Z\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + Z\frac{\sigma}{\sqrt{n}}$ 구간이라 볼 수도 있다. 

$\bar{X}$ 에서 뭐가 생성되든, $\bar{X}$ 는 $\bar{X}$ 이기 때문이다. 

100개 구간을 사실상 모두 같은 '하나의 구간' 이라고 생각해보자. 

빈도주의 관점에서, 구간 $\bar{X} - Z\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + Z\frac{\sigma}{\sqrt{n}}$ 구하기를 100번 반복했을 때, 그 중 95번은 모수를 포함한다(사건 발생횟수). 이렇게 빈도주의 관점에서 볼 때, 구간이 모수를 포함할 확률은 95% 다. 

---

한편 구간이 95% 확률로 모수를 포함할 수 있다면, 5% 확률로 모수를 포함 안 할 수도 있다는 걸 명심하자. 

---

예)

신뢰수준이 95% 이고 표본오차가 $\pm$ 5% 인 여론조사 결과 A 후보의 지지율이 25%, B 후보 지지율이 30% 라고 해보자. 

이를 해석하면, 

n개 표본 얻는 작업을 100번 반복할 때, 100번 중 95번은 A후보 지지율(모수) 이 20%~30% 사이에, B후보 지지율(모수) 이 25%~35% 사이에 포함된다는 말이다. 

한편 각각 25% 와 30%는 수많은 '가능한' 표본중에 '어쩌다가' 나온 모수 점추정치다. 

그 말은, 25%, 30%가 아닌 다른 값이 충분히 나올 수도 있었다는 말이다. (또 다른 평행세계에서는 다른 값이 나왔을수도 있다)

이렇게 점 추정치가 절대적이고 완벽한 값이 아니기 때문에, 표본오차를 통해 신뢰구간을 구하는 것(모수 구간추정 하는 것)이다. 

---

<img width="846" alt="Screen Shot 2021-10-02 at 22 54 17" src="https://user-images.githubusercontent.com/83487073/135719402-5e8abc5f-6f16-4625-b3a5-5320489f5927.png">

---

# 표본오차 

## $\bar{X} \pm Z\frac{\sigma}{\sqrt{n}}$ 에서,

## $\pm Z\frac{\sigma}{\sqrt{n}}$ 이 부분이 표본오차다. 

- 오차한계(margin of error) 라고도 한다. 

---

# 표준오차 (Standard Error of Mean : SEM)

## 정의: 

표본평균 값들이 모평균에서 떨어져 있는 정도.

## $= \frac{\sigma}{\sqrt{n}}$ 

또는 

표본평균 확률변수에서 실현되는 데이터들이 불확실하게 변화하는 정도를 말한다. 

```python
#표준오차 예
rv = sp.stats.binom(30, 0.6)

mean = []
for i in range(10000) : 
    mean.append(np.mean(rv.rvs(1000)))
sns.distplot(mean, kde=False, fit=sp.stats.norm)
plt.axvline(18, ls=':', c='r')
plt.annotate('', xy=[18.1,2], xytext=[18, 2], arrowprops={'facecolor':'black'})
plt.annotate('', xy=[17.9,2], xytext=[18,2], arrowprops={'facecolor':'black'})
plt.text(18.02, 1.5, '표준오차')
plt.text(17.92, 1.5, '표준오차')
plt.title('표준오차')
plt.xlabel('표본평균')
plt.show()
```
<img width="819" alt="Screen Shot 2021-10-02 at 23 14 51" src="https://user-images.githubusercontent.com/83487073/135720213-f724607b-a5d5-4661-ba29-0573008bba84.png">

- 표준오차가 작으면, 모수 점 추정 결과 정확도가 높다.
- 표준오차가 크면, 모수 점 추정 결과 정확도가 낮다. 














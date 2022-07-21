---
title : "[기록/메모] 공부하다 찾은 재밌는 사실들"
excerpt : "."

categories : 
- Data Science
- python
- matplotlib


tags : 
- [python, matplotlib, datascience]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-08-27
last_modified_at : 2021-08-27

---

# 공부하다 찾은 재밌는 사실들

파이썬과 데이터사이언스 공부를 하다가 찾은, 개인적으로 흥미로운 사실들을 기록하는 저장소. 

짜잘하고 사소한 것들 모두 기록할 것이다.

---

# 확률분포함수 시뮬레이션해서 얻은 표본들에 다시 확률밀도함숫값 대응시켜 matplotlib으로 그래프 그리면 다음과 같은 모양이 나온다. 

```python
rv = sp.stats.norm()
s = rv.rvs(1000, random_state=0)
pdf =rv.pdf(s)

plt.plot(s, pdf, 'ro-')
```
<img width="725" alt="Screen Shot 2021-08-27 at 11 32 03" src="https://user-images.githubusercontent.com/83487073/131062535-09994e6a-c3f3-4dc3-83a9-e4e04956fc84.png">

정규분포에서 표본 1천개를 시뮬레이션해서 뽑은 후 이 표본들의 정규분포 확률밀도함숫값을 계산했다. 

그리고 각 표본과 확률밀도함숫값을 대응시켜 라인플롯을 그려봤다. 

## 처음에 했던 내 생각

함수는 입력과 출력 사이 일정한 대응관계다. 

정규분포에서 추출한 표본들은 모두 특정한 정규분포 확률밀도함숫값이 애초에 대응되어 있을 것이다. 

그러면 정규분포에서 시뮬레이션 해서 뽑은 표본들을 각 확률밀도함숫값과 대응시켜 그래프 그리면. 당연히 다시 원래 이론적 확률분포(정규분포) 형상을 띄어야 한다.

## 그런데 왜? 

라인플롯을 그렸더니 위 이미지처럼 이상한 모양이 나오는가. 

## 내가 내린 결론

생각해보면, 확률분포에서 시뮬레이션해서 얻은 표본들은 무작위로 추출된다. 같은 표본이 여러번 나오기도 한다. 

각 표본과 거기 대응되는 확률밀도함숫값을 벡터로 엮어 2차원 공간상에 점으로 일일이 하나씩 찍으면, 분명 표본들 순서에 상관 없이 결국 완벽한 이론적 확률분포 모양을 이룰 것이다. 

$\Rightarrow$ 라인플롯에서 선을 제거하고 점들만 남기면 분명히 이론적 확률분포함수 모양 그대로 나올 것이다. 

그래서 실행해봤다. 

```python
rv = sp.stats.norm()
s = rv.rvs(1000, random_state=0)
pdf =rv.pdf(s)

plt.plot(s, pdf, 'ro')
```
<img width="622" alt="Screen Shot 2021-08-27 at 11 42 47" src="https://user-images.githubusercontent.com/83487073/131063566-53651ccb-90ea-4c5e-94bc-f4c9a5b52dc5.png">

## 역시!!! 무릎을 탁 쳤다. 십년묵은 체중이 싹 내려가는 기분. 

그러면 선이 있을 때는 왜 저런 모양이 나오는가? 

그건 라인플롯이 그려지는 원리와 관련이 있었다. 

```python
def f(x) : 
    return x**2+3*x+7
xx = np.linspace(-3,3,50)
plt.plot(xx, f(xx), 'ro-')
```
<img width="656" alt="Screen Shot 2021-08-27 at 11 45 29" src="https://user-images.githubusercontent.com/83487073/131063818-5991b1e9-e76e-42b2-9141-2f7f6af920f4.png">

점을 하나하나찍어 라인플롯을 그린다고 사고실험해보자. 

점 하나하나를 벡터로 보자. 

벡터 x좌표는 위 코드에서 xx 배열 원소 하나하나에 해당한다. 

그리고 벡터 y 좌표는 위 코드에서 f(xx) 값 하나하나에 해당한다. 

라인플롯은 점 하나하나를 순차적으로 찍어가면서, 찍히는 순서에 맞춰서 각 점을 선으로 연결한 거였다. 

무슨 말인가. 

예를 들어 위 라인플롯은 

```python
np.dstack([xx, f(xx)])
```
<img width="308" alt="Screen Shot 2021-08-27 at 11 48 54" src="https://user-images.githubusercontent.com/83487073/131064127-057e74a5-71d8-4fe9-9d6f-c653facf5f40.png">

위 배열의 각 행벡터들을 점으로 찍어 그린 것이다. 

플롯을 그린다 하고, 맨 처음 벡터 $[-3, 7]$ 을 점으로 찍었다. 

그리고 두번째로 배열의 2행 $[-2.87, 6.64]$ 를 점으로 찍고, $[-3, 7]$ 에서 출발해서 $[-2.87, 6.64]$ 까지를 선으로 잇는다. 

그리고 다시 3행에 있는 벡터를 점으로 찍고, 두번째로 찍었던 벡터와 선으로 연결한다. 

이렇게 점-선-점-선 이 연결되어서 라인플롯이 되는 것이다. 

그런데 시뮬레이션으로 뽑은 정규분포 표본들은 순서가 뒤죽박죽이었다. 

기댓값 근처 값이 나오기도 했다가, 분포 꼬리 쪽 값이 나오기도 하고. 적어도 시뮬레이션 표본들을 담고 있는 배열 안에서는 순서가 뒤죽박죽이었다. 

자, 이 뒤죽박죽인 점들을 가지고 점을 찍는다 생각해보자. 

```python
rv = sp.stats.norm()
s = rv.rvs(10, random_state=0)
pdf =rv.pdf(s)

plt.plot(s, pdf, 'ro-')
```
## 표본 10개 일 때
<img width="739" alt="Screen Shot 2021-08-27 at 11 58 36" src="https://user-images.githubusercontent.com/83487073/131064959-a81fdc64-2253-4709-b9f4-1bfeb92c1ea8.png">

순서가 뒤죽박죽이니 선을 그을 때 상하좌우 종횡무진하면서 선을 그을 것이다. 

## 표본 50개 일 때
<img width="662" alt="Screen Shot 2021-08-27 at 11 59 37" src="https://user-images.githubusercontent.com/83487073/131065034-f8606b6d-7b47-48db-bdde-d4748d82ba13.png">

자, 그래프 상에 점들을 잘 보자. 

분명 각 점들은 정규분포 형상을 분명하게 갖춰가고 있다. 

하지만 선들은 여전히 일정한 방향성을 잃은 채 계속 교차하고 있다. 

## 표본 100개 일 때
<img width="661" alt="Screen Shot 2021-08-27 at 12 02 38" src="https://user-images.githubusercontent.com/83487073/131065349-5e2a148c-9ef1-4541-8122-736c7b4e0275.png">

## 표본 1000개 
<img width="664" alt="Screen Shot 2021-08-27 at 12 03 16" src="https://user-images.githubusercontent.com/83487073/131065392-474d813e-23af-46d3-a229-b9450525824b.png">

...해서 맨 처음의 희한한 형상이 만들어진 거였다. 

결론을 내려보자. 

## 결론 
맷플롯립으로 확률분포함수 그릴 때는 

- 시뮬레이션해서 얻은 샘플 써서는 제대로 표현이 안 된다. 

- 정렬된 벡터공간 좌표들(예 : np.linspace())을 써야 한다. 

---




---
title : "[수학/확률과 통계] 결합확률, 조건부확률, 베이즈정리, pgmpy, 몬티 홀 문제"
excerpt : "확률과 통계 복습 후 기록"

categories : 
- Data Science
- python
- mathematics

tags : 
- [datascience, python, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-09-20
last_modified_at : 2021-09-21

---

# 결합확률 

## 정의 : 

사건 A와 사건 B가 동시에 발생하는 사건의 확률. (A,B 교집합의 확률)

$=$ $P(A,B)$

- 사건 A의 원소와 사건 B의 원소로 구성된 벡터의 확률이기도 하다. 
- 2차원 벡터들의 결합확률분포는 3차원 모양 함수로 표현된다. 

---

# 주변확률

## 정의 : 

개별 사건 A,B의 확률. 

$= P(A), P(B)$

- X,Y 확률변수의 결합확률분포함수에서, X의 주변확률분포는 각 X값에 대한 모든 Y값을 더한 값들의 분포다. 

---

# 조건부확률

## 정의 : 

사건 B를 새 전체집합으로 삼았을 때, 사건 A의 확률

또는 

조건 B가 있을 때, A의 확률. 

$= P(A\vert{B})$

- 조건부확률분포는 특정 값에서의 결합확률분포 단면을 특정 값에서의 주변확률분포값 으로 나눈 값들의 분포다. 

예) $P(X\vert{Y=3}) = \frac{P(X, Y=3)}{P(Y=3)}$

---

# 사건의 독립 (두 사건이 서로 영향 미치지 않는다)

사건 A,B 사이에 다음 관계가 성립하면, '사건 A,B는 서로 독립이다' 라고 한다. 

## $P(A,B) = P(A)P(B)$

- 사건 A,B가 독립이면 조건부확률이 원래 확률과 같아진다. 

$P(A\vert{B}) = P(A)$

즉, 사건 B가 사건 A 확률값에 아무 영향 못 미친다는 뜻이다. 

---

# 사슬법칙 

## 정의 : 

결합확률과 조건부확률 사이 관계를 나타내는 법칙이다. 

## $P(A,B,C,D) = P(A\vert{B,C,D})P(B\vert{C,D})P(C\vert{D})P(D)$

- 다양한 곳에서 응용이 가능한, 매우 유용한 법칙이다. 

사건 A,B,C,D가 서로 독립이면, 다음이 성립한다. 

## $P(A,B,C,D) = P(A)P(B)P(C)P(D)$

---

# 피지엠파이 패키지로 확률모형 시각화 하기 

pgmpy 패키지를 사용하면 확률모형을 시각화 할 수 있다. 

## - 결합확률모형 시각화 하기 

pgmpy 패키지의 JointProbabilityDistribution 클래스를 사용하면, 결합확률모형을 시각화 할 수 있다. 

```python
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

px = JPD(variables, cardinality, values)
```
- variables : 결합확률분포 모형 구성하는 확률변수들 이름 '리스트' 넣는다. 
- cardinality : 각 확률변수 가능한 표본 수를 '리스트' 형식으로 넣는다. 
- values : 확률변수의 모든 표본 조합에 대한 (결합)확률값의 배열

결합확률모형 시각화 예) 

```python
# 확률변수 X,Y의 주변확률분포 시각화 하기 
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

px = JPD(['X'], [2], np.array([12,8])/20)
print(px)

py = JPD(['Y'], [2], np.array([5,5])/10)
print(py)
```

<img width="204" alt="Screen Shot 2021-09-21 at 15 37 57" src="https://user-images.githubusercontent.com/83487073/134123219-9b76b636-cda7-47c6-a496-4586e92adc7a.png">

<img width="187" alt="Screen Shot 2021-09-21 at 15 38 15" src="https://user-images.githubusercontent.com/83487073/134123252-c2186d51-bbd9-4e34-898a-e7fcd418a027.png">

```python
# X,Y 확률변수의 결합확률분포 시각화 
pxy = JPD(['X','Y'], [2,2], np.array([3,9,7,1])/20)
print(pxy)
```
<img width="226" alt="Screen Shot 2021-09-21 at 15 40 44" src="https://user-images.githubusercontent.com/83487073/134123537-9077448c-540f-483d-83cc-8c7d159f697a.png">

---

## - 결합확률분포에서 주변확률분포, 조건부확률분포 구해서 시각화 하기 

- X의 주변확률분포는 X 각 점을 기준으로 Y와의 가능한 모든 결합확률을 더한 값들의 분포다. 
- X의 조건부확률분포는 Y값 고정한 결합확률분포 단면 / 고정한 Y값에서의 주변확률분포값 의 분포다. 

## 사용하는 메서드 

주변확률분포 구하는 메서드 

- marginal_distribution('주변확률분포 구할 확률변수 이름 리스트', inplace=JPD 객체를 주변확률분포 객체로 대체할 지 말지 입력)
- maginalize('주변화시킬(제거할) 확률변수 이름 리스트', inplace=)

조건부확률분포 구하는 메서드 
- conditional_distribution(values, inplace=)

values : 조건으로 삼을 확률변수 이름과 고정할 값 튜플로 묶은 것들의 리스트

---

## - 결합확률분포에서 두 확률변수가 서로 독립인지 확인하기 
- check_independence(\['확률변수 1 이름'\],\['확률변수 2 이름'\])

---

사용 예) 

```python
# 확률변수 X,Y의 결합확률분포 객체 정의
pxy2 = JPD(['X','Y'], [2,2], np.array([3,3,2,2])/10)
print(pxy2)
```
<img width="221" alt="Screen Shot 2021-09-21 at 15 53 25" src="https://user-images.githubusercontent.com/83487073/134124979-09a45fff-b134-4f5d-8903-3a0c45beb2f4.png">

```python
# 결합확률분포의 두 확률변수 X,Y가 서로 독립인가? 
print(pxy2)
ind = pxy2.check_independence(['X'],['Y'])
print(f'확률변수 X,Y 는 서로 독립인가? : {ind}') # 확률변수 X,Y는 서로 독립이다. 
```
True 

X,Y 확률변수는 서로 독립이다. 

```python
# x=0 일 때 Y의 조건부확률분포
print('X=0 일 때 Y의 조건부확률분포')
print(pxy2.conditional_distribution([('X',0)], inplace=False)) # X = 0 일 때 Y의 조건부확률분포
```
<img width="234" alt="Screen Shot 2021-09-21 at 16 02 45" src="https://user-images.githubusercontent.com/83487073/134126153-733b60e5-78aa-4cd7-8c46-46fdb8473389.png">

```python
# x=1 일 때 Y의 조건부확률분포
print('X=1 일 때 Y의 조건부확률분포')
print(pxy2.conditional_distribution([('X',1)], inplace=False)) # X = 1 일 때 Y의 조건부확률분포 
print()
```
<img width="227" alt="Screen Shot 2021-09-21 at 16 03 15" src="https://user-images.githubusercontent.com/83487073/134126194-e8237071-e850-4be1-8fad-17035af89b29.png">


```python
# Y의 주변확률분포
print('Y의 주변확률분포')
print(pxy2.marginal_distribution(['Y'], inplace=False))
print()
```
<img width="235" alt="Screen Shot 2021-09-21 at 16 03 37" src="https://user-images.githubusercontent.com/83487073/134126232-6e3cc98f-1675-4fb3-82f2-83be27d374ec.png">

```python
# Y=0 일 때 X의 조건부확률분포
print('Y=0 일 때 X의 조건부확률분포')
print(pxy2.conditional_distribution([('Y',0)], inplace=False))
```
<img width="229" alt="Screen Shot 2021-09-21 at 16 05 35" src="https://user-images.githubusercontent.com/83487073/134126439-d1c61b15-9244-4648-96cf-e05b9f57a2b9.png">

```python
# Y=1 일 때 X의 조건부확률분포
print('Y=1 일 때 X의 조건부확률분포')
print(pxy2.conditional_distribution([('Y',1)], inplace=False))
print()
```
<img width="217" alt="Screen Shot 2021-09-21 at 16 06 25" src="https://user-images.githubusercontent.com/83487073/134126559-360cfadf-96a8-4d16-be3f-4a3e36164025.png">

```python
# X의 주변확률분포
print('X의 주변확률분포')
print(pxy2.marginal_distribution(['X'], inplace=False))
print()
```
<img width="191" alt="Screen Shot 2021-09-21 at 16 07 15" src="https://user-images.githubusercontent.com/83487073/134126648-5bfa6243-099f-4746-94cc-d9425dd265cc.png">

---

## marginalize()를 사용하는 경우

```python
# Y를 주변화하고 X의 주변확률분포 구하기 
py = pxy2.marginalize(['Y'], inplace=False) # X의 주변확률분포 
print(py)
```
<img width="152" alt="Screen Shot 2021-09-21 at 16 09 39" src="https://user-images.githubusercontent.com/83487073/134126943-7f193239-9a3c-4f30-816e-d61b7d513820.png">


```python
# X를 주변화 하고 Y의 주변확률분포 구하기 
px = pxy2.marginalize(['X'], inplace=False) # Y의 주변확률분포 
print(px)
```
<img width="195" alt="Screen Shot 2021-09-21 at 16 10 04" src="https://user-images.githubusercontent.com/83487073/134127002-b820f961-dfce-441f-9afc-957422e800d7.png">

---

# 베이즈 정리 

## 정의 : 

사건 B라는 새로운 정보가 추가로 들어왔을 때, 사건 A의 사전확률이 어떻게 바뀌는가?(사건 A의 사후확률)

## $P(A\vert{B}) = \frac{P(B\vert{A})P(A)}{P(B)}$

조건부확률의 응용버전.
- $P(B\vert{A}) =$ 가능도
- $P(A)=$ 사전확률
- $P(A\vert{B})=$ 사후확률
- $P(B) =$ 정규화 상수 (중요도 낮음. 크기 조절 역할만 한다)

---

# 베이즈 정리의 확장 1

사건 $A_{i}$ 가 

- $A_{i}\cap{A_{j}}= \varnothing$
- $A_{1}\cup{A_{2}}...\cup{A_{n}} = \Omega$

일 때 전체확률의 법칙을 베이즈정리에 적용할 수 있다. 

사건 B가 표본공간 $\Omega$ 의 또 다른 부분집합일 때,

## $P(A_{i}\vert{B})$ 

$= \frac{P(B\vert{A_{i}})P(A_{i})}{P(B)}$

$= \frac{P(B\vert{A_{i}})P(A_{i})}{\sum{P(B\cap{A_{i}})}}$

## $= \frac{P(B\vert{A_{i}})P(A_{i})}{\sum{P(B\vert{A_{i}})P(A_{i})}}$

가 성립한다. 

위 식은 '다중분류 문제'에서 확용할 수 있다. 

---

## 다중분류 문제 

여러 배타적이고 완전한 사건 중에서 가장 확률이 높은 사건 하나를 고르는 문제가 '다중분류 문제'다. 

정답이 무조건 하나인 4지 선다형 객관식 문제를 푼다고 생각해보자. 정답이 될 수 있는 보기는 1,2,3,4 이다. 

'정답' 확률변수의 가능한 4개 표본(1,2,3,4) 하나하나를 그 표본으로만 구성된 단순사건이라 생각하자. 

그리고

$A_{1} = [1]$

$A_{2} = [2]$

$A_{3} = [3]$

$A_{4} = [4]$

로 생각하자. 

$A_{1}$ ~ $A_{n}$ 의 사건은 각각 1,2,3,4 가 정답인 사건이다. 

문제의 가능한 정답은 무조건 1,2,3,4 중에 있다. 따라서 $A_{1}$ ~ $A_{n}$ 사건의 합집합은 전체 표본공간을 구성한다. 

$\Rightarrow$ $A_{1}\cup{A_{2}}\cup{A_{3}}\cup{A_{4}} = \Omega$

그리고 각 단순사건은 교집합이 공집합이다. 

$\Rightarrow$ $A_{1}\cap{A_{2}}\cap{A_{3}}\cap{A_{4}} = \varnothing$

이때, 선생님께서 정답에 대한 힌트를 주셨다고 하자. 힌트에 대한 사건을 B 라고 하자. B는 표본공간 $\Omega$ 의 또 다른 부분집합이다. 

보기 1,2,3,4 중에서 주어진 힌트와 가장 비슷한 보기가 정답일 확률이 가장 높을 것이다. 

그러면 힌트가 주어졌을 때, 1,2,3,4 각각이 정답인 사건의 확률을 구해보자. 

$P(A_{1}\vert{B}) = P(B\vert{A_{1}})P(A_{1})$

$P(A_{2}\vert{B}) = P(B\vert{A_{2}})P(A_{2})$

$P(A_{3}\vert{B}) = P(B\vert{A_{3}})P(A_{3})$

$P(A_{4}\vert{B}) = P(B\vert{A_{4}})P(A_{4})$

분모는 모두 같아서 생략했다. 

$A_{1}, A_{2}, A_{3}, A_{4}$ 는 서로 배타적이며, 이들은 완전하다. 

이 문제는 힌트 B가 주어졌을 때, 서로 배타적이고 완전한 $A_{i}$ 중 확률이 가장 높은 $A_{i}$ 를 고르는 문제다. 

=  다중분류 문제다. 

이 다중분류 문제의 '정답'은 위 조건부확률값이 가장 큰 $A_{i}$ 로 분류된다. 

---

## 이진분류 문제 

만약 정답이 두 개의 서로 배타적이고 완전한 사건 중 확률값 더 높은 걸로 분류된다면, 이진분류 문제라고 한다. 

$A \cup{A^{C}} = \Omega$

$A \cap{A^{C}} = \varnothing$

$\Rightarrow$ 정답이 $A$ 또는 $A^{C}$ 로 분류된다면 '이진분류 문제'다. 

위 문제를 이진분류 문제로 바꾸면, 

$P(A\vert{B}) = P(B\vert{A})P(A)$

$P(A^{C}\vert{B}) = P(B\vert{A^{C}})P(A^{C})$

힌트 B가 있을 때 $A$ 또는 $A^{C}$ 중에 확률값이 더 높은걸로 '정답'을 분류하면 된다. 

곧, 위 조건부확률값이 더 큰 사건($A$ or $A^{C}$) 으로 '정답'을 분류하면 된다.

---

# 베이즈 정리의 확장 2

A에 추가 정보 B가 반영되고 나서 또다시 C가 반영되는 경우, A 사후확률은 다음과 같이 계산할 수 있다. 

$P(A\vert{B,C}) = \frac{P(C\vert{A,B})P(A\vert{B})}{P(C\vert{B})}$

또는 추가 정보 C가 반영되고 나서 또다시 B가 반영되는 경우, A 사후확률은 다음과 같다. 

$P(A\vert{B,C}) = \frac{P(B\vert{A,C})P(A\vert{C})}{P(B\vert{C})}$

증명은 결합확률과 조건부확률사이 사슬법칙을 사용해서 할 수 있다. 

$\Leftarrow P(A,B,C) = P(A\vert{B,C})P(B\vert{C})P(C)$

## 다양한 사전확률에 대해 사후확률 계산할 수 있다는 게 핵심 포인트다. 

---

# 몬티 홀 문제. 다중분류 문제로 풀기 

몬티홀 문제를 다중분류 문제로 풀어보자. 

내가 분류하고자 하는 건 '자동차의 위치'다. 

자동차의 위치 확률변수를 $C$ 라 하자. 

자동차의 위치는 0번 문, 1번 문, 2번 문 셋 중에 있다. 

자동차가 0번 문에 있는 사건을 $C_{0} (C=0)$

자동차가 1번 문에 있는 사건을 $C_{1} (C=1)$

자동차가 2번 문에 있는 사건을 $C_{2} (C=2)$

이라고 하자. 

$C_{0},C_{1},C_{2}$ 은 서로 배타적이고(교집합이 공집합), 완전하다(세 집합 합집합은 전체집합 $\Omega$ 다). 

자동차의 위치는 $C_{0}, C_{1}, C_{2}$ 셋 중 하나로 분류되므로, '다중분류 문제'로 풀 수 있다. 

한편, '내가 선택한 문' 확률변수를 $X$ 라고 하자. 

$X$ 가 가질 수 있는 표본도 0,1,2 이다. 

'사회자가 선택한 문' 확률변수를 $H$ 라 하자. 

$H$ 가 가질 수 있는 표본도 0,1,2 이다. 

---

맨 처음에 자동차가 0,1,2 각각에 있을 수 있는 확률은

$P(C_{0}) = \frac{1}{3}$ 

$P(C_{1}) = \frac{1}{3}$

$P(C_{2}) = \frac{1}{3}$

이다.

게임을 하면서 내가 문을 하나 선택하고, 사회자가 문을 선택해서 염소를 보여줄 것이다. 

내가 문을 고르고, 사회자가 염소를 보여줬을 때, $C_{0}, C_{1}, C_{2}$ 의 확률이 각각 얼마가 될 지 보자. 

$\Rightarrow$ $P(C_{i} \vert{H_{j}, X_{a}})$

예를 들어 내가 2번문을 골랐다. 사회자가 1번문을 열어 염소를 보여줬다. 이때 0,1,2 번 문 각각에 자동차가 있을 확률은 얼마일까?

### 자동차가 0번 문에 있을 확률)

## $P(C_{0}\vert{H_{1}, X_{2}}) = \frac{2}{3}$

### 자동차가 1번 문에 있을 확률) 

사회자가 1번문을 열여서 염소가 있다는 걸 보여줬기 때문에, 1번 문 뒤에 자동차가 있을 수. 없다. 

## $P(C_{1}\vert{H_{1}, X_{2}}) = 0$

### 자동차가 2번 문에 있을 확률) 

$C_{0} \cup C_{1} \cup C_{2} = \Omega$ 였다. 또, $C_{0} \cap C_{1} \cap C_{2} = \varnothing$ 이었다. 

따라서 $P(C_{0})+P(C_{1})+P(C_{2}) = 1$ 이다. 

조건부확률일 때도 똑같이 성립한다. 따라서, $1-\frac{2}{3} = \frac{1}{3}$ 이다. 

## $P(C_{2}\vert{H_{1}, X_{2}}) = \frac{1}{3}$

---

### 결론 1 : 다중분류 문제 정의에 충실하게 

다중분류 문제에서, '자동차의 위치'는 $C_{0}, C_{1}, C_{2}$ 중에서 가장 확률값이 큰 사건으로 분류된다. 

세 조건부확률 중 $P(C_{0}\vert{H_{1}, X_{2}})$ 의 확률값이 가장 크다. 

따라서 '자동차의 위치'는 $C_{0} =$ '0번 문 뒤에 자동차가 있다' 로 분류된다. 

내가 처음 선택했던 문은 1번 문 이었다. 다중분류 문제 분류 결과, 1번 문 뒤에는 자동차가 없었다. 

분류결과에 따르면 자동차가 있는 것은 0번 문이었다. 따라서 자동차가 당첨되기 위해서는 1번 문에서 0번 문으로 '선택을 바꿔야 한다'. 

---

### 결론 2 : 단순 확률값 비교해서 

위 세개 조건부 확률은 각각 최종으로 0번문, 1번문, 2번문 뒤에 자동차가 있을 확률이다. 

내가 선택했던 문은 1번 문이었다. 1번문의 확률값은 $\frac{1}{3}$ 이었다. 

한편, 0번 문에 자동차가 있을 확률은 $\frac{2}{3}$ 이었다. 

0번 문에 자동차가 있을 확률이 더 높으므로, 나는 '선택을 바꾸는 것이 자동차 당첨에 유리하다'. 

---

# 몬티 홀 문제를 pgmpy 베이지안 모형으로 풀어보자. 

몬티 홀 문제를 pgmpy 베이지안 모형 이용해서 풀어보자. 

내가 선택을 바꿀 지, 바꾸지 말 지 결정하기 위해 필요한 확률값은 다음 세 가지다. 

$P(C_{0}\vert{H_{1}, X_{2}})$

$P(C_{1}\vert{H_{1}, X_{2}})$

$P(C_{2}\vert{H_{1}, X_{2}})$

이 세 가지 확률을 베이지안 모형 패키지 사용해서 구하자. 

## 사전확률을 정의하자

먼저 사전확률을 정의해야 한다. 

사전확률은 자동차의 위치 $C_{0}, C_{1}, C_{2}$ 의 확률이다. 

사전확률 값이 모두 $\frac{1}{3}$ 이다. 

이제 피지엠파이 TabularCPD 클래스에 위 값들을 넣고, 사전확률 객체를 정의하자. 

```python
from pgmpy.factors.discrete import TabularCPD

# 사전확률 정의하자
pre = TabularCPD('C', 3, np.array([[1/3],[1/3],[1/3]]))
print('사전확률')
print(pre)
```
<img width="166" alt="Screen Shot 2021-09-21 at 21 10 02" src="https://user-images.githubusercontent.com/83487073/134167899-3782d431-81ba-4253-877d-ff919018ec85.png">

## 가능한 모든 가능도를 정의하자. 

사전확률을 정의했다. 그러면 이제 모든 조건에 대해 가능한 모든 가능도를 정의하자. 

$P(H,X\vert{C})$ 에 대해 가능한 모든 경우를 정의할 것이다. 

결합사건 (H,X) 는 하나의 문자 G 로 통일했다. 

$\Rightarrow$ $P(G\vert{C})$

$C$ 를 이용해 정의할 수 있는 가능한 모든 조건은 $C_{0},C_{1}, C_{2}$ 이다. 

각각의 $C_{i}$ 에 대해 $G$ 는 총 9가지 조건이 가능하다. $(H_{0}X_{0}, H_{1}X_{1},....H_{2}X_{2})$

이걸 생각하면서 가능도 객체를 정의하자. 

```python
# 가능한 모든 가능도 정의
values = np.array([
    [0,0,0],
    [0,1/6, 1/3],
    [0,1/3, 1/6],
    [1/6, 0, 1/3],
    [0,0,0],
    [1/3, 0, 1/6],
    [1/6, 1/3, 0],
    [1/3, 1/6, 0],
    [0,0,0]
])
likelihood = TabularCPD('G', 9, values, evidence=['C'], evidence_card=[3])
print('가능도')
print(likelihood)
print()
```
위에서 ndarray 객체 values 는 각 경우에 대해 직접 수기계산해서 확률값을 넣었다.

<img width="569" alt="Screen Shot 2021-09-21 at 21 16 33" src="https://user-images.githubusercontent.com/83487073/134168807-387df704-44ba-4091-b66b-2f2fbc4878c3.png">

정의된 가능도 객체는 위 이미지와 같다. 

$G$ 는 결합사건 $(H_{i},X_{j})$ 을 의미한다. 

## 베이지안 모형 만들기 

이제 그러면 베이지안 모형 객체를 생성하고, 앞에서 만든 사전확률과 가능도 객체를 넣어주자. 

```python
# 베이지안 모형 만들기 
from pgmpy.models import BayesianModel

model = BayesianModel([('C','G')]) # 확률변수 이름 순서도 중요하다. 
model.add_cpds(pre, likelihood) # 사전확률, 가능도 객체 투입!
model.check_model() # 모델이 정상으로 생성되었는지 점검
```
True 

모델이 정상으로 생성되었다. 

이제 추정 객체를 사용해서, 내가 알고싶은 위 조건부 확률(C의 사후확률) 3개를 구하자. 

```python
# 베이지안 모형 이용해서 조건부확률 계산
from pgmpy.inference import VariableElimination

infer = VariableElimination(model) # 추정객체
post = infer.query(['C'], evidence={'G': 5}) # G = 5 일 때(H1,X2), C에 대한 사후확률 계산
print(post)
```
<img width="186" alt="Screen Shot 2021-09-21 at 21 21 30" src="https://user-images.githubusercontent.com/83487073/134169447-2932e60e-dc6f-4c31-a0e1-418c63e812a5.png">

위에서 부터 차례로 

$P(C_{0}\vert{H_{1}, X_{2}})$

$P(C_{1}\vert{H_{1}, X_{2}})$

$P(C_{2}\vert{H_{1}, X_{2}})$

이다. 

내가 선택한 2번 문 뒤에 자동차가 있을 확률은 0.33 이고, 남은 0번 문 뒤에 자동차가 있을 확률은 0.666 이다. 

따라서 선택을 바꾸는 게 '자동차 당첨에 유리하다'.





























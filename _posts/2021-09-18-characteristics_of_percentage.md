---
title : "[수학/확률과 통계] 확률의 성질, 확률분포함수"
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

date : 2021-09-19
last_modified_at : 2021-09-19

---

# 확률의 성질 

## 1. 공집합의 확률

공집합의 확률은 0 이다. 

$P(\varnothing) = 0$

## 2. 여집합의 확률

여집합의 확률은 1 - 사건의 확률 과 같다. 

$P(A^{C}) = 1 - P(A)$

한편 여기에 콜모고로프의 정리 1번, $0 \leq P(A)$ 를 적용하면,

$0 \leq P(A) \leq 1$ 이 성립한다. 

### $\Rightarrow 0 \leq P(A) \leq 1$

## 3. 포함-배제 원리 (덧셈규칙)

$P(A\cup{B}) = P(A) + P(B) - P(A\cap{B})$

## 4. 전체 확률의 법칙

조건 : 

- $C_{i} \cap{C_{j}} = \varnothing$
- $C_{1}\cup{C_{2}}\cup{C_{3}}...\cup{C_{n}} = \Omega$ 일 때,

$P(A) = \sum{P(A\cap{C_{i}})}$

---

# 확률분포함수 

## 확률분포

## 정의 : 

확률값이 어디에, 얼마나 분포되어 있는지 나타낸 것을 확률분포라 한다. 

## 확률분포함수 

## 정의 : 

확률분포를 묘사해주는 함수를 확률분포함수라고 한다. 

## 확률분포함수 종류 

- 확률질량함수
- 확률밀도함수
- 누적분포함수

---

## 단순사건 

## 정의 : 

표본 1개로만 구성된 사건을 '단순사건' 이라 한다. 

---

# 확률질량함수 (pmf)

## 정의 : 

단순사건에 대한 확률값을 정의하는 함수를 '확률질량함수' 라고 한다. 

또는 

## 표본 1개에 대해 확률값을 정의하는 함수를 '확률질량함수' 라고 한다.


확률질량함수 예) 

```python
# 확률질량함수 예
xx = np.arange(1,7)
yy = [0.1]*5+[0.5]

plt.stem(xx, yy)
plt.xlim(0, 7)
plt.ylim(-0.01, 0.6)
plt.xlabel('주사위 눈금')
plt.ylabel('주사위 각 눈이 나올 확률')
plt.title('6이 잘 나오도록 조작된 주사위 확률분포의 확률질량함수')
plt.show()
```
<img width="630" alt="Screen Shot 2021-09-19 at 22 08 02" src="https://user-images.githubusercontent.com/83487073/133928681-cc9c889c-0e65-4fc2-8b79-126612a2115b.png">

---

## 확률함수가 표본이 아닌 사건에 대해서만 확률을 정의할 수 있는 이유

표본공간의 표본 수가 유한한 경우, 확률질량함수 이용하면 표본 하나하나에 대해서 확률값 정의할 수 있다. 

또는 확률질량함수 이용하면 표본 하나만 포함된 단순'사건'에 대한 확률값 정의할 수 있다. 

한편,

표본공간의 표본 수가 무한한 경우, 표본 하나하나에 대해서 확률값 정의할 수 없다. 

대신, 구간으로 설정된 '사건' 하나 하나에 대해서는 확률값 정의할 수 있다. 

확률함수가 언제나 확률값을 정의할 수 있으려면, 표본공간 표본 갯수가 유한하든. 무한하든 확률값 정의할 수 있어야 한다. 

## '사건'에 대해 확률을 할당하면 표본공간 표본 갯수가 유한하든. 무한하든. 상관 없이 언제나 확률값 정의할 수 있다. 

$\Rightarrow$ 이 같은 이유로 확률함수는 표본이 아닌 '사건'을 입력으로 받아 확률값을 할당한다. 

---

# 누적분포함수 (cdf)

## 정의 : 

$P([-\infty < X \leq x])$ 를 정의하는 함수. 

또는 

특수구간 $[-\infty < X \leq x]$ 의 확률 $P([-\infty < X \leq x])$ 를 구하는 함수다. 

## 특징 : 

1. $F(-\infty) = 0$
2. $F(\infty) = 1$
3. 누적분포함수는 전 구간에서 단조증가한다. 따라서 $x \leq y \Rightarrow F(x) \leq F(y)$ 

$\Rightarrow$ 누적분포함숫값은 0에서 시작해서 계속 증가하면서 1로 다가간다. 절대 감소하지 않는다. 

---
구간 $[a < x \leq b]$ 의 확률 $P(a,b)$ 는 누적분포함수를 이용하면 다음과 같이 정의할 수 있다. 

## $P(a,b) = F(b)-F(a)$

---

누적분포함수 예) 

```python
xx = np.linspace(-100,500, 1000)
ld2 = lambda x : 0 if x < 0 else (2/3)*(x/180) if 0 <= x <= 180 else (2/3)+(1/3)*((x-180)/180) if 180 < x < 360 else 1
yy = [ld2(x) for x in xx]
plt.plot(xx, yy)
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.title('조작된 원반의 누적분포함수')
plt.xticks([0, 180, 360])
plt.ylim(-0.1, 1.1)
plt.xlim(-100, 500)
```
<img width="630" alt="Screen Shot 2021-09-19 at 22 26 10" src="https://user-images.githubusercontent.com/83487073/133929274-c922ce78-5fef-4e01-9121-8ab324152961.png">

---

# 확률밀도함수 (pdf)

## 정의 : 

각 위치(점)에서의 '상대적 확률값' 묘사하는 확률분포함수다. 

$\Leftarrow$ 각 점에서의 '확률값'이 아니다! 

또한 각 개별구간(사건)의 확률 나타내는 함수다. 

## 특징 : 

- 누적분포함수 $F$ 의 1차 도함수 $=$ 확률밀도함수다. 

또는 누적분포함수 $F$ 각 지점에서의 기울기 $=$ 각 지점에서의 확률밀도함숫값 이다. 

- 확률밀도함수 면적 = 그 구간(사건)의 확률이다. 

- ### 누적분포함수는 단조증가 함수다. 기울기는 항상 0 또는 양수다. 따라서 항상 $0 \leq p(u)$ 이다. 

- 확률밀도함수를 $-\infty$ 부터 $\infty$ 까지 적분한 값은 1이다. (전체의 확률 = 1) 

### $\int_{-\infty}^{\infty}{p(u)du} = 1$

## 누적분포함수와의 관계 

미적분학의 기본정리 이용하면 확률밀도함수 면적(확률값)을 누적분포함수 이용해서 구할 수 있다. 

## $F(b) - F(a) = \int_{a}^{b}{p(u)du}$

---

확률밀도함수 예)

```python
# 확률밀도함수 예
xx = np.linspace(-100,500,1000)
ld2 = lambda x : 0 if x < 0 else 1/270 if 0 <= x <= 180 else 1/540 if 180 < x < 360 else 0
yy = np.array(list(map(ld2, xx)))

plt.plot(xx, yy)
plt.xlabel('$x$')
plt.ylabel('$pdf$')
plt.title('조작된 원반의 확률밀도함수')
plt.xticks([0, 180, 360])
plt.show()
```
<img width="630" alt="Screen Shot 2021-09-19 at 22 42 21" src="https://user-images.githubusercontent.com/83487073/133929743-911d1686-1447-4e2e-a2f8-dea8be195db2.png">


















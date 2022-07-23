---
title : "[수학/미적분] 범함수, 변분법, 최적제어 개념"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- mathematics

tags : 
- [datascience, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-08-30
last_modified_at : 2021-08-30

---

# 범함수 

## 정의 : 

## 함수를 입력으로 받아 스칼라 출력하는 '함수' 

범함수 또한 함수의 한 종류다. 

일반적인 함수는 실수를 입력받아 실수를 출력한다. 


## 표기 : 

알파벳 대문자와 대괄호로 범함수 표기한다. 

예) 엔트로피 $H[p(x)]$, 기댓값 $E[p(x)]$ 등 

대괄호 안에는 함수를 입력으로 받는다. 

---

# 범함수의 테일러 전개 

## 테일러 전개

어떤 함수 $f(x)$ 가 있다고 하자. 

이 함수의 도함수 또는 특정 점에서의 기울기를 알면, 

함수 $f(x)$ 의 근사함수를 구할 수 있다. 

$f(x) \approx f(x_{0}) + \frac{df(x_{0})}{dx}(x-x_{0})$ 

$f(x,y) \approx f(x_{0}, y_{0}) + \frac{\partial{f(x_{0}, y_{0})}}{\partial{x}}(x-x_{0})+\frac{\partial{f(x_{0}, y_{0})}}{\partial{y}}(y-y_{0})$

## 이렇게 함수 $f$의 도함수 또는 기울기 만으로 

## $f$의 근사함수를 구하는 과정 또는 그 결과를 테일러 전개 라고 한다. 

---

## 범함수의 테일러 전개 

그러면 범함수의 근사식은 어떻게 나타낼 수 있을까? 

우선, 일반적인 함수의 근사식을 조금 변형시켜 보자. 

$f(x) \approx f(x_{0}) + \frac{df(x_{0})}{dx}(x-x_{0})$ 

위 식에서 $x \Rightarrow x+\epsilon$ , $x_{0} \Rightarrow x$ 로 문자를 바꾸자. 

$\epsilon$ 은 매우 작은 어떤 상수다. 

그러면 

$f(x+\epsilon) \approx f(x) + \frac{df}{dx}\epsilon$

이 된다. 

이 식을 가지고 범함수의 테일러 전개 식을 유도할 것이다. 

함수 $f$ 를 함수 $F$로 기호를 바꾸자. 

변수 $x$ 는 $y$ 로 기호를 바꾸자. 

똑같은 함수 인데 기호만 각각 바꾼 것이다. 


$\Rightarrow F(y+\epsilon) \approx F(y) + \frac{dF}{dy}\epsilon$

만약 함수 $F$ 가 스칼라 $x_{0}, x_{1}, x_{2} ...$ 를 입력으로 받는 다변수함수라면, 어떻게 테일러 전개 할 수 있을까? 

$F(y_{0}+\epsilon_{0}, y_{1}+\epsilon_{1}, y_{2}+\epsilon_{2}, ... y_{n}+\epsilon_{n}) \approx F(y_{0}, y_{1}, y_{2}, ... y_{n}) + \frac{\partial F}{\partial y_{0}}\epsilon_{0} + \frac{\partial F}{\partial y_{1}}\epsilon_{1}+...\frac{\partial F}{\partial y_{n}}\epsilon_{n}$

이런 식으로 전개 될 것이다. 

여기서 $y_{i}$ 를 어떤 실수 $x_{i}$ 를 입력으로 받는 어떤 단변수함수 $y(x_{i})$ 의 출력이라고 보자. 

또 $\epsilon_{i}$ 를 어떤 실수 $x_{i}$ 를 입력으로 받는 어떤 단변수함수 $\eta(x_{i})$ 출력에 아주 작은 공통 상수 $\epsilon$ 을 곱한 값이라 생각하자. 

요점은, 

실수 $y_{i}$ 를 어떤 미지의 실수 $x_{i}$ 에 대한 어느 함수의 출력값으로 보고

아주 작은 실수 $\epsilon_{i}$ 를 어떤 미지의 실수 $x_{i}$ 에 대한 어느 함수의 출력값으로 본다는 거다. 

( $\epsilon_{i}$ 은 함수 $\eta(x_{i})$ 출력에 공통상수 $\epsilon$ 을 곱해서 모든 $\epsilon_{i}$ 값 크기를 아주 작게 조정해 준 값이다.)

## 새로운 관점을 가지고 

테일러 전개식의 기호를 바꿔보자. 

$F(y(x_{0})+\epsilon\eta(x_{0}), y(x_{1})+\epsilon\eta(x_{1}), y(x_{2})+\epsilon\eta(x_{2}), ... y(x_{n})+\epsilon\eta(x_{n})) \approx$

$F(y(x_{0}), y(x_{1}), y(x_{2}), ... y(x_{n})) + \frac{\partial F}{\partial y_{0}}\epsilon\eta(x_{0}) + \frac{\partial F}{\partial y_{1}}\epsilon\eta(x_{1})+...\frac{\partial F}{\partial y_{n}}\epsilon\eta(x_{n})$

$= F(y(x_{0}), y(x_{1}), y(x_{2}), ... y(x_{n})) + \sum_{i=0}^{n}{\frac{\partial F}{\partial y(x_{i})}}\epsilon\eta(x_{i})$

---

위 식에서 

함수 $F$ 는 입력으로 $y(x_{0}), y(x_{1}), y(x_{2}), ... y(x_{n})$ 수열을 받는 다변수함수다. 

만약 입력변수 갯수 $n$ 이 무한대가 된다면 어떻게 될까? 

그러면 함수 $y$의 입력값 $x_{i}$ 는 가능한 모든 실수가 된다. 

위 수열은 가능한 모든 $x_{i}$ 에 대한 특정한 출력값들로 구성되어 있다. 

$\Rightarrow$ 위 수열은 가능한 모든 입력에 대한 특정한 출력값들로 구성되어 있다. 

입력과 출력사이 특정한 대응관계를 '함수'라고 정의했었다. 

위 수열의 각 원소는 모든 입력에 대한 각각의 출력을 정의하고 있으므로, 하나의 함수를 표현한 것이라 볼 수도 있다. 

$y(x_{0}), y(x_{1}), y(x_{2}), ... y(x_{n}) \Rightarrow y(x)$

$x$ 는 모든 $x_{i}$ 를 대표하는 기호다. 

$y()$ 는 모든 $x_{i}$ 에 대한 출력을 대표하는 기호다. 

$n = \infty$ 일 때, 테일러 전개식도 바뀐다. 

## $F[y(x)+\epsilon\eta(x) ] \approx F[y(x) ] + \epsilon\int{\frac{\partial F}{\partial y(x)}}\eta(x)$

위 식이 범함수에 대한 테일러 전개다. 

---

# 범함수의 도함수 

위에서 구한 범함수의 테일러 전개를 조금 변형해보자. 

## $\frac{F[y(x)+\epsilon\eta(x) ] - F[y(x) ]}{\epsilon} = \int{\frac{\partial F}{\partial y(x)}}\eta(x)$

우변의 $F(x)$ 를 좌변으로 넘겼다. 

그리고 아주 작은 상수 $\epsilon$ 으로 양변을 나눴다. 

이제 좌변 식의 의미는 다음과 같다. 

" $\epsilon$ 값 변화에 의한 범함수 $F$ 값의 변화량 " 

## 한편, 만약 좌변이 항상 $0$ 이 되어야 한다고 해보자. 

$\frac{F[y(x)+\epsilon\eta(x) ] - F[y(x) ]}{\epsilon} = 0$

이 경우, 위 식의 의미는 " $\epsilon$ 값이 변화해도 범함수 $F$ 값 변화량이 0이다" 라는 의미를 갖는다. 

곧, 위 식이 성립한다는 건 $\epsilon$ 값이 변화해도 범함수 $F$ 값은 변화하지 않는다는 소리다. 


한편, 위 식의 분자를 "범함수 입력을 변화시켰을 때 출력값 변화량' 으로 정의할 수 있다. 

위 식의 분자에서 눈여겨 봐야 할 부분은 

$F[y(x)+\epsilon\eta(x) ]$ 이 부분이다. 


범함수 입력이 원래 $y(x)$ 였다고 생각한다면, $+\epsilon\eta(x)$ 만큼 입력을 변화시켰을 때 출력값이 '변화한 정도가 얼마인가' 측정한 것이 분자다. 

$y(x)$ 를 어떤 임의의 고정된 값이라 생각할 때, $x$도 고정되어 있을 것이다. 

$x$가 고정되어 있으면 $\eta(x)$ 값도 고정되어 있을 것이다. 

입력을 $+\epsilon\eta(x)$ 만큼 변화시키려 한다면, $\eta(x)$ 가 고정되어 있으므로 $\epsilon$ 값이 어떻게 변화하느냐에 따라 범함수 $F$ 의 입력값도 함께 변화할 것이다. 

### 곧, $\epsilon$ 값 변화는 범함수 $F$ 의 입력값을 변화시킨다. 


$\frac{F[y(x)+\epsilon\eta(x) ] - F[y(x) ]}{\epsilon} = 0$

이 식이 항상 성립한다는 말은 $\epsilon$ 값이 변화해도 범함수 $F$ 출력이 변화하지 않는다는 의미다. 

$\epsilon$ 값 변화는 범함수 $F$ 의 입력값 변화를 동반한다.

## $\Rightarrow$ 범함수 $F$ 입력값이 변화해도, 범함수 $F$ 출력값이 변화하지 않는다는 말이다. 

이는 모든 입력값에 대한 범함수 기울기가 0 이어야 한다는 의미다. 

범함수의 도함수 = 0 이어야 한다. 

## 한편

좌변 = 0 이면 우변도 = 0 성립해야 한다. 

## $\int{\frac{\partial F}{\partial y(x)}}\eta(x) = 0$

$\eta(x)$ 값에 상관없이 위 등식이 항상 성립하기 위해서는

## $\frac{\partial F}{\partial y(x)} = 0$ 이어야만 한다. 

따라서 우변 = 0 은 

## $\frac{\partial F}{\partial y(x)} = 0$ 을 의미한다. 

좌변 식의 의미는 범함수의 도함수 = 0 이었다. 

등식이기 때문에 좌변과 우변은 서로 동일해야 한다. 

그러면 $\frac{\partial F}{\partial y(x)}$ 가 도함수 $F$ 를 입력변수인 독립함수 $y(x)$ 로 미분한 범함수의 도함수라는 말이다. 

## $\Rightarrow$ 범함수의 도함수 : $\frac{\partial F}{\partial y(x)}$

범함수의 도함수는 라운드 $\partial$ 기호 대신에 델타 $\delta$ 기호를 쓴다. 

## $\frac{\delta{F}}{\delta{y(x)}}$

---

# 적분형 범함수의 도함수 

범함수는 일반적으로 $x$ 에 대한 적분 형태로 정의된다. 

적분 기호 안의 연산은 실수 $x$와 함수 $y(x)$ 를 입력으로 받는 또 다른 함수 $G$ 로 간주한다. 

$F[y(x)] = \int{G(y(x), x)} dx$

이렇게 $x$에 대한 적분형태의 범함수는 다음과 같이 미분한다.

## 적분형 범함수 미분 방법 : 

## $\frac{\delta{F}}{\delta{y}} = \frac{\partial{G}}{\partial{y}}$

범함수의 입력변수인 독립함수 $y(x)$ 를 일반 변수처럼 취급해서. 

적분 내부 연산인 함수 $G$ 를 편미분 한다. 

예) 

$L = \int{\frac{1}{2}(\hat{y}(x)-y(x))^{2}}dx$

위 함수 $L$ 을 $\hat{y}(x)$ 을 입력으로 보고 미분해보자. 

$G = \frac{1}{2}(\hat{y}(x)-y(x))^{2}$

$G = \frac{1}{2}(\hat{y}(x)^{2}+y(x)^{2}-2\hat{y}(x)y(x))$

## $\frac{\partial{G}}{\partial{\hat{y}}} = \hat{y}(x)-y(x)$

---

# 오일러-라그랑주 공식 

## 정의 : 

적분형 범함수의 적분 내부 연산에 실수 $x$, 함수 $y(x)$ 이외에도 함수 $y(x)$ 를 $x$로 미분한 $y'(x)$ 가 있는 경우, 적분형 범함수 미분하는 공식. 

## $F[y(x)] = \int{G(y, x, y')}dx$ 일 때

## $\Rightarrow \frac{\delta{F}}{\delta{y}} = \frac{\partial{G}}{\partial{y}} - \frac{d}{dx}(\frac{\partial{G}}{\partial{y'}})$

---

# 최적제어

## 정의 : 

범함수의 최적화. 

범함수 최대점, 최소점을 찾는 작업이다. 

---

## 범함수 최적제어 필요조건 : 

일반적인 함수 최적화 할 때와 조건이 같다. 

범함수 최적제어 필요조건은 

## '최적입력 $y^{*}(x)$에서 범함수의 1차 도함숫값이 0 되어야 한다' 이다. 

## $\Rightarrow$ $\frac{\delta{F}}{\delta{y}}[y^{*}(x) ] = 0$























 
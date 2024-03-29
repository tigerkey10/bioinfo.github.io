---
title : "[수학/확률론] 결합확률과 조건부확률, 결합확률분포 정의내리기- 확률론 기초 복습 (2)"
excerpt : "공부 중 정의내린 내용을 정리한 글"

categories : 
- Data Science
- mathematics
- python

tags : 
- [study, data science, mathematics]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-06-23
last_modified_at : 2021-06-23

---
2021-6월-23일 \<6장 확률론 복습 -2 \>

---
# 확률밀도함수
## 확률값과 똑같은 성질 갖는다. 
- 확률밀도함숫값 p(x)는 항상 0 이상이다. 
- -무한대~+무한대 구간 확률밀도함수 면적 구하면 1이다. 

---

# 결합확률 P(A,B) or P(AnB)
- 사건 A,B가 동시에 발생할 확률
결합확률 P(A,B) or P(AnB)
- 사건 A,B가 동시에 발생할 확률
- A,B 교집합 확률

---

# 주변확률 P(A),P(B)
- 개별 사건 확률을 주변확률이라 한다. 

---


# 사건의 독립 
- 사건이 서로의 확률에 영향 안 주면 서로 독립이다. 
- $P(A,B) = P(A)P(B)$ 성립한다. 
- 사건이 독립이면 \[조건부확률=주변확률\] 성립한다.

---

# 결합활률 & 조건부확률 간 관계 
- $P(A,B) = P(A \vert B)P(B)$
## 위 관계 확장하면 '사슬법칙'을 쓸 수 있다. 
- 사슬법칙 : X1 ~ XN 여러 개 사건 결합확률을 조건부 확률 이용해 나타내는 법칙 
- $P(X1,X2,X3) = P(X3 \vert X1,X2)P(X2 \vert X1)P(X1)$

---

# 확률변수 
- 확률적 실수 데이터 생성기 
### - 두 확률변수에서 나오는 사건들이 서로 독립 \<=\> 두 확률변수가 서로 독립
- 표본값 하나하나를 실수 데이터로 바꿔서 현실 세계로 보내준다.
- 예를 들어 X=0 은 확률변숫값이 모두 0인 데이터 집합을 말한다. {0,0,0,0,0,0,0,0,....}

---


# 결합확률분포함수 
- 확률변수 X,Y에서 나오는 사건 두 개 교집합 확률을 정의하는 함수


---
## - 결합확률질량함수 
- 이산확률변수 X,Y에서 나오는 \[표본 두 개 (단순사건_두개)\] \[결합확률\] 정의하는 함수
## - 결합누적분포함수
- 연속확률변수 X,Y의 \[특수구간사건 결합확률\] 정의하는 함수
## - 결합확률밀도함수
- 연속확률변수 X,Y에서 나오는 표본 두 개 결합확률 정의하는 함수
- 이중적분하면 x구간 & y구간_결합확률 구할 수 있다. 

---
# 결과적으로 결합확률질량함수 & 결합확률밀도함수는 확률변수벡터의 확률 할당하는 함수다. 

---


# 다시 
# - 결합확률질량함수 : 확률변수벡터(다변수확률변수 표본값)의 확률질량함수
# - 결합확률밀도함수 : 확률변수벡터(다변수확률변수 표본값)의 확률밀도함수
## 둘 다 기본적으로 다변수함수다. 


---
# 조건부확률질량/확률밀도함수 : '조건부확률' 확률질량함수, 확률밀도함수
# - '원인'에 해당하는 확률변숫값 고정, '결과'에 해당하는 확률변수값의 확률질량함수, 확률밀도함수다. 

---

# 피지엠파이 패키지 
- 계산. 연산 하지 않는다. 
- 결합확률분포를 시각적으로 나타내기만 한다. 


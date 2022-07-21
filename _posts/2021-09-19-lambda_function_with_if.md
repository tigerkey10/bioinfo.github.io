---
title : "[Python] Lambda 함수, Lambda 함수 안에서 if 문 사용하기"
excerpt : "나중에 다시 보려고 기록해 둔 글"

categories : 
- python
- syntaxes

tags : 
- [python, syntaxes]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-09-19
last_modified_at : 2021-09-19

---

# lambda 함수 

- 이름 없는 함수. 일회용 함수다. 
- 함수 이름, return 을 안에 통합하고 있다. (따로 지정 안 해줘도 된다)

문법 
```python
lambda a, b : a+b # 입력, :, 출력
```

## 인-라인 람다함수 

별도 라인에서 함수 호출하고 자시고 할 것 없이, 함수 호출문 내에서 람다함수를 바로 쓸 수 있다. 

이를 인-라인 람다함수 라고 한다. 

```python
# 인-라인 람다함수 예
(lambda a,b : a+b+5)(1,2)
```
결과: 8

## filter() 함수와 lambda 함수 같이 사용하기 

filter() 함수는 함수, 반복가능자 를 입력받는다. 그리고 각 요소에 함수를 적용, 결과가 true 인 것만 추출해준다. 



```python
list1 = [11,22,33,44,55,66]

list(filter(lambda x : x>20 and x<50, list1))
```
결과: [ 22,33,44 ]

## reduce 함수와 lambda 함수 같이 사용하기 

- reduce 함수: 누적연산 함수. functools 모듈을 불러와야 쓸 수 있다. 

- reduce(함수, 반복가능자)

반복가능자에서 요소를 받아와서, 누적연산 한 다음 하나의 값을 도출한다. 

```python
from functools import reduce

list1 = [1,2,3,4,5,6]
reduce(lambda x,y: x+y, list1)
```
결과: 21

계산원리: 

1. list1 에서 1과 2를 각각 x와 y로 받는다. 
2. 둘을 덧셈연산해서, x에 넣는다. 
3. y에 3을 받아와서 다시 덧셈연산한다. 
4. 3의 결과를 다시 x에 넣고 4를 받아온다. 
5. 과정 반복, 결국 sum(list1) 과 같은 결과 나온다. 


---

# lamdba 함수 안에서 if문 사용하기 

## lambda (식 1 if 조건문 1) (else 식 2 if 조건문 2) (else 식 3 if 조건문 3) else (식 4)

- elif 는 사용하지 않는다. 
- 항상 끝은 else 로 끝나야 한다. (모든 조건이 다 틀렸을 때 뭘 할건가)

```python

ld = lambda a, b : (2/3)*((b-a)/180) if b <= 180 else (2/3)*((180-a)/180) + (1/3)*((b-180)/180) if a < 180 and 180 < b else (1/3)*((b-a)/180)

ld(0, 270)
```

---

# lambda 함수를 map 함수와 함께 사용하기 

map 함수는 iterator 객체 원소 하나하나를 호출해서, 함수를 적용한다. 

```python
map(func, iterator)
```
- func : 함수 이름
- itererator : 리스트, 튜플 등 반복가능자

다음과 같은 방식으로, 람다함수와 맵 함수를 유용하게 사용할 수 있다. 

```python
# lambda + map

ld = lambda a : a+3
list(map(ld, [1,2,3]))
```

위 경우 맵 함수는 리스트 축약식과 같은 역할을 한다. 

```python
# 리스트 축약식과 같은 역할 한다
[ld(x) for x in [1,2,3]]
```


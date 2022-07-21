---
title : "[파이썬/넘파이] 데이터분석/머신러닝에 유용한 넘파이 명령어 정리"
excerpt : "빠르게 remind 하기 위한 기억 저장소"

categories : 
- Data Science
- python
- mathematics

tags : 
- [mathematics, python, datascience]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-08-25
last_modified_at : 2021-08-27

---

# 이 글은 데이터분석/머신러닝에 유용한 넘파이 명령어를 기록해둔 것이다. 

필요할 때 바로바로 꺼내쓸 수 있도록 기록해 둔 저장소. 

---

## np.argmax(반복가능자)
- 반복가능자 안 max 값의 인덱스를 반환해준다.

## np.argmin(반복가능자)
- 반복가능자 안 min 값의 인덱스를 반환해준다. 

## np.meshgrid(x값, y값)
- 그리드포인트 (2차원 평면상의 벡터점들) 생성해준다.
- 3차원 그래프 그릴 때 쓴다. 
- 인자로 각 벡터점들의 x값들, y값들 담는다. 
- X,Y = np.meshgrid(x,y) 이런 식으로 생성한다. 즉, X좌표 Y좌표 배열이 따로 생긴다.

## np.dstack([X,Y])

## np.vstack([X,Y])

## np.hstack([X,Y])

## np.count_nonzero(배열)
- 배열 내 데이터 중에서 0이 아닌 값 갯수를 센다. 

## np.bincount(배열)
- 배열 내에 0과 양의 정수가 각각 몇 개씩 있는지 세어 준다. 
- 리스트 형태로 반환한다. 
- minlength= : np.bincount 명령 결과로 나오는 리스트에 '최소' 몇 개 카테고리값까지 표시할 건지 지정해줄 수 있다. 

---

## np.where(조건-ndarray 객체, x,y)
- x,y가 없을 때 출력 : 조건에 맞는 ndarray 객체 원소의 인덱스 
- x, y가 있을 때 출력 : ndarray 객체 원소 중 조건에 맞으면(True) x 출력. 조건에 안 맞으면(False) y출력

예) 

```python
x = np.array([1,2,3,4,5])
np.where(x>3, 0, 1)
```
out : array([1,1,1,0,0])

혹은

```python
x = np.array([1,2,3,4,5])
np.where(x>3)
```
out : array([3,4])

ndarray의 3, 4번 인덱스 값이 조건에 맞는다~ 라는 뜻.

---

## np.roots([계수1, 계수2, 계수3 ...])
- 고차방정식 해(근) 구하는 함수 
- 사용법 : 

예를 들어 고차방정식 $ax^{2}+bx^{1}+c$ 가 있다고 하자. 

고차방정식의 고차항 계수 a,b,c 를 1차원 array로 만든다. 예) 리스트

그리고 np.roots() 에 계수들의 array를 넣어주면 바로 방정식의 근을 뱉어낸다. 

## np.flip() 
- 괄호 안에 array 배열을 받는다. 
- 입력받은 array 좌우를 뒤집는다. 







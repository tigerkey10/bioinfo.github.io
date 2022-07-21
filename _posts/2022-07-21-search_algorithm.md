---
title : "[알고리즘/검색 알고리즘] 선형검색, 이진검색 알고리즘"
excerpt : "공부한 내용을 정리한 글"

categories : 
- computer science
- algorithm
- study 

tags : 
- [computer science, algorithm, study]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-07-21
last_modified_at : 2022-07-21

---

'프로그래머가 알아야 할 알고리즘 40'(임란 아마드 지음, 길벗 출판사) 을 통해 선형, 이진검색 알고리즘을 공부. 복습하고나서, 알고리즘을 구현하기 위해 문제를 정의내리고 해결책을 구상한 과정. 사고흐름. 구현 결과를 기록한다. 

---

# 선형 검색(Linear search)

## 정의 및 특징

배열에서, 선형으로 데이터 하나하나 조회하기.

### 장점 

검색하기 위해서 사전에 배열을 정렬할 필요, 없다. 

### 단점 

느리다. 

## 구상 

1. 0번 인덱스에서부터 시작한다. 
2. 찾았는지 여부는 처음에 false 로 둔다. 
3. 반복조건: \<리스트 마지막 요소까지 반복\> + \<일치하는 값 찾으면 정지(=일치하는 값 찾을 때 까지 반복한다 = 아직 못 찼았는 동안 반복한다)\>
4. 만약 이번 인덱스가 일치하면: 찾았는지 여부 = true 로 변경한다
5. 인덱스가 일치하지 않으면: 다음 인덱스로 이동한다. 
6. 최종으로, 찾았는지 여부를 return 한다(true or false)


## while 문 사용해서 구현 

```python 
# 선형검색 
def linear_search(x, item) : 
    current_index = 0 # 현재 인덱스 
    found = False # 찾았는지 여부 

    while (current_index < len(x)) and (found == False) : 
        if x[current_index] == item : 
            found = True 
        else : 
            current_index += 1 
    return found 
```

### 테스트 

```python 
x = [12, 33, 11, 99, 22, 55, 90]
print(linear_search(x, 12))
print(linear_search(x, 91))
```
True 

False 

## for 문 사용해서 구현 

```python 
# 선형검색 -2 
def linear_search_2(x, item) : 
    found = False
    for cont in x : 
        if cont == item : 
          found = True 
          break 
    return found   
```

### 테스트 

```python 
print(linear_search_2(x, 12))
print(linear_search_2(x, 91))
```
True 

False 

### 테스트 -2

```python 
import numpy as np 
x2 = np.random.sample(100)
sample = x2[3]
print(linear_search_2(x2, sample))
print(linear_search_2(x2, 33))
```

True 

False 

---

# 이진 검색(Binary search)

## 왜 이진(binary) 검색인가? 

배열을 중간값 기준으로 왼쪽 오른쪽 끊임없이 둘로 나누면서 검색하기 때문에 이진검색 이라 부른다. 

## 사용 전제조건

데이터가 이미 정렬되어 있어야 한다. 

## 검색 과정 

### 아래 과정 반복한다. 
- 배열에서 중간값 찾는다. 
- 중간값과 찾으려는 값 크기 비교한다. 
- 중간값 = 찾으려는 값 이면: 검색 종료한다. 
- 중간값 \< 찾으려는 값 이면: 배열에서 중간값 기준 오른쪽 배열로 이동
- 중간값 \> 찾으려는 값 이면: 배열에서 중간값 기준 왼쪽 배열로 이동 

## 1차 구상 

1. 배열에서 가장 중앙에 위치한 값 찾는다. 
2. 중앙값과 찾으려는 값 비교해서, 값이 중앙값보다 크면 오른쪽 배열. 작으면 왼쪽 배열 따로 떼어낸다. 만약 중간값과 찾으려는 값 같으면 '찾았음'
3. 1과 2 과정, (원하는 값 못 찾았고) and (배열 크기가 1 이상일 때) 까지 반복한다. 

$\Rightarrow$ 

## 2차 구상 

while (원하는 값 못 찾았고) and (배열 크기가 1 이상 일 때) : \<반복 지속조건\>

1 배열 가장 중간값 찾는다. 

2 

만약 중간값 == 찾으려는 값 이면: $\Rightarrow$ '찾았음'

그렇지 않은 모든 경우에 대해서,

만약 중간값 \< 찾으려는 값 이면: x = x[ 중간값+1: ]

만약 중간값 \> 찾으려는 값 이면: x = x[ :중간값 ]

그 외 모든 경우에: break 

## 구현 

```python 
# 이진검색 
def binary_search(x, item) : 

    found = False # item을 찾았는가 여부 

    while (found is False) and (len(x) >= 1) : # 반복 지속조건 
        mid_point = len(x)//2 # 중간값 인덱스 지정
        if x[mid_point] == item : # 검사: 중간값이 내가 찾으려는 item 값과 같은가?
            found = True # 같다면 true(반복문도 종료)

        else : # item과 중간값이 다른 경우
            if x[mid_point] < item : # item이 더 크면 중간값 기준 오른쪽 리스트로 이동
                x = x[mid_point+1:]
            elif x[mid_point] > item : # item이 더 작으면 중간값 기준 왼쪽 리스트로 이동
                x = x[:mid_point]
    
    return found # item 찾았는가 여부 반환. true 또는 false 일 것이다. 
```

### 테스트 

```python 
test = [12, 33, 11, 99, 22, 55, 90]

# 이진검색은 데이터가 사전에 이미 정렬되어 있어야 한다. 
# 따라서 test 리스트 정렬 위해 버블정렬 사용했다. 

# 버블정렬 정의
def bubble_sort(x) : 
    for pas in range(len(x)-1) : # 패스 n-1 번 반복 
        for i in range(len(x)-1) : 
            if x[i] > x[i+1] : 
                x[i], x[i+1] = x[i+1], x[i] # 교환 
    return x 

# 결과 출력 
print(binary_search(bubble_sort(test), 33332))
print(binary_search(bubble_sort(test), 91))
```

False

False














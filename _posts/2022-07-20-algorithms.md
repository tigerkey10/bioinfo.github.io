---
title : "[알고리즘/정렬 알고리즘] 버블정렬, 삽입정렬, 병합정렬, 셸 정렬, 선택정렬 알고리즘"
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

date : 2022-07-20
last_modified_at : 2022-07-20

---

'프로그래머가 알아야 할 알고리즘 40'(임란 아마드 지음, 길벗 출판사) 을 통해 정렬 알고리즘을 공부. 복습하고나서, 알고리즘을 구현하기 위해 문제를 정의내리고 해결책을 구상한 과정. 사고흐름. 구현 결과를 2022년 7월 20일 최초 기록한다. 

---

# 버블정렬 

## 정의

가장 큰 값을 가장 오른쪽으로 보내기 

## 방법 

이웃한 값 비교해서 교환(패스)

- 패스 N-1 번 반복 

## 스니펫 구상 

```python 
# 버블정렬 구상 
for end in range(len(list), 0, -1) : 
    for i in range(0, end-1) : 
        # 1. 2개 비교 
        # 2. 만약 둘 중에 앞 값이 더 크면: 서로 자리 교체 
        # --> 
        if list[i] > list[i+1] : 
            list[i], list[i+1] = list[i+1], list[i]
```

## 구현 

```python 
# 버블정렬 구현 
def bubble_sort(x) : 
    for end in range(len(x)-1, 0, -1) : 
        for i in range(0,end) : 
            if x[i] > x[i+1] : 
                x[i], x[i+1] = x[i+1], x[i]
    return x
```

### 테스트 

```python 
x = [25, 21, 22, 24, 23, 27, 26]
bubble_sort(x)
```
[21, 22, 23, 24, 25, 26, 27]

---

# 삽입정렬 

## 구상 

1. N-1번 중 이번회차에 대해서
2. 이번 회사체 정렬해야 할 원소 위치: j
3. 이번 회차에 정렬해야 할 원소의 값: temp 
4. while(j-1 >= 0) and (list[ j ] < list[ j-1 ]) : list[ j ] = list[ j-1 ], j-=1
5. while 반복이 끝나면: list[ j ] = temp 

## 스니펫 

```python 
# 삽입정렬 스니펫 
for n in range(1, len(x)) : 
    j = n 
    temp = x[j]
    while (j-1 >= 0) and (temp < list[j-1]>) : 
        list[j] = list[j-1]
        j-= 1
    list[j] = temp 
```

## 구현 

```python 
# 삽입정렬 구현 

def insert_sort(x) : 
    for n in range(1, len(x)) : 
        j = n # 정렬 해야 할 가장 왼쪽 위치
        temp = x[j]
        while (j-1 >= 0) and (temp<x[j-1]) : 
            x[j] = x[j-1]
            j -= 1
        x[j] = temp 
    return x
```

### 테스트 
```python 
import numpy as np 
import matplotlib.pyplot as plt 

x = [25, 26, 22, 24, 27, 23, 21]
insert_sort(x)
x2 = list(np.random.sample(100))
plt.bar(range(100), insert_sort(x2))
```

<img width="369" alt="Screen Shot 2022-07-20 at 20 32 22" src="https://user-images.githubusercontent.com/83487073/179971733-3aa84be6-2e58-4b37-ae58-8f90d65ae779.png">

---

# 병합정렬 

## 구상 

전체 과정은 분리 $\Rightarrow$ 병합. 

### \<분리\>
- 분리조건: 리스트 크기 $> 1$ 일 때; 리스트 크기 1 되면 분리 정지(=크기 1될 때 까지 분리)
- 분리 기준점 설정 
- 분리 기준점 기준 왼쪽으로 분리 
- 분리 기준점 기준 오른쪽으로 분리 

왼쪽에 대해 다시 분리(+병합) 적용 

오른쪽에 대해 다시 분리(+병합) 적용 

### \<병합\>
- 왼쪽 오른쪽 크기 비교해서, 오름차순으로 원본리스트에 결합 

*분리된 상태 = 정렬된 상태. 

a = 0 , 왼쪽 인덱스

b = 0 , 오른쪽 인덱스 

c = 0 , 전체 인덱스 

while (a \< len(left)) and (b \< len(right)) : 

if left[ a ] \> right[ b ] : list[ c ] = right[ b ], b+= 1

else: list[ c ] = left[ a ], a+= 1 

c += 1 

while (a \< len(left)) : list[ c ] = left[ a ], a += 1, c+= 1

while (b \< len(right)) : list[ c ] = right[ b ], b+= 1, c+= 1

return list, 정렬 결과 출력 

## 구현 

```python 
# 병합 정렬 구현
def merge_sort(x) : 
    # 분리 
    if len(x) > 1 : # 분리 조건 
        separate_criterion = len(x)//2 
        left = x[:separate_criterion]
        right = x[separate_criterion:]

        merge_sort(left)
        merge_sort(right)

        # 병합 
        a = 0 
        b = 0 
        c = 0 

        while (a < len(left)) and (b < len(right)) : 
            if left[a] > right[b] : 
                x[c] = right[b]
                b += 1
            else : 
                x[c] = left[a]
                a += 1 
            c += 1 

        while (a < len(left)) : 
            x[c] = left[a]
            a += 1
            c += 1
        
        while (b < len(right)) : 
            x[c] = right[b]
            b += 1
            c += 1

    return x 
```

### 테스트 

```python 
list3 = [44, 16, 83, 7, 67, 21, 34, 45, 10]
merge_sort(list3)
```

[7, 10, 16, 21, 34, 44, 45, 67, 83]

### 테스트 2 

```python 
test = list(np.random.sample(100))
plt.figure(figsize=(100,50))
plt.subplot(1,2,1)
plt.bar(range(100), test)
plt.subplot(1,2,2)
plt.bar(range(100), merge_sort(test))
```

<img width="603" alt="Screen Shot 2022-07-20 at 20 41 32" src="https://user-images.githubusercontent.com/83487073/179973408-c65b24de-013e-4c87-b4ab-b1da7ec3f239.png">

---

# 셸 정렬 

삽입정렬 보완판. 

## 구상 

1. 거리 설정한다. 
2. 정렬해야 할 원소 j와 j-distance 사이 값 비교한다. 
3. j \< j-distance 이면, j와 j-distance 값 위치 교환 , j = j-distance 로 새로 할당
4. 3의 과정을 j-distance \>=0 일 때 까지 반복한다. 

## 구현 

```python 
# 셸 정렬 구현
def shell_sort(x) : 
    distance = len(x)//2 # 거리 지정 
    
    while distance > 0 : 
        # 부분 리스트 정렬 하는 코드 블럭
        for i in range(distance, len(x)) : 
            j = i 
            # 부분리스트 1개에 대해 정렬하는 코드 블럭
            while (j-distance >= 0) and (x[j] < x[j-distance]) : 
                x[j], x[j-distance] = x[j-distance], x[j]
                j = j - distance 
            
        distance = distance // 2 # 다 끝났으면 거리 조정 
    
    return x 
```

### 테스트 

```python 
x = list(np.random.sample(100))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(range(100), x)
plt.subplot(1,2,2)
plt.bar(range(100), shell_sort(x))
plt.show()
```

<img width="590" alt="Screen Shot 2022-07-20 at 20 45 14" src="https://user-images.githubusercontent.com/83487073/179974175-e03fdde7-8ffd-4d02-9abb-db6f6fcecca6.png">

---

# 선택정렬 

## 구상 

1. 정렬 안 된 부분에서 가장 큰 값 찾아서, 정렬 안 된 부분 가장 오른쪽 원소와 바꾼다. 
2. 정렬 안 된 부분 가장 오른쪽 위치는 교환 발생할 때 마다, 1씩 줄어든다. 

$\Rightarrow$

\<다시 정리\>

### 교환
- 정리 안 된 부분에서 가장 큰 값 찾는다. 
- 가장 큰 값과, 정렬 안 된 부분 가장 오른쪽 원소 맞바꾼다. 

### +
- 교환 발생할 때 마다 '정렬 안 된 부분 가장 오른쪽 위치'가 1씩 줄어든다.
- 교환은 총 n-1번 발생한다. 

## 스니펫 

```python 
for r in range(len(x)-1, 0, -1) : # 가장 오른쪽 요소의 인덱스 
    # n-1번 교환 발생 
    #<가장 큰 값 찾기>
    max_index = 0 
    for i in range(1, r+1) : 
        if x[max_index] < x[i] : 
            max_index = i
    
    # 이제 가장 큰 값(그것의 인덱스) 찾았다. 교환한다. 
    # <교환>
    x[max_index], x[r] = x[r], x[max_index]
```

## 구현 

```python 
# 선택정렬 복습 & 구현 
def selection_sort(x) : 
    for r in range(len(x)-1, 0, -1) : # 가장 오른쪽 요소의 인덱스 (1씩 감소)

        # 가장 큰 값 찾기 
        max_index = 0 
        for i in range(1, r+1) : 
            if x[max_index] < x[i] : 
                max_index = i 
                
        # 교환 
        x[max_index], x[r] = x[r], x[max_index]
    return x 
```

### 테스트 

```python 
x = np.random.sample(500) 
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(range(500), x)

plt.subplot(1,2,2)
plt.bar(range(500), selection_sort(x))
```

<img width="592" alt="Screen Shot 2022-07-20 at 20 51 18" src="https://user-images.githubusercontent.com/83487073/179975241-2ddd1a4a-79e6-42d3-95c3-bdad738e21f2.png">







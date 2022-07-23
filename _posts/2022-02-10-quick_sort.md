---
title : "[2021 인공지능전문가 교육과정 복습] 퀵 정렬, 기수 정렬 알고리듬"
excerpt : "부산대학교 인공지능전문가 교육과정 - 데이터사이언스:데이터 구조 수업 복습 후 정리"

categories : 
- Data Science
- python
- data structure

tags : 
- [data structure, computer science, python, study, data science]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-02-10
last_modified_at : 2022-02-11

---

# 퀵 정렬(Quick Sort)

피벗(기준) 정렬.

## 정의 

피벗값을 기준삼아. 피벗 왼쪽에는 피벗보다 작은 값들, 오른쪽에는 큰 값들 오도록 정렬하는 알고리듬.

## 특징 

- 피벗값은 아무거나 잡아도 되나, 기왕이면 중앙값 잡는 게 좋다. 그 값 중심으로 왼쪽 작은 부분과 오른쪽 큰 부분이 균등하게 나눠지도록 하기 위해서다. 양쪽이 균등하게 나눠 질 수록 정렬 수행 시간 빨라진다. 
- 일반적으로 가장 빠른 정렬 알고리듬이다. 
- 합병정렬이 정렬 과정에서 전체를 균등분할 했다면, 퀵 정렬은 대부분 경우 비균등분할한다. 

## 메커니즘 

1. 피벗값 잡는다. 
2. i, j 잡고 각각 오른쪽, 왼쪽 방향 이동하다가 각각 피벗보다 큰 값, 피벗보다 작은 값 만나면 멈춘다. 
3. i, j 위치의 값 서로 교환한다. 
4. 2~3 과정 반복. i = j(둘이 만나는 경우) 또는 j \< i(둘이 교차한 경우) 되면서 멈춘 경우 반복도 멈춘다. 
5. ### 피벗 왼쪽에는 피벗보다 작은 값만, 오른쪽에는 큰 값만 있어야 한다. 이 원칙에 기반해서 i, j 위치 값을 피벗과 비교한다. i, j 위치 값 중 원칙 어긋나는 값을 피벗과 교환한다. 이후 피벗 위치는 그 값의 최종위치다. 더 이상 정렬하지 않는다.
6. 피벗 기준 왼쪽과 오른쪽 부분리스트에서 1~5 과정을 재귀호출 함으로써 과정 반복한다. 매 재귀호출 마다 피벗값들이 최종위치에 놓여지면서 정렬된다. 

## 예시 - 1 

## 가장 왼쪽 레코드를 피벗(pivot)으로 잡는 경우 

<img width="371" alt="Screen Shot 2022-02-10 at 20 10 45" src="https://user-images.githubusercontent.com/83487073/153395352-965fae98-859c-4b06-abd6-e6f0ac15c2f0.png">

## 예시 - 2

## 중간 레코드를 피벗으로 잡는 경우 1

<img width="353" alt="Screen Shot 2022-02-10 at 20 12 57" src="https://user-images.githubusercontent.com/83487073/153395698-8b41fb2c-a7f5-4b5d-a81c-0b784bcadf0d.png">

## 예시 - 3

## 중간 레코드를 피벗으로 잡는 경우 2

<img width="366" alt="Screen Shot 2022-02-10 at 20 13 37" src="https://user-images.githubusercontent.com/83487073/153395829-c9e0575d-af4f-41c4-8d2e-59bb516be310.png">

## 예시 - 4

## 가장 오른쪽 레코드를 피벗으로 잡는 경우 

<img width="373" alt="Screen Shot 2022-02-10 at 20 14 18" src="https://user-images.githubusercontent.com/83487073/153395948-1b7ce496-ab65-45ab-b119-c3addef6d84b.png">

---

# 퀵 정렬 성능 

## 평균 경우: $O(N\log_{2}{N})$

## 최선 경우: 각 부분리스트가 균등분할 되는 경우 

이론상 가장 이상적 경우는 각 부분리스트가 모두 균등분할 되는 경우다. 

- 재귀호출 수: $\log_{2}{n}$

입력의 원소 개수를 $2^{k}$ 개 라고 가정하자. $n = 2^{k}$.

k가 3인 경우($n = 2^{3}$) 재귀호출 하면서 균등분할 한다고 했을 때. 

$2^{3} \Rightarrow 2^{2} \Rightarrow 2^{1} \Rightarrow 2^{0}$ 으로 총 $3$번 재귀호출 하게 된다. 

곧, 재귀호출 수 $= k$ 가 된다. 

$n = 2^{k} \Rightarrow k = \log_{2}{n}$ 이므로 재귀호출 수는 $\log_{2}{n}$ 이 된다. 

- 각 재귀호출에서 레코드 간 비교 횟수: $n$
- 이동횟수는 비교횟수에 비해 적어 무시가능

총 비교 횟수 $\Rightarrow n\log_{2}{n}$ 

### 따라서 

### 퀵 정렬 최선 경우 성능: $O(N\log_{2}{N})$

## 최악 경우: 극도로 불균등한 리스트로 분할되는 경우 

퀵 정렬은 리스트 분할이 불균등 할 수록 시간복잡도 증가한다. 

이미 정렬된 리스트 정렬의 예)

\[ 1,2,3,4,5,6,7,8 \] 이라는 이미 정렬된 리스트가 있다고 하자. 

가장 왼쪽 원소를 피벗 삼아 퀵 정렬 한다. 

이 경우 매 순환호출 마다 리스트가 오른쪽으로 전부 몰려서 분할된다. 왼쪽 부분리스트는 아예 없다. 

- 총 비교횟수: $n+(n-1)+(n-2)+...+2+1 = \frac{n(n+1)}{2}$
- 이미 정렬되어 있기 때문에 이동은 없다. 

### 따라서 

### 퀵 정렬 최악 경우 성능: $O(N^{2})$

---

# 퀵 정렬 구현 

## 퀵 정렬 정의(1) - pivot을 가운데 값으로 잡는 경우 

```python 
# pivot이 가운데 있는 경우 정의 

def partition(a, low, high) : 
    i = low 
    j = high 
    pivot = (high + low) // 2

    while True : 
        while (i < high) and (a[i] < a[pivot]) : 
            i += 1
        while (j > low) and (a[j] > a[pivot]) : 
            j -= 1
        if j <= i : break # 계속 가다가 멈춘 사유가 j <= i 이면 break.
        if (i == pivot) : 
            a[i], a[j] = a[j], a[i]
            pivot = j # pivot은 포인터에 불과 
            i += 1 ; j -= 1
        elif (j == pivot) : 
            a[i], a[j] = a[j], a[i]
            pivot = i
            i += 1 ; j -= 1
        else : 
            a[i], a[j] = a[j], a[i]
            i += 1 ; j -= 1
    
    if (a[j] <= a[pivot]) and (j >= pivot) : 
        a[pivot], a[j] = a[j], a[pivot]
        return j
    elif (a[i] >= a[pivot]) and (i <= pivot) : 
        a[pivot], a[i] = a[i], a[pivot]
        return i

# 퀵 정렬 

def qsort(a, low, high) : 
    if (low < high) : 
        pivot = partition(a, low, high)
        qsort(a, low, pivot-1)
        qsort(a, pivot+1, high)
```

## 알고리듬 테스트 

### (1)
```python
# 정렬 알고리듬 테스트 

a = [54, 88, 77, 26, 93, 17, 49, 10, 17, 77, 11, 31, 22, 44, 17, 20]

print(f'정렬 전:{a}')
qsort(a, 0, len(a)-1)
print(f'정렬 후:{a}')
```
정렬 전:\[54, 88, 77, 26, 93, 17, 49, 10, 17, 77, 11, 31, 22, 44, 17, 20\]

정렬 후:\[10, 11, 17, 17, 17, 20, 22, 26, 31, 44, 49, 54, 77, 77, 88, 93\]

### (2)

```python 
# 정렬 알고리듬 테스트 

random_list = list(np.random.randint(0,100,20))

print(f'정렬 전:{random_list}')
qsort(random_list, 0, len(random_list)-1)
print(f'퀵 정렬 후:{random_list}')
```
정렬 전:\[5, 83, 59, 31, 86, 89, 25, 91, 41, 15, 77, 6, 55, 55, 46, 59, 91, 40, 50, 46\]

퀵 정렬 후:\[5, 6, 15, 25, 31, 40, 41, 46, 46, 50, 55, 55, 59, 59, 77, 83, 86, 89, 91, 91\]

### (3)

```python 
# 정렬 알고리듬 테스트 

random_list = list(np.random.randint(0,1000,20))

print(f'정렬 전:{random_list}')
qsort(random_list, 0, len(random_list)-1)
print(f'퀵 정렬 후:{random_list}')
```
정렬 전:\[831, 388, 62, 442, 111, 668, 838, 7, 869, 445, 635, 793, 169, 843, 586, 405, 346, 892, 665, 950\]

퀵 정렬 후:\[7, 62, 111, 169, 346, 388, 405, 442, 445, 586, 635, 665, 668, 793, 831, 838, 843, 869, 892, 950\]

## 퀵 정렬 정의(2) - 가장 왼쪽 레코드가 pivot이 되는 경우

```python 
# 가장 왼쪽이 pivot이 되는 경우 정의

def partition_pl(a, pivot, high) : 
    i = pivot + 1
    j = high 

    while True : 
        while (i < high) and (a[i] < a[pivot]) : 
            i += 1
        while (j > pivot) and (a[j] > a[pivot]) : 
            j -= 1
        if j <= i : break # i, j 만났거나 i, j 교차해서 멈춘 경우
        a[i], a[j] = a[j], a[i]
        i += 1 ; j -= 1 
    # break 로 loop 깨진 경우 
    a[pivot], a[j] = a[j], a[pivot]
    return j # pivot 위치 반환 

# 퀵 정렬 정의
def qsort_pl(a, low, high) : 
    if low < high : 
        pivot = partition_pl(a, low, high)
        qsort_pl(a, low, pivot-1)
        qsort_pl(a, pivot+1, high)
```

## 알고리듬 테스트 

```python 
# 테스트 - qsort_pl
randoms = np.random.randint(0, 1000, 20)

print(f'정렬 전:{randoms}')
qsort_pl(randoms, 0, len(randoms)-1)
print(f'정렬 후:{randoms}')
```

정렬 전:\[468  69 341 122 406 909 912 904 327 811 448  80 379 333 960 478 652  48 913 386\]

정렬 후:\[ 48  69  80 122 327 333 341 379 386 406 448 468 478 652 811 904 909 912 913 960\]

## 퀵 정렬 정의(3) - 가장 오른쪽 레코드가 pivot이 되는 경우

```python 
# 가장 오른쪽이 pivot이 되는 경우 정의 

def partition_pr(a, low, pivot) : 
    i = low 
    j = pivot - 1

    while True : 
        while (i < pivot) and (a[i] < a[pivot]) : 
            i += 1 
        while (j > low) and (a[j] > a[pivot]) : 
            j -= 1
        if (j <= i) : break 
        a[i], a[j] = a[j], a[i]
        i += 1 ; j-= 1
    a[pivot], a[i] = a[i], a[pivot]
    return i # pivot 위치 반환 

# 퀵 정렬 정의
def qsort_pr(a, low, high) : 
    if low < high : 
        pivot = partition_pr(a, low, high)
        qsort_pr(a, low, pivot-1)
        qsort_pr(a, pivot+1, high)
```

## 알고리듬 테스트 

```python 
# 테스트 - qsort_pr 

randoms = np.random.randint(5, 100, 20)

print(f'정렬 전:{randoms}')
qsort_pr(randoms, 0, len(randoms)-1)
print(f'정렬 후:{randoms}')
```

정렬 전:\[78 78  8 38 33 31 12 71 95 78 99 54 29 65 65 15 28 33 10 58\]

정렬 후:\[ 8 10 12 15 28 29 31 33 33 38 54 58 65 65 71 78 78 78 95 99\]

---

# 정렬 알고리듬 별 비교 

시간복잡도, 안정성, 추가메모리 필요 여부 비교 

<img width="916" alt="Screen Shot 2022-02-10 at 22 22 07" src="https://user-images.githubusercontent.com/83487073/153416753-b0b4f6a1-bfa1-4edc-b9f6-a270a8f37b71.png">

---

# 기수 정렬(Radix Sort)

자릿수(기수) 정렬. 

MSD / LSD 방법 중 LSD 방법(작은 자릿수 부터 정렬하는 방법)에 근거해 설명한다. 

## 정의 

각 레코드를 자릿수 별로 정렬하는 정렬 알고리듬. 

## 특징 

- ### 비교연산 없이 정렬한다!
- 최선, 최악 경우 없이 매번 굉장히 빠른 정렬 알고리듬이다$O(N)$.
- 데이터 외 버킷(데이터 임시 저장 공간. 하나하나가 큐) 이 추가로 필요하다. 따라서 메모리 많이 잡아먹는다. 

## 메커니즘 

1. 10개 버킷 준비한다(0~9 의미)
2. 각 레코드 1 자릿수 별로 버킷에 순서대로 넣는다. 
3. 버킷에서 순서대로 가져온다(1 자릿수에 대해 정렬완료)
4. 10 자릿수, 100 자릿수... 에 대해 2~3 과정 반복한다. 

## 예시 

각 버킷은 큐(Queue)다. 선입선출(FIFO) 한다. 

## 1 자릿수 대로 정렬 

<img width="544" alt="Screen Shot 2022-02-11 at 10 53 05" src="https://user-images.githubusercontent.com/83487073/153526226-d450e9ae-b0d6-4e30-854d-12dd749c6d05.png">

<img width="566" alt="Screen Shot 2022-02-11 at 10 53 30" src="https://user-images.githubusercontent.com/83487073/153526258-3da01275-17a6-4eb3-9477-1e41a50d4857.png">

## 10 자릿수 대로 정렬 

<img width="549" alt="Screen Shot 2022-02-11 at 10 54 00" src="https://user-images.githubusercontent.com/83487073/153526287-eb0e50fe-bb61-4509-aa6d-641c49d41715.png">

## 100 자릿수 대로 정렬 

<img width="542" alt="Screen Shot 2022-02-11 at 10 54 29" src="https://user-images.githubusercontent.com/83487073/153526324-3c772d1b-7238-4034-ba4c-f565ceb75400.png">

<img width="554" alt="Screen Shot 2022-02-11 at 10 54 46" src="https://user-images.githubusercontent.com/83487073/153526352-4e543885-f50f-4f5b-a978-0b28ab7ce8de.png">

---

# 기수 정렬 성능 

## 시간복잡도 

### 항상 $O(N)$ 시간복잡도 보장된다. 

최선, 최악 경우 없이 항상 빠르다. 

위 경우 예로 들 때(데이터수: 11개)

- 1 자릿수 찾기 위해: 11번 탐색 
- 10 자릿수 찾기 위해: 11번 탐색
- 100 자릿수 찾기 위해: 11번 탐색

$\Rightarrow$ $k \times N$ 번 탐색 수행.

$\Rightarrow$ $O(N)$ 시간복잡도. 

## 정렬 알고리듬 안정성

### 안정성 보장

## 단점

### 메모리 많이 잡아먹는다

전체 데이터 저장공간 뿐 아니라, 버킷을 위한 메모리 공간도 필요하다. 

이것 자체로 메모리를 많이 먹는다. 

- 그리고 버킷에 데이터 담고 빼는 과정이 많아 질 수록, 무시할 수 없을 만큼 시간도 많이 잡아먹는다. 

---

# 기수 정렬 구현 

## 기수 정렬 정의 

```python 
# 기수 정렬 정의 

class Radix_sort : 
    def __init__(self, num) : 
        self.num = num # 정렬할 대상 
    
    def radix_sort(self) : 
        max1 = max(self.num) # 최댓값 찾는다: 최소한 최댓값 자릿수만큼 충분히 정렬하기 위해서다
        exp = 1
        while (max1/exp) > 0 : 
            self.count_sort(self.num, exp)
            exp *= 10 # 1의 자리, 10의 자리, 100의 자리, ... 돌아가며 정렬한다
        
    def count_sort(self, A, k) : 
        B = [0]*len(A) # 정렬 결과 담을 임시 리스트 
        C = [0]*10 # 0~9. 0~9인 N 자릿수 갯수 담을 리스트 

        # 정렬 위해 C를 먼저 만든다. 
        for i in range(0, len(A)) : 
            C[(A[i]//k)%10] += 1 # C완성 

        # C를 누적 값 리스트로 바꾼다. 
        for i in range(1, len(C)) : 
            C[i] = C[i-1] + C[i]

        i = len(A) - 1 # 정렬 다 된 상태에서. 더 안 움직이기 위해 뒤에서 부터 돈다. 
        while (i >= 0) : 
            B[C[(A[i]//k)%10]-1] = A[i] # 임시리스트에 정렬 
            C[(A[i]//k)%10] -= 1
            i -= 1
        
        # 임시리스트 결과 원본 리스트에 복사 
        for i in range(0, len(A)) : 
            A[i] = B[i]
```

- 1 자리, 10 자리, 100자리, 1000자리, ... 돌아가며 자릿수 별로 순서대로 정렬한다.
- 어떤 수 X의 k 자릿수 $= (X//k)\%{10}$ (몫 - 나머지)
- 임시 배열 B 만들고 거기다 정렬한 뒤 원래 배열에 옮긴다.
- C는 N 자릿수가 0부터 9인 것들 갯수 저장된 배열이다. 배열 C 각 인덱스 위치는 0~9를 의미한다. C 배열 0번 인덱스에 2가 들어가 있다면. 1의 자리가 0인게 (정렬 안 된 채로) 2개 있다는 말이다. 
- C를 만들고 나서. 0~9까지 갯수 누적값 저장한 배열로 바꾼다. 
- ### B에 '정렬' 할 때. 그 수가 반드시 있을 법한 자리에 집어넣는다. 

$\Rightarrow$ 예컨대 1 자리가 2인 수를 B에 임시로 정렬하려 한다. 한편 C의 2번 인덱스가 3이라 가정한다. 이건 1 자리가 0, 1, 2인 원소 갯수가 총 3개라는 말이다. 우리는 내가 넣으려는 수가 이 3개 안에 반드시 포함되어 있다는 걸 알고 있다. 3개 중에서 적어도 마지막 3번째에는 어느 상황에서든 반드시 1 자리 2인 수가 포함되어 있을 것이다. 내가 넣으려는 수는 1 자리가 2다. 따라서 3개 중 마지막 3번째에 해당하는 B의 2번 인덱스에 내가 넣으려는 수를 집어넣는다. 맨 마지막 while 루프 안에 B\[C\[(A\[i\]//k)%10\]-1\] = A\[i\] 코드는 그래서 나온 것이다. 이후 1자리 2인 원소 1개가 정렬되었으므로, 정렬되지 않은 것들 개수 누적 리스트 C의 2번 인덱스 원소에서 1을 줄인다. 그 다음 줄 C\[(A\[i\]//k)%10\] -= 1 코드가 그 역할 한다. 

## 알고리듬 테스트 - 1

```python 
# 테스트 

test = [5,2,8,4,9,1,11,33,22,12]

rd = Radix_sort(test)
rd.radix_sort()
rd.num
```
\[1, 2, 4, 5, 8, 9, 11, 12, 22, 33\]

## 알고리듬 테스트 - 2

```python 
# 테스트 

number = [170, 45, 75, 90, 802, 24, 2, 66]

print(f'정렬 전:{number}')
radix = Radix_sort(number)
radix.radix_sort() 
print(f'정렬 후:{radix.num}')
```
정렬 전:\[170, 45, 75, 90, 802, 24, 2, 66\]

정렬 후:\[2, 24, 45, 66, 75, 90, 170, 802\]

---

# 기수 정렬(LSD) 문자열 정렬에 응용 

## LSD 알고리듬 정의 - 1

```python 
# 1

def lsd_sort(a) : 
    width = 3
    n = len(a)
    r = 128
    temp = [None]*n
    for d in reversed(range(width)) : 
        count = [0]*(r+1)
        for i in range(n) : 
            count[ord(a[i][d])+1] += 1 
        for j in range(1, r) : 
            count[j] += count[j-1]
        for i in range(n) : 
            p = ord(a[i][d])
            temp[count[p]] = a[i]
            count[p] += 1
        for i in range(n) : 
            a[i] = temp[i]
        print(f'{d}번째 문자:', end='')
        for x in a : 
            print(x, '', end=' ')
        print()
```

---

## 참고

### ord() 함수 

ord() 함수는 어떤 문자열의 아스키코드 값을 반환 해준다. 

```python 
# ord()

print(ord('A'))
print(ord('a'))
```
65

97

### chr() 함수 

chr() 함수에 아스키코드 값 넣으면 그 아스키코드에 할당된 문자열 반환해준다. 

곧, ord() 함수와 정 반대다. 

```python 
# chr() 

print(chr(65))
print(chr(97))
```
'A'

'a'

---

## 알고리듬 테스트 - 1

```python 
a = ['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW']
print(f'정렬 전:{a}')
lsd_sort(a)
```
정렬 전:\['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW'\]

2번째 문자:FRA  SYD  HKG  JFK  PEK  ROM  ICN  SIN  SFO  LHR  BER  NRT  TLV  MOW  LAX  MEX  

1번째 문자:LAX  ICN  PEK  BER  MEX  JFK  SFO  LHR  SIN  HKG  TLV  ROM  MOW  FRA  NRT  SYD  

0번째 문자:BER  FRA  HKG  ICN  JFK  LAX  LHR  MEX  MOW  NRT  PEK  ROM  SFO  SIN  SYD  TLV  

## LSD 알고리듬 정의 - 2

```python 
# 2

def lsd_sort(a) : 
    width = 3
    n = len(a)
    r = 128
    temp = [None]*n
    for d in reversed(range(width)) : 
        count = [0]*r 
        for i in range(n) : 
            count[ord(a[i][d])] += 1 
        for j in range(1, r) : 
            count[j] += count[j-1]
        for i in range(n) : 
            p = ord(a[i][d])
            temp[count[p]-1] = a[i]
            count[p] -= 1
        for i in range(n) : 
            a[i] = temp[i]
        print(f'{d}번째 문자:', end='')
        for x in a : 
            print(x, '', end=' ')
        print()
```

## 알고리듬 테스트 - 2

```python 
a = ['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW']
print(f'정렬 전:{a}')
lsd_sort(a)
```
정렬 전:\['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW'\]

2번째 문자:FRA  SYD  HKG  PEK  JFK  ROM  SIN  ICN  SFO  BER  LHR  NRT  TLV  MOW  MEX  LAX  

1번째 문자:LAX  ICN  MEX  BER  PEK  SFO  JFK  LHR  SIN  HKG  TLV  MOW  ROM  NRT  FRA  SYD  

0번째 문자:BER  FRA  HKG  ICN  JFK  LHR  LAX  MOW  MEX  NRT  PEK  ROM  SYD  SIN  SFO  TLV  

## LSD 알고리듬 정의 - 3

```python
# 3

def lsd(a) : 
    width = 3 # 문자열 크기
    n = len(a) # 입력 크기 
    asch = 128 # 아스키코드 총 수 128
    for d in reversed(range(width)): # 2,1,0
        count = [0]*asch # 0~127
        for i in range(n) : 
            count[ord(a[i][d])] += 1 # aschii 코드 집계 
        for i in range(1, len(count)) : 
            count[i] = count[i-1] + count[i] # count를 누적 값들로 변환 
        temp = [None] * n # 정렬 결과 담을 임시 리스트 
        for i in range(0, len(a)) : 
            temp[count[ord(a[i][d])] - 1] = a[i] # temp에 정렬 
            count[ord(a[i][d])] -= 1 # 정렬 할 게 하나 줄었다: -1 
        for i in range(0, len(a)) : 
            a[i] = temp[i]
        print(f'{d}번째 알파벳: {a}')
```

## 알고리듬 테스트 - 3

```python 
a = ['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW']
print(f'정렬 전:{a}')
print()
lsd(a)
```
정렬 전:\['ICN', 'SFO', 'LAX', 'FRA', 'SIN', 'ROM', 'HKG', 'TLV', 'SYD', 'MEX', 'LHR', 'NRT', 'JFK', 'PEK', 'BER', 'MOW'\]

2번째 알파벳: \['FRA', 'SYD', 'HKG', 'PEK', 'JFK', 'ROM', 'SIN', 'ICN', 'SFO', 'BER', 'LHR', 'NRT', 'TLV', 'MOW', 'MEX', 'LAX'\]

1번째 알파벳: \['LAX', 'ICN', 'MEX', 'BER', 'PEK', 'SFO', 'JFK', 'LHR', 'SIN', 'HKG', 'TLV', 'MOW', 'ROM', 'NRT', 'FRA', 'SYD'\]

0번째 알파벳: \['BER', 'FRA', 'HKG', 'ICN', 'JFK', 'LHR', 'LAX', 'MOW', 'MEX', 'NRT', 'PEK', 'ROM', 'SYD', 'SIN', 'SFO', 'TLV'\]










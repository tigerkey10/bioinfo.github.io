---
title : "[2021 인공지능전문가 교육과정 복습] 이진 힙 개념, Downheap과 Upheap, 이진 힙 구현"
excerpt : "부산대학교 인공지능전문가 교육과정 - 데이터사이언스:데이터 구조 수업 복습 후 정리"

categories : 
- Data Science
- python
- data structure

tags : 
- [data structure, computer science, study, data science, python]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-01-22
last_modified_at : 2022-01-22

---

# 이진 힙(Binary Heap)

## 정의

데이터 간 우선순위 있는 완전이진트리

- 우선순위: 부모노드 > 자식노드 

## 종류

- 최대힙: 부모노드 키 값이 자식노드 보다 같거나 큰 값 갖는 이진 힙
- 최소힙: 부모노드 키 값이  자식노드 보다 같거나 작은 값 갖는 이진 힙 

## 특징

- 반 정렬 상태다. 
- 중복 값 허용한다. 

## 응용

- 우선순위 큐 구현

---

# 이진 힙 연산 (최소힙 기준)

## 삭제연산(Downheap)

루트로부터 아래로 '내려가면서' 정렬(엄밀히 말해 정렬은 아니지만)이 진행된다. 그래서 Downheap 이라 부른다. 

## 결과

루트의 키가 삭제된다. 그리고 힙 속성 유지 위해 정렬한다. 

## 과정 

1. 힙의 가장 마지막 노드를 루트 자리에 갖다 놓는다. 그 과정에서 루트는 삭제된다. 
2. 힙 크기를 1 감소시킨다.
3. 새 루트의 왼쪽 / 오른쪽 자식 중 크기 더 작은 것 찾는다.
4. 더 작은 자식과 새 루트 키 크기 비교해서, 루트가 더 작으면 가만히 있는다. 자식이 더 작으면 루트와 그 자식 자리를 바꾼다. 
5. 최소힙 속성 만족할 때 까지 3~4 과정 반복하며 이파리 방향으로 진행한다. 

## 시간복잡도 

$O(log{n})$ 

---

## 삽입연산(Upheap)

이파리부터 위로 올라가면서 정렬이 진행된다. 그래서 Upheap 이라 부른다. 

## 결과

이진 힙 마지막 노드로 데이터를 삽입한다. 그리고 힙 속성 유지 위해 정렬한다. 

## 과정 

1. 힙 마지막 노드로 데이터 삽입한다. 
2. 데이터 삽입한 노드와 그 부모의 키를 비교해서, 부모가 더 크면 자식과 부모 노드를 맞교환한다. 
3. 2번 과정 반복하며 루트 향해 올라간다. 

## 시간복잡도

$O(log{n})$ 

---

# 이진 힙 구현 - 배열 리스트 이용 1

완전 이진트리에서, [ 왼쪽 $\Rightarrow$ 오른쪽 방향 ] 으로 노드 별 번호를 붙인다. 

노드 별 번호를 배열 리스트의 인덱스로 삼는다. 

- 번호는 1번부터 시작한다. 따라서 리스트 0번 인덱스 자리도 비운다.

## 최대힙

```python 
# 배열 리스트 이용해 최대힙 구현 

class Maxheap : 
    def __init__(self) : 
        self.heap = [] 
        self.heap.append(0)

    # 힙 크기 정의
    def size(self) : return len(self.heap) -1 

    # 힙이 비었다'의 정의
    def isEmpty(self) : return len(self.heap) == 0 

    # i번째 노드의 부모 노드 
    def parent(self, i) : return self.heap[i//2]

    # i번째 노드의 왼쪽 자식 노드 
    def left(self, i) : return self.heap[2*i]

    # i번째 노드의 오른쪽 자식 노드 
    def right(self, i) : return self.heap[2*i+1]

    # 힙 출력 정의 
    def print_heap(self) : print(self.heap[1:])

    # 최대힙 삽입연산 정의
    def insert(self, n) : 
        self.heap.append(n)
        # 삽입한 데이터 현재위치 
        i = self.size()
        # 정렬
        while i != 1 and n > self.parent(i) : # 정렬 일어나는 조건 정의: 루트가 아니면서, 자식이 부모보다 클 때 
            # 정렬은 어떻게 발생하는가?의 정의
            self.heap[i] = self.parent(i) # 자식의 위치로 부모가 이동 
            i = i // 2 # 자식이 부모 위치로 이동  
        self.heap[i] = n 

    # 최대힙 삭제연산 정의
    def delete(self) : 
        parent = 1 
        child = 2 
        if not self.isEmpty() : # 힙이 비어있지 않을 때 
            heaproot = self.heap[1]
            last = self.heap[self.size()]
            # 정렬 
            while child <= self.size() : # self.size() 이하 인덱스. 모든 노드에 대해 반복. 특정조건 나오면 서라. 
                if (child < self.size()) and (self.left(parent) < self.right(parent)) : # 1. 왼쪽.오른쪽 자식 비교해서 더 큰 놈 가려낸다. 
                    child += 1 
                if last > self.heap[child] : break # 2. 자식과 부모 비교 -> 속성 만족하면 선다. 
                # 속성 만족 안 하면 정렬한다. 
                self.heap[parent] = self.heap[child] # 원래 부모가 있던 자리에 자식이 간다. 
                parent = child # 원래 자식이 있던 자리는 또 다른 부모의 자리다. 
                child = 2*parent # 새 부모자리의 왼쪽 자식(인덱스)
            self.heap[parent] = last 
            self.heap.pop(-1)
            return heaproot 
```

## 최대힙이 잘 작동하는 지 테스트 

```python 
if __name__ == '__main__' : 
    m = Maxheap()
    m.insert([90, 'watermelon']);m.insert([80, 'pear'])
    m.insert([70, 'melon']);m.insert([50, 'lime'])
    m.insert([60, 'mango']);m.insert([20, 'cherry'])
    m.insert([30, 'grape']);m.insert([35, 'orange'])
    m.insert([10, 'app']);m.insert([15, 'banana'])
    m.insert([45, 'lemon']);m.insert([40, 'kiwi'])
```

```python 
if __name__=='__main__' : 
    print(f'최대 힙:', end='')
    m.print_heap()
    print()
    print(f'최댓값 삭제 후:', end='')
    m.delete()
    m.print_heap()
    print()
    m.insert([5, 'apple'])
    print(f'5 삽입 후:', end='')
    m.print_heap()
    print()
```
최대 힙:\[\[90, 'watermelon'\], \[80, 'pear'\], \[70, 'melon'\], \[50, 'lime'\], \[60, 'mango'\], \[40, 'kiwi'\], \[30, 'grape'\], \[35, 'orange'\], \[10, 'app'\], \[15, 'banana'\], \[45, 'lemon'\], \[20, 'cherry'\]\]

최댓값 삭제 후:\[\[80, 'pear'\], \[60, 'mango'\], \[70, 'melon'\], \[50, 'lime'\], \[45, 'lemon'\], \[40, 'kiwi'\], \[30, 'grape'\], \[35, 'orange'\], \[10, 'app'\], \[15, 'banana'\], \[20, 'cherry'\]\]

5 삽입 후:\[\[80, 'pear'\], \[60, 'mango'\], \[70, 'melon'\], \[50, 'lime'\], \[45, 'lemon'\], \[40, 'kiwi'\], \[30, 'grape'\], \[35, 'orange'\], \[10, 'app'\], \[15, 'banana'\], \[20, 'cherry'\], \[5, 'apple'\]\]

---

# 이진 힙 구현 - 배열 리스트 이용 2

좀 더 직관적인 방식이다. 

downheap, upheap 의 정의를 삽입. 삭제 정의 내부에 두지 않고, 따로 빼서 정의했다. 

결과로, 삽입. 삭제 연산 정의할 때는 정의된 downheap, upheap을 갖다 쓰기만 하면 된다. 

## 최소힙 

```python 
# 이진힙 구현 - 2
# 좀 더 직관적 방식

class BinaryHeap : 
    def __init__(self, base) : 
        self.heap = base # 리스트
        self.n = len(base) - 1

    # 최초 힙 생성(초기정렬)
    def create_heap(self) : 
        for i in range(self.n//2, 0, -1) : 
            self.downheap(i)

    # 삽입연산 정의
    def insert(self, n) : 
        self.heap.append(n)# 삽입
        self.n += 1 # 힙 크기 + 1
        self.upheap(self.n)# 정렬

    # 삭제연산 정의
    def delete_minimum(self) : 
        if self.n != 0 : # 이진힙 비어있지 않을 때 
            minimum = self.heap[1] # 루트노드
            self.heap[1], self.heap[-1] = self.heap[-1], self.heap[1] # 루트와 맨 마지막 노드 서로 바꾼다
            self.heap.pop(-1) 
            self.n -= 1
            self.downheap(1) # 루트노드1 부터 정렬 
            return minimum
    
    # downheap 정렬 
    def downheap(self, i) : # i는 현재노드
        while (2*i <= self.n) : 
            k = 2*i
            if (k < self.n) and (self.heap[k][0] > self.heap[k+1][0]) : # i의 왼쪽 자식 vs 오른쪽 자식 중 더 작은 것. 
                k = k+1
            if self.heap[i][0] <= self.heap[k][0] : break # 최소힙 속성 만족하는 지 검사 
            self.heap[i], self.heap[k] = self.heap[k], self.heap[i] # 정렬 
            i = k 
    
    # upheap 정렬 
    def upheap(self, j) : # j는 현재노드 
        while (j > 1) and (self.heap[j//2][0] > self.heap[j][0]) : # 현재노드가 루트가 아닐 때 & 부모가 자식보다 클 때 
            self.heap[j//2], self.heap[j] = self.heap[j], self.heap[j//2] # 부모-자식 교환
            j = j//2

    # 힙 출력  
    def print_heap(self) : 
        for i in range(1, self.n+1) : 
            print(f'[{self.heap[i][0]}, {self.heap[i][1]}]', end='')
        print()
        print(f'힙 크기: {self.n}')
```

참고) 

create_heap() 은 '최소힙 상태로 초기정렬 하는' 메소드다. 

- 완전이진트리에서 각 단위 서브트리 돌면서 downheap 
- downheap(맨 마지막 노드의 부모~ 루트)

---

## 구현된 최소힙이 잘 작동하는 지 테스트 

```python 
# 위 스크립트를 모듈로 만들고, 저장했다. 
# 따라서 먼저 모듈을 호출한다. 

import sys 
import os
sys.path.append('/Users/kibeomkim/Desktop/datascience/DIY_modules')
os.listdir('/Users/kibeomkim/Desktop/datascience/DIY_modules')
```
```python 
# 이진힙 클래스 호출 

from binary_heap import BinaryHeap
if __name__ == '__main__' : 
    a = [None]
    a.append([90, 'watermellon']); a.append([80, 'pear'])
    a.append([70, 'melon']); a.append([50, 'lime'])
    a.append([60, 'mango']);a.append([20, 'cherry'])
    a.append([30, 'grape']);a.append([35, 'orange'])
    a.append([10, 'apricot']);a.append([15, 'banana'])
    a.append([45, 'lemon']);a.append([40,'kiwi'])
    bh = BinaryHeap(a)
```
```python 
if __name__ == '__main__' : 
    print(f'힙 만들기 전:', end='')
    bh.print_heap() 
    bh.create_heap()
    print(f'최소힙:', end='')
    bh.print_heap()
    print(f'최솟값 삭제 후:', end='')
    print(bh.delete_minimum())
    bh.print_heap() 
    bh.insert([5, 'apple'])
    print(f'5 삽입 후:', end='')
    bh.print_heap()
```

힙 만들기 전:\[90, watermellon\]\[80, pear\]\[70, melon\]\[50, lime\]\[60, mango\]\[20, cherry\]\[30, grape\]\[35, orange\]\[10, apricot\]\[15, banana\]\[45, lemon\]\[40, kiwi\]

힙 크기: 12

최소힙:\[10, apricot\]\[15, banana\]\[20, cherry\]\[35, orange\]\[45, lemon\]\[40, kiwi\]\[30, grape\]\[80, pear\]\[50, lime\]\[60, mango\]\[90, watermellon\]\[70, melon\]

힙 크기: 12

최솟값 삭제 후:\[10, 'apricot'\]
\[15, banana\]\[35, orange\]\[20, cherry\]\[50, lime\]\[45, lemon\]\[40, kiwi\]\[30, grape\]\[80, pear\]\[70, melon\]\[60, mango\]\[90, watermellon\]

힙 크기: 11

5 삽입 후:\[5, apple\]\[35, orange\]\[15, banana\]\[50, lime\]\[45, lemon\]\[20, cherry\]\[30, grape\]\[80, pear\]\[70, melon\]\[60, mango\]\[90, watermellon\]\[40, kiwi\]

힙 크기: 12























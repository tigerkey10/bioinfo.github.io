---
title : "[2021 인공지능전문가 교육과정 복습] 트리, 이진트리, 이진트리 순회"
excerpt : "부산대학교 인공지능전문가 교육과정 - 데이터사이언스:데이터 구조 수업 복습 후 정리"

categories : 
- Data Science
- python
- data structure

tags : 
- [data structure, python, computer science, study, data science]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-01-20
last_modified_at : 2022-01-20

---

# 트리(Tree)

비선형자료구조. 

## 정의

계층. 

또는 부모/자식/형제.

## 특징

- 계층구조 표현에 적합하다. (조직계층구조, OS 파일시스템 등)
- 일반트리와 이진트리로 구분된다. 
- Empty 이거나 루트노드R 과 서브트리 집합으로 구성된다. 단, 서브트리 집합은 공집합일 수 있다. 
- 각 서브트리 루트노드는 루트노드R 의 자식노드다. 

## 응용 

탐색트리, 힙, 구문트리 등 

---

# 트리 용어 

<img width="1231" alt="Screen Shot 2022-01-20 at 10 06 01" src="https://user-images.githubusercontent.com/83487073/150243696-eec4a42f-a0c1-4d56-a00e-0a49d1b742c1.png">

- 루트노드 R: 트리 최상단 1개 노드. 부모가 없다. 
- 단말노드(이파리): 끝노드. 자식이 없다. 
- 내부노드: 단말노드 아닌 모든 노드 
- 링크(가지): 각 노드 연결
- 형제: 같은 부모에서 나온 노드들
- 노드 크기: 자기자신 & 그 자손노드 갯수 
- 노드 깊이: 루트노드R에서 특정노드까지 가기 위해 거치는 가지 수 
- 노드레벨: 특정 노드 깊이 갖는 노드들의 집합 
- 노드 차수: 각 노드가 가진 가지 수 
- 트리 차수: $max($ 노드 차수 $)$
- 트리 높이: 노드레벨 수 
- 서브트리: 루트노드R 에 달려있는 모든 하위 트리 
- 조상: 어떤 노드에서 루트R 까지 경로에 있는 모든 상위노드 
- 자손: 특정 노드 아래 매달린 모든 노드 

---

# 일반트리 표현방법 

## 왼쪽 자식-오른쪽 형제 표현

부모/자식/형제 표현 가능하다. 

- 노드 왼쪽 링크필드에 가장 왼쪽 자식노드, 오른쪽 링크필드에 오른쪽 형제노드 넣는다. 
- 링크필드 메모리 아낄 수 있다. 

<img width="788" alt="Screen Shot 2022-01-20 at 10 22 55" src="https://user-images.githubusercontent.com/83487073/150245183-b60596c3-5769-4d4a-9b41-fa7bdb69d914.png">

왼쪽 자식-오른쪽 형제 표현법을 써서 일반트리 나타내면 비어있는(None) 링크필드를 최대한 줄일 수 있을 것이다. 위 오른쪽 그림에서는 '.' 이 없는 링크가 None 이 들어간 링크다. 

## N-링크 표현법

모든 노드가 트리차수 N 만큼 링크필드 갖는 방법이다. 

<img width="340" alt="Screen Shot 2022-01-20 at 10 30 19" src="https://user-images.githubusercontent.com/83487073/150245888-da402323-7718-4f80-81f4-144607f84ccc.png">

- None 링크필드가 너무 많이 생긴다. 메모리 비효율적 활용 한다. 
- 예컨대, 트리차수 N 인데 자식노드 수 N-3인 부모노드가 있다고 하자. 3개의 None 저장된 링크필드 생긴다. 이 3개만큼 메모리 낭비다. 
- 트리차수 N 커질 수록, 메모리 낭비 더 심해진다. 따라서 이 방법은 비효율적 방법이다. 

---

# 이진트리(Binary Tree)

## 정의

각 노드 자식 수가 2 이하 $(0,1,2)$ 인 트리 

## 종류

### 포화 이진 트리(Full binary tree): 각 레벨에 노드가 꽉 찬 이진트리 

<img width="168" alt="Screen Shot 2022-01-20 at 10 44 01" src="https://user-images.githubusercontent.com/83487073/150247222-48acac87-d35c-4f0b-9af4-c74fb7d6dafe.png">

### 완전 이진 트리(Complete binary tree): 단말노드가 꽉 차거나 비는 이진트리 

<img width="399" alt="Screen Shot 2022-01-20 at 10 44 46" src="https://user-images.githubusercontent.com/83487073/150247289-7960858c-3bdd-409f-ac45-21cfe1f318f2.png">

- 포화 이진 트리는 완전 이진 트리의 한 종류다. 하지만 반대는 성립 안 한다. 

### 편향 이진 트리(Skewed binary tree): 모든 노드 한쪽방향으로 치우친 이진트리. 

<img width="163" alt="Screen Shot 2022-01-20 at 10 45 13" src="https://user-images.githubusercontent.com/83487073/150247318-80845f62-b0c7-45b5-80d4-36337d49fbc3.png">

- 트리높이가 높아질 수록 메모리 낭비 심해진다. None이 들어간 링크필드가 많아져서 그렇다. 

## 규칙 

- 노드 개수가 $n$ 개 이면 가지 갯수는 $n-1$

예) 노드 7개, 가지 6개 

- 트리높이가 $h$ 이면 노드 갯수는 $h$~ $2^{h}-1$

예) 트리높이 3, 노드 3~7

- $n$ 개 노드 갖는 이진트리 높이는 $[\log_{2}{(n+1)}]$~$n$

예) 3개 노드 갖는 이진트리 높이: $[\log_{2}{(4)}]$~$3$

예) 7개 노드 갖는 이진트리 높이: $[\log_{2}{(8)}]$~$7$

---

# 이진트리 표현방법 

## 배열 리스트로 이진트리 표현하는 방법 

- 리스트 0번 인덱스는 비운다. 
- 1번 인덱스에 루트노드R 넣는다. 
- 짝수노드 i의 부모노드 인덱스: i/2
- 홀수노드 i의 부모노드 인덱스: i/2-0.5
- 노드 i 왼쪽자식 노드의 인덱스: 2i 
- 노드 i 오른쪽 자식 노드의 인덱스: 2i+1

### $\Rightarrow$ 부모/자식/형제 표현 가능하다. 

## 포화 이진트리 배열 리스트로 표현한 예

<img width="647" alt="Screen Shot 2022-01-20 at 10 59 15" src="https://user-images.githubusercontent.com/83487073/150248774-ccc36ff2-5388-41e1-b214-6430e4c14260.png">

## 편향 이진트리 배열 리스트로 표현한 예 

<img width="679" alt="Screen Shot 2022-01-20 at 11 03 39" src="https://user-images.githubusercontent.com/83487073/150250913-15abb218-bfe0-48e4-9cc4-4eb20822a0c4.png">

---

## 연결리스트로 이진트리 표현하는 방법 

### 왼쪽 자식-오른쪽 자식 표현 사용 

- 노드 왼쪽 링크필드에는 왼쪽 자식, 오른쪽 링크필드에는 오른쪽 자식 할당한다. 
- None 들어가는 링크필드 최대한 줄일 수 있다. 따라서 메모리 효율적 사용 가능하다. 

```python
class Node : 
    def __init__(self, item, left=None, right=None) : 
        self.item = item # 노드가 담고 있는 데이터 
        self.left = left # 왼쪽 링크 필드에 왼쪽 자식
        self.right = right # 오른쪽 링크 필드에 오른쪽 자식
```

<img width="830" alt="Screen Shot 2022-01-20 at 11 18 24" src="https://user-images.githubusercontent.com/83487073/150258011-2282d2e5-550c-41d3-9db8-7e17bc07b097.png">

<img width="638" alt="Screen Shot 2022-01-20 at 11 18 46" src="https://user-images.githubusercontent.com/83487073/150258172-382e8661-f7a0-4c83-9e1c-9810059525b1.png">

---
# 순회 

이진트리가 연산(데이터 저장, 삭제, 출력)하는 방법이다. 

## 정의

모든 노드 한번씩 다 '방문'하는 것. 

- 노드 그냥 지나가는 것과 '방문' 하는 건 다르다. 

## 종류

<img width="351" alt="Screen Shot 2022-01-20 at 11 33 55" src="https://user-images.githubusercontent.com/83487073/150261706-1cd9fc4b-a7c8-4ef0-a082-f85bdc942d90.png">

- 전위순회: NLR

<img width="105" alt="Screen Shot 2022-01-20 at 11 36 37" src="https://user-images.githubusercontent.com/83487073/150261978-77aa9cf9-3967-4798-b8f1-63cfc49589ca.png">

- 중위순회: LNR

<img width="117" alt="Screen Shot 2022-01-20 at 11 37 09" src="https://user-images.githubusercontent.com/83487073/150262041-4c5e9f8f-c8e9-4254-9694-c1af277aebe7.png">

- 후위순회: LRN

<img width="126" alt="Screen Shot 2022-01-20 at 11 37 28" src="https://user-images.githubusercontent.com/83487073/150262082-36313cda-7584-4ec8-8676-062cc1895d7f.png">

## 이진트리 순회하면서 데이터 출력 예시

이진트리가 아래와 같은 포화 이진트리 일 때,

<img width="358" alt="Screen Shot 2022-01-20 at 11 38 16" src="https://user-images.githubusercontent.com/83487073/150262167-74c4eb62-7db3-4384-8544-69362b27ff26.png">

- 전위순회 결과: A B C D E F G
- 중위순회 결과: C B D A F E G
- 후위순회 결과: C D B F G E A

그 외

- 레벨순회: 레벨 순서로 순회, 왼쪽에서 오른쪽으로. 
- 레벨순회 결과: A B E C D F G

# 이진트리 순회 구현 

함수 재귀호출 사용해 이진트리 순회(전위, 중위, 후위) 구현한다. 

- 레벨순회는 재귀호출 대신 큐를 이용해 구현한다. 

### 전위순회 구현 
```python 
# 전위순회 구현 

class BinaryTree : 
    def __init__(self) : 
        self.root = None # 이진트리 empty 상태 

    # 전위순회
    def preorder(self, n) : 
        if n != None : # 트리가 empty 상태가 아닐 때 
            print(f'{str(n.item)}', end=' ')
            if n.left: self.preorder(n.left)
            if n.right: self.preorder(n.right)
```

### 중위순회 구현 

```python 
# 중위순회 구현 
class BinaryTree : 
    def __init__(self) : 
        self.root = None # 이진트리 empty 상태 

    # 중위순회 
    def inorder(self, n) : 
        if n != None : 
            if n.left: self.inorder(n.left)
            print(f'{str(n.item)}', end=' ')
            if n.right: self.inorder(n.right)
```

### 후위순회 구현 

```python 
# 후위순회 구현 
class BinaryTree : 
    def __init__(self) : 
        self.root = None # 이진트리 empty 상태 

    # 후위순회
    def postorder(self, n) : 
        if n != None : 
            if n.left: self.postorder(n.left)
            if n.right: self.postorder(n.right)
            print(f'{str(n.item)}', end='')
```

### 레벨순회 구현 

- 함수 재귀호출 사용 안 하고 큐로 구현한다. 

```python 
# 레벨순회 구현 
class BinaryTree : 
    def __init__(self) : 
        self.root = None # 이진트리 empty 상태 

    # 레벨순회
    def levelorder(self, root) : 
        if root != None : # 트리가 empty 가 아닐 때 
            q = []
            q.append(root)
            while len(q) != 0 : 
                result = q.pop(0) ;print(f'{result.item}', end=' ')
                if result.left: q.append(result.left)
                if result.right: q.append(result.right)
```

# 이진트리 노드 수, 트리 높이, 그리고 동일성 여부 계산 알고리듬 

## 노드 수 계산 

이진트리 노드 수 = 1 + (루트노드R 왼쪽 서브트리 노드 수) + (루트노드R 오른쪽 서브트리 노드 수)

### 파이썬으로 알고리듬 표현 
```python 
# 노드 수 계산 
class BinaryTree : 
    # 이진트리 생성자
    def __init__(self) : 
        self.root = None # 트리가 empty 상태

    # 트리 노드 수 계산 
    def size(self, n) : 
        if n == None : return 0 
        return 1 + self.size(n.left) + self.size(n.right)
```

## size() 재귀호출의 이해 

함수 재귀호출 사용해서 노드 수 계산 알고리듬을 표현했다. 

위 알고리듬에서 함수 재귀호출이 어떻게 동작하는지, 아래와 같은 순서로 이해할 수 있다. 

예를 들어 아래와 같은 완전 이진트리가 있다고 하자. 

<img width="257" alt="Screen Shot 2022-01-20 at 13 12 41" src="https://user-images.githubusercontent.com/83487073/150272039-9ec9b167-103e-4b65-bf3c-faab224f9aa8.png">

1. self.size(A.left)
2. self.size(B.left)
3. self.size(D.left)
4. size(None) $\Rightarrow$ 0 
5. self.size(D.right)
6. size(None) $\Rightarrow$ 0 
7. 1 + 0 + 0 $\Rightarrow$ 1 
8. self.size(B.right)
9. self.size(E.left)
10. size(None) $\Rightarrow$ 0 
11. self.size(E.right)
12. size(None) $\Rightarrow$ 0 
13. 1 + 0 + 0 $\Rightarrow$ 1
14. 1 + 1 + 1 $\Rightarrow$ 3
15. self.size(A.right)
16. self.size(C.left)
17. size(None) $\Rightarrow$ 0 
18. self.size(C.right)
19. size(None) $\Rightarrow$ 0 
20. 1 + 0 + 0 $\Rightarrow$ 1
21. 1 + 3 + 1 $\Rightarrow$ 5

---

## 트리 높이 계산 

이진트리 트리 높이 = 1 + $max($ 루트R의 왼쪽 서브트리 높이, 루트R 오른쪽 서브트리 높이 $)$

### 파이썬으로 알고리듬 표현 
```python 
# 트리 높이 계산 
class BinaryTree : 
    # 이진트리 생성자
    def __init__(self) : 
        self.root = None # 트리가 empty 상태

    # 트리 높이 계산
    def height(self, n) : 
        if n == None : return 0 # 트리가 empty 상태면 높이 = 0
        return 1 + max(self.height(n.left), self.height(n.right))
```
## height() 재귀호출의 이해 

함수 재귀호출 사용해서 트리 높이 계산 알고리듬을 표현했다. 

위 알고리듬에서 함수 재귀호출이 어떻게 동작하는지, 아래와 같은 순서로 이해할 수 있다. 

예를 들어 아래와 같은 완전 이진트리가 있다고 하자. 

<img width="257" alt="Screen Shot 2022-01-20 at 13 12 41" src="https://user-images.githubusercontent.com/83487073/150272039-9ec9b167-103e-4b65-bf3c-faab224f9aa8.png">

1. self.height(A.left)
2. self.height(B.left)
3. self.height(D.left)
4. height(None) $\Rightarrow$ 0 
5. self.height(D.right)
6. height(None) $\Rightarrow$ 0 
7. max(0, 0) + 1
8. self.height(B.right)
9. self.height(E.left)
10. height(None) $\Rightarrow$ 0 
11. self.height(E.right)
12. height(None) $\Rightarrow$ 0 
13. max(0, 0) + 1 $\Rightarrow$ 1
14. max(1, 1) + 1 $\Rightarrow$ 2
15. self.height(A.right)
16. self.height(C.left)
17. height(None) $\Rightarrow$ 0 
18. self.height(C.right) 
19. height(None) $\Rightarrow$ 0 
20. max(0, 0) + 1 $\Rightarrow$ 1
21. max(2, 1) + 1 $\Rightarrow$ 3

---

## 트리 동일성 여부 계산 

비교하려는 두 트리 루트노드R 부터 시작해서 각 노드 메소드 인자로 전달하며 하나씩 같은지 검사 

### 파이썬으로 알고리듬 표현 
```python 
class BinaryTree : 
    # 이진트리 생성자
    def __init__(self) : 
        self.root = None # 트리가 empty 상태

    # 트리 동일성 검사 
    def is_equal(self, n, m) : 
        if n is None or m == None : return n == m # n, m 둘 다 None 인 경우도 여기 걸린다. 
        if n.item != m.item : return False # 내용물이 다른 경우 
        else : # n,m 둘 다 none 아니고 내용물도 같은 경우 
            return (self.is_equal(n.left, m.left) and self.is_equal(n.right, m.right)) # 둘 다 True 일 때만 True 반환한다. 
```
## is_equal() 재귀호출의 이해 

함수 재귀호출 사용해서 트리 동일성 여부 계산 알고리듬을 표현했다. 

위 알고리듬에서 함수 재귀호출이 어떻게 동작하는지, 아래와 같은 순서로 이해할 수 있다. 

## 트리 동일성 검사 - 1

<img width="516" alt="Screen Shot 2022-01-20 at 22 23 58" src="https://user-images.githubusercontent.com/83487073/150347061-c81add91-348f-4a35-87b6-cb27b2ddeeee.png">

1. self.isequal(A.left, A.left)
2. self.isequal(B.left, B.left)
3. self.isequal(D.left, D.left)
4. isequal(None, None) $\Rightarrow$ True
5. self.isequal(D.right, D.right)
6. isequal(None, None) $\Rightarrow$ True
7. True and True $\Rightarrow$ True
8. self.isequal(B.right, B.right)
9. self.isequal(E.left, E.left)
10. isequal(None, None) $\Rightarrow$ True 
11. self.isequal(E.right, E.right)
12. isequal(None, None) $\Rightarrow$ True 
13. True and True $\Rightarrow$ True 
14. True and True $\Rightarrow$ True 
15. self.isequal(A.right, A.right)
16. self.isequal(C.left, C.left)
17. isequal(None, None) $\Rightarrow$ True 
18. self.isequal(C.right, C.right)
19. isequal(None, None) $\Rightarrow$ True 
20. True and True $\Rightarrow$ True 
21. True and True $\Rightarrow$ True 

## 트리 동일성 검사 - 2

<img width="504" alt="Screen Shot 2022-01-20 at 22 30 03" src="https://user-images.githubusercontent.com/83487073/150348082-ec96fcb7-37ee-4e82-a7b9-2658c7aa1ad0.png">

1. self.isequal(A.left, A.left)
2. self.isequal(B.left, B.left)
3. self.isequal(D.left, D.left)
4. isequal(None, None) $\Rightarrow$ True 
5. self.isequal(D.right, D.right)
6. isequal(None, None) $\Rightarrow$ True 
7. self.isequal(B.right, B.right)
8. isequal(E, None) $\Rightarrow$ False 
9. True and False $\Rightarrow$ False
10. self.isequal(A.right, A.right)
11. self.isequal(C.left, C.left)
12. isequal(None, None) $\Rightarrow$ True 
13. self.isequal(C.right, C.right)
14. isequal(None, None) $\Rightarrow$ True 
15. True and True $\Rightarrow$ True 
16. False and True $\Rightarrow$ False 

## 트리 동일성 검사 - 3

<img width="495" alt="Screen Shot 2022-01-20 at 22 34 38" src="https://user-images.githubusercontent.com/83487073/150348784-48002104-c69c-48ed-a867-3cd334f81e3e.png">

1. self.isequal(A.left, A.left)
2. isequal(E, X) $\Rightarrow$ False
3. self.isequal(A.right, A.right)
4. isequal(G, Z) $\Rightarrow$ False
5. False and False $\Rightarrow$ False

## 트리 동일성 검사 - 4

<img width="470" alt="Screen Shot 2022-01-20 at 22 36 46" src="https://user-images.githubusercontent.com/83487073/150349092-a1a52a11-b33a-4f13-ae16-cd9aed23114d.png">

1. self.isequal(A.left, A.left)
2. self.isequal(B.left, B.left)
3. isequal(None, E) $\Rightarrow$ False
4. self.isequal(A.right, A.right)
5. isequal(G, T) $\Rightarrow$ False
6. False and False $\Rightarrow$ False 

---

## + 트리 복제 

이파리 노드 부터 복제시작한다. 결과값으로 (아래로 서브트리를 달고 있는) 복제된 루트노드R을 반환한다. 

파이썬으로 알고리듬 표현 
```python 
class Node : 
    def __init__(self, item, left=None, right=None) : # 왼쪽자식-오른쪽자식 표현
        self.item = item 
        self.left = left # 왼쪽자식 
        self.right = right  # 오른쪽 자식
    
    # 트리 복제 
    def duplicate_tree(self, n) : 
        if n is None : return None
        else:
            left = self.duplicate_tree(n.left)
            right = self.duplicate_tree(n.right)
            return Node(n.item, left=left, right=right) # 새 루트 노드 생성(복제본)
```
## duplicate_tree() 재귀호출의 이해 

함수 재귀호출 사용해서 트리 복제 알고리듬을 표현했다. 

위 알고리듬에서 함수 재귀호출이 어떻게 동작하는지, 아래와 같은 순서로 이해할 수 있다. 

예를 들어 아래와 같은 완전 이진트리가 있다고 하자. 

<img width="257" alt="Screen Shot 2022-01-20 at 13 12 41" src="https://user-images.githubusercontent.com/83487073/150272039-9ec9b167-103e-4b65-bf3c-faab224f9aa8.png">

1. self.duplicate_tree(A.left)
2. self.duplicate_tree(B.left)
3. self.duplicate_tree(D.left)
4. duplicate_tree(None) $\Rightarrow$ None
5. self.duplicate_tree(D.right)
6. duplicate_tree(None) $\Rightarrow$ None
7. Node(D.item, None, None) 생성 
8. self.duplicate_tree(B.right)
9. self.duplicate_tree(E.left)
10. duplicate_tree(None) $\Rightarrow$ None
11. self.duplicate_tree(E.right)
12. duplicate_tree(None) $\Rightarrow$ None
13. Node(E.item, None, None) 생성 
14. Node(B.item, Node(D, None, None), Node(E, None, None)) 생성 
15. self.duplicate_tree(A.right)
16. self.duplicate_tree(C.left)
17. duplicate_tree(None) $\Rightarrow$ None
18. self.duplicate_tree(C.right)
19. duplicate_tree(None) $\Rightarrow$ None
20. Node(C.item, None, None) 생성 
21. Node(A, 14번 노드, 20번 노드) 생성 

---
# 연결리스트로 이진트리 구현 

왼쪽 자식-오른쪽 자식 표현 사용 

```python 
# 연결리스트로 이진트리 구현 

# 노드 정의
class Node : 
    def __init__(self, item, left=None, right=None) : # 왼쪽자식-오른쪽자식 표현
        self.item = item 
        self.left = left # 왼쪽자식 
        self.right = right  # 오른쪽 자식

# 이진트리 정의 
class BinaryTree : 
    # 이진트리 생성자
    def __init__(self) : 
        self.root = None # 트리가 empty 상태
    
    # 전위순회
    def preorder(self, n) : 
        if n != None : # 트리가 empty 상태가 아닐 때 
            print(f'{str(n.item)}', end=' ')
            if n.left: self.preorder(n.left)
            if n.right: self.preorder(n.right)
        
    # 중위순회 
    def inorder(self, n) : 
        if n != None : 
            if n.left: self.inorder(n.left)
            print(f'{str(n.item)}', end=' ')
            if n.right: self.inorder(n.right)

    # 후위순회
    def postorder(self, n) : 
        if n != None : 
            if n.left: self.postorder(n.left)
            if n.right: self.postorder(n.right)
            print(f'{str(n.item)}', end='')

    # 레벨순회
    def levelorder(self, root) : 
        if root != None : # 트리가 empty 가 아닐 때 
            q = []
            q.append(root)
            while len(q) != 0 : 
                result = q.pop(0) ;print(f'{result.item}', end=' ')
                if result.left: q.append(result.left)
                if result.right: q.append(result.right)

    # 트리 높이 계산
    def height(self, n) : 
        if n == None : return 0 # 트리가 empty 상태면 높이 = 0
        return 1 + max(self.height(n.left), self.height(n.right))

    # 트리 노드 수 계산 
    def size(self, n) : 
        if n == None : return 0 
        return 1 + self.size(n.left) + self.size(n.right)

    # 트리 복제 
    def duplicate_tree(self, n) : 
        if n is None : return None
        else:
            left = self.duplicate_tree(n.left)
            right = self.duplicate_tree(n.right)
            return Node(n.item, left=left, right=right) # 새 루트 노드 생성(복제본)
    
    # 트리 동일성 검사 
    def is_equal(self, n, m) : 
        if n is None or m == None : return n == m
        if n.item != m.item : return False 
        else : 
            return (self.is_equal(n.left, m.left) and self.is_equal(n.right, m.right))
```
## 연결리스트로 구현된 이진트리 잘 작동하는지 테스트 

```python 
# 위에서 작성한 파이썬 스크립트를 모듈화 시켜서 따로 저장해뒀다. 
# sys 함수 이용해 파이썬 스크립트 저장해둔 경로를 시스템 경로에 추가한다. 

import sys
sys.path.append('/Users/kibeomkim/Desktop')

from binary_tree import Node, BinaryTree

#print(__name__) # __main__ 이다. 

if __name__ == '__main__' : # 현재 실행환경이 메인 스크립트이므로 항상 True 일 것이다. 한편 모듈 불러온 뒤 모듈 내에 있던 __name__ 을 출력하면 그 모듈 이름이 출력된다. 
    t = BinaryTree() # 이진트리 객체 생성 
    n1 = Node(100);n2 = Node(200)
    n3 = Node(300);n4 = Node(400)
    n5 = Node(500);n6 = Node(600)
    n7 = Node(700);n8 = Node(800)
    # 100부터 800까지 값 넣은 8개 노드 생성 (연결 전)
    
    n1.left = n2 # n1 노드 왼쪽자식 = n2 
    n1.right = n3 # n1 노드 오른쪽자식 = n3
    n2.left = n4 # n2 노드 왼쪽자식 = n4 
    n2.right = n5 # n2 노드 오른쪽자식 = n5 
    n3.left = n6 # n3 노드 왼쪽자식 = n6 
    n3.right = n7 # n3 노드 오른쪽자식 = n7 
    n4.left = n8 # n4 노드 왼쪽자식 = n8 

    t.root = n1 # n1을 이진트리 루트노드 R로 설정. 

    print(f'트리높이: {t.height(t.root)}')
    print(f'전위순회:', end='')
    t.preorder(t.root)
    print('\n중위순회:', end='')
    t.inorder(t.root)
    print('\n후위순회:', end='')
    t.postorder(t.root)
    print('\n레벨순회:', end='')
    t.levelorder(t.root)
    print()
    print(f'트리 노드 수: {t.size(t.root)}')

    new_root = t.duplicate_tree(t.root) # 트리 복제 
    t2 = BinaryTree() # 새 이진트리 객체 호출 (empty 상태)
    t2.root = new_root # 새 이진트리 루트노드 R에 기존 트리 t의 루트노드 할당 (이제 t2는 복제된 이진트리)

    print('두 트리가 같은가?: ', end='')
    print(t.is_equal(t.root, t2.root)) # 두 트리 (t, t2) 동질성 검사: 둘은 같은 트리인가? 
    print() # 두 트리가 같다. 
```

트리높이: 4

전위순회:100 200 400 800 500 300 600 700 

중위순회:800 400 200 500 100 600 300 700 

후위순회:800 400 500 200 600 700 300 100 

레벨순회:100 200 300 400 500 600 700 800 

트리 노드 수: 8

두 트리가 같은가?: True


















































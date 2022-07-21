---
title : "[2021 인공지능전문가 교육과정 복습] 스택, 큐, 덱, 우선순위 큐"
excerpt : "부산대학교 인공지능전문가 교육과정 - 데이터사이언스: 데이터 구조 수업 복습 후 정리"

categories : 
- Data Science
- python
- data structure

tags : 
- [data structure, computer science, study, data science]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-01-12
last_modified_at : 2022-01-14

---

# 스택(Stack)

쌓기나무.

선형자료구조다. 

## 추상자료형

속성

- 데이터 접근 방식: 후입선출(Last In First Out) 
- 0개 이상 유한개 원소를 포함한다. 
- 각 원소는 순서(인덱스)를 갖는다. 

연산

- push(): 스택 맨 위에 새 데이터 삽입한다. 
- pop(): 스택 맨 위 데이터를 사용자에게 한번 보여주고 삭제한다. 
- peek(): 스택 맨 위 데이터가 뭔지 확인한다. 
- isEmpty(): 스택이 비어있는지 확인한다. 
- size(): 스택 내 원소 수를 확인한다. 

## 스택 구현 방법 

- 파이썬 기본 리스트 
- 단순연결리스트

파이썬 기본 리스트 후단을 top 으로 삼아 거기서만 데이터 삽입, 삭제 가능케 하면 스택으로 쓸 수 있다. 

---

# 스택 구현 - 파이썬 기본 리스트 이용하는 방법 

## 1. pythonds 클래스 사용

pythonds 클래스는 파이썬 기본 리스트 이용해서 스택을 미리 구현해놓은 클래스다. 

곧, 클래스 호출만으로 스택 자료구조를 바로 쓸 수 있다. 

pythonds 에서 구현해놓은 스택 연산은 

- push(): 삽입
- pop(): 반환 후 삭제
- peek(): 조회
- isEmpty(): 비었는지 확인
- size(): 원소 갯수 확인
- items(): 들어있는 모든 요소 리스트 형식으로 출력

```python
from pythonds.basic import Stack # 스택 클래스 호출 

st = Stack() # 스택 객체 st 생성 
print(st.isEmpty()) # 스택이 비어있는지 확인: 아무것도 안 넣었으니까 True

st.push(3);st.push(4);st.push(5);st.push('A') # 스택에 차례로 3,4,5,'A' 삽입
print(st.items) # 스택에 넣은 요소들 리스트 형태로 출력 

st.peek() # 맨 위에 있는 데이터 조회: 'A'

while st.size() != 0 : # 스택이 빌 때 까지
    print(st.pop()) # 후입선출로 하나씩 꺼내서 반환하고 삭제. A-5-4-3 순으로 나올 것이다. 

print(st.isEmpty())# 스택이 비어있는지 확인. pop()으로 다 꺼냈다. True
```
<img width="459" alt="Screen Shot 2022-01-09 at 14 30 06" src="https://user-images.githubusercontent.com/83487073/148670608-61ece173-72f1-4c9c-b433-08a9ffb742dd.png">



## 2. 파이썬 기본 리스트 사용해서 직접 스택 클래스 짜기 

파이썬 기본 리스트 사용해서 직접 스택 클래스를 쓰고, 스택 자료구조 객체를 직접 생성할 수 있다. 

```python
# 스택 클래스 생성 - 1

class Stack : 
    # 객체 생성 할 때 자동 호출 
    def __init__(self): 
        self.items = []

    # push() 연산 구현 
    def push(self, value): 
        self.items.append(value) # 리스트 후단에 데이터 추가 
    
    # pop() 연산 구현 
    def pop(self): 
        print(self.items[-1])
        del self.items[-1]

    # peek() 연산 구현 
    def peek(self): 
        print(self.items[-1])

    # isEmpty() 연산 구현 
    def isEmpty(self): 
        if len(self.items) == 0 : 
            return True
        else : 
            return False
        
    # size() 연산 구현 
    def size(self): 
        return len(self.items)

st2 = Stack() # 스택 객체 생성 
```
하지만 위 방법은 스택이 비어 있을 때 pop(), peek() 연산에서 에러를 발생시킨다는 단점이 있다. 

더 나은 방법 

```python
# 스택 클래스 생성 - 2

class Stack : 
    def __init__(self) : 
        self.items = []

    # push()
    def push(self, item) : 
        self.items.append(item)

    # pop() : 스택이 비어 있을 때는 아무 동작도 하지 않는다. 
    def pop(self) : 
        if not self.isEmpty() : 
            self.items.pop()
    
    # peek() : 스택이 비어 있을 때는 아무 동작도 하지 않는다. 
    def peek(self) : 
        if not self.isEmpty() : 
            return self.items[len(self.items)-1]

    # isEmpty()
    def isEmpty(self) : 
        return self.items == []

    # size()
    def size(self) : 
        return len(self.items)


st2 = Stack()
```

---
# 스택 구현 - 단순연결리스트 이용하는 방법 

## 단순연결리스트 이용

- push 연산할 때 마다 매번 노드 생성 
- 새 노드 생성되면 곧바로 헤드포인터 부여 

### 연결리스트에 데이터 추가 절차 (push)
1. 노드 생성 
2. 데이터 투입 
3. 링크 설정
4. 헤드포인터 지정 

### 연결리스트에서 데이터 빼는 절차 (pop)
1. 헤드포인터만 이동시키면 끝난다.
2. 원래 top에 뭐가 들어있었는지 보여준다.

```python
# 단순연결리스트로 스택 구현하기 

# 노드 정의 
class Node : 
    def __init__(self, data, link) : 
        self.item = data # 데이터 필드 
        self.next = link # 링크 필드 

# 빈 단순연결리스트의 헤드포인터 
top = None
# 빈 단순연결리스트 사이즈 
size = 0

# push(): 삽입 
def push(data) : 
    global top # 전역변수 top 가져와서 쓸 것이다
    global size # 전역변수 size 가져와서 쓸 것이다 

    # 새 노드 생성, 데이터 삽입, 링크 지정, 그리고 헤드포인터 지정 
    top = Node(data, top)
    # 데이터 1 추가할 때. 연결리스트 크기 1 증가 
    size += 1 

# pop(): 삭제 
def pop() : 
    global top 
    global size 

    if size != 0 : # 스택이 차 있을 때 만
        top_item = top.item

        # 연결리스트에서 삭제: 헤드포인터만 다음 노드로 옮겨간다
        top = top.next 

        # 연결리스트 노드 수 -1
        size -= 1

        return top_item # 연결리스트에서 뺀 노드에 뭑가 들어있었는지 반환 

# peek(): 조회
def peek() : 
    global top 

    if size != 0 : # 스택이 차 있을 때 만
        return top.item

# print_stack(): 스택 내용물 전체 출력 
def print_stack() : 
    global top 
    global size

    print('top ->', end='')

    p = top # 현재 노드 
    while p : 
        if p.next != None : # 후단 노드가 아닐 때 
            print(f'{p.item} ->', end='')
        else : # 후단에 도달 했을 때 
            print(p.item, end='')
        p = p.next # 다음 노드로 이동 
    print()

# 스택 내용물 모두 제거 
def stack_clear() : 
    global top 
    global size 

    # 연결리스트 헤드포인터를 None에 지정하면 된다. 
    # 맨 처음 빈 연결리스트 상태로 돌아간다. 
    top = None
    size = 0
```

## 스택이 잘 생성되었는지 테스트 

```python
# 스택이 잘 생성되었는지 테스트 
push('apple')
push('orange')
push('cherry')
print('사과, 오렌지, 체리 스택에 push 후:', end='')
print_stack() # 스택 출력 
```
<img width="435" alt="Screen Shot 2022-01-09 at 16 53 22" src="https://user-images.githubusercontent.com/83487073/148673940-c50f347f-a498-47e3-b19b-a371ca63be81.png">

```python
print('top 항목: ', end='')
print(peek())
```

<img width="146" alt="Screen Shot 2022-01-09 at 16 53 57" src="https://user-images.githubusercontent.com/83487073/148673956-28dd2a31-82bd-4a2e-994e-38d4fe8cc296.png">

```python
push('pear')
print(f'배 push 후: ', end='')
print_stack()
```
<img width="378" alt="Screen Shot 2022-01-09 at 16 54 27" src="https://user-images.githubusercontent.com/83487073/148673975-25941422-b95b-4d37-9996-0d75298b5c7b.png">

```python
pop()
```
<img width="296" alt="Screen Shot 2022-01-09 at 16 54 53" src="https://user-images.githubusercontent.com/83487073/148674000-4f7b66b9-7f0c-4c1d-9c9e-ddf63b68ace2.png">

```python
push('grape')
print('pop(), 포도 push 후:', end='')
print_stack()
```
<img width="504" alt="Screen Shot 2022-01-09 at 16 55 23" src="https://user-images.githubusercontent.com/83487073/148674039-1a293b03-5bae-4de9-bfd0-6557acd7496e.png">

---

# 큐(Queue)

줄서기. 

선형자료구조다. 

데이터 삭제는 큐 전단에서, 삽입은 큐 후단에서 일어난다.

## 추상자료형 

속성 

- 데이터 접근 방식: 선입선출(First In First Out)
- 0개 이상 유한개 원소 포함한다. 
- 각 원소는 순서(인덱스)를 갖는다. 

연산 

- enqueue(): 큐 후단에 데이터 삽입한다.
- dequeue(): 큐 전단에서 데이터 꺼내 반환하고 삭제한다. 
- peek(): 큐 전단에 데이터 조회한다. 
- isEmpty(): 큐가 비어있는지 확인한다. 
- size(): 큐 안에 데이터가 몇개 들었는지 확인한다. 

## 큐 구현 방법 

- 파이썬 내장 queue 모듈의 Queue 클래스 
- 파이썬 기본 리스트
- 단순연결리스트 

---

# 큐 구현 - 파이썬 내장 queue 모듈의 Queue 클래스

## Queue 클래스 

queue 모듈의 Queue 클래스를 사용하면, 별도의 클래스 정의나 함수 정의 없이 큐 자료구조를 바로 활용 가능하다. 

Queue 클래스에서 구현해 놓은 큐 연산은 

- enqueue: 삽입(put(item))
- dequeue: 삭제(get())

```python
# queue 모듈 Queue 클래스 활용해서 큐 자료구조 사용하기 

from queue import Queue

que = Queue()
que.put(3)
que.put(4)
que.put(5)
que.put(8)
que.put(10)

for i in range(5) : 
    print(que.get(), end=',')
print()
```
결과: 3,4,5,8,10,

---

# 큐 구현 - 파이썬 기본 리스트 이용하는 방법 

데이터 저장은 파이썬 기본 리스트에 하되, 특별함수 만들어서 데이터 삽입과 삭제를 큐 처럼 하는 방법이다.

```python
# 파이썬 기본 리스트로 큐 구현 

que = [] # 큐

# 데이터 삽입
def enqueue(item) :
    global que
    que.append(item) # 데이터 삽입은 리스트 후단에서만 일어나도록 

# 데이터 삭제 
def dequeue() : 
    global que
    if len(que) != 0 : 
        item = que.pop(0) # 데이터 삭제는 리스트 전단에서만 일어나도록 
        return item # 삭제 한 항목 반환
```
---

# 큐 구현 - 단순연결리스트 이용하는 방법 

단순연결리스트로 큐 생성 전 명심해야 할 것

1. 큐는 전단과 후단이 있다. 
2. 새로 만든 노드는 항상 큐의 가장 후단이다.
3. 노드 생성은 데이터 삽입 할 때 마다 발생한다.  

```python 
# 단순연결리스트로 큐 구현 시도 1 

# 빈 큐
top = None
size = 0
pre_node = None

# 노드 정의
class Node : 
    def __init__(self, data, link) : 
        self.item = data # 데이터 필드 
        self.next = link # 링크 필드 

# 삽입 
def enqueue(item) : 
    global top 
    global size
    global pre_node
    if size == 0 : # 큐에 처음으로 뭘 넣을 때 
        # 헤드포인터 노드 생성 
        top = Node(item, None)
        size += 1
        pre_node = top
    else : # 그게 아닌 모든 경우 
        # 새 노드 생성 
        new_node = Node(item, None)
        size += 1
        # 새 노드를 이전 노드 링크와 연결 
        pre_node.next = new_node
        pre_node = new_node

# 삭제 : 헤드포인터만 이동하면 된다. 
def dequeue() : 
    global top 
    global size 
    if size != 0 : 
        deleted_one = top.item 
        top = top.next # 헤드포인터 변경 : 삭제 끝 
        size -= 1
        return deleted_one 

# 조회 : 헤드포인터 노드에 담긴 데이터 확인한다. 
def peek() : 
    global top 
    global size 
    if size != 0 : 
        return top.item 

# 큐가 비어있는지 확인 
def isEmpty() : 
    global size 
    return size == 0 

# 큐의 자료 수 확인 
def que_size() : 
    global size 
    return size 

def clear() : 
    global top 
    global size 
    top = None
    size = 0

def print_q() : 
    global top 
    global size 

    if size != 0 : 
        n = top 
        print(f'front ->', end='')
        for i in range(size) : 
            print(f'{n.item}', end=' ')
            n = n.next
        print()
```

삽입 연산 enqueue() 가 필요이상으로 번잡하다. 

큐에 대한 개념(concept) 및 이해를 좀 더 명료하게 해야 할 필요가 있었다. 

- 큐에 처음으로 데이터 삽입하면 그 큐는 전단이면서 동시에 후단이다. 
- 이후, 큐에 데이터 삽입하는 '작업'은 '큐 후단에 새 노드를 갖다 붙이는 작업'이다. 
- 새 노드는 항상 큐의 후단이 된다. 

```python 
# 단순연결리스트로 큐 구현 시도 - 2 

# 빈 큐
front = None
rear = None
size = 0 

# 노드 정의 
class Node : 
    def __init__(self, item, link) : 
        self.item = item 
        self.next = link 
    
# 삽입 
def enqueue(item) : 
    global front 
    global rear
    global size 

    p = Node(item, None)
    if size == 0 : # 첫 번째면 
        front = p 
    else : # 그 외에 
        rear.next = p 
    
    rear = p 
    size += 1 

# 삭제 
def dequeue() : 
    global front 
    global rear
    global size 

    if size != 0 : 
        deleted = front.item # 삭제된 항목 
        front = front.next 
        size -= 1 
        if size == 0 : 
            rear = None 
        return deleted 

# 조회 
def peek() : 
    global front 
    return front.item 

# 삭제 
def clear() : 
    global front 
    global rear 
    global size 

    front = None 
    rear = None 
    size = 0 

# 내용물 출력 
def print_que() : 
    global front 
    global rear 
    global size 

    p = front 
    print(f'front ->', end=' ')
    while p : # 큐 끝까지 돌아라 
        print(f'{p.item}', end=' ')
        p = p.next # 다음 노드로 가라 
    print() 
```

---

# 큐 구현 - 선형 큐의 문제점 

선형 큐 삽입 연산은 큐를 어떻게 구현하든 $O(1)$ 만큼 시간복잡도만 걸린다. 효율적이다. 

### 반면 삭제 연산 할 때 선형 큐는 시간 또는 공간을 비효율적으로 사용한다. 

## 높은 시간복잡도 $O(n)$ - 배열 리스트로 구현된 선형 큐 

삭제연산을 하면 배열에서 0번 인덱스가 삭제되고, 1번 인덱스부터 n번까지가 모두 한 칸씩 앞으로 물리적 이동을 한다. 당연히 시간복잡도는 $O(n)$ 이다. 

매 삭제연산마다 $O(n)$ 만큼 시간복잡도가 걸린다는 건 매우 비효율적이다. 

## 비효율적 공간 사용 - 단순연결리스트로 구현된 선형 큐 

삭제연산 하면 큐의 전단(front) 을 나타내는 헤드포인터가 한 칸 뒤로 이동한다. 

그러면 원래 데이터가 있었던 공간을 큐가 사용할 수 없게 된다. 

# 원형 큐 - 선형 큐 문제점 해결 

## 1차원 배열 리스트(파이썬 기본 리스트) 를 '원형'으로 '사용'해서 원형 큐 구현한다.

## 특징 

Front 포인터와 Rear 포인터 둘 다 이동한다. 

- 삽입연산: Rear 포인터 이동 
- 삭제연산: Front 포인터 이동 

두 포인터는 시계방향으로 이동한다. 

Front 포인터 위치(노드)는 항상 비워둔다. 

## 개념 정의 

## 1. 원형 큐가 비었다

front 포인터 위치 == rear 포인터 위치

## 2. 원형 큐가 포화상태다

(rear+1)%MAX_SIZE = front

## 3. 삽입연산(enqueue)

front 포인터 고정.

1. 새 rear 포인터 위치: rear = (rear+1)%MAX_SIZE 
2. 새 위치로 rear 포인터 이동 
3. 새 rear 포인터 위치에 데이터 삽입 

만약 새 rear 포인터 위치가 (rear+1)%MAX_SIZE == front 이면 큐가 포화상태, 삽입연산 수행하지 않는다. 

## 4. 삭제연산(dequeue)

rear 포인터 고정. 

1. 새 front 포인터 위치: front = (front+1)%MAX_SIZE
2. 새 위치로 front 포인터 이동 
3. 새 front 포인터 위치의 데이터 삭제 

만약 front 포인터 위치 == rear 포인터 위치 이면 원형 큐가 공백 상태다. 삭제연산 수행하지 않는다. 

# 원형 큐 구현 

## 구현방법

파이썬 기본 리스트 자료구조 이용 

## 개념 정의

1. 원형 큐가 비었다 $=$ front 포인터와 rear 포인터 위치가 같다. 
2. 원형 큐에 데이터 삽입한다 $=$ front 포인터 고정, rear 포인터만 새 위치로 이동 후 그 위치에 데이터 삽입 
3. 원형 큐에서 데이터 삭제한다 $=$ rear 포인터 고정, front 포인터만 새 위치로 이동, 원래 그 위치에 있던 데이터 반환 
4. 원형 큐가 포화상태다 $=$ (rear+1)%MAX_SIZE = front 
5. 원형 큐 front 포인터와 rear 포인터는 두 가지 상황만 갖는다. rear 가 front 앞에 있던지. 혹은 뒤에 있던지. 
6. front 포인터 위치는 항상 비어있다. 

```python
# 1차원 배열 리스트로 원형 큐 구현 

class Circularque : 
    def __init__(self) : 
        self.MAX_SIZE = 5 # 큐 최대 사이즈 5로 지정 
        self.front = 0 # front 포인터 시작점 
        self.rear = 0  # rear 포인터 시작점 
        self.q = [None]*self.MAX_SIZE  

    # 포화상태 정의 
    def isFull(self) : return (self.rear+1)%self.MAX_SIZE == self.front 
    
    # 공백상태 정의 
    def isEmpty(self) : return self.front == self.rear 
    
    # 삽입연산 
    def enqueue(self, item) : 
        if not self.isFull() : # 포화상태가 아닐 때 
            self.rear = (self.rear+1)%self.MAX_SIZE # 다음 리어 포인터 위치 
            self.q[self.rear] = item 
    
    # 삭제연산 
    def dequeue(self) : 
        if not self.isEmpty() : # 공백상태가 아닐 때 
            self.front = (self.front+1)%self.MAX_SIZE # 다음 프론트 포인터 위치 
            deleted = self.q[self.front]
            return deleted # 뭐가 삭제됬는지 반환 

    # 원형 큐 출력 연산 
    def print_cq(self) : 
        if not self.isEmpty(): 
            out = []
            # 리어 포인터가 프론트 포인터보다 앞에 있는 경우 
            if self.front < self.rear : 
                out = self.q[self.front+1:self.rear+1]
            # 리어 포인터가 프론트보다 뒤에 있는 경우 
            else : 
                out = self.q[self.front+1:self.MAX_SIZE+1]+self.q[0:self.rear+1]
            print(f'front:{self.front}, rear:{self.rear}', out)

    def size(self) : 
        return (self.rear-self.front+self.MAX_SIZE)%self.MAX_SIZE 
    
    # 원형 큐 비우기 
    def clear(self) : self.front = self.rear 

    # 원형 큐 가장 전단에 있는 항목 조회 
    def peek(self) : 
        if not self.isEmpty() : 
            return self.q[(self.front+1)%self.MAX_SIZE] # 현재 front 포인터 하나 뒤 위치의 항목 
```

## 속성과 연산을 정의. 구현한 원형 큐가 잘 동작하는 지 보자. 

```python 
# 원형 큐 호출 
cq = Circularque() 
```

```python 
# 원형 큐 테스트 
print(cq.isEmpty())

for i in range(1,4) : 
    cq.enqueue(i)
cq.print_cq()
```

<img width="234" alt="Screen Shot 2022-01-12 at 13 33 48" src="https://user-images.githubusercontent.com/83487073/149064353-cab831a9-28fc-47dc-8f2e-77207e54bef6.png">


```python 
for _ in range(2) : 
    cq.dequeue()
cq.print_cq()
```

<img width="177" alt="Screen Shot 2022-01-12 at 13 34 18" src="https://user-images.githubusercontent.com/83487073/149064393-d34a5da2-47ab-4f82-b361-4687df8874a2.png">

```python 
for i in range(4,7) : 
    cq.enqueue(i)
cq.print_cq()
print(cq.isFull())
```

<img width="301" alt="Screen Shot 2022-01-12 at 13 34 49" src="https://user-images.githubusercontent.com/83487073/149064431-d6d2be8a-2f39-4811-9d3f-82ea9748f73a.png">

---

# 덱(Deque)

## Double Ended QUEue

큐의 일종. 

## 추상자료형 

속성 

- 전단과 후단 모두에서 접근가능한 항목들의 모임 

연산 

큐의 연산

- enqueueRear(): 큐 후단에 데이터 삽입한다. (enqueue)
- dequeueFront(): 큐 전단에서 데이터 꺼내 반환하고 삭제한다. (dequeue)
- isEmpty(): 큐가 비어있는지 확인한다. 
- size(): 큐 안에 데이터가 몇개 들었는지 확인한다.
- isFull(): 큐가 비어있는지 화인한다. 

추가

- enqueueFront(): 큐 전단에 데이터 삽입한다.
- dequeueRear(): 큐 전단에서 데이터 꺼내 반환하고 삭제한다. 

## 덱 구현 방법 

- 파이썬 내장 collections 모듈의 deque 클래스 
- (1차원 배열 리스트로 구현된) 원형 큐에 덱 연산 추가 

---

# 덱 구현 - Collections 모듈의 deque 클래스 

## deque 클래스 

덱 연산이 이미 구현되어 있는 클래스다. 갖다 쓰기만 하면 된다. 

덱 구현에 1차원 배열 리스트를 사용한다. 리스트를 '덱 처럼' 이용한다. 

덱 객체 생성 시 클래스에 반복가능자를 받는다. 반복가능자 각 요소가 덱의 요소가 된다. 반복가능자 안 넣으면 빈 덱을 생성한다. 

deque 클래스에서 구현해 놓은 덱 연산은 

- enqueueRear: 큐 후단에 데이터 삽입 (append(item))
- enqueueFront: 큐 전단에 데이터 삽입 (appendleft(item))
- dequeueRear: 큐 후단에서 데이터 삭제 (pop())
- dequeueFront: 큐 전단에서 데이터 삭제 (popleft())

```python 
# collections 모듈 deque 클래스로 덱 구현하기 
# 덱을 직관적. 간단하게 사용할 수 있다. 

from collections import deque

dq = deque([88,22,11]);print(dq)

dq.append(3);print(dq)

dq.appendleft(76);print(dq)

dq.pop();print(dq)

dq.popleft();print(dq)
```
<img width="476" alt="Screen Shot 2022-01-12 at 14 50 26" src="https://user-images.githubusercontent.com/83487073/149071410-b036e5a5-a4e4-4a2c-ab3c-c7bf65386b31.png">

```python 
# deque 클래스 활용 

from collections import deque

dq = deque('data')

for elem in dq : 
    print(elem.upper(), end='')
```
DATA 

```python 
print() 
dq.append('r')
dq.appendleft('k')
print(dq) 
```
deque(['k', 'd', 'a', 't', 'a', 'r'])

```python 
dq.pop() 
dq.popleft() 
print(dq[-1])
```
a

```python 
print('x' in dq)# 소속검사: 문자열 'x'가 dq에 소속되어 있나? 
```
False

```python 
dq.extend('structure') # 문자열 'structure' 각 요소 하나하나를 후단에서 덱에 추가 
print(dq)
```
deque(['d', 'a', 't', 'a', 's', 't', 'r', 'u', 'c', 't', 'u', 'r', 'e'])

```python 
dq.extendleft(reversed('python')) # 문자열 'python' 문자열 순서를 반대로 뒤집고, 요소 하나하나를 차례로 덱 전단에 추가 
print(dq)
```
deque(['p', 'y', 't', 'h', 'o', 'n', 'd', 'a', 't', 'a', 's', 't', 'r', 'u', 'c', 't', 'u', 'r', 'e'])

---

# 덱 구현 - 원형 큐에 덱 연산 추가 

덱은 큐의 일종이다. 

원형 큐에 '전단에서 데이터 삽입', '후단에서 데이터 삭제' 연산을 가능하게 만들면 덱이 된다. 

원형 큐 클래스에 '전단 삽입', '후단 삭제' 연산을 추가하자. 

```python
# 원형 큐에 연산 추가해서 덱 구현 

class Deque : 
    def __init__(self) : 
        self.MAX_SIZE = 5
        self.front = 0 
        self.rear = 0 
        self.q = [None]*self.MAX_SIZE  
    
    # 추가된 덱 연산 

    # 전단 삽입 
    def enqueueFront(self, item) : 
        if not self.isFull() : # 포화상태가 아닐 때 
            # 1. front 포인터 위치에 데이터 삽입 
            self.q[self.front] = item 
            # 2. front 포인터 이동 
            self.front = self.front -1 
            if self.front < 0 : self.front = self.MAX_SIZE -1 
    
    # 후단 삭제 
    def dequeueRear(self) : 
        if not self.isEmpty() : # 공백상태가 아닐 때 
            # 리어 포인터 이동 = 큐에서 제외(삭제)
            deleted = self.q[self.rear] # 현재 리어에 담겨 있는 거. 
            self.rear = self.rear - 1 # 삭제: 리어 포인터 이동 
            if self.rear < 0 : self.rear = self.MAX_SIZE - 1 
            return deleted 

    # 후단 항목 조회 
    def peekRear(self) : return self.q[self.rear]

    # --------------------------- 원형 큐 연산과 동일-----------------------
    # 포화상태 정의 
    def isFull(self) : return (self.rear+1)%self.MAX_SIZE == self.front 
    
    # 공백상태 정의 
    def isEmpty(self) : return self.front == self.rear 
    
    # 삽입연산 
    def enqueueRear(self, item) : 
        if not self.isFull() : # 포화상태가 아닐 때 
            self.rear = (self.rear+1)%self.MAX_SIZE # 다음 리어 포인터 위치 
            self.q[self.rear] = item 
    
    # 삭제연산 
    def dequeueFront(self) : 
        if not self.isEmpty() : # 공백상태가 아닐 때 
            self.front = (self.front+1)%self.MAX_SIZE # 다음 프론트 포인터 위치 
            deleted = self.q[self.front]
            return deleted # 뭐가 삭제됬는지 반환 

    # 원형 큐 출력 연산 
    def print_cq(self) : 
        if not self.isEmpty(): 
            out = []
            # 리어 포인터가 프론트 포인터보다 앞에 있는 경우 
            if self.front < self.rear : 
                out = self.q[self.front+1:self.rear+1]
            # 리어 포인터가 프론트보다 뒤에 있는 경우 
            else : 
                out = self.q[self.front+1:self.MAX_SIZE+1]+self.q[0:self.rear+1]
            print(f'front:{self.front}, rear:{self.rear}', out)

    def size(self) : 
        return (self.rear-self.front+self.MAX_SIZE)%self.MAX_SIZE 
    
    # 원형 큐 비우기 
    def clear(self) : self.front = self.rear 

    # 원형 큐 가장 전단에 있는 항목 조회 
    def peek(self) : 
        if not self.isEmpty() : return self.q[(self.front+1)%self.MAX_SIZE] # 현재 front 포인터 하나 뒤 위치의 항목 
```

## 원형 큐로 구현된 덱이 잘 작동하는지 테스트 

```python 
# 덱 테스트 

dq = Deque() 
```

```python 
for i in range(1,4) : 
    if i%2 == 0 : # i가 짝수면 
        dq.enqueueRear(i) # 덱 후단에 추가 
    else : # 홀수면 
        dq.enqueueFront(i) # 덱 전단에 추가  
dq.print_cq()
```
front:3, rear:1 [3, 1, 2]

```python 
for i in range(2) : dq.dequeueFront()
dq.print_cq() 
```
front:0, rear:1 [2]

```python 
for i in range(3) : dq.dequeueRear()
dq.print_cq() 
```

```python 
for i in range(5,7) : dq.enqueueFront(i) 
dq.print_cq() 
```
front:3, rear:0 [6, 5]

---

# 우선순위 큐(Priority Queue)

큐의 일종.

출력(삭제) 우선순위가 있다. 

- 입력 순서는 중요치 않다. 

## 추상자료형 

속성 

- 출력 우선순위가 있는 항목들의 모임 

연산 

- enqueue(): 데이터와 출력 우선순위 함께 큐에 삽입한다. 
- dequeue(): 우선순위 대로 큐에서 데이터 삭제한다. 

## 우선순위 큐 구현 방법 
- 파이썬 내장 queue 모듈의 PriorityQueue 클래스 
- 리스트 
- 단순연결리스트 
- 힙트리 

---

# 우선순위 큐 구현 - 파이썬 내장 PriorityQueue 클래스 로 우선순위 큐 구현

우선순위 큐가 이미 구현되어 있는 클래스다. 갖다 쓰기만 하면 된다. 

- 기본 출력 우선순위는 오름차순 이다. 
- 내가 출력 우선순위 부여하려면 (우선순위, 데이터) 튜플 형식으로 데이터 우선순위 큐에 넣으면 된다. 

## 구현되어 있는 연산 

- put(): 우선순위 큐에 데이터 삽입 
- get(): 우선순위 큐에서 우선순위 대로 데이터 삭제 

```python 
# 파이썬 내장 PriorityQueue 클래스로 우선순위 큐 구현 

from queue import PriorityQueue

que = PriorityQueue(maxsize=8) # 우선순위 큐 객체 
```

```python 
# 큐에 데이터 삽입
# 별도 출력우선순위 부여 안 했다 = 출력시키면 오름차순으로 출력된다. 

que.put(2);que.put(8)
que.put(3);que.put(5)
que.put(4);que.put(55)

# 출력
for _ in range(que.qsize()) : 
    print(que.get(), end=',')
print()
```
2,3,4,5,8,55,

오름차순으로 낮은 것 부터 출력된 걸 볼 수 있다. 

```python 
que.qsize()
```
0

## PriorityQueue 에 데이터 넣을 때 사용자 정의 우선순위 부여하기 

```python 
# 사용자 정의 우선순위 부여해서 데이터 삽입하기 

# (우선순위, 데이터) 튜플 형식으로 큐에 삽입 
que.put((1,'가'));que.put((2,1))
que.put((3,'s'));que.put((4,'나'))
que.put((5,2))

# 데이터 우선순위대로 삭제 
for _ in range(que.qsize()) : 
    print(que.get(), end=',')
print()
```
(1, '가'),(2, 1),(3, 's'),(4, '나'),(5, 2),

뒤에 어떤 데이터가 있던 아랑곳하지 않고, 우선순위대로 삭제된 걸 볼 수 있다. 

---

# 우선순위 큐 구현 - 파이썬 기본 리스트로 우선순위 큐 구현 

우선순위 큐에서 중요한 것: 오직 출력 순위. 

```python 
# 파이썬 기본 리스트로 우선순위 큐 구현 

class priorityque : 

    def __init__(self) : 
        self.items = [] 

    # 우선순위큐가 비어있는지 확인하는 메소드 
    def isEmpty(self) : return len(self.items) == 0 # 비어있다'의 정의= 들어있는 아이템이 0개다. 

    # 우선순위큐에 들어있는 항목 갯수 확인하는 메소드 
    def size(self) : return len(self.items)

    # 우선순위 큐 비우는 메소드 
    def clear(self) : self.items = [] 

    # 삽입 메소드 
    def enqueue(self, item) : self.items.append(item) # 넣는 건 아무렇게나 넣어도 상관없다. 출력이 matter. 
    
    # 최우선순위 항목 찾는 알고리즘 
    def find_top_priority(self) :
        if self.isEmpty() : return None # 비어있다면, 최우선순위는 없다. 
        else : 
            top_priority = 0 
            # 전체 항목에 대하여: 2개씩 비교해서 최우선순위항목 나올때 까지 가린다. 
            for i in range(1, self.size()) : 
                if self.items[i] < self.items[top_priority] : # 새로운 도전자가 기존 것 보다 작을때 우선순위 부여 
                    top_priority = i # 갱신된 최우선순위 
            return top_priority #(도출된 최우선순위)
    
    # 삭제 메소드: 삭제 우선순위만 지키면 된다. 
    def dequeue(self) :
        # 삭제우선순위
        top_priority = self.find_top_priority()
        if top_priority is not None : 
            return self.items.pop(top_priority)
    
    # 최우선순위 항목 미리보기 
    def peek(self) : 
        # 삭제우선순위
        top_priority = self.find_top_priority() 
        if top_priority != None : 
            return self.items[top_priority]
```

## 우선순위 큐가 잘 작동하는지 테스트 

```python 
# 우선순위 큐 테스트 

pq = priorityque() 
```
```python 
pq.enqueue(34)
pq.enqueue(18)
pq.enqueue(27)
pq.enqueue(45)
pq.enqueue(15) # 작은 것 부터 우선순위 부여 받았으니 15,18,27,34,45 순으로 삭제될 것이라 예상. 
```
```python 
print(f'p_que:',pq.items) 
```
p_que: [34, 18, 27, 45, 15]

```python 
while not pq.isEmpty() : 
    print(f'삭제된항목:{pq.dequeue()}')
```
삭제된항목:15

삭제된항목:18

삭제된항목:27

삭제된항목:34

삭제된항목:45

---

# 우선순위 큐 구현 - 이진 힙 이용해 구현 

## heapq

파이썬 내장 모듈 중 heapq 는 파이썬 기본 리스트를 우선순위 큐로 쓸 수 있게 해준다. 

데이터 간 출력 우선순위 부여를 이진 힙 사용해서 한다. 

## heapq 에 이미 선언되어 있는 메소드 

아래 메소드들은 동작하며 최소힙 속성 유지 한다. 

- heappush(heap, item): 삽입연산 
- heappop(heap): 삭제연산
- heappushpop(heap, item): item 삽입 후 삭제 수행
- heapreplace(heap, item): 삭제 수행 후 item 삽입 

## heapq로 우선순위 큐 구현 
```python
# heapq - 이진 힙 이용해 우선순위 큐 구현

from heapq import heappush, heappop, heappushpop, heapreplace

heap = [] # 파이썬 리스트를 우선순위 큐 처럼 사용

heappush(heap, 3);heappush(heap, 8);heappush(heap, 1);heappush(heap, 0);heappush(heap,99);heappush(heap, 4);heappush(heap, 2)

heap
```
[0, 1, 2, 8, 99, 4, 3]

```python 
for i in range(len(heap)) : 
    print(heappop(heap))
```
0

1

2

3

4

8

99

```python 
heappushpop(heap, 5);print(heap) # 5 삽입하고 0 삭제 
```
[1, 5, 2, 8, 99, 4, 3]

```python 
heappushpop(heap, 4);print(heap) # 4 넣고 1 삭제 
```
[2, 5, 3, 8, 99, 4, 4]

```python 
heapreplace(heap, 11) # 2삭제하고 11 삽입 
```
2

```python 
print(heap)
```
[3, 5, 4, 8, 99, 4, 11]

```python 
heapreplace(heap, 17);print(heap) # 3나오고 17 삽입 
```
[4, 5, 4, 8, 99, 17, 11]

---

# 자료구조 별 삽입, 삭제연산 시간복잡도 비교

<img width="1235" alt="Screen Shot 2022-01-14 at 23 20 54" src="https://user-images.githubusercontent.com/83487073/149530281-e06f0bf2-e00c-4fc3-b899-5810d39dece6.png">

## 설명 

파이썬 리스트로 구현한 스택은 

- 리스트 후단에 삽입연산 한번이면 된다. $O(1)$
- 리스트 후단에서 삭제연산 한번이면 된다. $O(1)$

연결리스트로 구현한 스택은

- 새 노드 만들고 헤드포인터 지정하면 전단에서 삽입된다. $O(1)$
- 헤드포인터 다음 노드로 옮기면 연결리스트에서 삭제된다. $O(1)$

파이썬 리스트로 구현한 큐(원형 큐)는 

- 리어 포인터에서 삽입연산 한번이면 된다. $O(1)$
- 프론트 포인터에서 삭제연산 한번이면 된다. $O(1)$

원형 연결리스트로 구현한 큐(원형 큐)는 

- 리어 포인터에서 삽입연산 한번이면 된다. $O(1)$
- 프론트 포인터에서 삭제연산 한번이면 된다. $O(1)$

파이썬 리스트로 구현한 덱(원형 큐로 구현한 덱)은 

- 리어 포인터든, 프론트 포인터든 삽입연산 한번이면 된다. $O(1)$
- 리어 포인터든, 프론트 포인터든 삭제연산 한번이면 된다. $O(1)$

원형 연결리스트로 구현한 덱(원형 큐로 구현한 덱)은 

- 리어 포인터든, 프론트 포인터든 삽입연산 한번이면 된다. $O(1)$
- 리어 포인터든, 프론트 포인터든 삭제연산 한번이면 된다. $O(1)$

순서없는 파이썬 리스트로 구현된 우선순위 큐는

- append 연산으로 파이썬 리스트 후단에 계속 집어넣으면 된다. 삽입연산 $O(1)$
- 막 넣었기 때문에 삭제할 때는 항목들 비교해서 최우선순위 항목 찾아야 된다. n개 항목을 서로 비교하면 최대 시간복잡도 $O(n)$ 만큼 걸릴 것이다. 

순서없는 연결리스트로 구현된 우선순위 큐는 

- 새 노드 만들고 기존 최후단 노드랑 링크로 연결하면 삽입에 $O(1)$ 만큼 시간복잡도 걸린다. 
- 각 노드에 담겨 있는 데이터 필드를 서로 비교해서 최우선순위 항목 찾아야 된다. 연결리스트이므로 헤드포인터 붙은 0번 항목부터 순차 접근해가면서 항목들 우선순위 비교해야만 한다. 따라서 시간상한은 최대 $O(n)$ 까지 걸릴 수 있다. 

정렬된 파이썬 리스트로 구현된 우선순위 큐는 

- 삽입할 때 항목별 우선순위에 맞춰서 넣어야 한다. 만약 0번 인덱스에 뭔가 넣어야 된다면, 기존 항목들이 뒤로 1칸씩 물리적 이동하면서 삽입에 최대 $O(n)$ 만큼 시간 복잡도 걸릴 것이다. 
- 이미 모든 항목이 우선순위대로 정렬되어 있다. 따라서 삭제할 때는 이동하고 자시고 할 것 없이. 가장 앞 0번 인덱스부터 순서대로 삭제하면 된다. $O(1)$

정렬된 연결리스트로 구현된 우선순위 큐는 

- 삽입할 때 우선순위 지켜서 넣어야 한다. 따라서 뭔가 항목을 넣으려면 연결리스트의 0번 노드에서 시작해서 n번 노드까지 접근하면서 새 항목과 기존 항목 간 우선순위 비교해야만 한다. $O(n)$
- 모든 항목이 우선순위대로 정렬되어 있다. 헤드포인터 한칸씩 뒤로 이동하면서 전단노드부터 연결리스트에서 빼면 된다. $O(1)$

## 힙으로 구현된 우선순위큐는 힙 자료구조를 좀 더 공부한 뒤 다시 정리하겠다. 





























































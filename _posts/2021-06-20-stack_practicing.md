---
title : "[네이버블로그이관/데이터사이언스/자료구조] 스택 복습 & 구현"
excerpt : "05.31.2021. 스택Stack 자료구조 학습 및 복습 내용 옮겨와 기록"

categories : 
- Data Science 
- python
- data structure

tags : 
- [data structure, computer science, study, data science, computer engineering]

toc : true 
toc_sticky : true 

date : 2021-05-31
last_modified_at : 2021-08-31

---


글 원본 : https://blog.naver.com/tigerkey10/222376737824

이 글은 내가 지킬 블로그를 만들기 전, 

네이버 블로그에 공부 내용을 기록할 때 학습하고 복습하면서 기록했던 글을 옮겨온 것이다. 

네이버 블로그에서 글을 옮겨와 지킬 블로그에도 기록해둔다. 

모든 공부 과정을 기록으로 남겨 치열했던 시간들을 기록으로 남기고, 

미래에 전달해 스스로 동기부여할 것이다. 또한 반복해서 학습. 복습하는 과정에서 

어떤 개념에 대한 내 이해가 어떻게 변화.발전 했는지 관찰함으로써 스스로의 학습 정도를 돌아보는 거울로 삼고자 한다. 


---
# '쌓기'를 스택이라한다. 

- 스택 자료구조에서는 '한쪽으로만' 데이터 접근 가능하다. 

- 스택에 들어있는 데이터들 삽입.삭제.는 '후입선출' 된다. 

​
# 스택 추상자료형 

### 속성 >

- 0개 이상 원소 갖는, 순서가 있는 유한 리스트. 

- 후입선출되는 항목들의 모음. 

​
### 주요 연산 > 

1. 스택에 데이터 추가하는 push()연산

2. 스택이 비어있는지 확인하는 isEmpty() 연산

3. 스택에서 데이터 삭제하는 pop() 연산

4. 스택에 들은 데이터 갯수 알려주는 size() 연산

5. 스택 가장 위에 얹혀있는 원소 확인하는 peek() 연산


# 스택 구현하기 
## 1. pythonds 모듈 내부의 Stack() 클래스 이용하기 

```python
from pythonds.basic import Stack
st1 = Stack() 
st1.push('orange');st1.push('apple');st1.push('banana')
st1.items

```
out : 
```python
['apple','orange','banana']
```


```python
st1.isEmpty()
```
out : 
```python
False
```
```python
print(st1.peek())
print(st1.size())
print(st1.pop())
```
out : 
```python
banana
3
banana
```

## 2. 1차원 배열 리스트로 스택 구현하기 

```python
class Stack : 
def __init__(self) : 
self.items = []

def push(self, item) : 
self.items.append(item)

def pop(self) : 
if not self.isEmpty :
self.items.pop()

def peek(self) : 
if not self.isEmpty : 
return self.items[len(self.items)-1]

def isEmpty(self) : 
if self.size() == 0 : 
return True
else : 
return False

def size(self) : 
return len(self.items)
```

## 3. 단순연결리스트로 스택 구현하기 

- '전단'부분에서만 데이터 추가/삭제하는 연결리스트와 같다. 

- 각 노드의 링크는 바로 직전 노드를 가리킨다.

- 스택 가장 밑바닥 노드는 링크가 None이다.

```python
# 노드 클래스 생성
class Node : 
def __init__(self, item, link) : 
self.item = item
self.next = link 

# push연산
def push(item) : 
global top
global size # 전역변수 선언 
top = Node(item, top) # 데이터 추가할 때 마다 노드 생성
size += 1

# pop연산 # 가장 위에 쌓인 항목 연결리스트에서 뺀다
def pop() : 
global top
global size
if size != 0 : 
top_item = top.item
top = top.next # 직전 항목을 새 top으로 지정
size -= 1
return top_item

#peek연산 
def peek() : 
if size != 0 : 
return top.item

# 스택 출력
def print_stack() : 
p = top
while p : 
if p.next != None :
print(p.item, end= '')

else : 
print(p.item, end = '')

p = p.next
print()

```


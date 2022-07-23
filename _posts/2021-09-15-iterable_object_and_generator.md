---
title : "[Python] Iterable 객체, Iterator 객체, Generator 객체,함수,축약식"
excerpt : "Python 프로그래밍"

categories : 
- Data Science
- python
- mathematics

tags : 
- [datascience, mathematics, python]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-09-15
last_modified_at : 2021-09-15

---

# iterable 객체와 iterator 객체 

아래 내용은 '레벨업 파이썬' 문서를 보고 

이터러블, 이터레이터, 제너레이터 객체에 대해 공부한 내용을 정리한 글 입니다. 

집합론 기초 - 부분집합 생성 부분을 공부하다가 위 개념들을 만났고, 학습하였습니다. 

https://wikidocs.net/74399

---

# iterable 객체 

## 정의 : 

## 파이썬 객체 중 클래스 안에  __ iter __ 메소드를 포함하고 있는 객체를 iterable 객체 라고 한다. 

또는 

for 문, while 문 등 반복문에서 사용할 수 있는 객체를 iterable 객체 라고 한다. 

---

# iterator 객체 

## 정의 : 

## iterable 객체의 __ iter __ 메소드에 포함된 또 다른 객체를 'iterator 객체'라고 한다. 

$\Leftarrow$ iterator 객체는 반드시 __ next __ 메소드를 포함하고 있어야 한다. 

$\Leftarrow$ next() 내장함수로 next 메소드의 내용을 호출할 수 있다. 


## 특징 : 

iterable 객체의 __ iter __ 메소드를 호출하면 자동으로 튀어나온다. 

- iter(iterable) 을 사용하면 객체의 __ iter __ 메소드를 호출하기 때문에, 결과로 iterator 객체를 호출할 수 있다. 

반복문 원리는 iterable 객체가 포함하는 iterator 객체를 호출해서, 이 iterator 객체 가지고 도는 것이다. 

*iterable 객체 자기자신을 iterator 객체로 지정해 줄 수도 있다. 

__ iter __ 메소드에서 self(자기자신)을 반환하면 된다. 

```python
# iterable 자기자신이 iterator 객체가 되는 예 
class four() : 
    def __init__(self) : 
        self.data = ['봄','여름','가을','겨울']
        self.index = 0
    def __iter__(self) : 
        return self  # 자기자신이 iterator 객체라고 선언
```
---
# iterable 객체, iterator 객체 만들기 

```python
# 이터러블 객체, 이터레이터 객체 만들고 호출하기 

# 1. 이터레이터 객체의 클래스를 따로 만드는 경우

# iterable 객체 클래스
class Season() : 
    def __init__(self) : 
        pass
    def __iter__(self) : 
        iterator = iteration() # 이터레이터 객체
        return iterator 

# 이터레이터 객체 클래스 
class iteration() : 
    def __init__(self) : 
        self.data = ['봄','여름','가을','겨울']
        self.index = 0

    def __next__(self) : 
        if self.index < len(self.data) : 
            order = self.data[self.index]
            self.index += 1
            return order
        else : 
            raise StopIteration # for문이 stopiteration error 인식하고 알아서 멈춘다. 
s = Season()
for i in s : 
    print(i)

```
봄

여름

가을

겨울

```python
# 또는 : 다른 방법으로 이터레이터 객체 만들기 
class four() : 
    def __init__(self) : 
        self.data = ['봄','여름','가을','겨울']
        self.index = 0
    def __iter__(self) : 
        return self  # 자기자신이 iterator 객체라고 선언
    def __next__(self) : 
        if self.index < len(self.data) : 
            s = self.data[self.index]
            self.index += 1
            return s
        else : 
            raise StopIteration
a = four()
for i in a : 
    print(i)
```
- for문을 사용하면 iterable 객체의 iterator 객체를 호출한다. 
- 반복해서 iterator 객체의 next() 메서드를 호출한다. 
- 계속 반복하다가 stopiteration error 가 나오면 인식하고 자동으로 멈춘다. 

for문 사용하지 않고, next() 함수를 계속 호출해서 결과물을 출력하는 방법도 있다. 

이때는 iterator 가 가진 데이터를 모두 출력하면 stopiteration error가 발생한다. 

```python
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
```

한편 while 문은 

```python
class four() : 
    def __init__(self) : 
        self.data = ['봄','여름','가을','겨울']
        self.index = 0
    def __iter__(self) : 
        return self  # 자기자신이 iterator 객체라고 선언
    def __next__(self) : 
        s = self.data[self.index]
        if self.index < len(self.data) : 
            self.index += 1
        if self.index == 4 : 
            self.index = 0

        return print(s)
season = four()
a = 0
while a < 13 : 
    next(season)
    a += 1
```

for 문과 달리 next() 내장 함수를 자동 호출하지 않았다. 

따라서 while 문 안에 next() 내장 함수를 따로 적어줘야 원하는 대로 반복 출력이 가능했다. 

<img width="227" alt="Screen Shot 2021-09-15 at 19 14 55" src="https://user-images.githubusercontent.com/83487073/133415487-33b208ff-1aa6-4df9-9388-e4a322d7686c.png">

---

한편, 

iterator 객체를 iter() 내장 함수에 넣으면 그대로 '통과' 되어서 넣은 iterator 객체가 그대로 다시 출력된다는 것도 관찰할 수 있었다. 

```python
# iterable 객체를 iter() 내장 함수에 넣으면 iterator 객체만 추출해온다. 
# iterator 객체를 다시 iter() 내장 함수에 넣으면 그대로 '통과' 되어서 넣어준 iterator 객체가 그대로 다시 나온다. 
iterator = iter(s) # 호출된 이터레이터 객체 
print('\n')
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
```

---

# 제너레이터 함수 

## 정의 : 

return 대신 yield 키워드를 쓰는 함수. 

## 특징 : 

일반 함수는 return 하고 함수가 끝난다. 

함수를 다시 불러오면 코드를 처음부터 다시 다 로드 한다. 

하지만 제너레이터 함수는 이번 순서의 출력물을 yield 하고, 실행 멈추고 대기상태가 된다. 

다시 next() 내리면 대기상태를 풀고, 그 다음 순서 출력물을 yield 로 출력한다. 그 후 다시 대기. 

### 따라서 제너레이터 함수는 '코드를 중지 된 상태로 대기'시켜 놓을 수 있다. 

```python
# 제너레이터 함수 예 
def gen() : 
    for i in range(3) : 
        yield i

g = gen()
print(next(g)) # 0 출력 후 대기
print(next(g)) # 1 출력 후 대기
print(next(g)) # 2 출력 후 종료
print(next(g)) # yield 할 수 있는 범위 벗어나면 stopiteration error 발생시킨다. 
```

<img width="651" alt="Screen Shot 2021-09-15 at 20 34 58" src="https://user-images.githubusercontent.com/83487073/133426152-9867d04f-b0cd-4dad-8d01-6e6ea52401e4.png">

이렇게 쓸 수도 있다.

```python
# 또 다른 예 

def gen() : 
    yield 1
    yield 2
    yield 3
a = gen()
next(a)
next(a)
next(a)
next(a)
```

yield 1을 먼저 출력하면서 yield 1을 더이상 기억하지 않을 것이다. 

따라서 다시 next()로 제너레이터 객체에 출력 명령을 내리면, yield 2부터 간다. 

그 뒤 yield 2도 기억하지 않을 것이다. 

다시 next() 명령을 내리면 yield 3 부터 갈 것이다. 

---

## 제너레이터 함수 안에 return 을 함께 쓸 수도 있다. 

제너레이터 함수 안에서 return 이 실행되면 제너레이터 함수가 StopIteration Error 와 함께 종료된다. 

```python
# 제너레이터 함수 안에서 return 함께 사용하는 예 
def gen() : 
    for i in range(100) : 
        if i == 3 : 
            return '그만합시다'
        yield i
gen = gen()

next(gen)
next(gen)
next(gen)
next(gen)
```
<img width="639" alt="Screen Shot 2021-09-15 at 21 12 26" src="https://user-images.githubusercontent.com/83487073/133431122-ed1af980-e511-4ed8-9918-85608eb0f75c.png">

```python
# 또 다른 예 
def gen(x) : 
    for i in range(10) : 
        if x == 3 : 
            return '종료'
        yield i
generator = gen(3) # 실행대기 
for i in range(10) : 
    print(next(generator))
```
<img width="590" alt="Screen Shot 2021-09-15 at 21 19 06" src="https://user-images.githubusercontent.com/83487073/133432048-0f3a8753-d83e-434c-9e60-f51809daf204.png">


---

# 제너레이터 객체 

## 정의 : 

제너레이터 함수로 생성하는 객체. 

## 특징 : 

객체이름 = 제너레이터함수() 형태로 생성된다. 

next() 로 제너레이터 객체 내부 코드를 실행할 수 있다. 

- 제너레이터 객체는 값을 출력하고 나면 값을 '소비' 하고 더이상 기억하지 않는다. 따라서 iterable 객체 처럼 여러 번 같은 값을 가져올 수 없다. 

### iterable 객체 (리스트) 예
```python
gen = [i for i in range(10)]

for i in gen : 
    print(i)

print('\n')
for i in gen : 
    print(i)
```

<img width="437" alt="Screen Shot 2021-09-15 at 20 57 02" src="https://user-images.githubusercontent.com/83487073/133429033-e9003863-f41b-466a-b2f9-ed73ca29f173.png">
iterable 객체는 반복해서 여러 번, 같은 값을 가져올 수 있다. 



반면에 

### generator 객체 예 
```python
gen = (i for i in range(10))

for i in gen : 
    print(i)
for i in gen : 
    print(i)
```
<img width="396" alt="Screen Shot 2021-09-15 at 20 57 36" src="https://user-images.githubusercontent.com/83487073/133429108-9991b561-5c4a-4771-8ee5-efc914876d21.png">

제너레이터 객체로 for문을 한번 돌고, 다시 같은 방식으로 for문 돌려고 했지만 아무것도 출력되지 않았다. 

---
# 제너레이터 축약식 

## 정의 : 

제너레이터 함수 없이 제너레이터 객체를 생성하는 축약식. 

리스트 축약식과 생긴게 똑같다. 

대신, 리스트 축약식은 대괄호로 감싸줬다면, 제너레이터 축약식은 일반 괄호 () 로 감싼다. 

### 또, 리스트 축약식처럼 값을 바로 반환하지 않는다. 

### 제너레이터 축약식으로 제너레이터 객체를 생성하면 '대기상태'에 들어간다. 

### next() 명령이 떨어질 때 마다, 값을 하나하나 출력한다. 


```python
# 제너레이터 축약식
gen_comp = (combinations(omega, i) for i in range(5))

# 또는 
gen = (i for i in range(5)) 
```
---

## 정리 : 

iterable, iterator : '반복가능자'

generator : 비복원 추출만 가능한 '공 주머니'






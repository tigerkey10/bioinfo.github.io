---
title : "[2021 인공지능전문가 교육과정 복습] 매핑, 딕셔너리, 집합, any(), all()"
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

date : 2022-01-07
last_modified_at : 2022-01-08

---

# 매핑(Mapping)

키(key) 값(value) 쌍 

## 추상자료형 

속성

- 키 데이터와 값 데이터를 짝지어 저장한다. 
- 변형이 불가능한 임뮤터블(immutable) 자료형 만 키(key) 가 될 수 있다.

연산

- 각 값(value) 데이터는 키(key) 로만 접근할 수 있다. 

## 매핑 예

딕셔너리

---

# 매핑 - 딕셔너리(Dictionary)

대표적인 매핑 타입 자료구조다. 

## 추상자료형 
- 매핑의 속성, 연산 상속 

딕셔너리 만의 연산 

- clear(): 딕셔너리 내 모든 항목을 삭제할 수 있어야 한다. 
- copy(): 딕셔너리 복사본을 반환한다.
- get(key): 키에 해당 하는 값을 리턴한다. 값이 없으면 None을 리턴한다.
- update(dict): dict의 키와 값의 목록을 dict에 추가한다. (또는 키-값 pair 를 dict로 수정)
- items(): 딕셔너리 내 모든 키-값 쌍을 튜플로 묶어서 반환한다. 
- keys(): 키 목록을 반환한다. 
- values(): 값 목록을 반환한다. 
- popitem(): 요소 중 가장 오른쪽에 위치한 pair 삭제 후 반환한다. OrderedDict 이면 popitem() 안에 last=True 또는 False 로 각각 맨 마지막 순서 pair 를 삭제하거나 맨 처음 pair 삭제할 수 있다.
- pop(key): 특정 key의 value 반환 후 그 pair 삭제한다.  

## 딕셔너리 구현 방법

- 해시 테이블 

---

# 딕셔너리 생성하기 - dict() 또는 {} 이용하는 방법 

```python
# 빈 딕셔너리 생성 
dd = dict()
```

```python
# 또는
dd = {}
```

## setdefault() 로 특정 키의 기본값 지정하기

- dict_name.setdefault(key, 값)

```python
dd.setdefault(3, 'hello')
dd
```
<img width="105" alt="Screen Shot 2022-01-07 at 21 18 11" src="https://user-images.githubusercontent.com/83487073/148543090-5bffc690-1b0f-4584-89aa-5901acf75f35.png">

## setdefault() 에 지정해 준 key값을 딕셔너리가 이미 갖고 있는 경우

그 key값에 원래 할당되어 있던 값을 출력한다.

```python
a = {'a':3,'b':8}
a.setdefault('a',8)
```
출력: 3

## setdefault() 에 key값만 넣는 경우 

그 key값의 기본값으로 None 이 할당된다. 

```python
dd = {}
dd.setdefault(3) # 3이 key 된다. 
dd
```

<img width="78" alt="Screen Shot 2022-01-07 at 21 24 08" src="https://user-images.githubusercontent.com/83487073/148543730-c8a633d0-e119-457e-b35b-dbd7700231a2.png">

---

# 딕셔너리 생성하기 - Defaultdict Class 이용하는 방법

정의: 기본값 갖는 딕셔너리 만드는 클래스.

```python
from collections import defaultdict

dd = defaultdict(객체 이름, key=value1, key=value2,...)
```

- collections 모듈에서 defaultdict 를 불러와서 사용한다. 
- 기본 딕셔너리 는 자신이 포함하고 있지 않은 키 값이 들어오면 key error를 발생시킨다.
- 하지만 defaultdict 는 딕셔너리 기본값을 지정해줌으로써, 없던 키 값이 들어오면 기본값을 출력하고, 그 키 값에도 기본값 할당해준다. 
- 결국 예상치 못한 key 값 들어왔을 때 key error 발생하는 걸 막을 수 있다. 

## defaultdict '객체 이름' 자리에 넣은 객체 기본값이 '기본값'이 된다. 

예를 들어 int를 넣으면 0을 기본값으로 설정한다. list 를 넣으면 [] 가 기본값이 된다. 

한편 key= argument를 써서 '키 = 값' 을 넣어줄 수도 있다. 이건 딕셔너리 생성하면서 특정 키-값 쌍을 같이 만들기 위한 것이다. 

```python
dd = defaultdict(list)
dd
```

<img width="192" alt="Screen Shot 2022-01-07 at 21 26 11" src="https://user-images.githubusercontent.com/83487073/148543941-cdf3d75e-b7ef-4dbb-b533-0a0f486a5835.png">

defaultdict(기본값, 현재 딕셔너리 상태) 가 출력된다.

객체 자리에 list를 넣었으므로, 기본값은 [] 빈 리스트가 된다. 

```python
dd[3] # 빈 딕셔너리에 3이 가지고 있는 값 출력 요구 
```
<img width="91" alt="Screen Shot 2022-01-07 at 21 28 18" src="https://user-images.githubusercontent.com/83487073/148544210-4b7df2d3-1d35-4325-bbd6-0f5366d6a246.png">

기본값 빈 리스트 출력한다. 

그리고 딕셔너리를 확인해보면 key 3이 [] 빈 리스트를 값으로 갖고 있는 걸 볼 수 있다. 

<img width="220" alt="Screen Shot 2022-01-07 at 21 29 23" src="https://user-images.githubusercontent.com/83487073/148544317-2f48db48-2d95-4301-9895-c5adb23214af.png">

만약 기본값 자리에 문자열 str 객체를 넣으면 '' 공백이 기본값 된다. 

```python
dd = defaultdict(str, a=[1,2],b=3)
dd[6]
dd
```
<img width="373" alt="Screen Shot 2022-01-07 at 21 36 29" src="https://user-images.githubusercontent.com/83487073/148545059-86624809-2593-4aaa-9d4b-220d9c42cbf5.png">


---

# 딕셔너리 생성하기 - OrderedDict Class 이용하는 방법 

정의: key값 순서 기억하는 딕셔너리 만드는 클래스. 

쓸모: 딕셔너리로 for문 돌 때 '시퀀스 타입처럼' 일정한 순서 가지고 데이터에 접근할 수 있다. 

*하지만 파이썬 업데이트 후 기본 딕셔너리도 반복문 돌 때 일정 순서 유지하게 되면서, 별로 안 쓰인다. 

```python
from collections import OrderedDict

od1 = OrderedDict({'a':1, 'b':2,'c':3})

for i in od1 : 
    print(f'key:{i}')
    print(f'value:{od1[i]}')
    print('-'*3)
```

<img width="96" alt="Screen Shot 2022-01-07 at 21 53 50" src="https://user-images.githubusercontent.com/83487073/148546790-8e46f57f-606f-466c-bb0a-5764297117d7.png">

## key값 순서기억

```python
od1 = OrderedDict({'a':1, 'b':2,'c':3})

od2 = OrderedDict({'a':1, 'c':3, 'b':2})
print(od1 == od2)

od3 = OrderedDict({'a':1, 'b':2, 'c':3})
print(od1 == od3)
```

False

True

key 순서가 같아야 두 딕셔너리 같다고 하는 걸 볼 수 있다. 

기본 딕셔너리는 순서 개념 자체가 없어서 키-값 쌍만 모두 같으면 같은 딕셔너리로 분류된다. 

---

# 딕셔너리 정렬하기 

sorted() 를 사용해서 딕셔너리를 정렬할 수 있다. 

```python
sorted(딕셔너리.items(), key=정렬기준) # 정렬 기준은 lambda 함수 써서 지정한다.
```

## OrderedDict 정렬시키기

```python
d = OrderedDict({'milk':3, 'coffee':1, 'capuchino':4, 'tea':2})

print(OrderedDict(sorted(d.items(), key=lambda x : x[1]))) # value를 기준으로 정렬
print(OrderedDict(sorted(d.items(), key=lambda x : x[0]))) # key를 기준으로 정렬 
print(OrderedDict(sorted(d.items(), key=lambda t : len(t[0])))) # 영단어 알파벳 길이 순 정렬
```
<img width="566" alt="Screen Shot 2022-01-07 at 22 44 21" src="https://user-images.githubusercontent.com/83487073/148552541-04b35c77-4dbf-4173-a431-b022929589d9.png">

## 기본 딕셔너리 정렬시키기 

```python
d2 = dict(d) # 기본 딕셔너리로 만들기 
print(dict(sorted(d2.items(), key=lambda x : x[0]))) # key 기준으로 정렬
print(dict(sorted(d2.items(), key=lambda x : x[1]))) # value 기준으로 정렬
print(dict(sorted(d2.items(), key=lambda x : len(x[0])))) # key 알파벳 길이 짧은 순 오름차순 정렬
```
<img width="464" alt="Screen Shot 2022-01-07 at 22 42 54" src="https://user-images.githubusercontent.com/83487073/148552386-2740b5a3-01f9-4017-993a-3897eb3d0299.png">


---

# 집합(Set)

구별가능한 개체들의 모임

- 딕셔너리에서 key만 모아놓은 형태

## 추상자료형 

속성 

- 요소(element) 간 중복 안 된다. 
- 각 요소 별 key 도, 순서(인덱스)도 없다. (따라서 인덱싱도, 슬라이싱도 불가능하다)

연산 

- add(x): 집합 내에 특정 원소 x를 추가한다. 
- discard(x): 집합 내 특정 원소 x를 삭제한다. 
- clear(): 집합 내 모든 원소를 삭제한다. 
- union(s): s와의 합집합 새로 생성한다. 
- difference(s): s와의 차집합 새로 생성한다. 두 집합에 대한 - 연산과 같다. 
- intersection(s): s와의 교집합 새로 생성한다. 두 집합에 대한 & 연산과 같다. 
- symmetric_difference(s): s와의 대칭 차집합 새로 생성한다. 두 집합에 대한 ^ 연산과 같다. 
- issubset(s): 다른 집합 s의 부분집합인지 검사한다. True 또는 False 반환한다. 
- issuperset(s): 다른 집합 s를 포함하는지 검사한다. True 또는 False 반환한다. 
- isdisjoint(s): s 집합과 교집합이 하나도 없는지 검사한다. True 또는 False 반환한다. 

## 집합 구현 방법 

- 해시 테이블 

---

# 집합의 효율성 

## 집합은 데이터 소속검사에서 리스트보다 빠르다(효율적이다)

- 집합은 소속검사에 $O(1)$ 만큼 시간복잡도 걸린다. 
- 리스트는 소속검사에 $O(n)$ 만큼 시간복잡도 걸린다.

리스트는 요소들에 하나하나, 일일이. 순차접근해서 소속검사를 하는 데 반해, 집합은 해시 테이블 이용해서 한방에 소속검사 하기 때문이다. 

어떤 값에 대해 소속검사 명령이 들어오면, 집합은 그 값을 해시 함수에 통과시켜 해시 값을 얻는다. 그 해시 값을 주소로 해시 테이블에 접근해서 저장소에 True 가 저장되어 있는지, False가 저장되어 있는지 확인한다. 이렇게 한 번만 검사하면 값 소속 여부를 바로 알 수 있기 때문에, $O(1)$ 만큼 시간 복잡도가 걸린다. 

[출처: https://rexiann.github.io/2020/11/28/set-in-python.html ]

```python
# 집합은 리스트보다 소속검사 속도가 월등히 빠르다

set1 = {'k1', 'k2', 'k3', 'k4'}

'k1' in set1 # True
'k5' not in set1 # False
```

## 집합은 데이터 삽입, 삭제에서 배열로 구현된 리스트보다 빠르다(효율적이다)

- 집합은 삽입, 삭제에 $O(1)$ 만큼 시간 복잡도 걸린다. 
- 배열로 구현된 리스트는 삽입, 삭제에 $O(n)$ 만큼 시간 복잡도 걸린다. 

배열로 구현된 리스트는 데이터를 삽입, 삭제하면 각 항목들의 물리적 이동이 동반된다. 

예를 들어 0번 인덱스 항목을 삭제하면 그 뒤의 1,2,3,... 번 인덱스 항목들이 앞으로 한 칸씩 물리적 이동한다. 따라서 시간복잡도가 $O(n)$ 만큼 걸린다. 

하지만 집합은 해시 테이블 사용하기 때문에 삽입, 삭제에 $O(1)$ 만큼 시간복잡도 걸린다. 

어떤 값을 삽입 할 때는 해시함수로 그 값의 해시를 받아서, 저장소에 해시와 True 만 저장하면 될 것이다. 

어떤 값을 삭제 할 때는 그 값의 해시(주소)에 접근해서 저장소에 True를 False 로 바꾸면 될 것이다. 

결국 삽입을 하든 삭제를 하든 $O(1)$ 만큼 시간복잡도만 걸린다. 

### $\Rightarrow$ 리스트에 소속검사, 삽입, 삭제가 빈번하게 일어날 경우 리스트를 집합으로 변환해서 사용하는게 효율적이다. 

---

# all() 과 any()

반복가능자를 입력으로 받아서. 검사한 뒤 결과를 True / False 로 반환하는 논리연산 함수. 

## all()

1. 반복가능자 입력받는다. 
2. 요소가 모두 0 또는 '' 공백 문자열이 아닐 때 True 반환한다. 하나라도 0 또는 '' 가 있으면 False 반환한다. 
3. 만약 반복자가 비어있으면 True 반환한다. 

```python
# all(): 0 또는 공백이 모두 아닐 때 true

s1 = {1,2,3}
s2 = {''}
s11 = [1,2,3]
all(s2) # False 반환한다
all({}) # True 반환한다
all(s11) # True 반환한다
```

## any()

1. 반복가능자 입력받는다. 
2. 요소 중에 0 또는 '' 공백 문자열 아닌 게 하나라도 있으면 True 반환한다. 
3. 요소가 모두 0 또는 '' 공백 문자열 일 때 만 False 반환한다. 
4. 만약 반복자가 비어있으면 False 반환한다. 

```python
# any(): 0 또는 공백이 아닌게 하나라도 있으면 true

s3 = {0,"",1}
s4 = {}
any(s3) # 1 때문에 True 
any(s4) # False
```

---

# sorted() 함수로 집합 정렬시키기

집합은 순서가 없다. 따라서 정렬이 불가능하다. 

sorted() 함수는 반복가능자를 입력받아, 일단 리스트로 바꾸고 정렬시켜 결과값을 출력한다. 

집합도 반복가능자 이므로 sorted() 함수에 넣을 수 있다. 

다만 결과가 리스트로 바뀌어 나오므로, sorted(집합) 한 결과를 다시 집합으로 바꿔야 한다. 

이게 집합을 정렬시키는 방법이다. 

- sorted(반복가능자, key=lambda 함수(정렬기준), reverse=True)

```python
# sorted() 함수로 집합 정렬시키기

s1 = {'key1', 'key2', 'key3'}

sorted_s1 = sorted(s1, reverse=True)
```

<img width="193" alt="Screen Shot 2022-01-08 at 17 35 54" src="https://user-images.githubusercontent.com/83487073/148637757-3845b99c-d299-484d-a9f1-2c8d8b2aebd5.png">

보다시피 리스트로 정렬된 채 출력된다. 

집합으로 사용하려면 다시 집합으로 바꿔줘야 한다. 

```python
# sorted() 결과를 다시 집합으로 바꾸기 

s2 = set(sorted_s1)
s2
```
<img width="207" alt="Screen Shot 2022-01-08 at 17 37 46" src="https://user-images.githubusercontent.com/83487073/148637805-69a2337b-0833-44ed-9c5c-0d885d41d140.png">







































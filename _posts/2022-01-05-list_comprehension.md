---
title : "[Python] 리스트 축약식, 중첩 리스트"
excerpt : "헷갈리던 내용을 확실하게 복습 및 정리"

categories : 
- python

tags : 
- [python, study]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-01-05
last_modified_at : 2022-01-05

---

# 리스트 축약식 (list comprehension)

## [ 표현식 for 변수 in 반복자 if 조건 표현식 ]

- if 조건 표현식 부분은 생략해도 된다. 

## if 조건 표현식 없이 

```python
# if 조건 없이 
list1 = [1,2,3,4,5,6]

[x**2 for x in list1]
```

결과: [ 1,4,9,16,25,36 ]

```python
# 또는
[x**2 for x in range(1,7)]

# 또는 
list(map(lambda x : x**2, list1))
```
## if 조건 표현식과 함께 

```python
list1 = [1,2,3,4,5,6]

[x for x in list1 if x > 3]
```
결과: [ 4,5,6 ]

```python
# 또는 
list(filter(lambda x : x > 3, list1))
```

결과: [ 4,5,6 ]

---

# 리스트 중첩시키기 

```python
# 중첩리스트 예 1
list1 = [[0,1],[1,2],[3,4]]
```

```python
# 중첩리스트 예 2
list2 = [[0 for i in [1,2]] for i in [1,2,3]]

# or 

list2 = [[0 for i in range(2)] for i in range(3)]
```

결과: [[ 0,0 ], [ 0, 0 ], [ 0, 0 ]]

---

# 리스트의 슬라이싱

리스트 슬라이싱 결과는 원래 리스트의 '부분 복사본' 이다. 

원본 리스트는 손상되지 않는다. 

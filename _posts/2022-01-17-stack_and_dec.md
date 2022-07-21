---
title : "[2021 인공지능전문가 교육과정 복습] 스택 활용, 덱 활용 (괄호검사, 후위표기변경, 후위표기 계산, 그리고 회문분석)"
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

date : 2022-01-17
last_modified_at : 2022-01-17

---

# 스택 활용 - 괄호검사 

문제 정의: 

- 괄호 짝 모두 pair 맞는지 검사 하고자 함.
- '(' 나 '{' 여는 괄호면 무조건 스택에 집어넣는다. 
- ')', '}' 닫는 괄호면 스택 제일 위에서 하나 꺼내 짝이 맞는지 검사한다. 
- 괄호 짝 맞으면 True.
- 괄호 짝 하나라도 안 맞으면 모두 False. 

## 1차원 리스트로 스택 구현해서 검사하겠다. 

```python 
# 스택 활용 - 괄호검사 

a = ['(', ')', '{', '{', ')', ')', '}', '}', '(', '(']

stack = [] # 스택 
n = len(a)

for i in range(n) : 
    if a[i] == '(' or a[i] == '{' : 
        stack.append(a[i])
    else : # 닫힌 게 나오면 괄호검사 
        if len(stack) != 0 : 
            top = stack.pop() 
            if a[i] == ')' and top != '(' : 
                result = False 
                break 
            elif a[i] == '}' and top != '{' : 
                result = False 
                break 
        else : 
            result = False 
            break 

if len(stack) == 0 and result == True : 
    print(f'{True}')
else : 
    print('False')
```
False

## 함수로 묶기
```python 
# 함수로 만들어 사용 
def check(a) : 
    s = [] 
    n = len(a)
    result = True

    for i in range(n) : 
        if a[i] == '(' or a[i] == '{' : 
            s.append(a[i])
        else : # 닫힌 게 나오면 괄호검사 
            if len(s) != 0 : 
                top = s.pop() 
                if a[i] == ')' and top != '(' : 
                    result = False 
                    break 
                elif a[i] == '}' and top != '{' : 
                    result = False 
                    break 
            else : 
                result = False 
                break 

    if len(s) == 0 and result == True : 
        return True 
    else : 
        return False

check(a)
```
False

```python 
a = ['(', '(', '{', '}', '(', ')', ')', '{', '{', '}', '(', ')', '}', ')']
check(a)
```
True 

---

# 스택 활용 - 중위표기를 후위표기로 변경하기 

중위표기 예: $A+B-C*(D+E)$

후위표기 예: $AB+CDE+*-$

## 중위표기 후위표기로 변경 방법 

1. 연산우선순위에 따라 괄호로 묶는다.
2. 괄호 안 연산자를 괄호 오른쪽 밖으로 이동시킨다. 
3. 괄호 제거한다. 

## 스택을 활용해서 중위표기 후위표기로 변경하기 

문제 정의:

- 문자열(숫자)는 바로 출력 
- '(' 여는 괄호는 스택에 넣는다. 
- ')' 닫는 괄호 나오면 스택에서 '(' 직전까지 모두 순서대로 빼서 출력한다. '('는 스택에서 빼서 버린다. 
- 사칙연산 연산자는 나오면 스택에 넣는다. 자기자신보다 우선순위 낮은 연산자가 스택에 들어올 때 스택에서 빠져나와 출력된다. 

## 1차원 리스트로 스택 구현해서 문제 해결 

## 첫번째 시도

```python  
# 스택 구현
stack = []

# 삽입연산
def push(item) : 
    global stack 
    stack.append(item)

# 삭제연산 
def pop() : 
    global stack 
    if len(stack) != 0 : 
        return stack.pop(-1)

# 조회 
def peek() : 
    if len(stack) != 0 : 
        return stack[-1] 

# -----------------------

#시도 1. 중위표기를 후위표기로 변경 
def infix_to_postfix(expr) : 
    result = [] 
    p = {} 
    p['*'] = 3
    p['/'] = 3
    p['+'] = 2
    p['-'] = 2
    p['('] = 1

    input_expr = expr.split()  

    for i in input_expr : 
        if (i in 'ABCDEFGHIJKLMNOPQRSTUWXYZ') or (i in '0123456789') : 
            result.append(i)
        elif i == '(' : 
            push(i)
        elif i == ')' : 
            while True : 
                popped = pop()
                if popped != '(' : 
                    result.append(popped)
                elif popped == '(' : 
                    break 
        else : 
            while (len(stack) != 0) and (p[i] <= p[peek()]) : 
                result.append(pop())
            push(i) 

    # 마지막까지 스택에 남은거 전부방출 
    while len(stack) != 0 : 
        result.append(pop())
    return ' '.join(result)
```

```python
infix_to_postfix('( A + B ) * C - ( D - E ) * ( F + G )')
```
'A B + C * D E - F G + * -'

```python 
print(infix_to_postfix('A * B + C * D'))
```
A B * C D * +

```python 
print(infix_to_postfix('A + B * C / ( D - E )'))
```
A B C * D E - / +

## 중위표기 - 후위표기 변경 함수 바꾼 뒤 두번째 시도

```python 
# 2. 중위표기를 후위표기로 변경 
def infix_to_postfix(expr) : 
    result = [] 
    p = {} 
    p['*'] = 3
    p['/'] = 3
    p['+'] = 2
    p['-'] = 2
    p['('] = 1

    input_expr = expr.split()  

    for i in input_expr : 
        if (i in 'ABCDEFGHIJKLMNOPQRSTUWXYZ') or (i in '0123456789') : 
            result.append(i)
        elif i == '(' : 
            push(i)
        elif i == ')' : 
            top = pop() 
            while top != '(' : 
                result.append(top)
                top = pop() 

        else : 
            while (len(stack) != 0) and (p[i] <= p[peek()]) : 
                result.append(pop())
            push(i) 
    # 스택에 남은거 전부방출 
    while len(stack) != 0 : 
        result.append(pop())
    return ' '.join(result)
```

```python
infix_to_postfix('( A + B ) * C - ( D - E ) * ( F + G )')
```
'A B + C * D E - F G + * -'

```python 
print(infix_to_postfix('A * B + C * D'))
```
A B * C D * +

```python 
print(infix_to_postfix('A + B * C / ( D - E )'))
```
A B C * D E - / +

---

# 스택 활용 - 후위표기 계산하기 

위에서 중위표기를 후위표기로 바꿨다. 이제 후위표기를 직접 계산한다. 

문제 정의: 

- 숫자는 무조건 스택에 집어넣는다. 
- 사칙연산 연산자가 나오면 스택에서 2개 꺼내 그 연산자로 연산한다. 
- 2번 연산 결과를 다시 스택에 집어넣는다.
- 스택에 마지막 최후의 1값 남을 때 까지 1~3 과정 반복한다. 

## 1차원 리스트로 스택 구현해서 문제 해결

```python 
# 후위표기 계산 

stack = [] # 스택 

def push(item) : stack.append(item)

def pop() : 
    if len(stack) != 0 : return stack.pop(-1) 

def peek() : 
    if len(stack) != 0 : return stack[-1]

def compute(operator, operand1, operand2) :
    if operator == '*' : return (operand1 * operand2) 
    elif operator == '/' : return (operand1 / operand2) 
    elif operator == '+' : return (operand1 + operand2)
    else : return (operand1 - operand2)

# 계산 
def calculation(input_expression) : 
    expr = input_expression.split()
    for token in expr : 
        if token in '0123456789' : 
            push(int(token))
        else : 
            op2 = pop() 
            op1 = pop()
            push(compute(token, op1, op2))
    if len(stack) == 1 : 
        return stack.pop(-1)
```
```python 
calculation('7 8 + 3 2 + /')
``` 
3

```python 
calculation('2 3 5 * 6 4 - / +')
```
9.5

---

# 덱 활용 - 영단어가 회문인지 아닌지 검사하기 

회문이란, 가운데 중심으로 알파벳이 좌우대칭되는 영단어 말한다. 

문제 정의: 

- 영단어를 알파벳 단위로 분리해서 덱에 넣는다. 
- 덱 전단과 후단에서 원소 빼서 같은지 검사한다. 
- 2번 덱에 알파벳 1개 또는 0개 남을 때 까지 반복한다. 그 후 검사종료.
- 2번 반복 중 불일치 발생하면 곧바로 검사종료, 회문 아니다. 

## 1차원 리스트로 덱 구현해서 문제 해결

```python 
def check_palindrome(word) : 
    flag = True
    words = list(word)
    while len(words) > 1 :
        front = words.pop(0)
        rear = words.pop(-1)
        if front != rear : 
            flag = False 
            break 
    return flag 
```

```python 
check_palindrome('racecar')
```
True 

```python 
check_palindrome('tesesta')
```
False 

```python 
check_palindrome('raddar')
```
True 

```python 
check_palindrome('wasitacatisaw')
```
True













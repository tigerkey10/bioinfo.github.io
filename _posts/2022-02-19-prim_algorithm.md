---
title : "[2021 인공지능전문가 교육과정 복습] 최소신장트리, Prim 알고리듬, Diijkstra 알고리듬(최단 경로 찾기) "
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

date : 2022-02-19
last_modified_at : 2022-02-19

---

# 최소신장트리(Minimum Spanning Tree)

## 정의 

간선 가중치 합 최소인 신장트리. 

- 가중치 합 최소인. 싸이클 없이. 모든 정점이 최소 간선 수로 연결된. 그래프
- 최소 간선 수: 정점 수($N$) $-1$

---

# 그래프에서 최소신장트리 찾는 알고리듬

- ### Prim 알고리듬 
- Kruskal 알고리듬
- Sollin 알고리듬 

모두 최적화 알고리듬인 그리디(Greedy) 알고리듬에 속한다. 

그리디 알고리듬은 매번 '욕심내서' 지역 최저점을 찾고, 이를 반복하면서 전역 최저점에 수렴하는 최적화 방식이다. 

# Prim 알고리듬 

무향 그래프에서 최소신장트리 찾는 알고리듬이다. 

### 연결성분과 정점 사이 간선 가중치 최소인 정점 찾아 연결성분과 연결한다. 

## 예시 

<img width="547" alt="Screen Shot 2022-02-19 at 14 32 28" src="https://user-images.githubusercontent.com/83487073/154787793-0a13bc04-9889-4ebc-af85-d14c20f45a70.png">

- 싸이클이 있는 무향 그래프다. 

### 시작점 A로 임의 설정한다. 

<img width="526" alt="Screen Shot 2022-02-19 at 14 34 07" src="https://user-images.githubusercontent.com/83487073/154787846-48f107c8-4779-4de7-87c0-32b4d6746f0b.png">

### A 에 인접한 정점 중 간선 가중치 최소인 정점과 A를 연결한다. 

### A 인접 정점 B, C 중에 B가 간선 가중치 3으로 더 작다. 따라서 A와 B를 연결한다. 

<img width="546" alt="Screen Shot 2022-02-19 at 14 35 57" src="https://user-images.githubusercontent.com/83487073/154787899-0bbddbeb-2ed4-4c67-b9b8-1a52a0e31510.png">

### A와 B 연결성분에 인접한 정점들 중 간선 가중치 최소인 정점과 연결성분 연결한다. 

### B 인접 정점 C가 간선 가중치 1로 가장 작다. 따라서 A-B 연결성분과 정점 C 연결한다. 

<img width="498" alt="Screen Shot 2022-02-19 at 14 46 29" src="https://user-images.githubusercontent.com/83487073/154788177-8b6f5893-9e85-4673-b41c-d9644ecb672d.png">

### A-B-C 연결성분과 인접한 정점 중 간선 가중치 최소인 정점 찾는다. 

### B와 인접한 정점 D가 간선 가중치 6으로 가장 작다. 

### A-B-C 연결성분과 정점 D 연결한다. 

<img width="496" alt="Screen Shot 2022-02-19 at 14 48 37" src="https://user-images.githubusercontent.com/83487073/154788236-1e8ebf80-c7a3-48ed-a75b-1366ff7c77cf.png">

### A-B-C-D 연결성분과 인접한 정점 중 간선 가중치 최소인 정점 찾는다. 

### E가 간선 가중치 7로 가장 작다. 따라서 A-B-C-D와 간선 E 연결한다. 

<img width="515" alt="Screen Shot 2022-02-19 at 14 50 20" src="https://user-images.githubusercontent.com/83487073/154788288-993dbd33-cfc2-491d-a2d1-46c69d254737.png">

### A-B-C-D-E 연결성분과 인접 정점 중 간선 가중치 최소인 정점 찾는다. 

### D에서 F가 간선 가중치 13으로 가장 작다. 

### 정점 D와 F를 잇는다. 

<img width="512" alt="Screen Shot 2022-02-19 at 14 52 44" src="https://user-images.githubusercontent.com/83487073/154788332-4b893364-3d68-46b8-9658-05953f468766.png">

### A-B-C-D-E-F 연결성분과 인접한 정점 중 간선 가중치 최소인 정점 찾는다.

### F-I가 간선 가중치 12로 가장 작다. 따라서 F와 I를 잇는다. 

<img width="502" alt="Screen Shot 2022-02-19 at 14 54 12" src="https://user-images.githubusercontent.com/83487073/154788366-72c66dfc-100b-4814-b1ea-7bd89b8b9e4b.png">

### 연결성분 A-B-C-D-E-F-I 와 인접한 정점 중 간선 가중치 가장 작은 정점과 연결성분 잇는다. 

### I - H 가 간선 가중치 9로 가장 작다. 따라서 I와 H를 잇는다. 

<img width="501" alt="Screen Shot 2022-02-19 at 14 55 48" src="https://user-images.githubusercontent.com/83487073/154788403-b8dbd63c-440b-447e-9348-98fefcb5713d.png">

### 연결성분과 인접한 정점 중 간선 가중치 최소인 정점 찾는다. 

### H와 J가 간선 가중치 8로 가장 작다. 따라서 H와 J를 잇는다. 

<img width="502" alt="Screen Shot 2022-02-19 at 14 58 18" src="https://user-images.githubusercontent.com/83487073/154788468-d262a2d7-41f4-4f33-b6ce-30eae4eefeb8.png">

### 연결성분과 마지막 남은 정점 G를 간선 가중치 최소인 정점으로 잇는다. 

### H-G가 간선 가중치 10으로 가장 작다. 따라서 H와 G를 잇는다. 

<img width="532" alt="Screen Shot 2022-02-19 at 14 59 43" src="https://user-images.githubusercontent.com/83487073/154788511-f8b1bbea-4f88-4d18-972c-e51b5437f1cb.png">

위 결과를 모양을 트리처럼 그리면 아래와 같아질 것이다. 

<img width="319" alt="Screen Shot 2022-02-19 at 15 05 34" src="https://user-images.githubusercontent.com/83487073/154788691-a19dc68a-8cdb-42cd-9980-f17c869f70a1.png">

- 싸이클이 없는가?: 없다. 
- 최소 간선 수인가?: 맞다. $10-1 = 9$
- 모든 정점이 연결되어 있는가?: 연결되어 있다. 
- 간선 가중치 합 최소인가?: 맞다. $69$. 

## $\Rightarrow$ Prim 알고리듬 실행 결과 최소신장트리 찾았다. 

---

# Prim 알고리듬 구현 

Prim 알고리듬 구현하고, 위 예제를 넣어보자.

```python 
# Prim's Algorithm

for i in range(N): # 모든 정점 한번씩 찍는다. 정점 수 만큼 반복. 
    # 임시로 쓰는 값. 큰 의미 없다.
    m = np.random.randint(-5, -1)
    min_value = sys.maxsize

    for j in range(N): # 전체 정점 중에서 
        if not visited[j] and D[j] < min_value : # 방문 안 했고, 인접한 거. 
            min_value = D[j] # 최소 간선 가중치 
            m = j # 최소 간선 정점 
        
    visited[m] = True # 최소 간선 정점 방문 완료 

    # 방금 방문한 정점 m의 인접 정점과 가중치 
    for w, wt in g[m]: 
        if not visited[w] : 
            if (wt < D[w]): # 간선 w 가중치 D에 추가 or D 업데이트 
                D[w] = wt 
                previous[w] = m 
```

## 메커니즘 

1. 어느 정점 연결하지? 
2. 전체 정점 중에 1) 방문 안 했고 2) 연결성분과 인접 3) 간선 가중치 최소 인 정점 찾는다
3. 2번 정점과 연결 
4. 2번 정점의 인접 정점 찾고, 간선 가중치 조사한다. 

### 모든 정점 연결 될 때 까지 1~4 과정 반복한다. 


## 테스트 - 1

```python
# 위 별모양 그래프로 프림 알고리듬 테스트 

start = 0 
N = 10 
g = [None for x in range(N)]

g[0] = [(2,3),(3,5)]
g[1] = [(2,6),(5, 13)]
g[2] = [(0, 3), (1,6),(3,1),(5,14)]
g[3] = [(0,5),(2,1),(4,7),(6,15)]
g[4] = [(3,7),(6,16)]
g[5] = [(1,13),(2,14),(8,12),(7,17)]
g[6] = [(3,15),(4,16),(7,10),(9,11)]
g[7] = [(5,17),(8,9),(6,10),(9,8)]
g[8] = [(5,12),(7,9)]
g[9] = [(7,8),(6,11)]

visited = [False for x in range(N)]


D = [sys.maxsize for x in range(N)]


D[start] = 0 


previous = [None for x in range(N)]
previous[start] = None 

# 프림 알고리듬 

for i in range(N): 
        m = np.random.randint(-5, -1)
        min_value = sys.maxsize

        for j in range(N): # 전체 정점 중에서 
            if not visited[j] and D[j] < min_value : # 방문 안 했고, 인접한 거. 
                min_value = D[j] # 최소 간선 가중치 
                m = j # 최소 간선 정점 
        
        visited[m] = True # 최소 간선 정점 방문 완료 

        # 방금 방문한 정점 m의 인접 정점과 가중치 
        for w, wt in g[m]: 
            if not visited[w] : 
                if (wt < D[w]): # 간선 w 가중치 D에 추가 or D 업데이트 
                    D[w] = wt 
                    previous[w] = m 


# 결과 출력 

print(f'최소신장트리(간선):', end='')
mst_cost = 0 
for i in range(N): 
    print(f'({i}, {previous[i]})', end='')
    mst_cost += D[i]
print(f'\n최소신장트리 가중치 합:{mst_cost}')
```
최소신장트리(간선):(0, None)(1, 2)(2, 0)(3, 2)(4, 3)(5, 1)(6, 7)(7, 8)(8, 5)(9, 7)

최소신장트리 가중치 합:69

예상대로 잘 나왔다. 

## 테스트 - 2

이번엔 다음과 같은 그래프에서 최소신장트리를 찾아보자. 

<img width="440" alt="Screen Shot 2022-02-19 at 15 36 42" src="https://user-images.githubusercontent.com/83487073/154789697-a3f41c31-0ac8-4088-a73b-87a2fe773ddc.png">

```python 
# 시작 정점 
start = 1 

# 정점 수 
N = 10 

# 0~9 까지 각 정점의 인접 정점 리스트 
g = [None for x in range(N)]

# 각 정점의 (인접 정점, 그 사이 간선 가중치)
g[0] = [(1, 7), (2, 6)]
g[1] = [(0, 7), (2, 5), (6, 13), (3, 9)]
g[2] = [(0, 6), (1, 5), (5, 8), (4, 2)]
g[3] = [(1, 9)]
g[4] = [(2, 2), (9, 1)]
g[5] = [(2, 8)]
g[6] = [(1, 13), (8, 11), (7, 10)]
g[7] = [(6, 10)]
g[8] = [(6, 11)]
g[9] = [(4,1)]

# 각 정점 방문 완료 여부 표시 
visited = [False for x in range(N)]

# 정점 i와 연결 성분 사이 간선 가중치(최소가 우선)
D = [sys.maxsize for x in range(N)]

# 시작 정점과 연결성분 사이엔 간선이 존재 안 한다. 
D[start] = 0 

# 새로 발견된 정점의 그 이전 정점 (최소신장트리 간선 추출 위함)
previous = [None for x in range(N)]
previous[start] = None 
```

## 프림 알고리듬 정의

```python 
# 프림 알고리듬 정의 

def prim(N, start): 
    global g ; global visited ; global D ; global previous 

    for i in range(N): 
        m = np.random.randint(-5, -1)
        min_value = sys.maxsize

        for j in range(N): # 전체 정점 중에서 
            if not visited[j] and D[j] < min_value : # 방문 안 했고, 인접한 거. 
                min_value = D[j] # 최소 간선 가중치 
                m = j # 최소 간선 정점 
        
        visited[m] = True # 최소 간선 정점 방문 완료 

        # 방금 방문한 정점 m의 인접 정점과 가중치 
        for w, wt in g[m]: 
            if not visited[w] : 
                if (wt < D[w]): # 간선 w 가중치 D에 추가 or D 업데이트 
                    D[w] = wt 
                    previous[w] = m 
    
    span = [] 
    mst_cost = 0 
    for i in range(N) : 
        span.append((i, previous[i]))
        mst_cost += D[i]

    return span , mst_cost

prim(10, 1)
```

(\[(0, 2),
  (1, None),
  (2, 1),
  (3, 1),
  (4, 2),
  (5, 2),
  (6, 1),
  (7, 6),
  (8, 6),
  (9, 4)\],
 65)

---

# 최단 경로(Shortest Path) 찾기

가중치 그래프 출발점에서 어떤 정점 $w$ 까지 도달하는 최단 경로 찾기. 

# Diijkstra(데이크스트라) 알고리듬

가중치 그래프에서. 출발점~정점 $w$ 까지. 최단 경로 찾는 알고리듬. 

- 전체적으로 Prim 알고리듬과 거의 똑같다. 

## Prim 알고리듬과 차이점 

1. Prim 알고리듬은 출발점 주어지지 않지만, Diijkstra 알고리듬은 출발점이 주어진다. 
2. Prim 알고리듬은 리스트 D에 연결성분과 정점 $w$ 사이 간선 가중치가 저장되는 데 반해, Diijkstra 알고리듬은 D에 출발점~정점 $w$ 사이 경로 길이가 저장된다. 

*경로 길이: 출발점~정점 $w$ 사이 간선들 가중치 합. 

---

# Diijkstra 알고리듬 구현 

*가중치 중 음수 있으면 최단 경로 제대로 못 찾을 수 있다. 

```python 
# 데이크스트라 알고리듬 정의 

import sys 

for k in range(N) : 
    # 여기서 m과 min_value 는 임시 값.
    m = -1
    min_value = sys.maxsize 
        
    for i in range(N) : 
        if not visited[i] and (D[i] < min_value) : # 방문 안 했고, 출발점에서 경로 길이 가장 짧은 정점 j
            min_value = D[i] # 최소 경로 길이 , D[i] 는 출발점에서 정점 i 사이 간선들 가중치 합
            m = i # 출발점에서 가장 가까운 정점 m 
    visited[m] = True # 방문 완료: 최단 경로 확정. 갱신 더 이상 안 한다.
    
    for w, wt in g[m] : # 정점 m의 (인접 정점, 가중치)
        if not visited[w] : # 출발~정점 w 까지 최단거리 아직 확정 못 지었다면 
            # 최단경로 갱신(간선완화)
            if (D[m] + wt) < D[w] : # 기존 경로보다 (D[m] + wt)가 더 최단 경로면 
                D[w] = D[m] + wt # 출발점~w까지 최단 거리 갱신. 간선완화 
                previous[w] = m 
```

## 알고리듬 테스트 - 1

아래 그래프에서 출발점을 정점 0 삼아 0에서 각 정점 i 까지 최단 경로를 찾아보자. 

<img width="522" alt="Screen Shot 2022-02-19 at 21 55 36" src="https://user-images.githubusercontent.com/83487073/154801691-a6520855-9b69-443b-a323-eaaca33b8cec.png">

```python 
# 테스트 

# 정점 수 
N = 10 

# 출발점
s = 0 

# 정점 i의 (인접 정점, 간선 가중치) 리스트
g = [None for x in range(N)]

# 정점 w까지 최단 경로 확정 유무 
visited = [False for x in range(N)]

# 출발점부터 정점 w 까지 경로 길이 
D = [sys.maxsize for x in range(N)]
D[s] = 0 

# 정점 w의 이전 정점 리스트 
previous = [None for x in range(N)]
previous[s] = None 

# 그래프 
g[0] = [(2,3),(3,5)]
g[1] = [(2,6),(5, 13)]
g[2] = [(0, 3), (1,6),(3,1),(5,14)]
g[3] = [(0,5),(2,1),(4,7),(6,15)]
g[4] = [(3,7),(6,16)]
g[5] = [(1,13),(2,14),(8,12),(7,17)]
g[6] = [(3,15),(4,16),(7,10),(9,11)]
g[7] = [(5,17),(8,9),(6,10),(9,8)]
g[8] = [(5,12),(7,9)]
g[9] = [(7,8),(6,11)]


# 데이크스트라 알고리듬 정의 
import sys 

for k in range(N) : 
    m = -1
    min_value = sys.maxsize 
        
    for i in range(N) : 
        if not visited[i] and (D[i] < min_value) : 
            min_value = D[i] # 최소 경로 길이 
            m = i # 출발점에서 가장 가까운 정점 m 
    visited[m] = True # 방문 완료: 최단 경로 확정. 갱신 더 이상 안 한다.
    
    for w, wt in g[m] : # 정점 m의 (인접 정점, 가중치)
        if not visited[w] : # 출발~정점 w 까지 최단거리 아직 확정 못 지었다면 
            # 최단경로 갱신(간선완화)
            if (D[m] + wt) < D[w] : # 기존 경로보다 최단 경로면 
                D[w] = D[m] + wt # 간선완화 
                previous[w] = m 
```

## 출발점 0에서 정점 i 까지 최단 경로 길이 

시작점과 정점 i 사이 경로 없으면 $D[i]$ 는 $\infty$ 로 그냥 둔다.

```python 
# 최단 경로 길이
print(f'출발점 {s} 로 부터 정점 i의 최단거리')
for i in range(N) : 
    if D[i] == sys.maxsize : # 시작점과 i 사이 경로 없는 경우
        print(f'출발점 {s}와 정점 {i} 사이 경로 없음')
    else : 
        print(f'[{s}, {i}] = {D[i]}') # 시작점과 정점 i 사이 최단 경로길이 출력
```

출발점 0 로 부터 정점 i의 최단거리

\[0, 0\] = 0

\[0, 1\] = 9

\[0, 2\] = 3

\[0, 3\] = 4

\[0, 4\] = 11

\[0, 5\] = 17

\[0, 6\] = 19

\[0, 7\] = 29

\[0, 8\] = 29

\[0, 9\] = 30

)

## 출발점 0에서 정점 i 까지 최단 경로 

```python 
# 최단경로

print(f'출발점 {s} 로 부터 정점 i 까지 최단경로')
for i in range(N) : 
    vertex = i # 현재 정점 i 
    print(vertex, end='')
    while (vertex != s) : 
        print(f'<-{previous[vertex]}', end='')
        vertex = previous[vertex]
    print() 
```

출발점 0 로 부터 정점 i 까지 최단경로

0

1\<-2\<-0

2\<-0

3\<-2\<-0

4\<-3\<-2\<-0

5\<-2\<-0

6\<-3\<-2\<-0

7\<-6\<-3\<-2\<-0

8\<-5\<-2\<-0

9\<-6\<-3\<-2\<-0

## 알고리듬 테스트 - 2

이번엔 아래 그래프에서 시작점 0 삼아 각 정점까지 최단 경로 찾기를 해보자. 

<img width="440" alt="Screen Shot 2022-02-19 at 15 36 42" src="https://user-images.githubusercontent.com/83487073/154789697-a3f41c31-0ac8-4088-a73b-87a2fe773ddc.png">

```python 
# 테스트

N = 10 
s = 0 

g = [None for x in range(N)]

visited = [False for x in range(N)]

D = [sys.maxsize for x in range(N)]
D[s] = 0 

previous = [None for x in range(N)]
previous[s] = None 

g[0] = [(1, 7), (2, 6)]
g[1] = [(0, 7), (2, 5), (6, 13), (3, 9)]
g[2] = [(0, 6), (1, 5), (5, 8), (4, 2)]
g[3] = [(1, 9)]
g[4] = [(2, 2), (9, 1)]
g[5] = [(2, 8)]
g[6] = [(1, 13), (8, 11), (7, 10)]
g[7] = [(6, 10)]
g[8] = [(6, 11)]
g[9] = [(4,1)]



# 데이크스트라 알고리듬 정의 
import sys 

for k in range(N) : 
    m = -1
    min_value = sys.maxsize 
        
    for i in range(N) : 
        if not visited[i] and (D[i] < min_value) : 
            min_value = D[i] # 최소 경로 길이 
            m = i # 출발점에서 가장 가까운 정점 m 
    visited[m] = True # 방문 완료: 최단 경로 확정. 갱신 더 이상 안 한다.
    
    for w, wt in g[m] : # 정점 m의 (인접 정점, 가중치)
        if not visited[w] : # 출발~정점 w 까지 최단거리 아직 확정 못 지었다면 
            # 최단경로 갱신(간선완화)
            if (D[m] + wt) < D[w] : # 기존 경로보다 최단 경로면 
                D[w] = D[m] + wt # 간선완화 
                previous[w] = m 
```

## 출발점 0에서 정점 i 까지 최단 경로 길이 

시작점과 정점 i 사이 경로 없으면 $D[i]$ 는 $\infty$ 로 그냥 둔다.

```python 
# 최단 경로 길이 
print(f'출발점 {s} 로 부터 정점 i의 최단거리')
for i in range(N) : 
    if D[i] == sys.maxsize : 
        print(f'출발점 {s}와 정점 {i} 사이 경로 없음')
    else : 
        print(f'[{s}, {i}] = {D[i]}')
```

출발점 0 로 부터 정점 i의 최단거리

\[0, 0\] = 0

\[0, 1\] = 7

\[0, 2\] = 6

\[0, 3\] = 16

\[0, 4\] = 8

\[0, 5\] = 14

\[0, 6\] = 20

\[0, 7\] = 30

\[0, 8\] = 31

\[0, 9\] = 9

## 출발점 0에서 정점 i 까지 최단 경로 

```python 
# 최단 경로 

print(f'출발점 {s} 로 부터 정점 i 까지 최단경로')
for i in range(N) : 
    vertex = i # 현재 정점 i 
    print(vertex, end='')
    while (vertex != s) : 
        print(f'<-{previous[vertex]}', end='')
        vertex = previous[vertex]
    print() 
```

출발점 0 로 부터 정점 i 까지 최단경로

0

1\<-0

2\<-0

3\<-1\<-0

4\<-2\<-0

5\<-2\<-0

6\<-1\<-0

7\<-6\<-1\<-0

8\<-6\<-1\<-0

9\<-4\<-2\<-0

---

# Diijkstra 알고리듬 성능 

## $O(N^{2})$ 

- 알고리듬 시작부에서. 출발점에서 가장 가까운 정점 찾기 위해. D에서 $N$개 정점 비교한다. $\Rightarrow$ $O(N)$ 시간 소요 
- 출발점에서 가장 가까운 정점 $m$ 의 인접 정접 $M(M\leq N)$ 개를 검사해서, D의 원소 갱신한다. $\Rightarrow$ 추가로 $O(M)$ 시간 소요 

$O(N+M) = O(N)$

- 위 과정을 정점 갯수 $N$ 번 반복한다. $\Rightarrow$ 총 수행시간: $O(N^{2})$












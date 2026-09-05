---
title : "[기록] Inductive & Transductive on Hypergraphs"
excerpt : "공부한 내용 기록"

categories : 
- computer science
- machine learning
- hypergraph
- study 

tags : 
- [ML, DL, Hypergraph, study]

toc : true 
toc_sticky : true 
use_math : true

date : 2026-09-05
last_modified_at : 2026-09-05

---


2026.09.05 기록.

- Inductive learning (=model)
한 줄 정리 > '산출물의 정의역이 도메인 공간 X 전체'.
>> 입력으로 들어오지 않은 대상(같은 도메인)에 대해서도 산출가능
입력 :
1. 레이블 있는 데이터 집합 L = {(x₁, y₁), …, (x_n, y_n)}
출력 : 일반 함수 f : X → Y
정의 : L을 근거로 입력 도메인 공간 전체에서 작동하는 함수 f를 도출.
학습에 없던 x_new에서도 f(x_new)로 예측 수행.


- Transductive learning(=model)
 한줄 정리 > '산출물의 정의역이 학습 시점 입력에 국한됨'
>> 학습 시점에 들어갔던 대상에 대해서만 산출할 수 있다.
입력 :
1. Label 있는 데이터 집합 L = {(x₁, y₁), …, (x_n, y_n)}
2. Label 없는 데이터 집합 U = {x_{n+1}, …, x_{n+m}}
정의 : L과 U를 학습 시점에 모두 입력으로 받아, L의 Label + L과 U의 X들을 근거로 U의 Label 예측.
> L과 U는 기본적으로 같은 분포에서 나온 데이터로 가정.
> U 밖의 새 데이터에는 정의되지 않는다.


- Inductive 와 Transductive 차이
1. 레이블 없는 데이터 U가 입력으로 들어가는지 유무
2. Output이 Inductive는 함수 f, Transductive는 U의 레이블 vector.
3. 구현 층위 : 특정 객체에 배정된 파라미터가 모델 내부에 있으면 = Transductive. 없으면 Inductive.




Inductive link prediction :
- 훈련 때 없었던 노드로 구성된 하이퍼엣지 예측
- 훈련 때 없었던 관계 타입을 갖는 하이퍼엣지 예측
>> 내 HyperKG연구가 가야할 방향. Inductive link prediction.


논문 : Hyper(ICLR, 2026)
용어
- Relation : 하이퍼엣지 entity type.
- arity : 하이퍼엣지 relation을 구성하는 slot 갯수 >> relation 구성 요소 수.
- KG에서 하이퍼엣지 : 어떤 사실을 기호화/구조화 시킨 대상.
- 모델은 관계의 구조적 유사성을 학습, 유사한 새 관계에 이전할 수 있어야 한다.
- 'Fully generalizable' 한 Hypergraph Foundation model.
- 각 hyperedge가 갖는 relation(type)-relation(type) pairwise interaction pattern을 보고, Inductive하게 이전 가능한 구조적 지식을 추출한다. < 각 node의 위치 encoding이 중요하게 작용.
- 이건 마치 자연어 문장에서 각 개념들 사이 주어와 술어, 목적어 패턴 비교해서 각 단어 위상, 지식 패턴 유사성 파악하는 것과 유사하다.
---
title : "[데이터시각화/파이썬] matplotlib 을 이용해서 그래프 그리기"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- Data Science
- python
- matplotlib

tags : 
- [data structure, computer science, study, data science, computer engineering]

toc : true 
toc_sticky : true 

date : 2021-06-21
last_modified_at : 2021-08-24

---
---
공부 중인 내용을 기록으로 남겨, 
나중에 다시 빠르게 '불러오기' 위한 저장소.

# 데이터 시각화 하기 
### 데이터사이언스 공부 기반이 되는 필수 수학 공부 중 만난 시각화 방법들을 기록한다. 
- 사용한 라이브러리 : matplotlib 

```python
import matplotlib as plt # plt라는 약어로 불러오기 위함. 
# 지금 시점에서 나는 스타트업 파일을 적용해서, 파이썬3 실행할 때 마다 자동으로 matplotlib을 불러오도록 지정해뒀다. 

# 스타트업 파일은 내가 파이썬3를 실행할 때, 자동으로 가장 먼저 불러올 코드 뭉치, 모듈 등을 IDLE 등을 통해 기록. 파이썬 스크립트를 만든 뒤, 파이썬이 설치되어 있는 폴더 - 스타트업 폴더 - 에 붙여놓아두면 된다. 

# 00, 01 이런 식으로 파일 명을 해두는 것이 좋다. 00이면 가장 먼저 로드, 01 이면 그 다음으로 코드 뭉치를 자동 로드해온다. 
```

# 라인플롯 

- 가장 많이 사용한 그래프이다. 
- 구현이 간단해서 쉽게 사용할 수 있다. 
- 라인플롯 불러오기 
```python
plt.plot(x값, y값...)  #x값 y값 말고도 수많은 argument를 지정해 줄 수 있다. 가장 기본은 이거다. 
```
- 라인플롯의 가장 기본 원리는 x점, 그리고 각 x점에 해당하는 y점을 지정받아, (x,y) 점을 2차원 벡터공간 상에 일일이(!) 찍어서 선(라인)으로 된 그래프를 그리는 것이다. 
- 예 : 

```python
xx = np.linspace(-3,3,1000)
def f(a) : 
    return a**2
y = f(xx)

plt.plot(xx, y)
```
<img width="737" alt="Screen Shot 2021-06-21 at 15 15 50" src="https://user-images.githubusercontent.com/83487073/122715249-923e7100-d2a3-11eb-9e15-c5fed3c4c06b.png">

이렇게 라인 플롯을 그릴 수 있다. 

## 라인플롯으로 점 찍기 
- plt.plot(x좌표, y좌표, 'ro', ms=점 사이즈)

'ro' 뿐 아니라 색 이름 머릿글자 + 'o' 붙여 지정해주면 점 찍을 수 있다. 

참고 )

색 이름 머릿글자 : 

r : 빨강

g : 초록

b : 파랑

k : 검정

c : cyan

m : magenta

y : 노랑

---

# 3D 서피스 플롯
- 2차원 함수를 3차원 벡터공간 상에 나타내서 시각화 할 때 쓰기 좋다. 
- 말 그대로 'surface' 표면을 표현해준다. 
- 2차원 공간상의 x,y 점 \<그리드포인트\>라고 한다. 을 2차원 함수에 넣어서 나온 결과값인 Z값들을 3차원 공간상에 표현한 것이다. 
- 서피스플롯에 색 입히는 법 : cmap= 인자에 원하는 색상 넣어주면 된다. 색 이름 모르면 cmap=에 아무 값이나 넣고 오류 발생시키자. 가능한 색상 이름 다 알려준다. 
- ax.view_init(각도, 각도) : 각도값들을 넣어주면 3차원 그래프를 돌려가며 여러 각도에서 볼 수 있게 해준다. 
- ax.set_zlabel('') : Z 축 라벨 달 수 있게 해준다. 
- ax.set_xticks(\[\]) : X 축 막대들 설정할 수 있다. 리스트 안에 아무것도 안 넣으면 바bar 모두 제거 가능하다. 
- plt.xticks(\[\]) 도 위와 같이 작동한다. 
- ax.set_yticks(\[\]) : y 축 막대들 설정할 수 있다. 
- plt.yticks(\[\]) 도 위와 같이 작동한다. 
- ax.set_zticks(\[\]) : z 축 막대들 설정할 수 있다. 

```python
xx = np.linspace(-3,5,100)
yy = np.linspace(-3,5,100)
def f(a,b) : 
    return a**2+(3*b)
X,Y = np.meshgrid(xx,yy) #그리드포인트 생성 #각 (x,y)점들
Z = f(X,Y)
ax = plt.gca(projection='3d')
ax.plot_surface(X,Y,Z, linewidth=0.1)

plt.title('서피스 플롯 예시')
plt.show()
```
<img width="731" alt="Screen Shot 2021-06-21 at 15 30 36" src="https://user-images.githubusercontent.com/83487073/122716756-a3887d00-d2a5-11eb-80dd-13f638c4b70d.png">


# 컨투어 플롯(등고선 플롯)

```python
# 일반 컨투어 플롯
plt.contour(X,Y,Z, colors=, levels=, alpha=)
# colors = 색상
# levels = 등고선 간격 
# alpha = 컨투어플롯 진하고 연한 정도 조정 
```

- 서피스플롯은 말 그대로 3차원 형상으로 2차원 함수를 시각화했다면, 컨투어 플롯은 2차원 등고선 형상으로 2차원 다변수함수를 표현한다. 
- Z값이 같은 그리드포인트들은 하나의 등고선 라인에 포함된다(원이든. 타원이든 이룬다)
- 다변수 정규분포 등을 나타내는 데 유용하게 사용한다. 
- levels= : levels 인자에 넣어준 값들에 등고선 표시한다. 등고선은 같은 Z값들이다. 
- plt.clabel(CS=등고선그래프 이름, fmt=) : 등고선들에 해당하는 Z 값들을 등고선에 라벨로 붙여준다. fmt= 인자(format)에 표현 형식 지정해줄 수있다. 예) %d (정수꼴로 표현해라)

컨투어플롯을 변수에 담고, CS argument에 플롯을 담은 변수 이름을 넣어주면 된다. 

```python
xx = np.linspace(-20,20,100)
yy = np.linspace(-20,20,100)
X,Y = np.meshgrid(xx,yy) #그리드포인트 생성

def f(a,b) : 
    return (a-3)**2+(b-2)**2

Z = f(X,Y)

plt.figure(figsize=(10,5))
plt.contour(X,Y,Z)
plt.title('컨투어플롯 예시')
plt.show()
```
<img width="738" alt="Screen Shot 2021-06-21 at 15 38 49" src="https://user-images.githubusercontent.com/83487073/122717600-c8312480-d2a6-11eb-8121-6a82d466ef87.png">

# 색상 컨투어 플롯 
```python
# 각 등고선을 색상으로 채운다. 
plt.contourf(X, Y, Z, alpha=, levels=, colors=)
```
<img width="626" alt="Screen Shot 2021-09-01 at 19 33 26" src="https://user-images.githubusercontent.com/83487073/131656766-06f2045f-a4e2-4657-a323-b14da65ea589.png">

## 컨투어플롯 여러 개를 한번에 중첩할 수도 있다. 
```python
plt.contourf(X,Y,Z)
plt.contour(X,Y,Z)
```

---

# 바 플롯
- plt.bar(x값들, y값들)
- 막대그래프
- x값들에 해당하는 y값들을 막대로 표현한다. 라인플롯과 원리는 비슷하다. 

```python
xx = np.linspace(-3,3,10)

plt.bar(np.arange(10), xx)
plt.xticks(np.arange(10), ['A','B','C','D','E','F','G','H','I','K'])
plt.show()
```
<img width="626" alt="Screen Shot 2021-08-24 at 22 25 59" src="https://user-images.githubusercontent.com/83487073/130624593-89b4fe6e-d1f5-4fa5-9e48-26339198ab87.png">

# 바 플롯 - seaborn
- matplotlib 말고 seaborn으로 바 플롯 그릴 수도 있다. 기본색이 더 알록달록해서 좋다. 
- sns.barplot(x값들, y값들), matplotlib과 방식 똑같다. 

```python
xx = np.linspace(-3,3,10)
sns.barplot(np.arange(10),xx)
plt.xticks(np.arange(10), ['a']*10)
plt.show()
```
<img width="629" alt="Screen Shot 2021-08-24 at 22 35 09" src="https://user-images.githubusercontent.com/83487073/130626119-eb140ae3-536a-4f48-b27a-1cadfad8a3c4.png">


# 스캐터플롯 
- plt.scatter(x값들, y값들, 점 사이즈, 색상)
- 점들 흩뿌려 놓는 그래프 
- (x값, y값) 벡터로 만들어서 좌표평면에 점으로 찍는다. 
- 점 1개만 찍을 수도 있다. 

```python
plt.scatter(1,2, 1000, 'b')
```
<img width="629" alt="Screen Shot 2021-08-24 at 22 29 52" src="https://user-images.githubusercontent.com/83487073/130625215-8d210ebe-9767-4dcf-bf10-4a10b62f0256.png">

```python
xx = np.linspace(0.01, 1,50)
yy = np.linspace(4,5,50)

plt.scatter(xx, yy, 20, 'r')
```

<img width="625" alt="Screen Shot 2021-08-24 at 22 31 55" src="https://user-images.githubusercontent.com/83487073/130625579-d3989e9d-9260-45f6-bff3-06a13a80fe38.png">




---




# matplotlib 유용한 메서드 정리 
### plt.figure(figsize=(가로, 세로))
- 그래프 그리기 전 그래프 사이즈를 얼만큼으로 할 건지 지정해줄 수 있었다. 
### plt.text(x점, y점, '텍스트 내용', 등등)
- 벡터공간 x,y좌표 상에 텍스트를 적을 수 있다. 
### plt.xticks(반복가능자)
- 괄호 안의 반복가능자를 받아 x축에 수직인 축을 그려준다. 
- plt.xticks(np.arange(5), \['A','B','C','D','E'\]) 이런 식으로 설정해주면 0에서 4번 인덱스까지 해당하는 x축 이름을 'A'에서 'E'까지 알파벳으로 바꿀 수 있다. 

### plt.yticks(반복가능자)
- 괄호 안의 반복가능자 받아 y충게 수직인 축 그려준다. 
### plt.axvline(x좌표,ls='-.'(라인 스타일),c='k'(라인 색상)) 
- x축에 수직인 직선을 그어준다. (a x vertical line) 
- c = 'k' \<검정\>
- c = 'b' \<파랑\>
- c = 'r' \<빨강\> 등 색상 지정가능했다. 
- y최소, y최대 없이 좌표공간 전체를 가로질러 선 긋는다. 
### plt.axhline(y좌표, ls='--',c='b')
- x축에 수평인(=y축에 수직인) 직선 그어준다. 
- 다른 내용은 axvline() 과 같다. 
### plt.xlim(시작점, 끝점)
- x축 어디부터 어디까지 나타낼 건지 지정할 수 있다. 
- x.limit제한
### plt.ylim(시작점, 끝점)
- y축 어디부터 어디까지 나타낼 건지 지정할 수 있다. 
- y.limit제한
### plt.tight_layout() 
- plt.subplot() 등을 이용해서 여러 개 그래프 한 지면에 그리는 경우 유용하다. 
- 그래프 간 축. 제목 등이 좁은 지면 공간 때문에 겹치는 경우가 생기는데, 그래프 간 간격 적당히 조정해서 이런 문제 해결해준다. 
### plt.subplot(221,222,223,224...)
- 서브플롯, 그러니까 작은 플롯 여러개를 동시에 한 지면에 나타내준다. 
- 그냥 플롯을 그리면 한 개 플롯이 한 개 지면 전체를 차지하게 된다. 
- 여러 개 서브 플롯을 1개 셀 결과로 동시에 출력할 수 있다. 
- 괄호안에는 플롯 사이즈를 넣어준다. 예륻 들어 221 이면 서브플롯 세로로 2개, 가로로 2개 총 4개를 만들고, 1은 그 중에 첫번째 플롯이다~ 라고 지정하는 숫자다. 

이거는 plt.subplot(2,2,1) 로 지정해주는거랑 똑같았다. 마찬가지로 세로에서 2개, 가로에서 2개 해서 총 4개 플롯을 그릴거고, 1은 그 중에 첫번째 플롯에 그린다~ 는 의미다. 


### plt.show() 
- 그래프 다 그린 뒤에, 입력해주면 깔끔하게 정돈된 그래프를 출력해준다. 
### plt.title('그래프 제목')
- 그래프 1개의 제목을 붙여줄 수 있다. 
### plt.xlabel('x축에 붙일 이름')
- 그래프 x축이 뭘 말하는지 이름 붙여줄 수 있다.
### plt.ylabel('y축에 붙일 이름')
- 그래프 y축이 뭘 말하는지 이름 붙여줄 수 있다. 
### plt.suptitle('대 제목')
- 여러 개 서브플롯이 있는 경우, plt.title()로는 각 서브플롯마다 그래프 제목 붙여줄 수 있고, 
- plt.suptitle('대 제목')으로는 전체 서브플롯들의 1개 대 제목을 붙일 수 있다. 

### plt.annotate('', xy=\[a,b\], xytext=\[c,d\], arrowprops={'facecolor' : 'color'})
- 2차원 표준기저벡터로 이루어진 좌표평면상에 화살표를 그릴 수 있다. 
- xy : 종점(화살표가 가리키는 지점) $\vert$ 예 : \[2,3\] 
- xytext : 시점 $\vert$ 예 : \[0,0\]
- ' ' : 텍스트를 쓸 수 있다. xy 인자에 넣어준 종점 좌표에 텍스트가 찍힌다. 
- arrowprops : 화살표 색 등 설정할 수 있다. $\vert$ 예 : 딕셔너리 형태로 화살표 색상 지정 가능 {'facecolor' : 'red'}

---
### plt.arrow(원점좌표x, 원점좌표y, dx, dy, head_width=, head_length=, fc=화살표머리 색깔, ec=화살표몸통색깔, lw=굵기)

- plt.arrow는 평행이동해서 원점을 $(0,0)$ 에서 $(x,y)$로 변경한 벡터 $(dx, dy)$ 표현에 적합하다. 

예) 퀴버플롯에서 그레디언트 벡터 

설명)

벡터 $(dx, dy)$는 본래 $(0,0)$을 원점으로 삼는 벡터다. 

plt.arrow는 이 벡터 $(dx, dy)$ 를 평행이동해서, 원점을 $(x,y)$ 로 바꿔준다. 

결과적으로 plt.arrow의 결과물 벡터는 

## $(x,y)$ 를 원점, $(x+dx, y+dy)$ 를 종점으로 삼는 벡터

가 그려진다. 

---


### plt.vlines('수직선 그을 x좌표', ymin=, ymax=, colors=, ls=)
- x축에 수직인 vertical line 을 긋는 코드다. 따라서 수직선 그을 x 좌표 설정해준다.
- ymin : 수직선 시작할 y좌표 최솟값
- ymax : 수직선 끝날 y좌표 최댓값
- colors : 선 색상 
- ls : 선 스타일 지정

### plt.hlines('수직선 그을 y좌표', xmin=, xmax=, colors=, ls=)
- x에 수평인 horizontal line 긋는 코드다. 수평선 그을 y좌표 설정해준다. 
- xmin : 수평선 시작할 x좌표 최솟값
- xmax : 수평선 끝날 x좌표 최댓값
- colors : 선 색상
- ls : 선 스타일 지정

### plt.fill_between('x점', 'y1점', 'y2점', alpha=)
- $[x, y_{1}]$ 에서부터 $[x, y_{2}]$까지 사이 공간을 색으로 채운다. 
- alpha : 색 진한 정도 




# + 공부하면서 더 더 추가해 나가기 





---
title : "[R을 이용한 데이터분석] 데이터분석에 유용한 R 함수 정리"
excerpt : "공부 중인 내용을 정리한 글"

categories : 
- R

tags : 
- [R]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-10-05
last_modified_at : 2021-10-14

---

# which()

TRUE 값들의 벡터/데이터프레임 내 인덱스를 반환한다. 

- which(TRUE/FALSE 반환하는 어떤 조건)

---

# complete.cases()

벡터에서는 NA 값이 아닌 값들에 TRUE 반환, NA 값들에 대해서는 FALSE 반환한다. 

데이터프레임에서는 NA값을 포함하지 않는 데이터레코드(행)에 대해 TRUE 반환한다. 

NA값 포함하는 데이터레코드(행)에 대해서는 FALSE 반환한다. 

---

# is.na()

벡터/데이터프레임 내에 NA값에 대해서는 TRUE 반환, NA 아닌 값에 대해서는 FALSE 반환한다. 

---

# str()

데이터프레임의 구조를 출력해준다. 

structure() 를 뜻한다. 

---

# gsub(): 텍스트마이닝에 매우 유용할 듯 하다.

## gsub('a', '', 'abcdef') 

abcdef 에서 a 를 '' 로 변환해라는 말이다. 

## gsub(',','', '22,345') 

22,345 에서 , 찾아서 '' 로 바꾸라는 말이다.

---

# apply()

apply(행렬, MARGIN=, FUN=)
- MARGIN: 1을 하면 행, 2를 하면 열
- FUN: 행렬의 행. 열에 적용할 함수

---

# pnorm()

정규분포에서 $P(X \leq x)$ 계산한다. 

- pnorm(x, $\mu$, $\sigma$, lower.tail=TRUE)

- lower.tail=FALSE 이면, $P(X > x)$ 계산한다. 

예를 들어 정규분포에서 $X=5$ 와 $X=10$ 사이 수가 실현될 확률을 알고 싶다면, 다음과 같이 명령할 수 있다. 

$\Rightarrow$

$P(5 \leq x)$ and $P(x \leq 10)$

```R
pnorm(10, 평균, 표준편차, lower.tail=TRUE) - pnorm(5, 평균, 표준편차, lower.tail=TRUE)
```

 
# qnorm()

정규분포에서 특정 면적에 해당하는 $x$ 값을 계산해준다. 

- qnorm(면적넓이(확률값), $\mu$, $\sigma$)

예를 들어 정규분포에서 상위 10% 면적의 시작점에 해당하는 $x$ 값을 찾고 싶다면, 다음과 같이 하면 된다. 

```R
qnorm(0.9, 평균, 표준편차)
```
---

# quantile()

데이터셋 원소 중 각 % 별 분위수 알려주는 명령이다. 

- quantile(데이터, probs=seq(몇 분위수로 할 것인가?))

```R
quantile(cust$age, probs=seq(0,1,by=0.2))
```
probs에 0, 0.2, 0.4, 0.6, 0.8, 1 을 넣었다. 

곧, 데이터 중 0%, 20%, 40%, 60%, 80%, 100% 에 해당하는 데이터를 반환하라는 뜻이다. 

<img width="395" alt="Screen Shot 2021-10-14 at 23 27 53" src="https://user-images.githubusercontent.com/83487073/137337855-d7ca5cfc-9848-4dda-9f0d-f885a40a7cda.png">

---

# cut()

데이터셋을 몇 개 구간으로 나눠 묶어주는(쪼개주는) 명령이다. 

```R
cut(벡터, breaks=, labels=, include.lowest=TRUE, right=FALSE)
```

- breaks: bin 경계에 해당하는 값들 벡터 꼴로 입력한다. 또는 몇 개 구간으로 나눌건지 정수 n 넣어도 된다. (5개 구간으로 쪼갤 거면 breaks=5 넣으면 된다)
- labels: 각 bin에 label(이름, 카테고리값)을 붙인다
- right=FALSE: 간격이 왼쪽으로는 닫혀있고 오른쪽으로는 열려있게 하는 옵션이다
- include.lowest=TRUE: right=FALSE 옵션과 함께 쓴다. 가장 큰 값이 마지막 bin에 포함되도록 한다. 

# levels()

cut과 함께 

levels(cut(데이터셋, breaks=5)) 이런 식으로 쓰면

데이터셋을 쪼갠 각 5개 구간 범위(시작점-끝점)를 보여준다. 

---

# paste()

```R
paste(벡터1, 벡터2, 벡터3)
```

- 벡터 1,2,3 을 요소별로 '합친다'
- 요소들을 하나로 이어 붙여서 문자열 하나로 만든다. 

그림 보면 이해가 쉽다. 

<img width="749" alt="Screen Shot 2021-10-12 at 23 00 34" src="https://user-images.githubusercontent.com/83487073/136970582-d7d175cc-72c4-4420-976e-68ee29742cac.png">

이런 식으로 만든다는 거다. 

## 만약 

벡터1 다음에 벡터2가 아니라 문자 하나가 오면 어떻게 될까? 

```R
paste(벡터1, '%')
```
 
일종의 브로드캐스팅해서, 벡터1의 모든 원소에 %를 붙여준다. 

---

# table()

도수분포표 출력해준다. 

- 각 변숫값 별로 총 몇번 실현됬는지 알려준다. 

---

# file.choose()

interactive 하게 데이터셋을 불러올 수 있다. 

```R
file.choose()

# 또는 

read.csv(file.choose()) -> 객체이름

# 이렇게 쓴다. 
```

---

# round()

반올림 함수다. 

```R
round(데이터, digits=2) # 반올림해서 소수 둘째자리까지 나타내라
```

---

# log()

log(벡터) 이런 식으로 쓰면 벡터 각 요소에 자연로그 취해준다. 

```R
log(vector1)
```

# log10()

log10(벡터) 이런 식으로 쓰면 벡터 각 요소에 상용로그 취해준다. 

---

# as.Date()

데이터를 날짜 자료형으로 바꿔준다. 

```R
as.Date(날짜형으로 바꿀 데이터, format='%m/%d/%Y')
```

format 에서 

- %m : R에게 '날짜형으로 바꿀 데이터' 에서 / 맨 앞자리를 '월'로 인식해라 
- %d : R에게 '날짜형으로 바꿀 데이터' 에서 / 중간 자리를 '일'로 인식해라 
- %Y : R에게 '날짜형으로 바꿀 데이터' 에서 / 맨 뒷자리를 '연'로 인식해라 

날짜형으로 바꿀 데이터가 '1990-08-21' 이렇게 생겼으면 format에 '/' 대신 '-' 써야 한다. 

'%Y-%m-%d' 이런 식이다. 이래야 똑바로 인식한다. 

만약 '18900623' 이라면 format 에는 '%Y%m%d' 이렇게 넣어야 똑바로 인식하고, date 자료형으로 바꿔준다. 

---

# difftime()

두 date 사이에 '몇 일'의 간격이 있는지 계산해주는 함수다. 

```R
difftime(끝나는date, 시작하는date)
```

---

# floor()

내림 함수다. 곧, 실수에서 자기자신보다 작지만 그중 가장 큰 수 반환한다. 

---

# fivenum()

벡터가 담고 있는 데이터 중 최솟값, 1분위수, 중앙값, 3분위수, 최댓값 반환하는 함수다. 

fivenum이 사분위 수 구하는 알고리듬은 다음과 같다. 

- 전체 표본 중에 중앙값을 찾는다. (50% 사분위수)
- 중앙값 이하 왼쪽 데이터를 찾는다. 그것들 중 다시 중앙값을 찾는다(25% 사분위수)
- 중앙값 이상 오른쪽 데이터를 찾는다. 그것들 중 다시 중앙값을 찾는다(75% 사분위수)
- 2에서 계산한 값을 1분위수(lower hinge), 3에서 계산한 값을 3분위수(upper hinge) 라고 한다. boxplot도 이거랑 똑같은 알고리듬 써서 1,3 사분위수 구한다. 

summary, quantile 은 훨씬 복잡한 알고리듬 써서 1,3 사분위수 구한다. 

---

# months()

date 자료형 갖는 데이터에 적용할 수 있다. date 벡터에 적용한다면, 벡터 각 요소에서 '월'에 해당하는 문자열만 모아서 새 벡터 만든다. 

```R
months(cust$BirthDate) -> cust$birthmonth

print(cust$birthmonth)
```


<img width="748" alt="Screen Shot 2021-10-16 at 13 28 02" src="https://user-images.githubusercontent.com/83487073/137573753-db12ae8a-7445-4bd0-8629-900895fa38f2.png">

---

# match()

```R
match(첫번째 벡터, 두번째 벡터. nomatch=0)
```

첫번째 벡터의 각 요소가 두번째 벡터에서 몇 번째에 있는지 알려준다. 

곧, 인덱스 넘버를 반환한다. 

만약 첫번째 벡터 요소가 두번째 벡터에 없으면 NA 값 반환한다. 

nomatch= argument 를 써서, NA 대신 다른 값 반환하게 할 수도 있다(위 경우는 0 반환)

예를 들면

아래 데이터를

<img width="748" alt="Screen Shot 2021-10-16 at 13 28 02" src="https://user-images.githubusercontent.com/83487073/137573753-db12ae8a-7445-4bd0-8629-900895fa38f2.png">



```R
match(cust$birthmonth, c('1월', '2월','3월', '4월', '5월', '6월', '7월','8월','9월','10월','11월','12월'))-> cust$birthmonth
head(cust$birthmonth)
```
위 코드를 써서

<img width="792" alt="Screen Shot 2021-10-14 at 23 25 13" src="https://user-images.githubusercontent.com/83487073/137337375-f4e4a982-7cab-4b37-8e27-dcf21eec1a46.png">

이렇게 바꿀 수 있다. 

---

# View()

전체 데이터프레임을 생략없이 엑셀-스프레드시트 느낌으로 보여준다. 

```R
View(df)
```

---

# weekdays()

해당 날짜가 무슨 요일인지 알려준다. 

```R
weekdays(as.Date('2000-12-25'))
```

월요일을 반환할 것이다. 

---

# format()

아래와 같이 사용하면 date 값에서 특정 부분만 떼어서 반환해준다. 

```R
format(as.Date('2000-12-25'), '%Y')
format(as.Date('2000-12-25'), '%d')
format(as.Date('2000-12-25'), '%m')
```
'2000' 

'25'

'12'

---

# Sys.Date() & Sys.time()

sys.date 는 오늘 날짜(연도-월-일) 알려준다. 

sys.time 은 오늘 날짜와 시간 알려준다. 

```R
Sys.Date()
Sys.time()
```
<img width="368" alt="Screen Shot 2021-10-14 at 23 44 25" src="https://user-images.githubusercontent.com/83487073/137340993-3584e096-dfb7-4a23-b3e7-4dfae493e49d.png">

---

# assign()

iterator 를 어떤 객체에 할당시켜주는 함수 

```R
assign('x', c(1,2,3,4,5))
```

5차원 열벡터를 x라는 객체에 할당하라는 의미다. 

---

# rep()

반복 명령이다. 

기본 쓰임새는 아래와 같다. 물론 사용방법도, 옵션도 다양하다.

```R
x5 <- rep(x(벡터이름), times=5) # 벡터 x를 다섯번 반복해서 x5 객체에 넣어라
```

객체 이름 x5인 25차원 열벡터 하나가 생성될 것이다. 

또 다른 사용 예

```R
x6 <- rep(x(벡터이름), each=7) # 벡터 x 각 요소를 7번씩 반복한 다음 x6에 넣어라
```

<img width="553" alt="Screen Shot 2021-10-16 at 16 13 07" src="https://user-images.githubusercontent.com/83487073/137577734-a1d3d9b2-e89d-450f-ac91-1cb9322af4fc.png">

---

# cbind()

열벡터 끼리 묶어라는 명령이다

```R
cbind(열벡터1, 열벡터2)
```

<img width="191" alt="Screen Shot 2021-10-16 at 16 43 42" src="https://user-images.githubusercontent.com/83487073/137578790-23d232ba-9cc5-49d4-a88c-79094a0b8239.png">

이런 식으로 묶인다. freq, relative_freq 둘 다 열벡터다. 

# rbind()

행벡터 끼리 묶어라는 명령이다. 

열벡터를 전치연산 해서 행벡터로 만들고, 두개를 묶는다. 

```R
rbind(열벡터1, 열벡터2)
```

<img width="587" alt="Screen Shot 2021-10-16 at 16 46 06" src="https://user-images.githubusercontent.com/83487073/137578897-cdee5ef0-91eb-417a-ad52-11de04e7b8b2.png">

---

# pie()

파이차트 그리는 명령이다. 

```R
pie(x, labels=값 별로 붙일 라벨, main='파이차트 이름')
```

<img width="459" alt="Screen Shot 2021-10-16 at 16 52 06" src="https://user-images.githubusercontent.com/83487073/137579191-9f4f5bbd-c790-4119-a3ad-26534bf6c343.png">

R 내장 pie chart 는 별로 예쁘진 않다. 

---

# cumsum()

벡터를 넣으면 각 요소의 누적합을 출력해준다. 

```R
x<- c(1,2,3,4,5)
cumsum(x)
```

1,3,6,10,15

이런 식 이다. 

---

# stem()

줄기-잎 그림 그리는 명령이다. 

```R
stem(벡터)
```

<img width="515" alt="Screen Shot 2021-10-16 at 17 06 19" src="https://user-images.githubusercontent.com/83487073/137579813-31a7f978-ea20-4113-9384-857cdabe4ed8.png">

이런 식의 그림이 그려진다. 

가장 왼쪽 숫자들은 10단위수다. 오른쪽 숫자들은 1단위수들이다. 

위 그림에서 같은 10단위수가 2개 존재하는 건, 길이가 너무 길어서 길이 줄이려고 이렇게 된거다.

- 장점 : 모든 자료 각각의 값을 세세하게 알 수 있다. (최소, 최대 등)
- 히스토그램은 구간에 몇 개가 있는지만 알 수 있다. 
- 또, 히스토그램 역할도 한다. 대강의 히스토그램 모양이다. 
- 이름의 유래 : 왼쪽에 있는 값들을 stem 이라고 부른다(가지, 줄기) / 오른쪽 숫자들을 leaf (잎) 이라고 부른다. 

---

# boxplot()

표본값들을 이용해서 box-whisker plot 을 그린다. 

*box-whisker plot 그리는 데 필요한 정보 

- 1사분위수(1Q)
- 2사분위수(2Q, = 중앙값)
- 3사분위수(3Q)
- IQR(표본 사분위수 범위, Inter Quartile Range): 3사분위수 - 1사분위수
- 1.5 $\times$ IQR
- 1.5IQR + 3사분위수 구간 내 최댓값 
- 1사분위수 - 1.5IQR 구간 내 최솟값 
- 그 외 아웃라이어 값들 (일반 박스 플롯은 아웃라이어 값 표시하지 않는다)

```R
boxplot(벡터, range=1.5)
# range=1.5가 디폴트 값이다. 최댓값 - 최솟값(표본범위) 구간 조정한다. 
# range=0 주면 일반 박스 플롯 나온다(최소-최대 구간에 아웃라이어도 모두 포함한다)
# range 값 *IQR에 따라 최댓값-최솟값 구간 형성한다. 
```

예를 들어 range가 0이면, 이런 식이다. 

```R
# range=1.5 일 때
test<- c(15, 17, 18, 18, 19, 20, 20, 22, 23, 25, 26, 99)
# 99는 아웃라이어
boxplot(test, range=1.5)
```
<img width="406" alt="Screen Shot 2021-10-16 at 19 49 31" src="https://user-images.githubusercontent.com/83487073/137584450-e349f956-42df-4eb8-a9fa-fe9df5128956.png">

```R
# range=0 일 때
boxplot(test, range=0)
```
<img width="406" alt="Screen Shot 2021-10-16 at 19 50 19" src="https://user-images.githubusercontent.com/83487073/137584466-47c79614-e020-482f-9540-f50757f734cd.png">

range=0 이 되니까 표본범위 안에 아웃라이어까지 포함하고 있다. 

---
















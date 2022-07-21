---
title : "[데이터수집/파이썬] 인스타그램 텍스트 데이터를 수집해보자"
excerpt : "파이썬으로 데이터 수집해보자"

categories : 
- python
- 데이터수집

tags : 
- [데이터수집, python, 인스타그램, selenium, 크롤링]

toc : true 
toc_sticky : true 

date : 2021-06-21
last_modified_at : 2021-06-21

---
이 글은 네이버 블로그 [https://blog.naver.com/tigerkey10]에 포스팅 했던 글을 옮겨 온 것 입니다. 

---
인스타그램 댓글을 수집하는 크롤러를 만들었다. 개발 환경은 맥 os 빅 서 11.2.2, 아나콘다가 설치된 주피터 노트북이다. 

그리고 인스타그램 등에서 크롤링 한 데이터를 임의로 이용하거나, 특히 상업적 목적으로 사용할 경우 해당 행위자가 법적 제제를 받을 수 있음을 

명백하게 밝혀둔다. 

---
# 1. 모듈 & 라이브러리 호출
```python
from selenium import webdriver
import selenium
import time
import os
import random
from bs4 import BeautifulSoup 
from selenium.webdriver.common.keys import Keys
import pandas as pd
```
브라우저를 조작하면서 웹페이지에 접근할 수 있는 셀레니움 라이브러리와 셀레니움으로 도달한 웹페이지 태그를 파이썬에서 읽어오도록 도와주는

뷰티풀 수프 라이브러리가 핵심이다. 판다스는 결과물로 데이터프레임을 만들기 위해 호출했다. 

인스타그램 크롤링을 위해서는 로그인을 해야 한다. 자동 로그인을 위한 알고리즘을 짰다. 자동 로그인 관련 코드는 지난번 포스팅

- 인스타그램 이미지 파일 수집하기- 를 참조하면 된다.  

# 
# 2. 해당 웹페이지 접속 후 데이터 수집하기 
인스타그램 페이지는 크롤링하기 매우 까다로웠다. 웹페이지가 동적 페이지여서, 스크롤을 내릴 때 마다 태그가 보였다가 안 보였다가 하면서

변하기 때문이다. 예컨대 특정 인스타 페이지에 도착했을 때 해당 페이지 게시물에 대한 태그가 10개가 보였다고 가정해보자. 

하지만 보통 웹 크롤링을 할때, 고작 10개 정보를 수집하고자 크롤러를 이용하진 않을 것이다. 즉, 우리는 더 많은 태그를 로드해올 필요가 있다. 

인스타그램과 같은 동적 페이지에서 태그를 더 많이 로드해 오기 위해서는 스크롤을 내려야 한다. 보통 다른 웹사이트의 경우에는

스크롤을 내리면 이미 로드된 태그는 유지되고, 더 많은 태그가 불러와진다. 이런 경우에는 크롤링하기 쉽다. 내가 원하는 태그 갯수가 나올 때 까지 

페이지 스크롤을 내리고, 태그 갯수를 세어서 내가 원하는 갯수만큼 나왔거나, 그보다 큰 경우 스크롤을 중단하고 BeautifulSoup로 페이지 파싱해서 

for문이나 while문을 이용해 돌면서 페이지 정보를 수집하면 된다. 그리고 한개 데이터 수집 할 때 마다 갯수를 체크해서, 내가 원하는 갯수에 

도달하면 반복문도 break로 종료해주면 된다. 

하지만 언급한 것처럼, 인스타그램 페이지는 스크롤을 내리면 기존에 로드된 태그들은 사라지게 된다. 즉, 스크롤을 내리면 내릴수록 맨 처음에

로드된 태그는 사라지고, 새로운 태그들이 그 자리를 채우게 된다. 따라서 [스크롤 내리기 --> 페이지 파싱 --> 태그 수집] 하던 예전 방식으로는

전혀 크롤링할 수 없었다. 새로운 방식의 시도가 필요했다. 

# 
# 3. 시행착오 
이전에 인스타그램 이미지 크롤러에서도 위와 비슷하게, 이미지 src 주소들이 페이지 스크롤할 수록 사라졌었다. 

그래서 페이지 스크롤 내린 후 수집된 src 주소들을 리스트에 계속 추가했다. 그리고 다시 스크롤을 내리고 src 주소 수집 후 다시 리스트에 추가하는

과정을 반복 수행했다. 그리고 매번 리스트에 추가할 때 마다 src들이 추가된 리스트를 set()집합 자료형으로 변환해주어 중복을 제거했다. 

딕셔너리의 한 일종으로, 중복을 허용하지 않는 set()집합 자료형이 중복 수집된 이미지 src 주소 제거에 도움이 될 것이라 생각되어 실행했고,

효과적이었다. 

그래서 이번에 인스타그램 댓글 크롤링을 할 때도, set()방식을 이용하고자 했다. 하지만, set방식은 여기 적합하지 못했다.

인스타그램 이미지 수집과 댓글 수집이 메커니즘이 전혀 달랐기 때문이다. 이미지 수집의 경우[ list --> set--> list --> for문으로 반복 ]

위 프로세스대로 하면 크롤링할 수 있었다. 하지만 댓글 수집의 경우 인스타그램 페이지 특성상 이미 크롤링한 태그들은 리스트에서 

제거 되어야 했다. set()방식을 이용하면 중복항목만 제거될 뿐, 이미 크롤링한 기존항목들이 사라지지 않고 남아있게 된다. 

이 리스트를 for문으로 반복했을 때, 항목들이 결과적으로 중복 크롤링되게 된다. 수행 시간도 많이 걸리고, 저장 공간도 조금 더 쓰게 된다. 

무엇보다 결과물을 받아보는 사람 입장에서는 '이게 뭐야, 중복된건 없앨 수 없나?' 할 것이다. 

크롤링 할 새로운 방식이 필요하다. 

그래서 나는 '중복을 없애고, 무엇보다 데이터를 한 번 수집한 태그는 리스트에서 빠지는, 그래서 매번 리스트에 새로운 태그만 들어가 있게 되는'

방식을 고안해냈다. 리스트가 선입선출 방식을 따른다는 점에서 '큐' 자료형과도 개념상 비슷하다고 생각한다. 

```python
img_list2 = []
pre_list = []
while True : 
    time.sleep(10)
    img_tag_list = driver.find_elements_by_class_name('_9AhH0')
    for b in img_tag_list : 
        img_list2.append(b)
    for unique in img_list2 : 
        if unique not in pre_list : 
            unique_list.append(unique)
    pre_list = img_tag_list
```

- 사진이 함께 있는 인스타그램 게시물 특성상, 인터넷 환경에 따라 사진 로드 시간이 다르게 걸렸다. 사진 로드 시간이 길게 걸리면

옆에 게시내용과 댓글 부분 태그들이 시간이 부족해서 제대로 로드되지 못하고 넘어가는 경우가 발생할 수 있을 것이다. 

결과적으로 셀레니움 웹드라이버가 태그를 못 찾아서 Error를 발생시킬 것이다.(Nonetype has no attribute 'find'이런 식이다. 페이지에서

내가 찾도록 시킨 태그를 못 찾아서 find함수를 사용할 수 없다는 것이다.)

따라서 time.sleep(10)으로 대기시간을 10초를 줘서 충분히 사진과 게시물 내용 태그가 로드될 수 있도록 코드를 짰다. 

그 후 driver.find_elements_by_classs_name()으로 각 게시물 elements의 포인트들을 찾아 리스트로 생성하도록 했다. 

이 포인트들은 웹드라이버가 나중에 돌아가면서 click()으로 클릭할 수 있는 지점이 된다. 

그 후 for반복문으로 수집한 포인트를 돌면서, img_list2라는 새 리스트에 '이번에 찾은 포인트'들을 추가해줬다. 

이 '이번에 찾은 포인트'는 [이전에 이미 수집한 포인트 + 새로 발견한 포인트]가 섞여있다. 

일단 이 둘 모두를 img_list2에 추가했다. 

그 다음이 중요한데, for unique in img_list2 : 부분이다. 이는 unique라는 변수가 img_list2를 돌면서, 

만약(if) unique변수가 not in pre_list (pre_list는 '이전에 이미 수집한 포인트들 리스트'다. 위에서 pre_list를 [] 빈 리스트로 정의해주고,

맨 첫번째 시행은 []으로 처리한 다음, for문이 끝나고 '이번 시행에 수집한 포인트들'을 pre_list에 담아줬었다.)

이라면, - 즉 '완전히 새로 등장한 포인트라면' - unique_list라는 새 리스트에 추가하도록 했다. 이렇게 함으로써 이미 수집한

태그 포인트들은 unique_list에서 빠져나가고(코드 말미에 unique_list=[]으로 정의해줘야 한다) 새롭게 수집한 태그 포인트들만

unique_list에 포함될 것이다. 그리고 처음에 [] 빈 리스트였던 pre_list에 '이번 시행에서 수집한 태그 포인트들'을 담아줬다. 

개인적으로 이 부분이 가장 어렵고 시간이 많이 걸렸다. 

```python
if unique_list == [] : 
        break
```
그리고 위 코드를 통해 '새로운 태그 포인트가 더 이상 수집되지 않은 경우 == 게시물이 더 없는 경우' 에는 break로 

전체 while문을 중단하도록 했다. 

# 
# 4. 크롤링 시작
위 unique_list 생성까지 무사히 마쳤다면, 나머지 과정은 한결 수월하다. 페이지 기본 규칙만 파악하면 된다. 

unique_list를 for문으로 돌면서 각각 태그를 [클릭 --> 창 뜬다 --> 게시물 텍스트 및 좋아요 갯수, 댓글 수집] 이 과정을 반복하면 된다. 

그리고 페이지는 페이지 태그 순서 상 위에서 아래로, 왼쪽에서 오른쪽으로 순서대로 접근하게 된다. 

```python
for c in unique_list : 
        try : 
            c.click() #태그 포인트 클릭해서 게시물 창을 연다
            time.sleep(10)
        except : 
            print('태그 못 찾아서 건너뛰었음')
            continue
        else : 
            html1 = driver.page_source
            soup1 = BeautifulSoup(html1, 'html.parser')
            
            try : 
                ti_cont = soup1.find('div', 'C4VMK').get_text().replace('\n', "").strip()
                f.write('*게시글 제목 : %s' %ti_cont+'\n')
                print('*게시글 제목 : %s' %ti_cont)
                contents1.append(ti_cont)
            except : 
                f.write('*게시글 제목 : 제공된 정보 없음'+'\n')
                print('*게시글 제목 : 제공된 정보 없음')
                contents1.append(None)
            
            try : 
                suggested_time = soup1.find('time', '_1o9PC Nzb55').get_text().replace('\n', "").strip()
                f.write('1.게시 시간 : %s' %suggested_time+'\n')
                print('1.게시 시간 : %s' %suggested_time)
                time1.append(suggested_time)
            except : 
                f.write('1.게시 시간 : 제공된 정보 없음'+'\n')
                print('1.게시 시간 : 제공된 정보 없음')
                time1.append(None)
            
            try : 
                likes = soup1.find('a', 'zV_Nj').find('span').get_text().replace('\n', "").strip()
                f.write('2.좋아요 갯수 : %s' %likes+'\n')
                print('2.좋아요 갯수 : %s' %likes)
                likes1.append(likes)
            except : 
                try : 
                    views = soup1.find('span', 'vcOH2').find('span').get_text().replace('\n', "").strip()
                    f.write('2.조회수 : %s' %views+'\n')
                    print('2.조회수 : %s' %views)
                    likes1.append(views)
                except : 
                    f.write('2.좋아요 갯수 및 조회수 : 제공된 정보 없음'+'\n')
                    print('2.좋아요 갯수 및 조회수 : 제공된 정보 없음')
                    likes1.append(None)
                    
            
            f.write('-'*100)
            f.write('\n')
            print()
    
```
게시물에 따라 '좋아요 갯수'가 표시된 게시물이 있고, '몇 명의 사람이 시청했는지'를 나타내주는 게시물이 있었다. 

좋아요 갯수를 우선으로 찾아 수집하되, 만약 좋아요 갯수가 없다면(try-except) 조회수로 찾아서 수집해라고 코드를 작성했다. 

그리고 그 마저도 없다면(try-except) 그때는 '제공된 정보 없음'으로 표시하고, 데이터프레임 만들 리스트 likes1에는 None, 그러니까

공값을 넣어라고 설정했다. 공값을 넣는 이유는 데이터프레임이 각 칼럼 별 행 갯수가 모두 맞아야 생성되기 때문이다. 

pd.Series()로 행 갯수를 맞춰준 경우에는 엑셀 결과 파일에 표기되지 않는 데이터들이 있었기 때문에, 최대한 사용하지 않고자 했다. 

```python
try  :
                ul_list = soup1.find_all('ul',  'Mr508')
            except : 
                driver.find_element_by_xpath('/html/body/div[5]/div[3]/button').click()
                continue
            else : 
                for d in ul_list : 
                    user_id = d.find('h3', '_6lAjh ').find('a').get_text().replace('\n', "").strip()
                    f.write('1.댓글 게시자 id : %s' %user_id+'\n')
                    print('1.댓글 게시자 id : %s' %user_id)
                    dd_id1.append(user_id)
            
                    d_time = d.find('a', 'gU-I7').find('time').get_text().replace('\n', "").strip()
                    f.write('2.댓글 게시 시간 : %s' %d_time+'\n')
                    print('2.댓글 게시 시간 : %s' %d_time)
                    dd_time1.append(d_time)
            
                    d_cont = d.find('div','C4VMK').find_all('span')[1].get_text().replace('\n', "").strip()
                    f.write('3.댓글 내용 : %s' %d_cont+'\n')
                    print('3.댓글 내용 : %s' %d_cont)
                    dd_cont1.append(d_cont)
                    
                    f.write('-'*100)
                    print()
```
그리고 위 코드로 게시물에 달린 댓글 정보를 수집했다. 맨 첫줄 ul, mr508을 찾아 리스트로 만드는 코드의 경우, try-except로 

'만약 댓글이 없으면 게시물 창에 X를 눌러 창을 닫은 뒤(인스타그램 PC버전은 게시물 닫는 X버튼이 있다) continue로

아래 코드 실행하지 말고 다음 순서로 건너가라'고 지정해줬다. 만약 댓글이 있다면? try 결과를 else에서 받아서 for문 반복하면서 

관련 댓글 정보들을 수집할 것이다. 

```python
no += 1
            f.write('-'*100)
            print()
    
            if no > cnt : 
                break
            driver.find_element_by_xpath('/html/body/div[5]/div[3]/button').click()
            time.sleep(3)

    
    if no > cnt : 
        break
    unique_list = []
    for a in range(3) : 
        driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
        time.sleep(3)
```

이렇게 한 개 게시물의 게시물 관련 정보 및 댓글 정보를 모두 수집했다면, 한 개 게시물에서 데이터 수집을 완료한 것이다. 

no값에 += 1을 통해 no를 증가시켜주고(1개 찾았다는 의미) 만약 찾은 갯수가 내가 찾도록 시킨 갯수(cnt)보다 많아질 경우

break로 전체 반복을 멈추도록 했다. 그래야 '내가 수집하도록 시킨 갯수'만큼 정확하게 수집하고, 크롤러가 종료될 것이다. 

만약 그게 아니라면, 데이터 수집을 마친 뒤 다시 게시물 X버튼을 클릭해서 인스타 메인 페이지로 나오게 했다. 

이렇게 페이지 전환 해줘야 다음 인스타 게시물 데이터도 수집할 수 있다. 

코드 말미에는 위에서 언급한 것과 같이 unique_list를 다시 빈 리스트[] 로 만들어줬다. 

그리고 만약 ' no > cnt'로 종료되지 않은 경우, 즉 다시 말해 '아직 내가 시킨 갯수만큼 다 못 찾은 경우'는 

driver.find_element_by_tag_name('body').send_keys로 페이지 스크롤 다운을 시켜서 새 태그 포인트를 가져오도록 했다.

만약 새 태그 포인트를 찾아서 가져온다면, 새 포인트들이 unique_list에 들어가서 [클릭--> 창 뜬다 --> 내용 수집] 과정을

계속 반복할 것이고, 

만약 새 태그 포인트를 못 찾거나 혹은 새 태그 포인트가 없는 경우(기존 태그 포인트들만 찾아지는 경우) 는 

'더 이상 게시물이 없다'는 뜻이므로 종료시켜야 한다. 이 경우가 unique_list가 [] 빈 리스트 인 경우로, 

위에서 unique_list = []이면 break(종료) 해라~ 고 지정해 준 이유다. 

# 
# 5. 결과
<img width="884" alt="Screen Shot 2021-06-21 at 16 23 38" src="https://user-images.githubusercontent.com/83487073/122722733-0a5d6480-d2ad-11eb-98c8-69abc9368a27.png">

<img width="880" alt="Screen Shot 2021-06-21 at 16 24 05" src="https://user-images.githubusercontent.com/83487073/122722796-1ba67100-d2ad-11eb-8a2d-5fc9ec33b94b.png">

결과적으로 위와 같이 txt, xlsx, csv형식으로 데이터를 저장할 수 있었다. txt형식 데이터를 이용하면 okt 함수 등을 활용해

자연어 처리 후 키워드 수집 등의 분석을 할 수 있을 것이고, 수치로 된 데이터 일 경우 csv 형식 데이터를 이용하면 pandas, matplotlib라이브러리 등을 써서 다양한 시각화, 기초통계량 분석, 추세 분석, 상관관계 분석, 인과관계 분석 등을 진행할 수 있을 것이다. 


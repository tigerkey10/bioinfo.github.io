---
title : "[데이터수집/파이썬] 인스타그램 사진 이미지를 수집해보자"
excerpt : "파이썬으로 데이터 수집해보자"

categories : 
- python
- selenium
- BeautifulSoup
- 데이터수집

tags : 
- [데이터수집, python]

toc : true 
toc_sticky : true 
use_math : true

date : 2021-06-21
last_modified_at : 2021-06-21

---
이 글은 제가 네이버 [https://blog.naver.com/tigerkey10/222260053529]에 포스팅 했던 글을 옮겨 기록한 것입니다. 

---
인스타그램 사진 수집 크롤러를 만들어봤다. 이번 크롤러는 인스타 내용이나 댓글 정보 수집이 아닌, 이미지만 수집하는 크롤러다. 

코드 작성을 위한 환경은 아나콘다가 설치된 주피터 노트북, Mac OS Big Sur이다. 

---

# 모듈 호출

```python
from selenium import webdriver
import selenium
from bs4 import BeautifulSoup
import pandas as pd
import os
from selenium.webdriver.common.keys import Keys
import urllib.request
import urllib
import time
```

웹페이지 접속, 페이지 조작을 위해 셀레니움 라이브러리에서 웹드라이버를 호출하고, 셀레니움도 호출했다. 

한편 웹페이지 태그 중 원하는 태그만 뽑아와서 데이터를 추출하기 위해 뷰티풀 수프 라이브러리도 함께 호출했다. 

파일 디렉토리를 생성해주고, 저장 경로 지정을 위해 os함수를 호출했다. 

인스타그램 웹페이지는 사진을 가져오려면 페이지 스크롤을 내려야 다음 사진 이미지를 볼 수 있다. 

따라서 자동 스크롤 다운을 위한 Keys도 불러왔다. 

urllib는 인스타에서 추출한 이미지 소스 주소에서 파일을 다운받게 해준다. 

time함수도 호출해, 크롤링 시작 및 종료 시간을 측정하고, 전체 소요시간을 계산할 것이다. 

# 인스타그램 자동 로그인

웹드라이버로 인스타그램에 접속했을 때, 로그인을 하지 않으면 사진을 볼 수가 없다. 따라서 드라이버가 자동 로그인을 하도록 코드를 작성해 주어야 한다. 

```python
def login(id_, password) : 
    driver.find_element_by_xpath('/html/body/div[1]/section/nav/div[2]/div/div/div[3]/div/span/a[1]/button').click()
    time.sleep(5) #로그인 버튼 클릭 후 페이지 전환 5초 대기
    
    driver.find_element_by_name('username').send_keys(id_) #아이디 칸에 미리 입력해준 id 입력하기 
    driver.find_element_by_name('password').send_keys(password) #비밀번호 칸에 미리 입력해준 password 입력하기 
    time.sleep(3) #페이지 전환 3초 대기 
    
    driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div/div[1]/div/form/div/div[3]/button').click()
    time.sleep(5) #로그인 버튼 클릭 후 5초 대기 
    
    driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div/div/section/div/button').click()
    time.sleep(5) #아이디 저장 버튼 클릭 후 5초 대기 
    
login(id_, password) 
```
자동 로그인을 위해 위 코드처럼 login이라는 함수를 정의했다. 

# 인스타그램 접속 후 
인스타그램 접속 후에는 페이지를 반복해서 내리고, 수집할 이미지 갯수 대로 이미지 태그를 가져와야 한다. 
```python
while True : 
    time.sleep(5)
    for a in range(3) : 
        driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
        time.sleep(3)
```
while문 아래에서 for문으로 3번 반복해 가면서 페이지를 내린다. 

그 후 페이지를 파싱하고, img 태그를 모두 가져온다. 그리고 내가 원하는 갯수만큼, 새로운 리스트를 만들어 추가하면 된다. 
그 후, for문으로 새 리스트를 돌면서 이미지 태그 속에 이미지 소스 주소를 찾아야 한다. 

```python
for img in div_list2 : 
    img_src = img['src']
    img_src_list.append(img_src)
    no += 1
    if no > cnt : 
        break
```
각 이미지 태그 내에서 ['src'] 주소만 다 찾아서, img_src_list에 추가했다. 추가하는 갯수는 내가 위에서 미리 입력해준 cnt개 만큼만 추가한다. 
```python
def download(url, filename) : 
    urllib.request.urlretrieve(url, filename)
```
그 후 download() 함수를 만들어서 이미지 다운을 위한 장치를 만들었다. 
```python
img_no = 1
for down in range(0, len(img_src_list)) : 
    try : 
        url = img_src_list[down]
        filename = str(img_no)+'.jpg'
        download(url, filename)
        img_no += 1
    except : 
        continue
```
for문으로 img_src_list를 돌면서 지정해준 파일 경로에 download함수를 이용해 이미지 파일을 순서대로 내려 받으면 된다. 




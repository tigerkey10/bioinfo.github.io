---
title : "[마케팅애널리틱스 프로젝트] Imagenet 데이터로 사전학습된 CNN 모델 사용한 데이터 분석 프로젝트"
excerpt : "사전학습된 CNN 모델 분류기 학습 및 모델 일부 층 미세조정 통한 이미지 분류 프로젝트 - 논리, 가설, 시행착오"

categories : 
- Project
- Data Science
- Python
- Deep Learning
- CNN

tags : 
- [Project, Data Science, Python, Deep Learning, CNN]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-04-24
last_modified_at : 2022-04-24

---

2022년 1월 ~ 2월 사이에 부산대학교 경영대학 송태호 교수님과 함께 마케팅애널리틱스 프로젝트를 진행했다. 이 프로젝트는 지금도 현재진행형이다. 

내가 설정한 프로젝트 목적은 Imagenet 데이터로 사전학습된, CNN 딥러닝 모델(vgg16, xception, inception_v3) 을 사용해서 브랜드 별 이미지 분류를 해 내는 것 이었다. 

여러 마케팅애널리틱스 논문을 찾아본 결과, 각 브랜드 별 인스타그램 및 Pinterest 이미지 수집해서, 그 이미지를 통해 각 브랜드가 주는 '느낌'을 분류하고자 한 시도들이 있었다. 

나는 논문들로부터 아이디어를 얻어 '딥러닝 모델을 사용해 이미지들 시각적 특징을 추출하고, 이를 기반으로 각 브랜드 이미지를 그 특징 별로 분류할 수 있다'는 가설을 세웠고, 이를 기반으로 삼성과 애플 관련 이미지들을 CNN 모델로 분류하고자 시도했다. 

아래 내용들은 그 과정과 경과를 기록한 것이다. 

내용들은 모두 내가 논리와 가설을 세우고 시행착오 겪은 과정만 기록되어 있으며, 이 내용에 대해 송태호 교수님께서 조언 해 주셨던 내용은 교수님의 지적재산권에 해당하므로, 모두 제외했다. 

해당 프로젝트를 진행하며 작성한 코드들은 깃허브 디파짓토리에 커밋 되어 있다. 

https://github.com/tigerkey10/tigerkey10.github.io

---

# 최초 계획, 가설, 논리, 그리고 시행착오

<img width="1108" alt="Screen Shot 2022-04-24 at 17 30 27" src="https://user-images.githubusercontent.com/83487073/164967591-52b8df2a-d9a8-47c8-b96b-2f5c46f82888.png">

<img width="1111" alt="Screen Shot 2022-04-24 at 17 31 54" src="https://user-images.githubusercontent.com/83487073/164967661-f912b9a9-6a5c-4c26-be3d-29c17cebea8e.png">

<img width="1110" alt="Screen Shot 2022-04-24 at 17 32 15" src="https://user-images.githubusercontent.com/83487073/164967675-6995db74-1104-4747-97c2-bbf019744499.png">

<img width="1111" alt="Screen Shot 2022-04-24 at 17 32 35" src="https://user-images.githubusercontent.com/83487073/164967689-70c67553-2fd3-4974-b3d6-0d670eb042ca.png">

<img width="1114" alt="Screen Shot 2022-04-24 at 17 33 02" src="https://user-images.githubusercontent.com/83487073/164967705-298f67d3-69ed-498c-9e5f-d88f1057b141.png">

<img width="1112" alt="Screen Shot 2022-04-24 at 17 36 41" src="https://user-images.githubusercontent.com/83487073/164967865-41ecd9dd-8403-4c87-a891-aa1f509e2567.png">

<img width="1113" alt="Screen Shot 2022-04-24 at 17 37 14" src="https://user-images.githubusercontent.com/83487073/164967890-afec6c90-0fc7-446a-8db5-f79f7d550f30.png">

<img width="1110" alt="Screen Shot 2022-04-24 at 17 37 53" src="https://user-images.githubusercontent.com/83487073/164967909-ef080974-2079-4259-ba0b-8be668d0f3e5.png">

<img width="1108" alt="Screen Shot 2022-04-24 at 17 38 15" src="https://user-images.githubusercontent.com/83487073/164967922-4e181482-b9a1-47dc-a62d-0bf14b8ee47d.png">

<img width="1115" alt="Screen Shot 2022-04-24 at 17 38 37" src="https://user-images.githubusercontent.com/83487073/164967945-2a5ef9a1-e6fe-4ecb-9d0f-bef9f33151fc.png">

<img width="1108" alt="Screen Shot 2022-04-24 at 17 39 14" src="https://user-images.githubusercontent.com/83487073/164967969-cee3e282-c400-4da9-98a3-eb28f38b07bd.png">

<img width="1112" alt="Screen Shot 2022-04-24 at 17 39 39" src="https://user-images.githubusercontent.com/83487073/164967987-c9003c7c-1774-4180-8dee-e92fe6615c7c.png">

<img width="1112" alt="Screen Shot 2022-04-24 at 17 40 05" src="https://user-images.githubusercontent.com/83487073/164968004-70e8a850-6ae7-401d-ba27-1215c2555e68.png">

<img width="1105" alt="Screen Shot 2022-04-24 at 17 40 32" src="https://user-images.githubusercontent.com/83487073/164968023-330a7d3f-6b2e-42d9-977b-d7c4ec157776.png">

<img width="1113" alt="Screen Shot 2022-04-24 at 17 40 56" src="https://user-images.githubusercontent.com/83487073/164968037-13a14011-561f-4de4-a224-883d8e04d9ce.png">

---

# 모델 학습 실패 후 수정된 계획, 아이디어, 시행착오

<img width="1154" alt="Screen Shot 2022-04-24 at 17 43 45" src="https://user-images.githubusercontent.com/83487073/164968146-389bb985-fae3-46fd-aaba-61b14ab35054.png">

<img width="1155" alt="Screen Shot 2022-04-24 at 17 44 11" src="https://user-images.githubusercontent.com/83487073/164968161-138ec058-4338-4cc7-acc3-074feed9464e.png">

<img width="1146" alt="Screen Shot 2022-04-24 at 17 44 34" src="https://user-images.githubusercontent.com/83487073/164968177-7cf19ff5-07f4-4926-bdc7-134dbddb3918.png">

<img width="1157" alt="Screen Shot 2022-04-24 at 17 45 03" src="https://user-images.githubusercontent.com/83487073/164968201-43f97a16-8c82-4df8-89fe-a94706af6b66.png">

<img width="1154" alt="Screen Shot 2022-04-24 at 17 45 31" src="https://user-images.githubusercontent.com/83487073/164968217-e7fff10b-7003-458c-80de-413676b24d36.png">

<img width="1151" alt="Screen Shot 2022-04-24 at 17 46 04" src="https://user-images.githubusercontent.com/83487073/164968234-babde8bf-c6ac-4c9f-8af8-f14a680b74fd.png">

<img width="1151" alt="Screen Shot 2022-04-24 at 17 46 31" src="https://user-images.githubusercontent.com/83487073/164968251-d67d0821-68ad-49a7-b1eb-48c55c82f070.png">

<img width="1115" alt="Screen Shot 2022-04-24 at 17 47 30" src="https://user-images.githubusercontent.com/83487073/164968289-f62aa63c-6d81-464c-92b5-02051baa401a.png">

<img width="1116" alt="Screen Shot 2022-04-24 at 17 47 51" src="https://user-images.githubusercontent.com/83487073/164968303-2d3b35a6-9045-416d-b456-f1c70954e6a9.png">

<img width="1114" alt="Screen Shot 2022-04-24 at 17 48 19" src="https://user-images.githubusercontent.com/83487073/164968325-07c8b29b-1c5c-4bb6-8a4e-04883d8e626f.png">

<img width="1112" alt="Screen Shot 2022-04-24 at 17 48 46" src="https://user-images.githubusercontent.com/83487073/164968348-2f596b79-be80-45fe-88db-bbb4006f72fe.png">

<img width="1115" alt="Screen Shot 2022-04-24 at 17 49 12" src="https://user-images.githubusercontent.com/83487073/164968368-c552f04e-bd5b-406b-a000-ec3a79d4239d.png">

<img width="1115" alt="Screen Shot 2022-04-24 at 17 49 38" src="https://user-images.githubusercontent.com/83487073/164968380-88630ef8-5c1e-4189-9ab9-54e09c5c82f9.png">

<img width="1114" alt="Screen Shot 2022-04-24 at 17 51 59" src="https://user-images.githubusercontent.com/83487073/164968459-8bf6c158-767b-4bfa-8817-a6a1cf02617b.png">









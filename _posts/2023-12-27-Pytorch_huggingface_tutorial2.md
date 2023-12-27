---
layout: post
title: Pytorch_huggingface tutorial 실습 (2)
date: 2023-12-27 10:00 +0900
last_modified_at: 2023-12-27 10:00:00 +0900
tags: [deeplearning, Pytorch, huggingface]
toc:  true
---

*이전 포스팅*<br>
<a href = 'https://streetlamplee.github.io/2023/12/26/Pytorch_huggingface_tutorial/'>Pytorch_huggingface tutorial 실습 (1)</a>

전에 했던 코드는 RAM이 부족한건지 자꾸 블루스크린이 떴다.

컴퓨터 교체를 하고 싶지만 할 수 없는 슬픈 현실을 묻어두고, 일단 Local에서 돌려도 돌아가게끔 코드를 수정해보자.

![Alt text](\..\img\DL4-12.png)

일단 가상환경을 conda로 바꾸었다.

poetry에서 conda로 바꾼건 다른 의미는 없고, GPU 환경 설정이 가능하기 때문이다.

GPU관련 환경설정에 관한 것은 <a href ='https://streetlamplee.github.io/2023/12/26/Pytorch_env_config/'>여기</a>를 확인

이 base 환경을 clone해서, deeplearn이라는 환경설정을 다시 만들어 줬다.

이후 여기에 필요한 library를 설치해줬다.

![Alt text](\..\img\DL4-13.png)

그리고 희망을 가지고 돌려보니..

![Alt text](\..\img\DL4-14.png)

GPU인식은 잘 된 것 같았으나, Out of Memory 에러를 뱉었다.

그렇다고 지금 에러가 나는게 하드웨어의 스펙이 부족해서는 아니고, 설정 상 limit가 걸려있는 느낌이다.

이 부분을 해결해보자
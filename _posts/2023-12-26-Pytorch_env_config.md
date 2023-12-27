---
layout: post
title: Pytorch_환경설정
date: 2023-12-26 11:00 +0900
last_modified_at: 2023-12-26 11:00:00 +0900
tags: [deeplearning, Pytorch]
toc:  true
---

# Pytorch 환경설정

## Anaconda 설치

아나콘다 검색 후, OS에 맞게 다운로드 하면된다.

window는 그냥 설치 파일 다운로드 하고, 실행시켜주자

---

Linux의 경우에는, 해당 다운로드 링크를 복사한 후,<br>
`wget (다운로드 링크)`를 통해 웹 파일 확인 후,<br>
`bash (파일 이름)`을 통해 설치하자

## Pytorch 설치

### NVIDIA GPU가 없는 경우

anaconda prompt에서,

    conda create --name pytorch_test --clone base

를 이용해서 새로운 가상환경을 만들어준다.

    conda activate pytorch_test

    conda install pytorch==2.0.0 torchvision==0.15.0 cpuonly -c pytorch

이후, 가상환경을 실행하고, pytorch 설치

### NVIDIA GPU가 있는 경우

conda prompt에서, 엔디비아 gpu가 있는 경우, 이를 확인하는 명령어가 있음

    nvidia-smi

현재 컴퓨터의 gpu를 확인하고자하면 아래의 명령어를 입력해보자

    nvidia-smi -L

GPU가 있는 경우에는, CUDA 버전을 고려해서 pytorch를 설치해야한다.

> CUDA : GPU에서 사용하는 수천 개의 코어를 활용하여 병렬로 코드를 실행하게 해주는 기술

> CUDA 버전찾기
>
><a href='https://en.wikipedia.org/wiki/CUDA>CUDA Wiki'</a>에서 `ctrl + F`를 이용해 GPU이름을 검색해보자
>
> ex, NVIDIA GFORCE GTX 1060의 Compute Capability는 6.1 버전인 것을 볼 수 있다.
>
> ![Alt text](\..\img\DL4-7.png)
>
>해당 페이지에서 바로 위의 표를 보면, 이렇게 정리되어있다.<br>CUDA SDK Version은 8.0 부터, 6.x를 지원하는 것을 볼 수 있다.

<a href='https://pytorch.org/get-started/previous-versions/'>파이토치 버전별 설치 페이지</a>

위의 홈페이지로 들어가서, 본인에게 맞는 CUDA SDK Version을 선택해서 설치하자<br>
나는 11.8 버전을 설치했다.

    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

![Alt text](\..\img\DL_4-8.png)

실치 후, python을 실행해서 잘 설치되었는지 확인해보자

    python

    import torch

    print(torch.cuda.is_available())

`True`가 나온다면 잘 되었다는 뜻

![Alt text](\..\img\DL_4-9.png)


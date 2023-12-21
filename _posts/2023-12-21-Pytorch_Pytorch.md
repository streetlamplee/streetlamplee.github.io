---
layout: post
title: Pytorch_Pytorch
date: 2023-12-21 18:16 +0900
last_modified_at: 2023-12-21 18:16:00 +0900
tags: [DL, Pytorch]
toc:  true
---

# Pytorch

## 1. DL Framework

### 1.1 왜 딥러닝 프레임워크를 사용할까?

코드로 구현하면,

1. layer를 직접 구현
2. loss function 구현
3. 모든 layer의 weight, bias에 대해 gradient를 계산
4. 최적화 알고리즘을 구현해야함

위의 과정을 간단하게 만들어줄 수 있다.

### 1.2 취업 시장에서의 딥러닝 프레임워크

많은 기업에서 딥러닝 프레임워크를 활용하고, 지원자격으로 능숙한 활용 능력을 요구하는 경우가 많다.

## 2. DL Framework trend

### 2.1 다양한 딥러닝 프레임워크

많은 프레임 워크가 오픈소스로 공개되어있음

1. tensorflow
2. pytorch
3. JAX
4. MXNet

### 2.2 딥러닝 프레임워크 트렌드

SOTA 모델은 대부분 Pytorch로 구현이 되어있다.

### 2.3 Why Pytorch

#### community

1. NLP
<br>허깅페이스가 가장 대중적인 NLP 커뮤니티
<br>여기에 공개된 모델들의 대부분이 Pytorch

2. Computer Vision
<br>timm, segmentation_models_pytorch 라이브러리 등등
<br>Vision에서도 Pytorch가 더 나음

3. LLM
<br>최근 LLM 모델들 또한 pytorch (GPT, LLaMa)


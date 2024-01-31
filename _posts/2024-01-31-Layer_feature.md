---
title: CNN Layer별 특징 탐구
layout: post
date: 2024-01-31 11:19 +0900
last_modified_at: 2024-01-31 11:19:00 +0900
tag: [DL, CV]
toc: true
---

## 1. CNN Layer별 특징 탐구

### 1.1 CNN 모델의 구성

- Convolution Block<br>
Convolution Layer + Batch Normalization Layer + Activation Layer
- Pooling Layer

### 1.2 Convolution Layer

network가 vision task를 수행하는 데에 유용한 feature를 학습할 수 있도록 함

conv layer를 여러 개 쌓을 경우, 뒤 layer의 결과값 하나를 만드는데에 사용되는 이미지의 범위가 넓어진다.<br>
**뒷 레이어로 갈수록 Receptive field가 커진다!**

conv layer의 초반 layer : edge와 같은 low-level feature 학습<br>
conv layer의 후반 layer : shape와 같은 high-level feature를 주로 학습


### 1.3 Batch Normalization

deep network가 잘 학습될 수 있도록 함



### 1.4 Activation Layer

모델에 비선형성을 부여해주기 위해서 사용됨

선형 함수 layer들로만 구성해도, 많이 쌓아도 선형함수 하나로 표현되는 모델이 되므로 activation func이 필요하다.

- Sigmoid<br>
[0, 1] 사이의 값으로 변경해준다.<br>
$\sigma(x) = \frac{1}{1+e^{-x}}$<br>
$\sigma(x)$의 값이 0이나 1에 매우 가까운 경우, 미분 값이 0에 가까워져 학습이 어려울 수 있음<br>
sigmoid 함수의 결과값은 항상 양수값이라서 0에 중심을 두지 않음
- tanh<br>
여전히 미분값이 0에 가까운 부분이 존재함<br>
0에 중심을 둠
- ReLU<br>
입력값이 0 이상일 경우, gradient가 0이 되지 않음<br>
computational cost가 매우 적음<br>
sigmoid나 tanh함수보다 매우 빠르게 수렴함<br>
0-centroid가 아니다<br>
음수값을 가질 경우, 절대 업데이트가 되지 않는다.
- Leaky ReLU<br>
ReLU에서 입력값이 음수일때, 0으로 처리하지 않고 매우 작은 수준으로 scaling<br>
입력이 음수일때에는 업데이트가 되지 않는 ReLU의 단점을 보완

### 1.5 Pooling Layer

feature map에 spatial aggregation을 시켜줌

이를 통해 파라미터 수를 줄이고, 더 넓은 receptive field를 가지게 해준다.

- Max Pooling<br>
정보의 손실이 일어날 수 있음<br>
데이터에 따라, 정보의 손실이 거의 없이(중요한 정보를 남기면서) feature map을 줄일 수 있음
- Average Pooling<br>
어느 정도의 정보를 유지할 수 있음<br>
그러나 데이터에 따라, 중요한 정보가 희석될 수 있음
---
title: CV vs. 고전 컴퓨터 비전
layout: post
date: 2024-01-31 10:00 +0900
last_modified_at: 2024-01-31 10:00:00 +0900
tag: [DL, CV]
toc: true
---

## 1. CNN vs. 고전 컴퓨터 비전

### 1.1 고전 컴퓨터 비전

#### 고전 컴퓨터 비전에서의 filter

**Sobel Filter** : 정해진 Sobel filter를 통해 x방향과 y방향으로 변화율을 계산해서 edge를 검출 (학습 불가능)

### 1.2 현대의 컴퓨터 비전

#### 학습가능한 filter의 등장

CNN(Convolutional Neural Network)

Convolution Filter : Conv 연산을 통해 산출한 결과를 정답지(Ground Truth)와 비교하여 오차를 줄여나가는 방식으로 계속 업데이트 되는 학습 가능한 필터를 많이 사용

#### 학습가능한 filter의 장점

고전 CV만으로는 성능이 좋지 않거나 해결이 불가능했던 task를 가능

ex. noise에 대해서도 강건하게 edge를 검출할 수 있음

정답지만 있다면, filter 그 자체를 학습하게 되면서 task가 간단해짐

### 1.3 학습 가능한 파라미터란?

#### 학습가능한 parameter를 가진 layer 예시

* Convolution Layer<br>
$Parameter = (F_m \times F_m \times C_m + 1) \times C_{out}$
* Batch Normalization Layer<br>
$Parameters = 2 \times C_m \dots (\gamma, \beta)$
* Fully Connected Convolution Layer<br>
$Parameters = (N_m \times N_m \times C_m + 1) \times M$

#### 학습이 불가능한 layer 예시

* Activation Layer<br>
    1. Sigmoid
    2. tanh
    3. ReLU
    4. Leaky ReLU
    5. Maxout
    6. ELU
* Pooling Layer<br>
    1. Max-Pooling
    2. Average Pooling

---
title: backbone 이해하기 _ AlexNet, VGGNet
layout: post
date: 2024-01-30 17:00 +0900
last_modified_at: 2024-01-30 17:00:00 +0900
tag: [DL, CV, CNN, AlexNet, VGGNet]
toc: true
---

# Backbone 이해하기 : CNN

## 1. AlexNet

### 1.1 딥러닝 모델의 등장

CNN 모델을 기반으로한 모델이 각광을 받으면서, 사람보다 좋은 성능을 가지기 시작

### 1.2 AlexNet의 구조

**Convolution - Pooling - Batch Normalization** 구조


### 1.3 Convolution Layer

![Alt text](\..\img\CV3.png)

### 1.4 Active Function

ReLU를 사용하는 것이 tanh를 사용하는 것보다 훨씬 빠르게 성능을 올릴 수 있었다.

### 1.5 Pooling Layer

Pooling 중에는, stride를 2로 두면서 한 pixel 씩 overlapping 하는 것이 더 성능이 좋다고 기술

### 1.6 Local Response Normalization

너무 강하게 활성화된 뉴런이 있을 경우, 주변 뉴런에 대해서 normalization을 진행한다.<br>
Normalize 중에는, 너무 강하게 활성화된 뉴런의 값을 조금 낮추면서 특정 뉴런만 활성화 되는 것을 막음

이후에는 Batch Normalization을 주로 사용한다.

### 1.7 Overfitting 방지

**Overfitting** : 학습 데이터에 너무 과적합되어 학습된 경우를 말함<br>
학습 데이터에 대한 성능이 좋지만, 학습데이터에 없는 데이터에 대해서는 매우 낮은 성능을 보이게 된다.<br>
즉, <mark>일반화</mark>되어 있지 않다고 표현한다.

- 판단 방법<br>
training error와 test error를 비교하여,<br>
training error가 줄어듦에도 불구하고, test error가 증가하는 포인트에서 overfitting 시작

---

#### Data Augmentation

학습 데이터에 변형을 가해서 좀 더 다양성을 지닌 데이터로 학습될 수 있도록 하는 방법

ex. flip, brightness, contrast, 등등

#### Dropout

뉴런 중 일부를 일정 비율로 생략하면서 학습을 진행하는 방법

다만, 학습 시에만 Dropout을 하고, 테스트를 할 때는 모든 뉴런을 사용해야한다.

## 2. VGGNet

### 2.1 Small Filters, Deeper Networks

AlexNet에 더 많은 Layer를 쌓아 성능을 개선한 모델

![Alt text](\..\img\CV4.png)

모든 Conv filter가 3x3을 연속적으로<br>
이 layer는 3x3 conv 3개가 7x7 conv 하나와 비슷한 효과를 가질 수 있다.<br>
(= effective receptive field)<br>
또한 더 적은 파라미터로 구현이 가능하다. ($3^3 < 7^2$)<br>
더 많은 layer를 쌓으면 비선형성을 높일 수 있다.
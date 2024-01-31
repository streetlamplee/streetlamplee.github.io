---
title: Image Classification
layout: post
date: 2024-01-31 16:00 +0900
last_modified_at: 2024-01-31 16:00:00 +0900
tag: [DL, CV, Classification]
toc: true
---

## 1. Image Classification

### 1.1 Image Classificatoin이란?

CV 분야에서 대중적인 task

목표 : 이미지를 input으로 받아, 주어진 class를 잘 예측하는 문제

구조 : Backbone(CNN) + Classification head(FC layers)

---

#### Logits & Softmax

Logits : 각 클래스에 대한 예측을 수치(실수값)로 나타내는 중간 단계

Softmax : 실수 전체 범위를 가지는 logits을 지수 함수를 이용해서 클래스 간의 상대적 확률 계산

---

Rule-based는 인간의 관점을 하나하나 제시하는 것은 너무 비용이 크다.

따라서 방법론 자체가 Data-driven으로 바뀌었다.

### 1.2 Image Classification Dataset

일반적으로 여러개의 이미지와, 이를 표현하는 class 쌍으로 구성되어있다.



### 1.3 Training Process

#### Preprocessing (전처리)

agumentation을 한다.

대표적으로 grayscale, rotation, flip, resize 등등이 있다.

#### modeling

학습 안정성을 증대 시키는 방법 : batch normalization, dropout

softmax를 이용해서 각 class의 확률값으로 변환한다.

#### Loss

loss func : 실제 class와 예측한 class의 차이를 줄이기 위해 사용

보통 Cross-entrophy loss를 이용해서 loss func을 정의한다.

이후, 역전파를 통해서 모델의 weight를 업데이트한다.

### 1.4 Test Process

#### Preprocessing (전처리)

학습에서 사용한 전처리 기법을 그대로 test data에 적용한다.

#### Model

학습 때 구했던 batch normalization의 평균 분산을 그대로 사용해서 정규화를 한다.

또한 dropout을 사용하지 않는다.

#### Prediction

가장 높은 확률을 가지는 class를 예측의 결과값으로 사용

### 1.5 Metric

정량 평가로 Acc와 Precision을 사용한다.

#### Accuracy

$\displaystyle \frac{TP+TN}{TP+TN+FP+FN}$

#### Precision

$\displaystyle \frac{TP}{TP+FP}$
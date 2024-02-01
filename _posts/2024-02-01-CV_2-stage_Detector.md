---
title: CNN 2-Stage Detector
layout: post
date: 2024-02-01 10:32 +0900
last_modified_at: 2024-02-01 10:32:00 +0900
tag: [DL, CV, object detection, 2-stage detector]
toc: true
---

## 1. R-CNN

### 1.1 R-CNN이란?

2-stage Detector 중 하나로서, R-CNN의 발전 = 2-stage Detector의 발전이다.

구조<br>
1. 객체가 있을법한 구역을 Region Proposals로 정함<br>
Selective Search 기법을 사용 (Rule-based)<br>
인접한 영역끼리 유사성을 축정해 큰 영역으로 차례대로 돌림
2. 이를 정해진 size로 resize (Warped Region)
3. backbone을 통과시켜 SVM classifier와 Bbox Regressor로 보내 결과 도출

---

#### 한계

1. CPU 기반의 Selective Search 기법으로 인해 많은 시간 필요
2. 2000개의 RoI로 인해서, 2000번의 CNN 연산이 필요해서 많은 시간이 필요하다.

## 2. Fast R-CNN

### 2.1 Fast R-CNN이란?

R-CNN에 비해 속도와 성능면에서 큰 개선을 이루게 된다.

Selective Search를 이용해서 2000개의 RoI를 만드는 것은 동일하다.<br>
하지만, CNN의 연산은 전체 input image를 바탕으로 한 후, 결과로 도출된 feature map에 대해서 RoI의 영역에 대응되는 feature map 원소의 위치를 알 수 있다.<br>
이 부분을 RoI pooling을 통해 (max pooling) 고정된 크기의 vector를 생성하게 된다.<br>
이후, FC layer를 거쳐 feature vector를 생성한 후, softmax classifier와 bbox regressor를 통과해 결과를 도출한다.

---

#### 한계

1. CPU 기반의 Selective Search기법은 그대로 사용하기에, CPU 연산 속도가 느리다.
2. RoI Pooling의 정확도가 떨어진다.

## 3. Faster R-CNN

### 3.1 Faster R-CNN이란?

Fast R-CNN + Region Proposal Network의 구조

RPN을 통해서 CPU 연산을 GPU 연산으로 변환한다.

---

#### Region Proposal Network

Feature Map을 기반으로 물체의 위치 예측

k개의 anchor box를 이용<br>
*anchor box* : RoI의 기준이 될 크기를 정해둔 box들의 집합

이 anchor box를 사용하면서, 모델은 기존의 RoI 자체를 예측하는 것이 아니라, 이 anchor box에서 얼마나 차이가 나는지를 학습하면서 학습할 데이터의 양이 줄어서 더 잘 예측하게 된다.

### 3.2 Training Process

각 위치에서 anchor box 마다 GT와 IoU 비교 : Positive는 0.7 이상, Negative는 0.3 이하

Positive Anchor : Classification Loss + Regression Loss<br>
객체가 있는지 여부를 분류

Negative Anchor : Classification Loss<br>
객체가 없는지 여부를 분류

### 3.3 Test Process

1. Preprocessing<br>
resize, augmentation 등을 이용
2. Model<br>
Non-Maximum-Suppression(NMS)<br>
중복된 경계 Bbox를 제거하여 최종 객체 감지 결과를 정리하고 정확도를 높이는 기술<br>
각 클래스마다 score가 가장 높은 박스와 IoU가 특정 기준 이상(e.g. 0.7)인 박스 제거
3. 각 객체의 위치(Bbox)와 클래스 레이블에 대한 예측<br>
모델의 출력으로부터 객체 감지의 임계값(ex. 0.5)을 초과하는 객체만을 최종 감지

### 3.4 Experiments

속도: 초당 17개의 이미지를 객체 인식을 할 수 있다.

2-stage detector vs 1-stage detector: 성능면에서 훨씬 2-stage detector가 낫다.

### 3.5 Limitation & Future Work

2-stage detector로 연산량이 많아서, 실시간 사용에 부적합하다.

1-stage detector인 YOLO는 실시간 사용에 적합함
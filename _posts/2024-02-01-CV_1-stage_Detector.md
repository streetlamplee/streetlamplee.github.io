---
title: CNN 1-stage_Detector
layout: post
date: 2024-02-01 18:06 +0900
last_modified_at: 2024-02-01 18:06:00 +0900
tag: [DL, CV, object detection]
toc: true
---

## 1. 1-stage_Detector

### 1.1 1-stage_Detector란?

Region Proposals없이, Feature Extractor 만을 이용한 object Detection 수행

Feature Extractor : 입력 이미지를 특성으로 변환, 해당 특성을 이용하여 추후 Classification 및 Bounding Box를 예측하는 작업 수행

---

#### 장점

간단한 파이프라인 : 2-stage Detector와 달리 Region Proposals가 없음

빠른 속도 : 2-stage Detector에 비해 연산이 효율적이며, 실시간으로 사용가능

## 2. You Only Look Once (YOLO v.1)

### 2.1 YOLO란?

CVPR 2016에 출판한 논문, 1-stage Detector 분야의 초기 모델

Single Shot Architecture : YOLO는 객체 감지를 위한 단일 신경망 아키텍처를 사용

이미지를 그리드로 나누고, 그리드 셀 별로 B Box와 해당 객체의 클래스 확률 예측

![Alt text](\..\img\cv12.png)

---

#### Grid Image

이미지를 SxS grid 이미지로 분할

객체의 중심 좌표가 특정 셀 안에 있으면 그 셀의 예측 박스 해당 객체를 검출해야함

각 셀은 Bounding Box와 Confidence, Lcass Probablity Map을 예측하는데 사용

---

#### Backbone

학습 데이터의 이미지가 YOLO 모델로 입력

Backbone은 CNN으로 구성

백본을 통과한 이미지에서 Feature Map 생성

* Decoder : FC layer 2개를 쌓아 진행

---

#### Bounding Box + Confidence

각 셀마다 Bounding Box 예측

각 셀은 여러 Bounding Box를 나타낼 수 있음

이후, Bounding Box의 Confidence를 예측

표현은 아래와 같다.

$[p_c\; b_x \; b_y \; b_h \; b_w]$

이때 $p_c$는 물체가 사물을 포함할 확률 (신뢰도 값)<br>
$b_x \; b_y \; b_h \; b_w$들은 각 bounding box의 중심 좌표 (x,y), bounding Box의 너비 w, 높이 h를 뜻하게 된다.

---

#### Class Probablity Map

각 셀마다 Class의 조건부 확률을 예측<br>
$P(C_i \| object)$

---

#### Output

예측한 Bbox, Confidence, Class Probability로, Object Detection 결과 산출

낮은 Confidence의 Bbo를 지운다.

각 클래스마다 Non-Maximum Suppression을 진행한다.

결국은, 하나의 Bbox는 하나의 물체를 나타내게 한다.



### 2.2 Training Process

전처리 - 모델링 - Loss 구하기

- 전처리 : 비슷하다. (크기변형, 정규화)

- 모델링<br>
YOLO 모델 초기화 - 가중치 무작위 설정 - Bbox의 위치와 객체 클래스 예측

- Loss<br>
Localization, Confidence, Classification의 제곱합으로 구성된다.<br>
>Localization: Bbox의 위치에 대한 Loss<br>
>Confidence : 각 Bbox에 대한 신뢰도의 Loss<br>
>Classification : Class 분류에 대한 Loss<br>


### 2.3 Test Process

전처리 - 모델링 - 예측

#### Prediction

바운딩 박스의 좌표, 객체 클래스 및 해당 객체에 대한 신뢰도 점수 예측

$Pr(Class_i \| Object) \times Pr(Object) \times IOU^{truth}_{pred} = Pr(Class_i) \times IOU^{truth}_{pred}$

NMS로 중복된 상자를 제거하고, 신뢰도 점수가 가장 높은 상자를 선택

### 2.4 Experiment

1-stage Detector vs. 2-stage Detector

YOLO는 45 FPS

2-stage Detector는 17 FPS

속도가 빨라 실시간 객체 검출에 사용됨

또한 성능이 좋은 편이라 그 당시에는 굉장히 매력적인 선택지
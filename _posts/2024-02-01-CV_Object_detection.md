---
title: CNN Object Detectoin
layout: post
date: 2024-02-01 10:06 +0900
last_modified_at: 2024-02-01 10:06:00 +0900
tag: [DL, CV, object detection]
toc: true
---

## 1. Object Detection

### 1.1 Object Detection이란?

사물 각각의 **Bounding Box** (Bbox)위치와 **Category**를 예측한다.

Bounding Box : {$x_0,y_0,x_1,y_1$}

Category : 사울의 class label

구조 : Backbone(CNN) + Decoder (Detection Head)

---

#### Image Classification vs. Object Detection

- 이미지 분류 : 이미지 내에 어떤 물체가 있는지 분류
- 객체 인식 : 이미지 내의 Bbox마다 객체의 class 분류 및 Bbox 위치 추론

---

#### Localization

*(=Bbox regression)*

각 Bbox의 {$x_0,y_0,x_1,y_1$} 예측

이때, 정답과의 차이를 비교하면서 *회귀 문제를 풀듯이* 학습하게 된다.

### 1.2 2-stage Detector vs. 1-stage Detector

- 2-stage Detecto<br>
Region Proposals 및 Feature Extractor를 거치면서 Object Detection을 수행<br>
*Region Proposals* : 다양한 크기와 모양의 Bbox로 물체의 위치를 제안<br>
*Feature Extractor* : 제안한 Region (Bounding Box)에 대하여 물체의 특성추출
- 1-stage Detector<br>
Region Proposal없이, **Feature Extractor만** 이용해서 object detection 수행



### 1.3 Object Detection Dataset

#### COCO

*Common Object in COntext*

91개의 클래스로 이루어진 사물 및 동물 모음

최대 640x480 RGB 이미지

330k 이미지

#### Pascal VOC

20개의 클래스로 이루어진 사물 및 동물 모음

500x375 RGB이미지

11k 이미지 데이터

#### KITTI

8개의 클래스로 이루어진 자동차 및 사물 모음

1248x384 RGB 이미지

15k 이미지 데이터

### 1.4 성능 평가 방법

#### Intersection of Union (IoU)

우리가 예측한 Bbox와 실제 Bbox 값의 차이를 어떻게 비교할 수 있을까

**정답 구역과 예측 구역의 교집합 부분의 비율이 높으면 정답에 가깝다**

$\therefore \; IOU = \displaystyle \frac{Area\;of\;Intersection}{Area\;of\;Union}$

Area of Intersection : 교집합 구역<br>
Area of Union : 합집합 구역

임계점을 선택해서, 어느 수준에서 구역이 일치하는지를 선택할 수 있다.

#### Average Precision (AP)

IoU 임계점을 통해 박스의 일치 여부를 판단해서 TP, FP, TN, FN을 구할 수 있고, 이를 이용해서 Precision, Recall을 계산할 수 있다.

이때 AP는 Precision, Recall curve의 넓이를 계산한 값이 된다.

Precision과 Recall은 서로 Trade-off 관계이므로, 둘다 적절히 좋아야 AP가 높다는 것을 알 수 있다.
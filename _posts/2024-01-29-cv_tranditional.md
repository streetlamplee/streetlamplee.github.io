---
title: 고전 컴퓨터 비전
layout: post
date: 2024-01-29 18:00 +0900
last_modified_at: 2024-01-29 18:00:00 +0900
tag: [DL, CV]
toc: true
---

## 1. 고전 컴퓨터 비전

### 1.1 고전 컴퓨터 비전이란?

*규칙 기반의 이미지 처리 알고리즘 (OpenCV)*

c.f. DL : 데이터 학습 기반의 이미지 처리<br>
즉 데이터로 학습된 neural network를 바탕으로 output을 도출

### 1.2 고전 컴퓨텆 비전의 활용

그러타 로보틱스나 가상현실에서는 DL로 해결하기 어렵다.

그리고, DL 결과의 후처리를 할때에도 고전 컴퓨터 비젼이 활용하다.

마지막으로, DL 모델 없이 데이터를 가공할 때 활용할 수 있다.

## 2. Morphological Transform

### 2.1 Morphological Transform이란?

*이미지에 기반한 연산, 흑백 이미지에서 일반적으로 수행한다.*

input : 원본 이미지, 커널(연산자)

---

#### 중요성

Morphological Transform은 이미지 전처리 영역에서 유용하게 사용<br>
ex. 이미지의 노이즈를 제거할 때 사용

### 2.2 Erosion이란?

*물체의 경계를 침식*

이미지의 특징을 축소할 때도 사용가능

---

#### 동작원리

홀수 크기의 커널이 이미지와 컨볼루션 연산을 수행

커널 아래 **모든 픽셀이** 1이면 1, 그 외에는 0

결계 근처의 픽셀은 침식

### 2.3 Dilation이란?

*Erosion과 반대로 동작*

사물의 크기를 팽창할 때도 사용가능

---

#### 동작원리

홀수 크기의 커널이 이미지와 컨볼루션 연산을 수행

커널 아래의 **하나 이상의 픽셀이** 1이면 1, 그 외에는 0

결계 근처의 픽셀은 팽창

### 2.4 Opening이란?

*Erosion 커널과 Dilation커널 순서대로 동작되는 연산*

>반대로 동작시키면 (Dilation $\rightarrow$ Erosion) **Closing 커널**이라고 부른다.

노이즈를 제거하는데 사용된다.

## 3. Contour Detection

### 3.1 Contour Detection

목표 : <br>Contour : 같은 색깔 및 intensity를 가지는 연속적인 결계점들로 이루어진 curve (물체의 경계)<br>고전 컴퓨터 비전을 활용하여 Raw image에서 객체의 contour를 추출

---

#### 중요성

DL 모델을 사용하지 않는다.<br>
즉, DL 모델의 단점을 그대로 학습하지 않고, rule-base로 찾기 때문에 DL 모델과 동시에 진행한다<br>
또한, 하드웨어적 이득을 볼 수도 있다.

---

#### 과정

**Edge detection** $\rightarrow$ **Dilation** (optional) $\rightarrow$ **Contour detection**

### 3.2 Canny Edge Detector

*Edge detection*의 일부

장점 : 정확도 높음<br>
단점 : 실행시간이 느리고 구현이 복잡함

---

#### 과정

1. 노이즈 제거<br>
이미지 내에 노이즈가 있다면, 엣지를 찾는데 어려움이 있음<br>
일반적으로 가우시안 필터 사용

2. 이미지 내의 높은 미분값 찾기<br>
미분 : 행 또는 열 방향으로 픽셀 값의 변화 정도

3. 최대값이 아닌 픽셀 값 0으로 치환<br>
목표 : 엣지 검출에 기여하지 않은 픽셀을 제거<br>
Gradient의 최대값을 가진 픽셀을 찾고, 주변 값과 비교해서 엣지가 아닌 픽셀 값을 0으로 치환(제거)

4. 하이퍼파리미터 조정을 통한 세밀한 엣지 검출<br>
2가지의 threshold를 정의 (low threshold, high threshold)

### 3.3 Contour edge Detection with OpenCV

Contour Detection

raw image를 binary image로 변환 $\rightarrow$ OpenCV 의 `findContours()` 함수 이용


---
title: Computer Vision introduction
layout: post
date: 2024-01-29 16:00 +0900
last_modified_at: 2024-01-29 16:00:00 +0900
tag: [DL, CV]
toc: true
---

# Computer Vision이란?



## 1. 컴퓨터 비전의 정의

#### Vision?

시각적인 정보들의 집합 (사진, 비디오)<br>
즉, 시각으로 보이는 것을 숫자로 데이터화 하여 저장한 모든 것들을 포함하는 개념

일반적으로, (너비, 높이, 3)의 tensor로 이루어진 데이터로 그림을 표현할 수 있다.

ex. (233, 255, 233) $\rightarrow$ RGB

### 1.1 Computer Vision이란?

AI의 한 종류로, vision 데이터들에서 의미있는 정보를 추출하고, 이를 이해한 것을 바탕으로 여러가지 작업을 수행하는 것

#### CV vs. Computer Graphics

* Computer Graphics : 컴퓨터 모델을 활용해서 이미지를 렌더링하는 과정

CV는 input을 이미지로 받고, 이를 통해 얻은 정보를 통해 task를 수행하는 과정

그러나 최근 기술의 발전에 따라, 둘 사이의 overlap이 발생하고 있기에 크게 분리하여 생각하지 않아도 무방하다.

### 1.2 Types of Computer Vision

Low-level로 갈수록 pixel 단위로 이미지를 처리한다.<br>
ex. 이미지 특징 추출

high-level로 갈수록 이미지 전반을 하나의 entity로 묶어서 처리하게 된다.
ex. 분류, 개체 인식 등

## 2. 컴퓨터 비전 훑어보기

### 2.1 Low-Level

#### image processing

* Resize<br>
이미지의 크기를 조절하는 작업
* Color Jitter<br>
이미지의 색을 바꾸는 작업<br>
특정 pixel마다 독립적으로 연산한다.
* Feature Extraction<br>
**edge**를 찾는 과정<br>
**edge** : 급격한 색의 변화가 있는 pixel을 의미함
* Feature Extraction<br>
ex. watershed<br>
이미지를 특징을 가지는 이미지로 분할하는 작업

### 2.2 Mid-Level

#### images to images

* Panorama Stitching<br>
파노라마 풍 사진 2개를 이어 붙이는 작업<br>
low-level 보다 훨씬 많은 pixel을 참조해야 하므로 mid-level이다.

#### images to world

* Multi-view Stereo<br>
여러 각도에서 찍은 이미지를 바탕으로 3D 모델링으로 변환해주는 작업
* Depth Estimation<br>
2차원 이미지에서, 3차원의 깊이를 판단하는 작업

### 2.3 High_Level

#### Semantics

* image Classification<br>
이미지 전체의 데이터를 활용해서, 해당 이미지를 feature에 따라 분류
* Object Detection<br>
이미지에서 객체를 검출, 객체가 어떤 객체인지 판단하는 작업
* Segmentation<br>
이미지에 있는 객체를 pixel 단위로 세분화하는 작업
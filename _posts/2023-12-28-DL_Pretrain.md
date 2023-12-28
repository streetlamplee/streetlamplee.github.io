---
layout: post
title: DL 전이학습
date: 2023-12-28 17:00 +0900
last_modified_at: 2023-12-28 17:00:00 +0900
tags: [deeplearning, pretrained_model]
toc:  true
---

# 전이학습

## 1. 전이 학습이란?

### 1.1 Pretrained Model이란?

대규모 데이터셋을 기반으로 학습된 모델<br>
학습한 task에 대한 일반적인 지식을 갖고 있음

ex. GPT, PALM, Stable-Diffusion

### 1.2 전이학습이란?

모델이 이미 학습한 **일반적인** 지식을 기반으로 더 빠르게 **새로운 지식**을 학습할 수 있음

### 1.3 Fine-Tuning이란?

전이학습의 한 방법

pretrained model을 그대로 혹은 layer를 추가한 후, <br>새로운 작업에 맞는 데이터로 모델을 추가로 더 훈련하는 방법

### 1.4 Domain Adaptation이란?

전이학습의 한 방법

A라는 도메인에서 학습한 모델을 B라는 도메인으로 전이하여<br>
도메인 간의 차이를 극복하는 것이 목적

### 1.5 유사한 다른 학습 방법들

- Multi-task learning : <br>하나의 모델을 사용하여 여러 개의 관련된 작업을 동시에 학습하면서 공통으로 사용되는 특징을 공유하는 학습방식
- Zero-shot learning : <br>
기존에 학습되지 않은 새로운 클래스나 작업에 대해 예측을 수행하는 기술<br>
(e.g. CLIP)
- One/few-shot learning : <br>
하나 또는 몇 개의 훈련 예시를 기반으로 결과를 예측하는 학습 방식

### 1.6 전이 학습 전략

#### 도메인이 비슷할 때, dataset 크기에 따른 전략

- 비교적 작을 때 : 마지막 classifier만 추가 학습 (나머지 freeze)
<br> $\because$ 데이터셋의 크기가 작아서, 기존의 일반적인 지식을 전달하는데 집중

- 비교적 클 때 : classifier 뿐만 아니라, 다른 일부 layer도 추가 학습
<br> $\because$ 기존의 일반적인 지식을 유지하면서, 몇개의 layer만 추가 학습해서
<br> 새로운 데이터에 대한 지식도 학습할 수 있도록

#### 도메인이 매우 다를 때, dataset 크기에 따른 전략

- 데이터 크기가 클 때 : 꽤 많은 layer를 학습해야함
<br> $\because$ 도메인이 매우 다르면, 일반적인 지식을 많이 수정해야하므로

- 데이터 크기가 매우 작을 때 : 학습이 잘 되지 않음

#### learning rate 전략

pretrained model의 일반적인 지식을 크게 업데이트하지 않기 위해 작은 learning rate로 학습

## 2. Pretrained Model Community

### 2.1 Pretrained Model Community의 필요성

- 대규모 데이터 셋을 사전 학습한 모델들이 매우 발전
- 이런 대규모 모델을 쉽게 커스터마이징해서 활용하고자 하는 수요 증대
- 그럼 이런 모델을 올릴 community의 필요성 대두

### 2.2 Timm for CV

Timm (py**t**orch **Im**age **M**odel)

```python
timm.list_models()
```

위의 코드로 제공되는 모델 리스트를 볼 수 있음

### 2.3 HuggingFace for NLP, CV

<a href = 'https://huggingface.co'>Hugging Face Website</a>

#### 사용법

사용하고자 하는 모델을 찾아야함

<a href = 'https://huggingface.co/docs'>hugging face doc</a>

모델을 찾아, tutorial을 천천히 따라가보도록 하자

```cmd
pip install transformer
```
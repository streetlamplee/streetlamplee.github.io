---
layout: post
title: DL 모델 학습법 I, 다층 퍼셉트론
date: 2023-12-19 16:48 +0900
last_modified_at: 2023-12-19 17:29:00 +0900
tags: [DeepLearning, MLP]
toc:  true
---

# 다층 퍼셉트론

## 1. 뉴럴넷 개요

### 1.1 사람과 기계

* 사람이 무언가를 인식하는 과정은 1,400 만개의 뉴런들과 뉴런 간의 수 십 억개의 시냅스를 이용함

* 기계가 무언가를 인식하는 과정 (전통방식)

1. 일단 sample을 많이 보고 패턴을 파악한다.

ex. 9는 위에 동그라미, 오른쪽에 직선이 있다

2. 주요 특징을 프로그래밍 언어로 표현한다.
3. 의사결정 률을 정한다.

* 기계가 무언가를 인식하는 과정 (뉴럴넷 방식)

학습데이터를 바탕으로 뉴럴넷에서 자동으로 특징들과 의사 결정 룰을 찾아낸다.

### 1.2 뉴런과 인공뉴런

인공 뉴런 : 사람의 뉴런을 모방한 것

![Alt text](\..\img\artificialneuron.png)

## 2. 퍼셉트론 (perceptron)

### 2.1 퍼셉트론의 구조

<mark>퍼셉트론</mark>

1957년 코넬 항공 연구소의 프랑크 로젠블라트에 의해 고안됨

가장 간단한 형태의 feed-forward 네트워크

![Alt text](\..\img\DL2-1.png)

각 input에 대해서 weight를 부여하고, 이를 transfer func를 이용해서 하나의 값을 출력한다.

이때 출력된 값이 특정 threshold보다 크다면 1, 아니라면 0을 출력하는 프로그램

이때, threshold를 받으면 0 또는 1을 출력하는 함수를 *Activation func*이라고 한다.

**가중치와 편향**
---
퍼셉트론에서 transfer func의 마지막에 편향 (bias) $w_0$를 더해주는 것,

결국, transfer func의 값이 커지는 것이므로, <mark>bias가 클수록 출력값은 쉽게 1이 된다.</mark>

이때 bias와 threshold T를 합쳐 한번에 표현하기도 한다.

$(transfer \;func) + w_0 > T$

$(transfer \;func) + w_0 - T > 0$

$(transfer \;func) + b > 0 \;\; where\; b = w_0 - T$

### 2.2 퍼셉트론의 원리

**퍼셉트론과 선형분리**
---
각 p개의 input에 $w$를 곱하고 더해주게 됨

이는 변수가 p개인 1차식으로 표현할 수 있음

결국 이를 threshold를 기준으로 크고 작은지를 구분하기 때문에, 어떤 p차원의 space에서

선 (또는 면)으로 분리되는 효과가 있다.

선형 분리 가능성 때문에, AND, OR, NAND 등의 로직을 구현할 수 있게 되었다.

(하나의 퍼셉트론으로 NAND GATE를 표현할 수 있다.)

즉 퍼셉트론을 여러 개 모으면 어떠한 연산이든 가능하다!

BUT

XOR 논리 게이트는 하나의 퍼셉트론으로는 구현이 불가능하다.

하지만 여러층의 퍼셉트론을 활용하면 분리가 가능하다. (*다층 퍼셉트론의 비선형 분리 가능성*)

## 3. 다층 퍼셉트론

### 3.1. 모델 개요

**다층 퍼셉트론 (Multi-Layer Perceptron, MLP)**

![Alt text](\..\img\DL2-2.png)

(=Fully-Connected Layers)

특정 뉴런은 이전 레이어의 모든 뉴런과 연결이 되어있다.

### 3.2 활성화 함수

입력 신호의 총합을 출력 신호로 변환하는 함수

사용하는 이유 : <mark>활성화 함수가 없는 다층 퍼셉트론 모형 = 단순한 선형식</mark>일 뿐이다.

> 층마다 사용되는 $w$의 집합을 $W$라고 한다면, 2개의 layer를 가지는 MLP는
>
> 결국 $y=W_1 W_2 X$ 가 된다.
>
> $W = W_1 W_2$라고 한다면 결국 $y=Wx$가 된다.

활성화 함수의 종류
---
* sigmoid
> 연산 과정 전체를 미분 가능하게 만드는 것이 큰 목적이다. (*역전파 기법*)

* tanh
* ReLU
* Leaky ReLu
* Softmax



### 3.3 다층 퍼셉트론 (MLP)

MLP의 구조
---
* node / neuron : Transition (z) + Activation Function (그래프로는 원으로 표현된다.)
* Layer / 층 : MLP에서 Layer의 수는 (은닉층 수 + 1) 개를 의미
> 입력층
> 은닉층
> 출력층
* edge, connection : 가중치를 의미한다. (그래프로는 선으로 보통 표현된다.)
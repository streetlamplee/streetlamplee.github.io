---
layout: post
title: DL 모델 학습법 II, 경사 하강법
date: 2023-12-19 17:33 +0900
last_modified_at: 2023-12-20 10:45:00 +0900
tags: [DeepLearning, MLP]
toc:  true
---

# 경사 하강법 (Grandient Descent)

## 1. 미분과 기울기

미분 : 변수의 움직임에 따른 함수값의 변화를 측정하는 도구

최적화에서 가장 많이 사용하는 기법이며, 극한에 의해 정의된다.

$\displaystyle \frac{df(x)}{dx}=\displaystyle \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}$

2차원 평면에서, $x$에서의 미분값은 접선의 기울기와 동일하다.

<mark>어떤 점에서 함수의 기울기를 알면, 어느 방향으로 점을 움직여야 함수 값이 증가 or 감소하는지 알 수 있음</mark>

**편미분** : 벡터를 입력으로 받는 다변수 함수의 경우, 각 변수에 대해 미분한 편미분을 사용한다.

$\partial_{x_i}f(x)=\lim_{h \rightarrow 0} \frac{f(x+he_i) - f(x)}{h}$

$\nabla f = (\partial_{x_1}f,\partial_{x_2}f,\cdots,\partial_{x_d}f)\;\;\leftarrow$ **Gradient Vector**

## 2. 경사 하강법 (GD)

기울기가 감소하는 방향으로 $x$를 움직여서 $f(x)$의 최소값을 찾는 알고리즘

딥러닝에서는 loss function의 그래디언트($\nabla L_\theta$)가 감소하는 방향으로

파라미터 $\theta$를 움직여 손실함수의 최소값을 얻는다.

    이때 미분 계산을 위해 '$\theta$ 주변이 부드럽게 감소하는 구간이다'라는 가정 필요


---
*학습과정*

1. 모델 파라미터 $\theta=\{ W,b\}$ 초기화
2. 전체 학습데이터셋에서의 손실 함수 값을 구하고, 미분을 통해 이를 최소화하는 모델 파라미터 W,b를 찾는다.
3. (2)를 반복하다가 종료조건이 충족되면 학습 끝

---
*한계점*
- Local minimum은 무조건 찾으나, Global minimum을 무조건 찾지 않는다.
> 해결방안
>
> 1. 파라미터 초기화를 굉장히 잘한다.
> 2. 모델 구조를 바꿔서 그래프 모양을 바꾼다.
> 3. <mark>Learning Step을 변경한다.</mark>

---
*경사 하강법의 파이프라인*

![Alt text](\..\img\DL2-3.png)

## 3. 확률정 경사 하강법 (SGD)

모든 데이터를 ㅏ용해서 구한 경사를 통해 파라미터를 한 번 업데이트하는 대신,

데이터 한 개 또는 일부를 활용하여 구한 경사로 여러 번 업데이터를 진행

M개의 데이터를 randomize한 후, m개 씩 쪼갠다.

**1 step**에서, m개의 batch size 마다 평균 loss를 구해 parameter를 한 번 업데이트

**1 epoch** (1 iter)는 전체 학습 데이터셋을 한 바퀴 모두 학습하는 것을 의미

---
*확룰적 경사 하강법의 파이프라인*

![Alt text](\..\img\DL2-4.png)

학습을 위해서 손실함수의 그래디언트 계산이 불가피

따라서 step function 대신 sigmoid function을 퍼셉트론에서 이용한다.

---
*SGD vs GD*

SGD는 학습에 활용하는 data set이 계속 달라짐에 따라 손실 함수의 형태가 약간 바뀐다.

이를 통해 <mark>local minima에 빠지는 것을 방지하여 최적해에 도달할 가능성을 높이는 방법</mark>으로 이해 가능
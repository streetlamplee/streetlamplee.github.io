---
title: CNN은 이미지의 어느 위치를 보고 있는걸까?
layout: post
date: 2024-01-31 14:01 +0900
last_modified_at: 2024-01-31 14:01:00 +0900
tag: [DL, CV]
toc: true
---

## 1. CAM

### 1.1 Class Activation Map

CNN의 모델이 어디를 보고 예측하는지 시각화하는 방법을 제시

FC layer에서는 flatten을 하기 때문에, pixel의 위치정보를 잃게된다.

그래서 대신 **Global Average Pooling**을 해서, feature map 하나당 하나의 특징 변수로 변환하게 된다.

### 1.2 CAM의 구조

![Alt text](\..\img\cv8.png)

마지막 과정에서, FC layer를 이용해서 결과를 뽑는 것이 아니라, 각 채널에 평균을 낸 값을 tensor로 변환한다.<br>
이때, 채널의 수는 class의 수와 같으며, 각 채널은 `Class에 속하는 특징을 알아볼 수 있는 feature map으로 해석할 수 있다.`

이때, Global Average Pooling을 하고 나서, 결과를 내기 위해 사용되는 가중치($w_n$)를 Global Average Pooling을 하기 이전의 channel에 곱하게 되면, 어떤 부분에서 활성화가 크게 이루어졌는지를 볼 수 있다.<br>
이 feature map(channel)의 크기를 이미지의 크기로 키우게 된다면, 위의 예시와 같은 활성부분을 볼 수 있게 된다. (Class Activate Map)

수식으로 표현하면 아래와 같다.

$f_k(x,y)$ : Feature Map k에서의 x,y 값<br>
$F_k$ : channel k에서 GAP된 특징값<br>
$w^c_k$ : 특징 변수 $F_k$가 클래스 c에 기여하는 가중치<br>
$S_c$ : 클래스 c에 대한 score<br>
$P_c$ : 클래스 c로 분류될 확률<br>
$M_c(x,y)$ : 클래스 c에 대한 activation map<br>


$\begin{aligned}   F_k &= \displaystyle \sum_{x,y}f_k(x,y) \newline S_c &= \sum_k w^c_k F_k \newline P_c &= \frac{exp(S_c)}{\sum_c exp(S_c)}                          \end{aligned}$

<br><br>
$\begin{aligned} S_c &= \displaystyle \sum_k w^c_k F_k \newline &= \sum_k w^c_k \sum_{x,y} f_k(x,y) \newline &= \sum_{x,y} \sum_k w^c_k f_k(x,y) \newline &= \sum_{x,y} M_c(x,y) \newline M_c(x,y) &= \sum_k w^c_k f_k(x,y)                           \end{aligned}$

### 1.3 CAM의 결과

![Alt text](\..\img\cv9.png)

위와 같이 각 class마다 어디를 중점적으로 보는지는 조금 차이가 있다. 

### Global Average Pooling vs. Global Max Pooling

Global Max Pooling을 통해서도 CAM을 그릴 수 있으나, max pooling 자체가 feature map에서 가장 뚜렷한 특징만 찾아내는 방법이므로, localization 능력이 GAP보다는 낮다.

## 2. Grad-CAM



### 2.1 Gradient_weighted CAM

*CAM의 단점* : GAP가 적용되지 않은 모델에서 적용을 하기 위해서는, 따로 conv layer 이후에 GAP를 추가해야하는 단점이 있었음

**Grad_CAM** : CAM과 마찬가지로, 각 feature 멥에 대한 각 class의 계수를 사용하긴 한다.<br>
하지만 그 계수를 계산할 때, GAP가 아니라 Gradient를 사용한다.<br>
재학습이 필요없고, 대부분의 모델에서 사용할 수 있다.

![Alt text](\..\img\cv10.png)

임의의 모델이 하는 것처럼 layer를 진행하고, feature map을 flatten한 후, FC layer를 2개 쌓는다.(한개는 hidden?)

이후, 모델이 학습하면서, 역전파를 통해 node에는 자동적으로 softmax 이전의 class y에 대한 각 feature map 원소의 편미분 값이 저장되게 된다. (각 feature map의 원소가 특정 class에 주는 영향력)

따라서, 각 원소별로 얻어진 영향력(gradient)를 feature map으로 옮기고, 그 feature map의 평균을 가중치로 곱해서 각 채널별로 영향력을 알 수 있는 feature map을 만들 수 있다.<br>
그 수는, 마지막으로 conv를 실행한 layer의 output channel의 수와 같다.

이를 pixelwise sum을 이용하여 확인할 수 있다.



### 2.2 Grad_CAM의 결과

![Alt text](\..\img\cv11.png)
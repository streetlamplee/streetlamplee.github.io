---
title: backbone 이해하기 _ EfficientNet
layout: post
date: 2024-01-30 18:00 +0900
last_modified_at: 2024-01-30 18:00:00 +0900
tag: [DL, CV, CNN, EfficientNet]
toc: true
---

# Backbone 이해하기 : CNN

## 1. EfficientNet

### 1.1 Baseline Model

#### MBConv Block

*Mobile inverted Bottleneck Convolution*

- Depthwise Separable Convolution : Depthwise conv + Pointwise conv
- Squeeze and Excitation : Feature를 압축했다가 증폭하는 과정을 통해 feature의 중요도를 재조정

**spatial한 정보와 channel의 정보를 conv 연산에서 활용하는 방식을 분리하자**


* Depthwise Separable Convolution<br>
spatial 정보 : Depthwise conv<br>
channel 별로 나누어서 conv 연산을 수행한다. 따라서, 기존의 conv 연산보다 적은 수의 parameter로 feature를 추출할 수 있다.<br>
즉, channel 별로 각각의 feature map이 생성이 된다.<br><br>
channel의 정보 : Pointwise conv<br>
각 channel 별로 처리를 하게 되니깐, channel의 수를 조절할 수 없게 되었다.<br>
이를 처리하기 위해서 pointwise Conv를 진행하게 된다.<br>
1x1 kernel을 처리해서, 채널의 수를 조절해서 전체 파라미터 수가 적어질 수 있도록 한다.
* Squeeze and Excitation<br>
feature를 압축했다가 증폭해서 feature의 중요도 재조정<br>
즉, Depthwise Separable convolution의 결과를 추가로 재조정 한다.<br>
$\because$ Depthwise Separable convolution은 파라미터 수를 줄이기 위해 성능적인 부분을 조금 포기한 module임<br>
이를 보완하고자 Squeeze and Excitation을 실행<br>
    1. Global Average Pooling으로 `1x1xC`의 크기로 압축<br>
    (= 각 채널의 중요한 정보만 가지도록 압축)
    2. `1x1xC` 를 이용해서 각 channel의 **중요도**를 추출, 그리고 기존 feature map에 곱함
    3. output은 **channel의 중요도가 학습된 새로운 feature map**

#### Baseline Model Scaling

해상도가 커지면 성능이 좋더라 이것을 해석해보자

**해상도가 커지면**
- 더 큰 Receptive Field가 필요하므로 더 많은 layer를 쌓은 깊은 모델이 높은 정확도를 가짐
- 이미지의 디테일(high frequency information에 대한 정보)가 포함되는 장점이 있음<br>
이를 효과적으로 사용하기 위해서는 더 큰 kernel size를 가지는 channel이 필요함

### 1.2 Compound Scaling

![alt TEXT](\..\img\CV6.png)

각 방향으로 모델의 크기를 키워도 한 방향으로만 키워도 결국 어느 수준에서 모델의 성능이 상승하지 않는다.

즉, 모델의 성능을 향상시키는 방법은

1. 모델의 **Depth**를 늘리기
2. Channel **Width** 늘리기
3. Input Image의 **해상도** 늘리기

따라서 모든 방향에서의 모델 크기 향상을 위해 **Compound Scaling** 사용

---

- **individual Scaling**<br>
깊이($d$), 넓이($w$), 해상도($r$)를 각각 독립적으로 scaling 하는 방법
- **Compound Scaling**<br>
깊이($d$), 넓이($w$), 해상도($r$)를 동시에 scaling 하는 방법

이때, $\phi$는 Compound Coefficients이고, 한번에 scaling 한다.

$\begin{aligned}depth:\;d&=\alpha^{\phi} \newline width:\;w &= \beta^{\phi} \newline resolution: \;r&=\gamma^{\phi}                          \end{aligned} $

---

*small grid search*

$\phi$를 1로 고정한 후, $\alpha , \beta , \gamma$를 구한다.

이때 구한 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$가 되는지 확인한 후,

$\alpha , \beta , \gamma$를 고정하고 $\phi$를 조절하여 scaling 한다.



### 1.3 EfficientNet의 결과

![Alt text](\..\img\cv7.png)

다른 모델보다 압도적으로 성능이 향상된 것을 볼 수 있다.
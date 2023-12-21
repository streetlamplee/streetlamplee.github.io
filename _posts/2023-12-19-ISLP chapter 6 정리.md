---
layout: post
title: ISLP chapter 6 정리
date: 2023-12-19 12:09 +0900
last_modified_at: 2023-12-19 12:09:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

# 6. Linear Model Selection and Regularization

## 6.1. 

### 6.1.1.

### 6.1.2.

## 6.2. Shrinkage Methods

계수를 제한하거나, 정규화를 시킴으로써 모든 p개의 predictor를 이용해서 모델을 학습할 수 있다.

다시 말하면, 계수의 예측값을 0에 수렴하도록 만든다는 것이다.

ex. ridge regression, lasso

### 6.2.1. Ridge Regression

기본적으로, Least Square에서는 RSS를 최소로 하는 것을 목표로 한다.

이때 $RSS = \displaystyle\sum^n_{i=1}\left(y_i - \beta_0 - \displaystyle\sum^p_{j=1}\beta_j x_{ij}\right)^2$

Ridge는 least square과 굉장히 유사하지만, ridge의 계수는 아래의 값을 최소화하도록 학습한다.

$\displaystyle\sum^n_{i=1}\left(y_i - \beta_0 - \displaystyle\sum^p_{j=1}\beta_j x_{ij}\right)^2 + \lambda \displaystyle\sum^p_{j=1} \beta^2_j$

$= RSS+\lambda \displaystyle\sum^p_{j=1} \beta^2_j$

$\lambda$는 0이상의 실수이며, tuning parameter이다.

RSS를 최소로 하도록 계수를 학습하면서, 우항의 $\lambda \displaystyle\sum^p_{j=1} \beta^2_j$(<mark>shrinkage penalty</mark>)는 각 계수가 전부 0에 가까울 수록 값이 작아진다.

따라서 이 값을 최소화하기 위해서는 $\beta_j$를 0과 가깝게 하는 shrinking 효과가 발생하게 된다.

$\lambda$는 shrinking 효과의 정도를 결정하는 파라미터로, 0이라면 shrinking 효과가 없을것이고, $\inf$라면 계수를 모두 0으로 예측할 것이다.

ridge는 각 $\lambda$마다 하나의 계수 set을 제공하게 되는데 어떤 $\lambda$를 사용하는지는 이후의 CV에서 확인

rigde에서 shrinking 효과를 받는 것은 intercept을 제외한 나머지 계수들인데, 이는 intercept가 단순하게 모든 predictor가 0일 때, 나오는 값이라서 ridge regression의 $\lambda$가 극한으로 커질 때, 그 평균을 쉽게 알아보기 위함


---
#### 왜 Least Squares 보다 나은가?

*<mark>bias-variance trade off</mark>*

$\lambda$가 증가할수록, ridge regression의 자유도는 줄어들게 된다.

(variance의 감소, bias의 증가)

![Alt text](\..\img\ridge1.png)

>검은색 : bias,  초록색 : variance,  보라색 : test MSE
>>좌측의 그래프에서, $\lambda$가 증가함에 따라 bias는 증가하고, variance는 감소한다. 그 중에서, test MSE는 그 값이 비슷할 때 더 최소의 값을 가진다.
>
>>우측의 그래프에서, $\| \hat \beta_\lambda^R \|_2/\| \hat \beta \|_2$의 증가에 따라 마찬가지로 움직인다.

ridge 또한 컴퓨팅 성능적인 이점이 있다.

$\lambda$ 하나만 fix할 수 있다면 모델을 한개만 학습시키면 되기 때문이다.

또한 결국 시뮬레이션을 통해 모든 $\lambda$에 대한 ridge를 하게 되면 least square과 비슷해진다.

### 6.2.2. The Lasso

ridge는 $\lambda$가 무한대가 아닌 이상, 각 계수가 0에 수렴하지, 0이 되지는 않는다. (모든 predictor를 사용한다.)

이게 모델의 성능면에서는 문제가 없을 수 있으나, 모델의 해석력에는 문제가 있을 수 있다.

lasso는 이러한 불리한 점을 해결할 수 있다.

lasso는 아래의 값을 최소화하면서 학습한다

$RSS + \lambda \displaystyle \sum^p_{j=1} \|\beta_j\|$

ridge와 다른점은 $\beta_j^2$ 가 $\|\beta_j\|$가 된 것

$l_2$ 페널티 대신, $l_1$ 페널티를 사용한다고 말할 수 있다.

$l_2$를 사용하기 때문에, lasso에서 계수는, $\lambda$가 충분히 크다면, 정확히 0이 될 수 있다.

<mark>이 과정이, 변수의 selection이 될 수 있다.</mark>

**sparse model** : predictor의 부분 집합을 이용하는 모델

$\lambda$의 선택은 이후 CV에서 확인

![Ridge regression & Lasso의 다른 표현식](\..\img\ridgelasso.png)

    위의 식은 Ridge의 다른 표현식, 아래는 Lasso의 다른 표현식

    각자 subject to 에 들어가는 수식이 다르다는 점을 알 수 있다.

    s가 커진다면, 매우 약하게 제한하게 된다. >> least square를 수행하게 된다.

ridge나 lasso는 best subset selection의 관점에서는, 아래의 식으로 모두 표현할 수 있다.

![Alt text](\..\img\6.10.png)

위의 식을 통해 best subset selection을 처리하는 것은 컴퓨팅 성능을 많이 잡아먹는다.

따라서 우리는 ridge나 lasso를 통해 이를 해결한다.

---

The Variable Selection Property of the Lasso

![Alt text](\..\img\6.7.png)

> 왼쪽의 그래프에서, $\hat \beta$ 는 least square의 계수 예측값, 하늘색 구역은 lasso로 인해 만들어진 계수 구역

> 오른쪽의 그래프에서, $\hat \beta$ 는 least square의 계수 예측값, 하늘색 구역은 ridge로 인해 만들어진 계수 구역

> 빨간 타원은 RSS가 같은 구역을 표현한 등고선

RSS가 증가함에 따라 타원의 크기가 커지게 되고, 각 계수 구역에 만나는 지점이 존재함

lasso는 기본적으로 절대값을 이용해서 구역을 생성하기 때문에, 둥근 모양이 아니라 각을 이루게 된다.

이 각이 툭 튀어나오게 되므로 기본적으로 등고선이 이 각에 만날 가능성이 매우 높다.

이 각은 결국, 어느 계수의 축에서 생성되기 때문에, 어떠한 계수의 값이 0이 되게 될 것이다.

이러한 과정에서 lasso는 계수의 값이 정확히 0이 되므로서 계수의 select을 하게 되는 효과를 가진다.

---

Comparing the Lasso and Ridge Regression

해석력에서는 Lasso > Ridge

정확도 성능의 측면에서는 어떻게 되는가?

![Alt text](\..\img\f6.8.png)

> 검정색 : bias, 초록색 : variance, 보라색 : test MSE

> 좌측 그래프 : lasso, 우측 그래프 : 직선은 lasso, 점선은 ridge

우측의 그래프를 보면 ridge가 조금 더 낮은 MSE를 얻을 수 있다고 볼 수 있다.

다만 이게 일반적인 경우는 아니고, 이 경우에서는 그렇다고 생각

다르게 생각하면, lasso는 ridge에 비해 사용하는 predictor의 개수가 더 적다.

그럼에도 불구하고 비슷한 성능을 낸다는 점에서 더 좋다고 생각할 수 있다.

<mark>결국 lasso도 ridge도 서로를 압도하는 성능을 내지 않는다.</mark>

다만 일반적으로 predictor가 적을수록 lasso가, predictor가 많을수록 ridge가 성능이 좋다.

하지만 현실에서는 predictor의 수가 얼마나 될지 모르기 때문에, CV를 통해 방법을 선택한다.

---

A Simple Special Case for Ridge Regression and the Lasso

> 가정 : intercept가 없다

그러면 least square의 목표는 아래의 값을 최소화하는 것이다.

$\displaystyle \sum^p_{j=1}(y_j - \beta_j)^2$

ridge와 lasso는 아래의 식을 최소화한다.

$\displaystyle \sum^p_{j=1}(y_j - \beta_j)^2 + \lambda \displaystyle \sum^p_{j=1} \beta^2_j$

$\displaystyle \sum^p_{j=1}(y_j - \beta_j)^2 + \lambda \displaystyle \sum^p_{j=1} \|\beta_j\|$

![Alt text](\..\img\f6.10.png)

그래프로 표현하면 다음과 같다.

왼쪽 그래프는 ridge regression이며, 모든 구간에서 계수의 예측값이 같은 비율로 shrink되는 것을 볼 수 있다.

오른쪽 그래프는 lasso이며, $\lambda / 2$를 기준으로 soft-thresholding이 되는 것을 볼 수 있다.

---

Bayesian Interpretation of Ridge Regression and the Lasso

이해불가

### 6.2.3. Selecting the Tuning Parameter

6.1의 subset selection과 마찬가지로, ridge와 lasso도 파라미터를 결정해야한다.

$\lambda$ 혹은 $s$를 결정한다. (둘 다 같은 의미를 가진다.)

간단하게 말하면, 이 문제는 Cross validation으로 쉽게 풀 수 있다.

>각기 다른 $\lambda$를 선택하고, 이를 각각 CV를 돌려 가장 작은 CV error를 내는 $\lambda$를 선택하면 된다.

![Alt text](\..\img\f6.13.png)

> 좌 : lasso에서 cv를 돌린 결과 (cv error)
>
> 우 : lasso에서 계수의 예측 값 (회색은 target과 관계가 없는 계수의 예측 선)
>
> 이때, 회색 점선은 cv mse가 최소가 되는 지점

우측 그래프에서, <mark>색이 있는 predictor는 signal, 회색 predictor는 noises</mark>라고 한다.

이렇듯, 좌측의 그래프 처럼 cv 를 통해 $\lambda$를 예측하게 되면, signal과 noises를 찾을 수 있으며,<br>
이때의 $\lambda$를 선택하게 된다.
<br><br><br><br><br><br><br><br><br><br><br>

---

---

## 6.3.

## 6.4.


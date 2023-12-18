---
layout: post
title: ISLP chapter 4 정리
date: 2023-11-21 19:00 +0900
last_modified_at: 2023-11-21 19:00:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

# 4. Classification

categorical 데이터에 대해 예측하는 것

종류로는 다음과 같음

- logistic regression
- linear discriminant analysis
- Quandratic discriminant analysis
- naive Bayes
- K-nearest neighbors
- Poisson regression

그외에 트리기반 모델들은 chapter 7,8,9에서 배울 예정

## 4.1. An overview of Classification

regression과 마찬가지로, 데이터 셋이 있다면 (Supervised) 분류를 돌릴 수 있다.

## 4.2. Why Not Linear Regression?

1. non-binary

    category 데이터에 대해 상태 1을 1, 상태 2를 2, 상태 3을 3으로 인코딩할 경우, (binary가 아닐 경우)

    linear regression은 상태 2를 상태 1과 상태 3의 중간 상태로 두게 된다.

    그리고, 상태 1과 상태 2의 차이와 상태 2와 상태 3의 차이가 같다고 학습한다.

    실제로 그런 상황이면 상관x 그러나, 상태 1이 뇌졸중, 상태 2가 약물 중독, 상태 3이 간질 발작인 경우에는

    약물 중독이 뇌졸중과 간질 발작의 중간 상태가 아니게 되고,

    뇌졸중과 약물중독의 차이, 뇌졸중과 간질 발작의 차이가 같지 않으므로 이와 같이 regression을 하게 되면 안된다.

    또한, 만약 각 범주의 순서를 다르게 코딩해도 같은 결과가 나와야 하지만, regression에서는 불가능

2. binary

    물론, binary 하다면 dummy 변수의 방법으로 0과 1로 범주를 나누어 regression 할 수 있다.

    그러나 이 방법도 그다지 유의미하지는 않음

    그 이유로는 해석의 어려움이 있는데, XB^이 Pr((상태 1) | X) 의 예측값으로 해석될 수 있지만,

    ![Alt text](\..\img/image-4.png)

    이 값이 0 과 1 사이의 구간을 벗어나게 되는 경우가 생김

    결국 예측값이 상태의 순서를 만들게 되면서 추정치로 사용하기엔 조잡함

    이런 방법론은 4.4의 linear discriminant analysis와 같음

**요약하면 결국 regression을 통해 분류를 하면 안되는 이유는 2가지**

1. regression 방법론은 category 결과값이 3가지 이상이 되면 그다지 유용하지 않으며

2. category 결과값이 2가지일지라도, Pr(Y|X) 에 대한 의미있는 예측값을 제공하지 못하기 때문

## 4.3 Logistic Regression

logistic regression model은 단순히 Y가 Yes / No 를 예측하는 것이 아니라, 데이터를 통해 Y가 가질 값의 확률을 modeling 함

책에서는 Pr(Y = 1 | X) 를 p(X) 로 표현함

이때 p(X) 가 일정 수치를 넘을 경우, Y = 1이라고 한다.

이때 일정 수치 (threhold) 는 개인의 판단 기준에 따라 다르다.

### 4.3.1 The Logistic Model

그럼 우리는 p(X), 즉 Pr(Y = 1 | X)와 X의 관계를 어떻게 모델링할 것인가

앞서 말했듯이 regression을 사용하면 확률이 0과 1 사이에 존재하지 않을 수 있음.

이러한 문제를 해결하기 위해, logistic regression에서는 logistic 함수를 사용함.

![model 4.2](\..\img/image-5.png)

위의 모델을 학습하기 위해 **maximum likelihood** 를 사용하게 된다. (자세한 설명은 4.3.2)

이 모델을 사용하면 0과 1에 근접할 수는 있지만, 절대 0과 1 사이의 값을 벗어나지는 않는다.

이 모델은 항상 S 자 그래프를 그리면서 확률의 범위를 한정할 수 있고,

심지어 (확률의 평균) = (분포의 비율) 이다.

![Alt text](\..\img/image-6.png)

위의 모델을 좀 더 살펴보면 위의 방법으로 정리할 수 있다.

좌측항, p(X) / (1 - p(X)) 를 **odd**라고 부르며, 0과 inf 사이의 값을 가짐

이 odd 값이 0에 가까우면 확률이 매우 낮은 것이고, inf에 가까우면 확률이 매우 높다는 것

(1 이면 p(X) = 0.5)

![Alt text](\..\img/image-7.png)

위의 수식에 log를 씌우면 위와 같음

좌측항, log(p(X) / (1 - p(X)))은 **log odd** 혹은 **logit** 이다.

이 logit은 X에 대한 선형식이고, logistic model에 들어가 있다.

이때, B_1 은 linear regression에서는 X의 1단위 증가에 따른 Y의 평균 변화량으로 해석됐지만,

logistic regression에서는 

- X의 1단위 증가에 따른 **logit의 변화량**으로 해석
- X의 1단위 증가 -> **odd의 e^(B_1) 증가**

하지만 결국 그래프의 모양이 선형이 아니라서, B_1이 X의 단위 증가에 대한 p(X)의 변화량과 일치하다고 이야기할 수는 없다.

(X의 단위 증가에 대한 p(X)의 변화량은 현재 X의 값에 영향을 받는다.)

하지만 B_1의 부호에 따라 이야기할 거리는 있다.
- B_1이 양수라면, X가 증가할 때 p(X) 값도 증가
- B_1이 음수라면, X가 증가할 때 p(X) 값은 감소

### 4.3.2 Estimating the Regression Coefficients

비선형 least squares를 사용할 수 있지만, 보통은 **maximum likelihood**를 사용한다.

maximum likelihood를 사용할 때는 이런 식으로 접근한다고 생각하자

    logistic model을 이용해서,
    ^p(x_i)가 상태 0일 때 0과 최대한 가깝게 / 상태 1일 때 1과 최대한 가깝게 만드는
    B_0와 B_1에 대한 예측값을 찾을 거야

위의 접근법을 수학적으로 적으면 아래와 같다.

![Alt text](\..\img/image-8.png)

maximum likelihood는 비선형 모델을 학습하기 위한 보편적인 방법

(least square도 maximum likelihood의 특별한 상황 중 하나임)

하지만 대부분의 통계 tool에서는 이 함수에 대한 계산을 진행해주기 때문에 굳이 알 필요는 없고, 보고싶다면 책 맨뒤의 Scope를 참조

![Alt text](\..\img/image-10.png)

### 4.3.3 Making Predictions

4.3.2에서 예측한 계수를 바탕으로 X에 값을 집어 넣어 p(X)를 계산할 수 있다.

책의 예시에서 X가 1,000일땐 0.00576, 2,000일땐 0.586으로 계산되었다.

마찬가지로 정량적 변수가 아닌 정성적 변수를 모델에 넣어 학습할 수도 있다.

이 경우, dummy 변수를 생성해서 logistic regression을 진행할 수 있다.

책에서는 이를 통해 확를로 무언가를 계산하는 것보단, 계수의 부호와 p-value를 확인해서 해석하는 것으로 마무리했다.

### 4.3.4 Multiple Logistic Regression

지금까지 한거 : 1개의 변수, binary response를 가지는 경우에 대한 logistic regression

변수가 다양해지면 어떻게 될까? (multiple logistic regression)

multiple linear regression과 마찬가지로, 단순하게 logit에 변수와 계수의 곱 항이 늘어날 뿐임

![Alt text](\..\img/image-11.png)

결국 multiple logistic model의 식은 아래와 같음

![Alt text](\..\img/image-12.png)

똑같이 maximum likelihood 방식으로 각 계수를 예측함

![Alt text](\..\img/image-13.png)

위는 chapter 3에서 사용한 예제 데이터를 이용해 logistic regression을 진행한 결과임

책에서는 Student[Yes]의 계수 부호가 앞서 4.3.3에서 진행한 결과와 다르다는 점에 주목함

![Alt text](\..\img/image-14.png)

주황이 Student[Yes] = 1, 파랑이 Student[Yes] = 0

위의 결과에서 Student[Yes]의 계수가 음수였기 때문에, 다른 변수가 고정이라면 Student[Yes]가 1이라면 0일 때보다 낮아야 하고, 실제로 그래프에서도 그렇게 잘 나타났음

근데, horizontal broken line (모든 변수에 대한 p(X)의 평균을 그은 선) 은 그래프 상에서 Student[Yes] = 1일 때 더 높다고 나옴

우측의 그래프는 x 축이 Student, y 축이 balance인 boxplot

보다시피 Student와 balance 사이의 관계성이 있음(Student[yes] = 1일 때 더 큼).

balane는 또 default와 양의 관계

그래서, 다른 요소를 고정해둔 그래프에서는 Student[yes] = 1일 때 더 낮게 보이지만,

(Student[yes] 변수 하나만 고려하므로)

모든 변수를 고려한 평균에서는 Student[yes] = 1일 때 증가하게 된다.

(balance 변수가 Student[yes]에 영향을 주었으므로)

결국엔 학생 변수 하나로만 logistic regression을 진행할 때엔 학생의 체납이 더 많아 위험하다고 결론지었지만,

balance와 같은 연관 변수를 포함해서 logistic regression을 진행하면, **같은 balance에선 학생이 덜 위험하다** 라고 결론 지어진다.

linear regression 에서도 변수 1개와 2개 이상의 차이는 존재했음을 알 수 있다.

이러한 현상을 **confounding**이라고 부른다.

위의 표에서 나온 계수를 바탕으로 이제 변수를 집어 넣어 확률을 예측할 수 있다.

### 4.3.5 Multinomial Logistic Regression

지금까지 한 것
 - 1개의 변수, binary class를 가지는 logistic regression (4.2)
 - 2개 이상의 변수, binary class를 가지는 logistic regression (4.3.1 ~ 4.3.4)

이제 non binary class로 확장시켜보자

K개의 class로 나누어서 logistic regression을 진행한다고 가정하면 모델은 아래와 같다.

![Alt text](\..\img/image-16.png)

특이하게도 1 ~ K-1 개의 class에 대한 model과 K에 대한 model이 다른데, 이는 K번째 class를 baseline으로 잡았기 때문

1 ~ K-1 개의 class에 대한 model의 logit은 아래와 같다.

![Alt text](\..\img/image-17.png)

$log$ 안의 $Pr(Y = K | X = x)$은 $ 1 - Pr(Y = k | X = x)$와 같음

결국 logit은 우리가 아는 그 모양 그대로 나온다.

baseline을 어떻게 잡든간에, 같은 데이터를 활용해 학습을 한다면 계수나 모델의 결과는 같게 나온다.

하지만 해석을 할때는 주의를 해야하는데 baseline에 묶여서 해석이 되기 때문

예를 들어 baseline을 상태 K라고 잡은 상태로 모델을 만들었다면,

각 계수 $\beta_{kj}$는 $X_j$의 1단위 증가에 따라, 상태 $K$에 대한 상태 $k$일 확률이 $e^{kj}$ 증가하는 것으로 해석된다.

이러한 요소의 대안책으로 **softmax** coding이 있다.
 
    앞서 소개한 방법은 그대로 진행되나, 임의의 baseline K를 정하지 않는다.

    대신 모든 class K에 대해 동일하게 생각하고 수식을 아래와 같이 적는다.

![Alt text](\..\img/image-18.png)

    이러면 앞서 baseline을 제외한 K-1개의 class에 대한 계수를 학습하는게 아니라,

    K개의 class의 변수 모두를 학습하게 된다.

    chapter 10에서 더 자세하게 배움

## 4.4 Generative Model for Classification

여태 $X$에 대한 $Y$의 조건부 확률로 modeling 했음

이젠 각 $Y$에 대한 $X$의 분포를 modeling해보자

이 문제를 Bayes의 정리로 풀어야함.

각 class에 대한 $X$의 분포가 정규분포라고 가정한다면, 이 문제는 logistic regression과 비슷해진다.

왜 우리는 위의 이러한 방법이 필요하고, 또 언제 logistic regression을 사용해야하는가
- 2개의 class에 대해 상당한 수의 변수가 있을 경우, logistic regression은 굉장히 불안정하지만, 이 section에서 다룰 방법은 그렇지 않음
- 독립변수 X의 분포가 정규분포에 근접하고, 각 class에 대한 관측치 수가 적을 때, logistic regression보다는 이 section에서 다룰 방법이 더 좋다.
- 이 section에서 다를 방법은 자연스럽게 3개 이상의 class를 가지는 문제에 대입할 수 있다.

**가정 : $K$개의 class를 가지고 category인 종속변수 $Y$가 있음. 이때 $K\geq2$ 이다.**

이때 $\pi_k$는 Class k에 대한 사전 확률

    사전확률 : 베이즈 추론에서, 관측자가 관측하기전에 가지고 있는 확률 분포

그리고 $f_k(X)\equiv PR(X|Y = k)$이다. (class k에서 나온 density function of X)

-> k class에서 $X$가 $x$에 근사할 확률이 높을수록, $f_k(x)$가 비교적 높게 나올 것이다.

-> k class에서 $X$가 $x$에 근사할 확률이 엄청나게 낮다면, $f_k(x)$가 낮게 나올 것이다.

![Alt text](\..\img/image-19.png)

위는 Bayes의 정리를 통해 작성한 수식

(책에서는 $Pr(Y=k|X=x)$ -> $p_k(x)$로 축약해서 표현)

이때, $p_k(x)$는 사후확률 ($X$의 값이 $x$로 주어질 때, $Y$의 값이 $k$로 (관측 class가 $k$일) 확률)

위의 수식에서 보다싶이, 이전 regression에서 bayes를 사용할 때보다, $\pi_k$와 $f_k(x)$를 통해 좀더 수식을 간결하게 적을 수 있음.

$\pi_k$는 단순히 모든 데이터에서 class k 가 가지는 비율을 통해 사전확률을 유추할 수 있어서 쉬움

그러나 $f_k(x)$는 예측하기 어렵기 때문에, 각 방법 별로 가정을 가지게 됨

Bayes classifier인 아래의 3개를 배움

- linear discriminant analysis
- quadratic discriminant analysis
- naive Bayes

### 4.4.1 Linear Discriminant Analysis for $p = 1$

1개의 독립변수 만이 존재한다고 가정

$f_k(x)$를 가정해서 $p_k(x)$를 예측하고, $p_k(x)$가 가장 큰 class로 분류함

**가정**

    독립변수 X가 1개이다.
    
    $f_k(x)$는 정규 분포, 혹은 가우시안 분포이다.

    모든 class의 분산은 같다.

![Alt text](\..\img/image-20.png)

![Alt text](\..\img/image-21.png)

위의 식은 정규분포로 가정한 $f_k(x)$이고, 아래 식은 위의 식을 $p_k(x)$에 넣은 식

![Alt text](\..\img/image-22.png)

정리하면 다시 위의 식이 된다. 이 값이 가장 큰 class로 분류함.

![](\..\img/image-23.png)

그래프를 통해 LDA의 원리를 살펴보자면, 검은 점선을 기준으로 분류가 된다는 것을 알 수 있다.

물론, 모든 현실에서 데이터의 실제 분포를 모르고, 모수도 모르기 때문에 bayes classifier를 사용할 수 없다.

원래라면, 정규/가우시안 분포를 따른다는 가정에 맞다고 확신이 들더라도, Bayes classifier를 사용할려면 아래의 모수를 예측해야한다.

$\mu_1,\dots,\mu_K,\pi_1,\dots,\pi_K\;and\; \sigma^2$

하지만 LDA에서는 가정을 통해 $\mu_k,\;\pi_k,\;\sigma^2$만 사용해서 $\delta_k(x)$를 만들어 Bayes classifier와 비슷하게 만들어 냈다.

- $\mu_k$ : $\hat\mu_k$를 이용하며, 이는 k class의 샘플평균이다.
- $\pi_k$ : $\hat\pi_k$를 이용하며, 이는 $\frac{k\;class\;데이터의\;개수}{전체\;데이터의\;개수}$이다.
- $\sigma^2$ : $\hat\sigma^2$를 이용하며, 이는 k class의 샘플분산이다.

![Alt text](\..\img/image-24.png)

잘 보면 위의 식도 결국 x에 대한 선형식을 확인할 수 있다.

하나의 x에 대해 각 k를 통해 $\hat\delta_k(x)$의 값을 얻어서, 그 값이 가장 높은 k를 선택하는 것으로 분류한다.

이런 식으로 진행할 때에, Bayes classifier와 LDA의 error rate는 그다지 차이가 나지 않는 것도 확인할 수 있음

### 4.4.2 Linear Discriminant Analysis for $p>1$

LDA지만, 가정을 하나 지워보자

    $f_k(x)$는 정규 분포, 혹은 가우시안 분포이다.

    모든 class의 분산은 같다. (공분산 matrix가 같다.)

이 과정을 진행하기 위해 한번 보고 가야하는 개념이 `multivariate Gaussian`

$X=(X_1,X_2,\dots,X_p)$는 p개의 독립변수에 대한 set 

**multivariate Gaussian**

가정 : 각 독립변수는 정규분포를 가진다.

독립변수가 여러 개면 독립변수 간 상관관계가 존재할 수 있다.

이때 우리는 $X\sim\;N(\mu,\sum)$로 표현
- $\mu$는 X의 기대값 (평균)
- $\sum$은 공분산을 나타내며 $p\times p$의 행렬이다.


보통 다변량 가우시안의 density function은 아래와 같다.

![Alt text](\..\img/image-25.png)

위의 식은 앞서 본 $p = 1$ 일때의 $f_k(x)$에서 크게 확장된 버젼임

이를 정리하면 아래와 같다.

![Alt text](\..\img/image-26.png)

위의 식에 변수 $x$를 집어 넣을 때에 가장 큰 값을 가지는 $k$ class를 선택하게 된다.

![Alt text](\..\img/image-27.png)

좌측 그래프는 각 class의 분포 (95% 수준)를 그린 것

검은선은 모두 $\delta_k(x)$와 $\delta_l(x)$이 같은 지점을 의미한다.

즉, 검은 선을 기준으로 class를 나눌 수 있다.

마찬가지로, X = x 이라면, x를 $\delta_k(x)$에 넣어 값을 비교한 후, 가장 큰 값을 가지는 class k 로 분류

$\delta_k(x)$는 마찬가지로 x에 대한 선형식임

우측의 그래프는 20개의 sample을 찍어봄과 동시에 bayes classifier를 이용한 decision line이 그려져있다.

대부분 비슷한 것을 볼 수 있다.

다만 실생활에서 적용할 때에는 조심할 사항이 2가지
- overfitting
- null classifier : 분류하고자하는 class 간 데이터의 갯수 차이가 극명할 때, classifier가 하나의 class로만 예측하는 경우

그리고 계속해서 error rate로만 모델의 성능을 확인하지 않고 나아가 **confusion matrix**를 확인함

또한 sensitivity와 specificity의 개념에 대한 설명도 나옴, 모두 classifier의 내용에 대한 

- sensitivity : 실제 Positive 중에서 True Positive일 확률 
- specificity : 실제 negative 중에서 True negative일 확률

LDA의 경우에는 total error rate를 줄이는 방향으로 분류를 하게 됨 (Bayes classifier의 특성을 따라감)

따라서 어떤 class가 우리가 명백히 분류해내야하는 class인지 모르고 전체적인 error를 줄이기 때문

이러한 점을 해결하기 위해서 threhold를 조정할 수 있다. (binary의 경우에만)

threhold를 조정함에 따라, 기존 50%을 넘어야 분류를 하던 점에서 나아가, 우리가 명백히 분류를 해야하는 class에 대해서는 좀 더 넓은 범위에서 분류해낼 수 있기 때문

![Alt text](\..\img/image-28.png)

검은 선이 total error rate, 파란 dash가 중요한 class에 대한 오분류율

threshold가 줄어들수록 total error rate는 증가할 수 있지만, 중요한 class에 대한 오분류율은 떨어지는 것을 확인할 수 있다.

![Alt text](\..\img/image-29.png)

**ROC Curve (receiver operating characteristics) : 이진분류모델에 사용되는 하나의 모델평가지표**

좋은 ROC curve는 좌상단에 치우쳐있음 (False Positive rate가 낮을 때에 True Positive rate를 높인다는 의미이므로)

**AUC : (0,0), (1,1) 사이에서 ROC curve의 아래 영역** (적분 공간)

높을수록 좋음 (최대 1)

이 그래프를 threshold가 0.5일때와, 혹은 0.2일때를 각각 그려보면서 AUC를 기준으로 모델을 평가할 수 있다.

### 4.4.3 Quadratic Discriminant Analysis

LDA와 마찬가지로 QDA 또한 가우시안 분포를 이용해서 각 class당 $\pi_k$와 $f_k(x)$를 만들어서 $\delta_k(x)$의 값을 통해 분류를 진행하는 것은 같다.

근데 가정에서 등분산성을 깬다.

    가정

    - 각 class의 분포가 가우시안 분포이다.

즉 각 class k 마다 하나의 공분산 matrix, $\sum_k$를 가진다.

다시 표현하면 4.4.2에서는 $X\sim N(\mu_k,\sum)$이었지만,

4.4.3 에서는 $X\sim N(\mu_k,\sum_k)$이다.

![Alt text](\..\img/image-30.png)

그럼 $\delta_k(x)$는 위와 같이 수식이 적힌다. 이 값이 가장 큰 class k로 분류하게 된다.

수식을 보면 더이상 $\delta_k(x)$은 x에 대한 선형식이 아니게 되고, 2차식이 된다. (그래서 이름도 Quadratic DA)

그럼 왜 여태 우리가 한 LDA를 냅두고 QDA를 배우고 써야하는가

그 이유는 bias-variance trade-off 때문이다.

LDA는 유연하지 못하다. 선형적이기 때문에 결국엔 기타 분류 모델에 비해서 높은 성능을 기대하기 어렵다.

그러나 LDA는 적은 데이터의 수에서도 충분한 성능을 낼 수 있다.

하지만 QDA는 LDA보다 유연하다. 2차식이기 때문에 LDA보다는 조금 더 dicision line을 그릴 때에 유연함이 발휘된다.

그렇지만 QDA는 다른 유연한 모델이 그러하듯, variance가 높다. 즉, 예측해야할 param가 많아지기 때문에 전체적인 모델의 분산이 커지게 된다.

공분산 matrix의 값을 예측한다고 생각했을 때, 일단 1개의 공분산 matrix에 $\frac{p(p+1)}{2}$개의 요소가 들어간다. 

그걸 k개가 있으니 예측해야할 파라미터는 $k\frac{p(p+1)}{2}$개다. 기존 LDA에서 $\mu_k,\;\sigma_k,\;\pi_k$ 이렇게 예측하던것과는 차이가 난다.

![Alt text](\..\img/image-31.png)

### 4.4.4 Naive Bayes

이전 우리는 LDA와 QDA를 통해 bayes' theorem을 이용했다. ($p_k(x)$, $\pi_k$, $f_k(x)$ 등등)

이중에서 $\pi_k$는 training data 중에서 class k의 비율을 통해 간단하게 예측할 수 있었다.

그러나 $f_k(x)$는 어렵다.

LDA에서는 가정을 많이 넣어서 이 작업을 쉽게 진행했다. ($\mu_k$, $\sum$을 모수로 가지는 다변량 정규분포로)

QDA에서는 가정 중 일부를 풀어 이 작업을 진행했다. (($\mu_k$, $\sum_k$을 모수로 가지는 다변량 정규분포로))

둘다 어쨌든 가정을 넣어서 density function이라는 것을 풀기보다, 분포로서 이를 풀려고 노력했다.

`naive Bayes classifier`는 density function인 $f_k(x)$를 풀기위해 하나의 가정을 가진다.

    가정 : 모든 클래스 k에 대해서, 모든 독립변수 p는 독립이다.

이렇게 가정하면 $f_k(x)$는 다음과 같다.

![Alt text](\..\img/image-32.png)

$f_{kj}(x)$는 class k에서, j 번째 독립변수가 가지는 density function을 의미한다.

원래 우리가 density function을 계산하기 어려웠던 이유가 1. marginal 분포(주변확률분포)와 2. joint 분포(결합확률분포)인데 독립이라면 이부분이 없음

물론 현실에 대입해서, 모든 X가 독립이라고 가정을 하기는 어렵지만, 모델의 사용성을 위해서 사용한다.

그리고 특정상황에서는 결과가 또 괜찮게 나오기도 한다.

그렇기 때문에 `Naive Bayes`는 넓은 범위를 아우르는 좋은 선택지가 되기도 함.

![Alt text](\..\img/image-33.png)

Naive Bayes를 통해 예측한 $f_k(x)$를 $p_k(x)$에 넣으면 위의 수식이 된다.

이 수식을 풀려면 몇 가지를 생각해야한다.

- 만약 $X_j$가 수치형 변수라면, 우린 $X_j|Y=k\sim N(\mu_{jk},\sigma^2_{jk})$ (정규분포)로 예측한다
- 만약 $X_j$가 범주형 변수라면, 그냥 데이터의 비율에 따라 $f_{kj}(x_j)$를 만들면 된다.

![Alt text](\..\img/image-34.png)

각각 1인 경우가 32개, 2인 경우가 55개, 3인경우가 13개

책에서는 이후, Naive Bayes를 이용해서, 3개의 변수, 2개의 class를 가질 때의 분류를 시행하고자 함.

![](\..\img/image-35.png)

각각 $\hat f_{11}(0.4) = 0.368$, $\hat f_{12}(1.5) = 0.484$, $\hat f_{13}(1) = 0.226$,     $\hat f_{21}(0.4) =
0.030$, $\hat f_{22}(1.5) = 0.130$, $\hat f_{23}(1) = 0.616$

이를 이용해서 만든 $p_k(x)$에 $x^*=(0.4,1.5,1)^T$를 넣어 예측함

결과는 Y = 1일 때 0.944, 2일 때 0.056

Y가 1로 분류됨

![Alt text](\..\img/image-36.png)

위 classifier의 confusion matrix

LDA와 마찬가지로, threshold를 통해 분류 정도를 적용할 수 있음

Naive bayes는 feature vector가 매우 크고(독립변수가 굉장히 많고) , 데이터의 갯수가 적을 때에 더 강력한 성능을 보임

## 4.5 A Comparison of Classification Methods

### 4.5.1 An Analytical Comparison

pass

###
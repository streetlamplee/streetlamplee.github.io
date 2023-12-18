---
layout: post
title: ISLP chapter 5 정리
date: 2023-11-28 19:00 +0900
last_modified_at: 2023-11-28 19:00:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

# 5. Resampling Methods

resampling method는 training set에서 sample을 뽑고, 이를 모델에 재학습을 시키면서 추가적인 정보를 얻을 수 있도록 한다.

기본적으로 resampling 방법은 비싸다 >> 같은 모델을 여러 번 학습시켜야 하기 때문

하지만 최근 컴퓨팅 성능의 향상으로 인해 굳이 금지되어야 할 정도도 아니다.

이 chapter 에서는 가장 보편적인 방법 2가지를 배운다.

- cross-validation : 통계적 학습방법의 성능을 평가하기 위해 (test error를 예측) (model assessment)/ 적당한 자유도를 찾기 위해 사용 (model selection)

- bootstrap : parameter 예측값 or 주어진 통계적 학습 방법의 정확도 수치를 알기 위해 사용

## 5.1. Cross-Validation

test error : 새로운 데이터(학습에 사용되지 않은 데이터)에 대해, 통계적 학습 방법을 통해 나온 예측값의 평균 error

test error가 낮으면 우리는 그 학습 모델이 검증되었다고 할 수 있다.

물론 test error가 이미 준비되어 있는 상황이라면 이를 쉽게 계산할 수 있지만, 주로 그렇지 않다.

그러나 train error는 모델을 학습한다. > train data가 있다. > error를 계산한다와 같이 간단하게 계산되어 질 수 있다.

train data로만 학습한 모델은 test data에 대해서는 제대로 된 예측을 할 수 없다는 점을 이미 배웠다. (chapter 2)

이렇게 제대로 평가하기 위한 큰 test data가 없는 상황에서도, 가용한 train data를 이용해서 test error를 계산할 수 있다.

그중 몇몇 방법은 training error에 장난을 쳐서 test error로 계산하는 것인데 이건 chapter 6에서 배우도록 하고,

여기서는 test error를 **train data 중 일부를 학습에 사용하지 않는 방법**을 이용해서 구해보도록 하자. (**holding out**)

c.f. 5.1.1 ~ 5.1.4 까지는 정량 변수를 이용한 regression을, 5.1.5는 category 변수를 이용한 분류를 활용한다. (**정량이든 category든 상관이 없다.**)

### 5.1.1. The Validation Set Approach

![Alt text](\..\img/image-37.png)

Validation Set approach는 단순하게 사용가능한 training data를 2개의 set으로 나누는 것을 의미한다.

이때 하나의 set은 training set, 나머지 하나의 set은 validation set (hold-out set)이라고 부른다.

모델을 학습할 때에는 training set을, 모델을 평가할 때에는 validation set을 활용한다.

**이때 validation set을 이용해 error rate를 내게 되면, 그 값이 test data에 대한 error rate의 예측값이 된다.**

![Alt text](\..\img/image-38.png)

위는 chapter 3의 data를 이용해 1:1로 train set과 validation set을 나누어서 Polynomial Linear Regression 모델을 만들고, 이를 평가한 그래프이다.

x 축은 Polynomial Linear Regression의 차수, y 축은 validation MSE이다.

    #### 왼쪽 그래프

    책에서 중점으로 다룬 관점은 아래와 같다.

    1. 차수가 1일 때(Simple Linear Regression) 보다 차수가 2 이상일 때 더 모델이 좋음을 알 수 있다. (MSE가 낮으므로)
    2. 차수가 3 이상일 때에는 차수가 2일 때보다 좋다고 평가할 수 없다. (MSE가 높거나 같으므로)

    #### 오른쪽 그래프

    각 선은 각기 다른 방법으로 train set과 validation set을 나누어 모델을 만들고, validation set을 이용해 test error rate를 예측한 그래프

    x 축은 Polynomial Regression의 차수, y축은 validation MSE

    책에서 중점으로 다룬 관점은 다음과 같다.

    1. 각기 다른 train, validation으로 학습하고 평가되어서, 같은 선이 존재하지 않는다.
    2. 모든 모델에서, 1차보다, 2차에서 성능이 좋아진다. (MSE가 감소하는 pattern이 동일)
    3. 모든 모델에서, 3차 이상의 모델이 2차보다 좋다고 판단할 수 없다. (MSE가 감소하는 pattern이 일정하지 않음)
    4. 이 모델들 중에서, 무슨 모델이 가장 좋다고 판단하는 것은 문제가 있다.

validation approach는 간단하고, 쉽지만 단점이 있다.

1. test error rate를 예측하기 위해 사용한 validation MSE (혹은 다른 평가지표)의 분산이 너무 크다. (어떤 data가 train, validation에 들어가는지에 따라 달라짐)
2. 보통의 통계적 학습 방법은 train data의 수가 적을수록 성능이 나빠진다. (기존의 train data를 train/validation으로 나누었으므로)

위의 2가지 단점을 보완하기 위한 방법을 아래에서 살펴보자

### 5.1.2 Leave-One-Out Cross-Validation (LOOCV)

validation set approach와 정말 유사하지만 단점을 보완하는 방법

위에서는 train과 validation을 일정한 비율을 이용해서 나누었지만, `여기서는 validation set에 단 1개의 data만 할당한다.`

n개의 training data가 있다면, n-1개를 학습하고 1개를 이용해 맞추는 셈

![Alt text](\..\img/image-39.png)

1번째 데이터를 validation set에 넣게 되면, $MSE_1=(y_1-\hat y_1)^2$ 이며, 이때 $MSE_1$은 unbiased 이다. (1개의 값만을 가지므로)

그렇다고 이게 완벽하지만은 않은게, $MSE_1$이 validation set에 들어간 data가 뭔지에 따라 굉장히 가변적이기 때문이다.

결국 이때 $MSE_1$은 validation set에 들어간 그 1번째 데이터에 영향을 크게 받게 된다.

그후, 2번째 데이터를 이용해 $MSE_2$를 계산한다. 식은 $(y_2 - \hat y_2)^2$

이렇게, n번째 데이터를 이용해 $MSE_n$까지 계산한다. 식은 $(y_n - \hat y_n)^2$

이때 나온 $MSE_i$를 산술 평균을 내어 test MSE의 예측값으로 한다.

$CV_{(n)}=\frac{1}{n} \sum_{i=1}^{n}{MSE_i}$

![Alt text](\..\img/image-40.png)

    LOOCV 의 장점 (validation set approach 비교)

    1. bias가 굉장히 적다.

        1개를 뺀 나머지 data에 대해 모두 학습하기 때문에, validation set approach보다 훨씬 bias가 작다.

        결과적으로 test error rate를 과대평가하지 않게된다.

    2. random성이 없다.

        validation set approach에서는 우리가 랜덤으로 train / validation을 나누었기 때문에, 반복해서 진행하면 그 값이 계속 바뀌게 된다.

        하지만 LOOCV는 각 1개의 data만을 뽑아서 반복하기 때문에, 다시 반복을 해도 그 값이 같다.

        (model이 학습하는 data, 모델이 평가하는 data가 동일하기 때문)

    LOOCV의 단점

    1. 비싸다

        여기서 비싸다는 의미는 사용되는 컴퓨팅 성능, 시간을 의미

        각 model을 데이터의 개수만큼 반복해서 학습하고, 예측해야하므로 진행하는데에 많은 시간이 소요된다.

        즉 데이터가 크면 클수록, 사용하기에 부담스러워진다.

$h_i = \frac{1}{n} + \frac{(x_i-\bar x)^2}{\sum_{i^\prime = 1}^{n}{(x_{i^\prime}- \bar x)^2}}$

$CV_{(n)}=\frac{1}{n} \sum_{i=1}^{n}{\left( \frac{y_i - \hat y_i}{1-h_i} \right)^2}$

linear regression에서 배운 영향도 $h_i$를 이용해서, LOOCV가 data가 클때 느려지는 것을 조금 완화할 수 있다.

위의 두 식은 least square linear, polynomial regression에서 사용가능하다.

위의 $CV_{(n)}$을 이용하면, 모델 1개로 돌리는 시간과 같아진다.

위의 $\hat y_i$는 OLS 모델의 $i$번째 예측값과 같다.

$h_i$는 leverage (영향도)

$CV_{(n)}$는 기존의 MSE와 같지만, 추가적으로 i 번째 잔차가 나뉘어진 것. ($1-h_i$)

$h_i$는 $1/n$ 와 1사이에 존재하는 값이고, 이는 i 번째 data가 모델의 학습에 얼마나 영향을 주었는지를 나타낸다.

영향도가 높은 data에 대한 잔차가 증가함에 따라, $h_i$도 증가하면서 해당 식이 효용성을 가지게 된다.

LOOCV는 일반적인 방법이고, 예측 모델 대부분에 사용가능하다.

### 5.1.3. k-Fold Cross-Validation

LOOCV의 다른 대안책.

![Alt text](\..\img/image-41.png)

train data를 대략 비슷한 크기의 무작위 k개의 그룹으로 만든다. (`fold`)

이 때 1번째 그룹을 validation set으로 간주하여 나머지 k-1개의 data 그룹을 이용해 모델을 학습하고, $MSE_1$을 1번째 그룹으로 만든다.

이렇게 k번 반복

결과로서 k개의 $MSE$가 생기게 된다.

결국, k-fold CV는 test error rate의 예측값으로 아래의 식을 사용한다.

$CV_{(k)}=\frac{1}{k} \sum_{i=1}^{k}{MSE_i}$

결국 k-fold cv의 k가 데이터의 개수 n과 같으면 LOOCV라는 것을 알 수 있다.

LOOCV보다 k-fold가 나은 점은 컴퓨팅 성능을 덜 먹는다는 것이다.

k-fold는 거의 모든 상황에서도 사용할 수 있다. (모델이 무엇이든, 데이터의 크기가 얼마든)

![Alt text](\..\img/image-42.png)

실제 test MSE가 파랑, LOOCV가 검은 점선, 10-fold cv가 오렌지

모든 그래프에서 LOOCV와 10-fold cv의 그래프가 비슷하다.

특히, 우측 그래프에서는 true MSE와 cv 값이 모두 비슷하다고 볼 수 있다.

중간 그래프에서는 낮은 자유도에서는 true MSE와 cv 값이 비슷하고, 높은 자유도에서는 더 높게 평가한다.

죄측에서는 일반적인 모양을 그리지만, true MSE를 더 낮게 예측한다.

================================================================

Cross-Validation을 하면서, 통계 모델이 얼마나 다른 data에 효용이 있는지 알고 싶은 것이 목표

그래서 결국은 true MSE를 예측한 값을 중점으로 봐야한다.

근데 어떨때는 우리는 `test MSE의 예측값의 최솟값`을 볼 수 있어야한다.

이는 `우리가 어떤 모델을 결정해야할지를 결정해야하는 상황`일때에 그렇게 한다.

그렇기 때문에 위의 그래프에서 최소값인 위치를 확인할 수도 있어야한다.

물론 모델이 true test error rate를 조금 낮게 예측하더라도, true test error rate가 최소가 되는 값은 엇비슷하게 잡아내는 것을 알 수 있다.

### 5.1.4. Bias-Variance Trade-off for K-fold Cross-Validation

5.1.3에서 우리는 k-fold cross-validation이 LOOCV보다 컴퓨팅 방면에서 더 좋다는 것을 알수 있었다.

근데 컴퓨팅 성능을 제쳐두고서라도, LOOCV보다 더 정확한 test error rate를 내놓을 수도 있다.

이 관점은 bias-variance trade-off에서 설명이 가능하다.

5.1.1에서 validation set approach에서 test error rate를 너무 높게 예측한다는 것을 배웠다.

이는 겨우 절반의 data를 이용해서 학습하고 나머지를 예측해서 MSE를 계산해 예측값을 만든다.

이 관점에서, LOOCV는 1개씩 처리하면서 계산을 하므로, unbias하다는 것을 알 수 있다.

k-fold cv는 적정한 수준의 bias를 유지하기 때문에, LOOCV와 validation set approach의 중간 수준이다.

그럼 bias의 관점에서는 LOOCV가 k-fold cv보다 선호될 것이다.

그러나 우리가 예측할때 하나더 고려해야할 요소로서 variance가 있다.

이 때, LOOCV가 k-fold cv보다 훨씬 높은 variance를 가진다.

이는 LOOCV가 각각의 data에 대해 모두 계산을 하기 때문에, 이러한 결과는 쉽게 상관관계를 가질 수 있다.

그러나 k-fold cv는 k개의 모델을 평균을 내기 때문에 덜 관계성을 가지게 된다. 

관계성이 높을수록 결국 variance까지 높아지게 된다.

결국 LOOCV의 test error rate 예측값은 k-fold test error rate보다 variance가 높다.

```
결국, k-fold cv에서 k의 값을 결정하는 것은 bias-variance trade-off가 발생하게 된다.

k의 값을 결정하면서, 높은 bias / 높은 variance를 대조하게 된다.
```

### 5.1.5. Cross-Validation on Classification Problems

이 chapter에서는 예측값이 category인 경우를 설명한다.

위에서는 정량 변수였기 때문에 MSE를 사용했지만, category일 때에는 그냥 `잘못 분류된 data의 개수`를 사용한다.

식으로 적게 되면 아래와 같다.

$CV_{(n)} = \frac{1}{n} \sum_{i=1}^{n}{Err_i}$

$where\;Err_i = I(y_i \neq \hat y_i)$

![Alt text](\..\img/image-43.png)

위의 그래프는 logistic regression을 이용한 분류이며, 각각 polynomial하며 1,2,3,4 이다.

이 데이터는 simulated 되어있기 때문에 실제 데이터의 분류를 알 수 있다. (bayes)

검은선은 모델의 분류선이며 각각 polynomial의 차수가 오를수록, 정확도가 높아진다.

![Alt text](\..\img/image-44.png)

실제 데이터에서는 우리가 실제 데이터의 분류선을 알수 없기 때문에, polynomial을 사용하더라도 차수를 얼마로 설정해야하는지, 실제 test error rate와 같은 부분을 정하기 쉽지 않다.

이러한 경우, cross-validation을 이용해서 예측할 수 있다.

    노랑 : test error

    파랑 : training error

    검정 : 10-fold cv error
    
    좌측 그래프

    x : polynomial의 차수

    y : error rate

    우측 그래프

    x : KNN에서 1/K의 값

    y : error rate

우리가 여기서 봐야할 것은, test error인 노랑과 10-fold error 인 검정색이다.

좌측 그래프에서는 10-fold error가 꽤나 잘 예측한 것을 볼 수 있다.

물론 실제로는 차수가 3이 더 작지만, 그래도 비슷한 차수인 4를 선택한 것을 볼 수 있다.

(실제로 데이터에서 3차 4차 5차 6차의 error rate의 차이는 미미하다.)

우측의 K 는 KNN에서 사용된 neighbor의 수

마찬가지로 error rate를 더 적게 예측하긴 했지만, k의 값은 근접하게 선택했다.

이는 training error는 test error와는 무관하게, 모델이 복잡할 수록 error rate가 떨어지므로 training error로는 모델을 선택할 수 없는 문제를 k-fold cv와 같은 cross-validation으로 선택이 가능한 것을 볼 수 있다.

## 5.2. The Bootstrap

모델의 불확실성을 측정하기 위해 사용한다. (ex. linear regression의 계수의 표준편차 계산)

사실 단순한 모델의 불확실성은 R과 같은 통계모델에서 계산을 해주지만, Bootstrap을 이용하면 더 복잡한 경우에도 사용할 수 있다.

    예시

    우리 돈을 상품 X와 상품 Y에 나누어서 투자한다고 가정 (a : 1-a)

    이때, risk가 가장 적은 a는 아래 수식과 같다.

![Alt text](\..\img/image-45.png)

$where\;\sigma^2_X=Var(X),\sigma^2_Y=Var(Y),\;and\;\sigma_{XY}=Cov(X,Y)$

그렇다곤 해도, 우리는 Var(X), Var(Y), Cov(X,Y)의 값을 모른다.

그렇기 때문에, 우리는 각 값을 예측해서 $\hat \sigma^2_X, \hat \sigma^2_Y,\hat \sigma_{XY}$를 사용하게 된다.

100개의 시뮬레이션을 통해 각 예측값을 얻을 수 있었고, 이를 통해 $\hat \alpha$를 구할 수 있었다.

이것의 정확도는 이 작업을 1000번 반복해서 나오는 standard deviation(표준편차)를 계산해서 정확도를 얻을 수 있다.

이런식으로 계산을 할 수 있지만 이 작업은 `시뮬레이션`된 것이라는 것을 알 필요가 있다.

현실에서 우리가 가진 data를 통해, 시뮬레이션을 돌리는 것은 불가능하다.

그렇기 때문에 Bootstrap을 사용.

bootstrap은 `원래의 데이터에서 반복적으로 복원추출하는 것`을 의미한다.

![Alt text](\..\img/image-46.png)

각 bootstrap set에서 나온 $\hat \alpha$를 $\hat \alpha^{*i}$이라고 한다면, $\hat \alpha$는 아래의 식으로 계산된다.

![Alt text](\..\img/image-47.png)

![Alt text](\..\img/image-48.png)

실제로 bootstrap set을 이용하면 거의 비슷한 수준의 $\hat \alpha$를 얻을 수 있다.
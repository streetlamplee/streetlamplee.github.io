---
layout: post
title: ISLP chapter 2 정리
date: 2023-10-24 19:00 +0900
last_modified_at: 2023-10-24 19:00:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

## 2. Statistical Learning
### 2.1 What Is Statistical Learning?

예시 : ‘광고’ 데이터를 통해 ‘매출’을 예측하거나 추론 할 수 있음
이 경우, ‘광고’ 데이터는 독립변수 (independent variable), ‘매출’은 종속변수 (dependent variable)이라고 할 수 있음
또한 우리는 종속변수 Y가 독립변수 X 마다 다른 영향을 받음을 알 수 있고 이를 가장 간단하게 표현한다면 선형식을 통해 작성할 수 있음
선형식이 아니더라도 Y = f(X) + e 를 대부분 만족함
#### 2.1.1 Why Estimate f?
예측
f를 알 수 있다면, Y^ = f^(X)를 이용해서 Y값을 예측할 수 있다.
그러나 이 경우 오차가 발생할 수 있고, 이는 1. 제거할 수 있는 error, 2. 제거 불가능한 error로 나눌 수 있음

제거할 수 있는 error : error가 특정한 분포를 띄고, 평균이 0이 아닌 경우, Y를 결정하는 새로운 변수가 있다고 생각, 이를 도출하여 새로운 독립변수를 만들어 error를 제거할 수 있음
제거할 수 없는 error : e를 통해서 표현할 수 있으며, 어떠한 독립변수에 의해 발생하는 error가 아님. 또한 정성적인 변수에 의한 error일 수 있음

따라서 결국 f를 제대로 예측한다고 해도 X를 통해 Y에 대한 완벽한 값을 알 수 없음. (추정치는 가능)js/dist

추론
p개의 독립변수 X1, X2, … , Xp 와 종속변수 Y의 관계를 이해하고자 할 때도 f 추론이 필요. 단, 이 경우 f의 형태를 명백히 추론해야함 (선형인지 비선형인지 등)

Y에 영향을 주는 변수 X는 무엇인지
각 변수 Xi 와 Y의 관계는 무엇인지 (양의 상관관계, 음의 상관관계)
각 변수 Xi 와 Y의 관계가 어떤 모양인지 (선형 or 비선형)

#### 2.1.2 How do We Estimate f?
모수적 방법 (Parametric Methods)
아래의 2단계 방법을 포함
먼저 f의 모양 (functional form)을 가정
예를 들어 선형모델의 경우, f(X) = b0 + b1*X1 + … + bp * Xp의 함수모양을 가질 것이라는 가정
이 경우, 추정해야하는 모수는 b0 ~ bp의 (p + 1)개
가정한 f의 모양에 따라 training data를 fit.
가장 common 방법 :  최소제곱법(최소자승법, OLS : Ordinary Least Square)
이외에 많은 방법이 있음

	
장점 : 
이 방법은 f의 모양을 가정하고 모수를 추정함으로써 굉장히 간단하게 사용할 수 있다

단점 : 
가정한 f의 모양이 진짜 f의 모양이랑 다를 가능성이 존재하고, 이 경우 f를 예측한 결과가 그다지 좋지 않을 수 있음.
좀 더 일반적인 경우를 파악할 수 있는 ‘flexible model’을 이용할 수 있으나, 더 많은 모수를 추정해야한다

비모수적 방법 (non-Parametric Methods)
f에 대한 가정을 하지 않음. 대신 data point가 너무 이상하지 않을 정도로만 f를 추정함
이를 통해 조금 더 넓은 범위로 train data를 fit할 수 있음
대신 적은 수의 모수를 예측하기 힘들고, 많은 관측치가 필요로 함
다만 명백한 모수가 없기 때문에, smoothness를 조정하지 않을 경우 overfitting의 위험이 있음

#### 2.1.3 The Trade-off Between Prediction Accuracy and Model Interpretability
선형 모델 : 직선/평면 모양으로 이해하기 쉬움, 그러나 정확도는 떨어짐
flexible : 곡면 모양으로 직관적인 f 의 모양을 이해하기 어려움, 정확도는 높음

**Model Interpretability : 모델 이해도**

선형 모델처럼 더 restrictive한 모델을 이용하고자하는 이유는 모델에 대한 이해도가 높아 ‘추론’을 하기 더 쉽기 때문임

Least Squared 선형 회귀 : 조금 제한적인 대신 모델 이해도가 더 높음
Lasso : 선형 모델과 비슷하지만 fitting 시 선형 회귀와는 다른 방법을 사용해서 더 제한적임. 대신 모델의 이해도는 매우 높다

GAMs(Generalized Addictive Models) : 선형 모델에서 비선형 관계를 허용함으로써 정확도는 높이지만 모델 이해도는 떨어짐

bagging, boosting, SVM (완전한 비선형 모델들) + neural network (Deep Learning) : 진짜 flexible해서 정확도는 높지만 이해하기가 힘들다

결국 목적에 따라서 모델을 선택할 수 있어야함
- 추론 : 각 독립변수와 종속변수의 관계를 파악하기 위해서 모델 이해도가 높은 모델을 사용하는 것이 좋음 (조금 더 restrictive한 모델 : 선형 모델 등)

- 예측 : 관계 파악에는 관심 없으므로 정확도를 높이기 위해 flexible한 모델을 사용하는 것이 좋음 ( boosting, neural network 등)

그러나 overfitting의 위험이 있기 때문에 완전히 flexible한 모델보단 조금은 restrictive 한 모델을 선택할 수도 있다

#### 2.1.4 Supervised VS Unsupervised Learning
Supervised (지도학습) : 관측치 xi 가 존재하고, 이에 대한 결과 yi가 존재해서 이를 모델에 학습시키는 방법
Unsupervised (비지도학습) : 관측치 xi는 존재하지만, 모델이 학습할 수 있는 결과는 없음.
당연히 독립변수에 대한 결과 (종속변수) 가 존재하지 않으므로 예측은 할 수 없음
각 독립변수에 대한 관계를 이해하는 수준으로 진행됨 (ex. clustering)
(여기서 관계의 이해는 변수끼리 어느정도 비슷한지 정도고 명확한 특징을 모델 단계에서 잡아내지 못함)
-> 사실 그럼 그냥 변수별로 그래프에 점찍어서 비슷한지 아닌지 보면 되지 않나요?
변수가 n개면 n(n-1) /2 개의 2차원 그래프에 점 찍어서 비교할 수 있으면 해라
그래서 clustering과 같은 비지도학습을 사용
 
-> 200만 개의 데이터에서 100만 개에만 예측치가 있으면 어떻게 되나요?
semi-supervised learning problem : 예측치를 가져오는 비용이 상대적으로 비싸면 나오는 상황임
이후에 어떻게 할지 알아보자

#### 2.1.5 Regression VS Classification Problems
변수는 정성적 변수와 정량적 변수로 나뉨 (정성적 변수 : categorical)
정량적 변수/종속변수는 regression로 처리할 것이라 생각하지만 사실 regression은 정성적 변수를 통해 classification으로 풀 수도 있음 (구분이 그리 좋진 않음)
Least Squared linear regression: 정량적 변수와 사용됨
Logistic regression : 종종 정성적 변수 (categorical)과 사용될 수 있음
그래서 logistic regression은 분류의 방법으로 분류되지만, 각 class의 확률을 예측하기때문에 regression의 방법으로도 생각할 수 있음
K-nearest neighbors, boosting : 정성적, 정량적 종속변수 모두 사용 가능하다.

결국 변수가 정성적인지 정량적인지 확인하고 어떤 방법을 쓸 것인지 정하는게 아님
(결국 코딩을 통해 정성적 변수를 처리가능함)

### 2.2 Assessing Medel Accuracy
	특정 data set에서 어떤 statistical learning method를 고를지 결정하는 것을 알아보자
#### 2.2.1 Measuring the Quality of Fit
얼마나 관측치에 대해 잘 예측하는지를 평가할 수 있어야 한다.
regression에서는 가장 많이 사용되는 방법이 mean squared error (MSE)
error (관측치에서 예측치를 뺀 값) 을 제곱한 값의 평균으로 계산
MSE가 작으면 작을수록 잘 예측했다고 판단할 수 있음

MSE를 이용해서 model을 fit하는 수단으로 사용할 수 있으나, 이때의 MSE는 train data를 통해 나온 train MSE 이므로 모델을 선택할 때에는 test data를 이용한 MSE를 통해 모델을 선택할 수 있도록 한다. >> test MSE가 작은 모델 선택
(이 때 test data는 train data에 포함되지 않은 전혀 새로운 data를 의미한다)
단, 대부분은 statistical learning method가 training MSE를 최소화하는 방법으로 계수들을 예측하기 때문에, training MSE가 작다고 test MSE가 작다는 보장은 없다.
책에서는 예시로 smoothing spline method를 사용함
smoothness를 통해 flexible을 조정해 본 결과, 가장 flexible한 모델의 train MSE가 가장 작음
그러나 이 모델은 overfitting 되어서 test MSE는 커짐.
결론적으로 flexibility가 특정 수치를 넘어서면 test MSE가 높아지는 것을 볼 수 있음
**degree of freedom : flexibility를 나타냄**
이는 대부분의 data set과 statistical learning method에서 통용됨
overfitting : train MSE는 매우 작은 대신 test MSE가 큰 상황
**대부분의 statistical learning method는 train MSE가 test MSE보다 작다**


다만 df가 커질수록 train MSE는 계속해서 감소하는 대신, test MSE는 U 모양을 그리며 감소하다가 증가하는 걸 볼 수 있다.
(이때, test MSE의 변곡점이 가장 좋은 model의 df 다)

따라서 우리는 training data를 통해 가장 작은 test MSE를 예측하는 cross-validation이라는 걸 배울거다. (ch.5)

#### 2.2.2 The Bias-Variance Trade-off
test MSE가 U 모양을 그리는 것은 2개의 요소때문에 그렇다.
variance : statistical model의 varience (statistical model의 df가 높을수록 varience가 크다)
variance가 클수록 data의 변화에 대해 예측값이 급격하게 움직인다.

bias : 실제 문제에 근접함으로써 생기는 편차
실제 현실문제가 선형이 아님에도 선형으로 가정한다면 bias가 생기게 된다.
따라서 간단한 모델을 사용할수록 bias는 커지게 된다.
그리고 MSE = varience + (bias) ** 2 + var(e)의 식을 가진다.

varience와 bias는 0 이상의 값을 가지고 (분산이라서) 
일반적으로 df가 커질수록 (flexible할수록) varience는 커지고 bias는 작아진다.

어떤 모델의 df를 증가시킨다면 varience는 1차항의 수준으로 줄어들고 bias는 2차항의 수준으로 줄어들기 때문에 test MSE가 작아질것이라고 예상할 수 있다.

그러나 어느 수준에 이르러서는 bias가 작아지는 것에 대한 영향이 줄어들고, varience가 커지는 것에 대한 영향이 커져서 df가 증가할수록 test MSE가 증가할 것이다.

결국 df가 커질수록 bias와 varience 사이에는 상충 관계가 생기게 된다.
작은 bias와 작은 varience를 가지는 df를 찾을 수 있도록 해야한다.

#### 2.2.3 The Classification Setting
2.2.1과 2.2.2에서는 정량적인 data에 대한 정확도를 이야기했음
그럼 정성적인 data (categorical data) 에서는? error rate를 사용함
error rate : 관측치에 f^ 를 적용할 경우 잘못 분류할 확률
결국 모델을 학습할 때에는 train error가 최소가 되도록 학습할 것이고
이렇게 학습된 모델들을 선택할 때에는 test error가 최소가 되는 모델을 선택할 것임

The Bayes Classifier
분류 class가 적을 경우 사용할 수 있다.
조건부 확률로써, predictor value x0이 주어질 때, reponse value Y가 나타날 확률이 가장 높은 Y를 test 관측치로 두어서 test error를 계산할 수 있다.
이 관측치가 어느 수준 (class가 2개일 경우 50%)을 기준으로 두는 가에 따라 Bayes decision boundary가 만들어진다.
이렇게 만들어진 error rate는 Bayes error rate라고 불리고 가장 작은 test error rate가 될 수 있다.

K - Nearest Neighbors
Bayes 분류가 정확하기에 현실에서 잘 쓰일 수 있다면 좋겠지만, 현실에서 우리는 주어진 관측치에 대한 조건부 확률을 구하기 어렵다.
따라서 Bayes 분류는 다른 방법들과 비교하기 위한 가장 이상적인 방법으로 치부함

주어진 X에 대한 Y의 분포를 예측하고, 가장 예측된 확률이 높은 수준으로 각 class에 맞게 관측치를 분류하는 것으로 많은 방법이 시도되었음
그중 하나가 K - Nearest Neighbors (KNN)
임의의 data, x0를 결정하고 그 point와 가까운 K개의 train data 들을 묶어 N 으로 표현하며, N에 대해서 j에 포함될 조건부 확률이 가장 큰 방향으로 분류함

K 값을 잘 조정하면 심지어 Bayes 분류와 비슷한 성능을 보인다.
K의 값이 작을수록 overfitting, 클수록 pattern을 무시하며 boundary가 직선형
(K가 작을수록 high-varience, low-bias, K가 클수록 low-varience, high-bias)

KNN도 train error 와 test error의 상관관계가 없음
다만 regression과 마찬가지로 varience가 높으면 train error는 - , test error는 + 경향
	책에서는 1/K로 표현해서 1/K가 커지면 (K가 작아지면) varience가 작아진다 라는 식으로 표현
	마찬가지로 test error 가 U 자 모양을 가짐

**regression 이든 classification 이든 결국 적정 수준의 flexibility 를 선정해서 varience-bias trade-off를 해결하는 것이 중요하다!** (ch5)

### 2.3 Lab : Introduction to Python
#### 2.3.1 Getting Started
파이썬 관련 설치법

#### 2.3.2 Basic Commands
간단한 명령어에 대한 설명
int, float, str 처리법, print, list 등을 사용하는 것을 확인

#### 2.3.3 introduction to Numerical Python
내장 라이브러리인 numpy를 활용하는 방법
np.array() : 행렬
np.array().ndim : 행렬의 차원 출력
np.array().dtype : 행렬 내의 요소 type 출력
np.array().shape : 행렬의 모양 출력
np.array().sum() : 행렬의 모든 요소 합
np.array().reshape(tuple) : 행렬의 모양을 변경
np.array().reshape[int] : index에 맞는 요소 값을 출력
np.array().reshape[int] = a : index에 맞는 요소 값을 a 값으로 update
** reshape가 된 array에 대해서 요소 값을 update할 경우, reshape 되기 전의 array도 요소 값이 update됨 ⇒ reshape를 하더라도 array는 같은 메모리에 저장된 상태이기 때문

np.array().T : transpose 행렬로 변환
np.sqrt(np.array()) = array의 모든 요소의 제곱근 값
np.random.normal(loc = float1, scale = float2, size = int) : 평균 float1, 분산 float2인 정규분포에서 int개의 random data를 만들어서 array로 출력
np.corrcoef(x, y) : x와 y의 상관계수를 행렬 (array)로 출력
np.random.default_rng(int) : random seed를 고정
np.mean(array) , np.var(array), np.std(array) : array의 평균, 분산, 표준편차 출력
**분산을 처리할 때 ddof가 n으로 계산함**
np.array().mean(int) : int 행의 평균을 구함

#### 2.3.4 Graphics
	matplotlib 라이브러리를 사용 (from matplotlib.pyplot import subplots)
	이하 내용은 ipynb에서 처리


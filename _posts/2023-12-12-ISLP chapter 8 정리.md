---
layout: post
title: ISLP chapter 8 정리
date: 2023-12-12 19:00 +0900
last_modified_at: 2023-12-12 19:00:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

# 8. Tree-Based Methods

Tree-Based Methods는 회귀 분류 모두 사용이 가능하다.

구역을 나누어서 (segmenting) 평균/최빈값을 이용한다.

## 8.1 The Basics of Decision Trees

### 8.1.1 Regression Trees

classification 처럼 tree method를 이용해서 나누되, 그 구역의 평균값을 예측값으로 설정

![Alt text](\..\img/image-49.png)

tree의 상위 terminal에서 나뉜 것일수록, 더 중요한 feature라고 해석할 수 있음.

이러한 tree method는 예측을 하기에는 너무 간단하지만, 해석이 매우 쉽다는 점이 용이하다.

```
예측하기

1. 주어진 p개의 feature를 이용해서, 겹치지 않게 구역을 j개로 나누게 된다.

2. 만약 j번째 구역으로 데이터가 들어오게 된다면, 그 데이터의 reponse 값은 항상 동일하다.

(그 구역 내, 관측된 response 값의 평균)

```

이 때, tree method의 목표는, RSS를 최소로 가지게 하는 구역을 나누는 것이다.

$RSS = \displaystyle\sum_{j=1}^J \displaystyle\sum_{i\in R_j}(y_i - \hat y_{R_j})^2$

$where\; \hat y_{R_j} = (mean\;response\;for\;the\;training\;observation\;within\;jth\;box)$

컴퓨팅 성능을 너무 많이 잡아먹으므로, recursive binary splitting을 한다.

top-down의 방향으로, 2개의 가지로 분화하게 되는데, 이는 임의의 level에서 다른 좋은 성능을 가지는지를 구분하지 않고, 그 level에서 제일 좋은 split만을 고려한다. (greedy)

그러면 우리는 그 특정한 split을 어떻게 고려해야하는가

j번째 feature를 s에서 나눈다고 note하면,

$\displaystyle\sum_{i:\;x_i \in R_1 (j,s)} (y_i - \hat y_{R_1})^2 + \displaystyle\sum_{i:\;x_i\in R_2(j,s)}(y_i-\hat y_{R_2})^2$

위의 식을 최소로 하는 $j$와 $s$를 찾아야한다. -> 각 region에서의 RSS의 합이 최소가 됨

위의 식은 feature의 수가 그리 크지 않으면 금방 계산이 된다.

위의 방법을 반복하지만, 첫 번째처럼 모든 dataset을 기준으로 나누지 않고, 이미 앞에서 한번 나뉘어진 dataset을 이용해서 나누게 된다.

**가지 치기**

이 방법은 좋은 예측 성능을 기대할 수 있지만, overfitting의 위험이 있다.

tree가 조금 덜 split되었다면, 오히려 낮은 variance와 나은 해석력을 가지게 된다.

조금 덜 split되게 할려면, split을 할 때에 낮아지는 RSS값에 임계점을 줘서 그 임계점을 넘을때만 split하게 하면 된다.

근데 이 방법도 처음에 나뉘는 split이 그리 좋지 않다면 오히려 중간에 멈춰버려서 별로다.

그래서 그냥 우리는 완전한 tree를 만들고, 너무 complex해지는 시점에서 가지를 치는 방법을 이용한다.

(다만 모든 subtree에 test error rate의 하락을 확인하기는 어려우므로, `Cost complexity pruning (weakest link pruning)`을 이용한다.)

```
Regression Tree 만드는 법

1. recursive binary splitting을 한다. (leaf에 일정량의 data가 있을 때까지)

2. cost complextiy pruning을 한다. ($\alpha$를 이용해서)

3. $\alpha$를 선택하기 위해 k-fold cv를 진행한다.

    a) k-1개의 fold에 1, 2의 작업을 수행
    b) MSE를 계산한다.

    이중에서 MSE가 최소로 되는 $\alpha$를 선택한다.

4. 선택된 $\alpha$을 이용해서 subtree를 선택한다.
```

### 8.1.2 Classification Trees

response가 category이다.

그러므로 각 구역의 평균을 예측값으로 가지지 않고, 각 구역에서 가장 많이 나타난 값을 response로 가지게 된다.

regression tree와 마찬가지로 recursive binary split을 진행하지만, RSS가 아니라 `classification error rate`를 이용하여 나누게 된다.

$Classification\;error\;rate = 1 - max_k(\hat p_{mk})$

$where\;\hat p_{mk} = mth\;region에서\;kth\;class의\;비율$

근데 classification error rate는 tree를 만들기에는 적합하지 않음 (나뉘는 것에 비해 값의 차이가 크게 움직이지 않음)

그래서 쓰는게 `Gini index` (지니계수) 이다.

$Gini\;index=\displaystyle\sum_{k=1}^K \hat p_{mk}(1 - p_{mk})$

(k 번째 class가 포함될 확률 $\times$ 포함되지 않을 확률의 합)

불순도를 의미하기도 하는데, 일단 기본적으로 어느 한쪽의 class의 비율이 높다면, 0과 1로 가까워지게 되므로, 이 값이 낮을수록, node가 순수하다는 것을 의미한다.

비슷하게 쓰이는 요소로 `Entropy` (엔트로피)도 있다.

$Entropy = - \displaystyle\sum_{k=1}^K \hat p_{mk} \log \hat p_{mk}$

$\hat p_{mk}$가 0과 1사이의 값이므로, $- \hat p_{mk} \log \hat p_{mk}$은 0 이상의 값을 가지게 된다.

엔트로피 또한, 낮으면 낮을수록 node가 순수하다는 것을 의미한다.

따라서 classification tree에서는 classification error rate보다는 `Gini계수`나 `엔트로피`를 사용한다.

### 8.1.3 Tree Versus Linear Models

tree 모델과 linear model 중에서 무엇이 더 나을까?

그건 경우에 따라 다르다.

feature와 response의 관계가 선형이라면 물론 linear model이 더 좋게 나타날 것이다.

비선형이라면 tree model이 더 나을 것이다.

결국 케바케다


### 8.1.4 Advantages and Disadvantages of Trees

장점

- 선형회귀보다 설명하기 쉽다.

- 회귀나 분류보다 tree method가 실제 사람이 선택하는 것과 유사한 방식을 띈다

- 더미 변수 없이 매우 쉽게 정량 변수를 다룰 수 있다.

단점

- tree는 다른 회귀나 분류 방법과는 정확도면에서 차이가 있다.

- 너무 예민하다 (작은 변화에도 최종 tree가 크게 변할 수 있다.)

위의 단점을 보강하기 위해, bagging이나 RF, boosting을 이용해서 tree의 성능을 조금 더 올릴 수 있다.

## 8.2 Bagging, Random Forest, Boosting and Bayesian Additive Regression Trees

위의 방법은 트리기반 앙상블 모델이다.

### 8.2.1 Bagging

통계적 학습 방법의 분산을 줄이는 대표적인 방법

일반적으로 우리가 모델 분산을 줄이고 test 정확도를 높이는 방법은 train data를 크게 늘리는 것

그러나 일반적으로 다수의 train set을 얻기 어렵지만, 우리는 bootstrap을 통해 train data에서 sample을 많이 뽑을 수 있다.

그리고 이를 단순히 평균을 취해 예측값으로 삼는다.

가지치기 (제약이 없는) tree는 분산이 높고 bias는 적다. 이걸 몇 백개씩 만들어서 평균을 취해 성능의 향상을 노릴 수 있다.

classification에서는 각 tree의 예측값을 평균을 내는 것이 아니라, major 투표를 통해 가장 자주 나타나는 class로 분류할 수 있다.

**Out of Bag Error 예측**

bootstrap을 통해 부분집합을 만들 경우, 그 집합에 포함되지 않은 data를 통해 예측 성능을 파악하는 방법 (test error 예측)

**variable importance measures**

1개의 의사결정나무는 해석하기 쉽지만, 이를 많이 붙여놓은 vagging model은 알아보기 힘들다.

그러나 하나의 predictor에 대한 RSS / Gini index를 계산해서 중요도를 요약할 수 있다.

* RSS : 이 predictor를 사용해서 split 될 경우, 떨어지는 RSS의 값을 측정
* Gini index : 이 predictor를 사용해서 split 될 경우, Gini index가 떨어지는 경우

### 8.2.2. Random Forests

bagged tree의 개선

각 tree를 만들때 bootstraped data set을 이용하고, `추가적으로 tree의 split에서 predictor 또한 random 으로 sampling된다.` 보통 전체 predictor 개수의 제곱근

각 predictor가 상관관계가 있을 경우, predictor의 random sampling으로 인해 이러한 점을 더 보완할 수 있다. (decorrelating)

상관관계가 있는 predictor가 많으면 많을수록 random sampling의 크기를 줄이는 편이 좋다.

### 8.2.3 Boosting

bagging이랑 마찬가지로, tree를 연속적으로 쌓는다. 쌓는데, 독립적인 tree가 아니다. (이전 tree의 정보를 받아 학습한다)

각 tree는 기존 data에서 이전 tree의 학습내용이 적용된 data를 받아 학습하니까 bootstrap도 필요가 없다.

앞의 tree가 data를 받아 RSS (혹은 gini) 를 줄이는 방향으로 학습을 한다.

그후, 그 tree에서 나온 잔차 (residual)을 다음 tree가 학습한다. (0이 되도록)

이를 반복하고, $\lambda$를 통해 천천히 학습하도록 유도한다.

![Alt text](\..\img/image-50.png)

parameter 3가지

1. tree의 개수 (n_estimators)
2. shrinkage parameter (learning rate)
3. depth

### 8.2.4 Bayesian Additive Regression Tree (BART)

bagging과 boosting의 특징이 동시에 있음

각각의 tree가 기존 data를 통해 학습되지만, 각 tree가 이전 tree의 residual을 통해 학습된다.



### 8.2.5 Summary of Tree Ensemble Methods

* bagging : data에 random sampling을 한다. 그리고 그 subset마다 하나의 tree를 만들어 학습시키고, 이 예측값들의 평균(최빈값)을 사용한다.
* random forest : data에 random sampling을 한다. 그리고 트리를 만들 때, 각 split 마다 predictor를 random sampling 한다. 이 tree들의 예측값들의 평균(최빈값)을 사용한다.
* boosting : 기존 data를 이용하지만, 첫 tree가 학습한 내용에서 잔차를 받아 이를 최소화하는 tree를 계속해서 쌓는다. 이를 weight sum을 해서 합치는 방법
* BART : 기존 data를 이용해서 연속적으로 학습한다. (boosting 처럼) 하지만 다른 model처럼 local minima에서 멈추지 않고, 추가적인 과정을 통해 space를 탐색하면서 추가적인 minima가 있는지 확인한다.

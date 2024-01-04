---
layout: post
title: ISLP chapter 12 정리
date: 2024-01-02 12:09 +0900
last_modified_at: 2024-01-02 12:09:00 +0900
tags: [Statistics, ISLP]
toc:  true
---

## 12.2 Princpal Components Analysis

### 12.2.1

### 12.2.2

### 12.2.3

### 12.2.4 More on PCA

#### Scaling the Variable

PCA를 수행하기 전에는 변수의 중심 (평균)이 0이 되도록 하고, 결과도 변수별로 스케일링에 따라 다르다

책의 예시에서는 데이터의 크기가 줄어들게 되면, 편차가 변동될 수 있기 때문에<br>
PCA를 하기 전에 표준편차가 1이 되지 않도록 해야한다.

#### Uniqueness of the Principal Component

이론적으론 component가 유일하지 않아도 되는데, 대부분의 실제 환경에서는 유일하다.

$\rightarrow$software package는 같은 벡터를 산출할 수 있으나, 그 부호는 다를 수 있다.

부호를 바꾸는 것은 결국 방향에는 아무런 변화를 안주기 때문이다.

마찬가지로 score vector 또한 부호의 변동에 대해선 unique하다.

#### Deciding How Many Principal Components to Use

visualize를 하기 위해 몇 개의 principal components를 써야할까

*scree plot*을 검정하면서 알아볼 수 있다.

![Alt text](\..\img\12.1.png)

데이터의 분포를 설명할 수 있는 적절하지만, 가장 최소의 principal components의 개수를 골라야한다.

elbow를 고르면 되는데, 이는 위의 그림에서 2번째 component 까지는 설명하는 데이터가 크게 늘지만,<br>
3번째 부터는 설명을 크게 못한다는 것으로 이해할 수 있다.

다만 이 방법은 임시방편이고, 아직까지 얼마나 어떻게 principal component를 선택해야하는지는 불분명하다.

다만 PCA로 데이터의 특정 패턴을 찾는 일이므로, 도메인과 데이터에 크게 영향을 받는다.

하지만 특정 component에 대해 패턴이 나오냐 안나오냐 차이로 component를 선택할 수 있어서, EDA에서는 일반적인 방법론으로 꼽힌다.

---

만약 우리가 component의 개수를 알고 있다면, 그냥 PCA의 principal component score vector의 개수를 회귀의 tuning parameter로 생각하고, CV와 같은 방법을 쓰면 된다.

supervised에서 component의 개수를 고르는게 단순한 것은 supervised가 unsupervised보다 더 잘 정의되고, 객관적인 경향임을 보여준다.

### 12.2.5 Other Uses for Principal Components

n x M 행렬을 가지기 때문에 component score vector를 파라미터로 해서 회귀를 할 수 있다.

## 12.3 Missing Values and Matrix Completion

matrix에 결측치가 있다면?

해당 행을 지우는 것은 결측 비율과 데이터의 중요도에 따라 다르다.

아니면 $x_{ij}$가 결측이라면, $j$의 평균으로 해당 값을 처리할 수 있다.

하지만 변수 간의 상관관계를 이용해서 더 나은 방법을 구현할 수도 있다.

---

**matrix completion**

통계적 학습 방법론에서 사용할 수 있는 결측치 처리 방법

결측치가 완전한 random이면 효과적이다.<br>
ex. 체중계가 고장나서 결측됨 $\rightarrow$ random<br>
ex. 체중계에 못 오를만큼 무거워서 결측됨 $\rightarrow$ not random

#### Principal Component with Missing Values

data matrix $X$의 요소인 $x_{ij}$가 결측되었다고 가정하자.

이때에, 결측치를 처리하면서, principal component 문제를 한번에 푸는 수식은 아래와 같다.

$\underset{A \in R^{n \times M}, B \in R^{p \times M}}{minimize} \Biggl\{ \displaystyle \sum_{(i,j) \in O} \left( x_{ij} - \sum^M_{m=1} a_{im}b_{jm}\right)^2 \Biggr\}$

$where\;O=(모든\;관측된\;pair의\;indices)$

위의 식을 풀게 되면
1. $\hat x_{ij} = \sum^M_{m=1} \hat a_{im} \hat b_{jm}$을 이용해 결측값 $x_{ij}$를 예측한다.
2. 대략적으로 M개의 principal component score를 풀게 된다.

위의 식을 완벽하게 푸는 것은 당연히 어렵다. 대신 위의 식을 푸는 것과 비슷한 효과를 보이는 알고리즘이 있다.

![Alt text](\..\img\12.2.png)

먼저 결측치를 평균으로 채워넣는다.

12.13 식을 풀고, $\overset{\sim}{x_{ij}} \leftarrow \sum^M_{m=1} \hat a_{im} \hat b_{jm}$을 설정

이후, 12.13 식을 풀게된다.

위의 순서를 12.14가 값이 더이상 떨어지지 않을 때까지 처리

이후, 책의 예제를 이용해서 설명

100개의 결측치 포함 데이터 vs. 20개의 완전한 데이터

$\begin{aligned}\hat x_{ij} = z_{i1} \phi_{j1} \newline where \;z{i1} &= (첫번째 principal component score의 요소) \newline \phi_{j1} &= (완전한 데이터의 loading vector) \end{aligned}$<br>를 계산하게 된다.

$0.63 \pm 0.11 versus 0.79 \pm 0.08$

완전한 데이터의 표준편차가 적기 때문에 더 낫긴하지만, 100개의 결측치 포함 데이터로도 충분히 괜찮게 계산할 수 있음을 볼 수 있음

일반적으로 위의 알고리즘을 이용하기 위해서는, M (principal component의 숫자)를 결정해야했다. 이부분은 앞서 배운 CV를 이용해 결정할 수 있다.

#### Recommender Systems

ex. 넷플릭스에는 i번째 고객이 j번째 영화에 평점을 준, 거대한 i x j 행렬이 있었다. 하지만 대부분의 고객이 17,000여개의 영화 중 200여 개를 보았다.<br>
즉, 대부분이 결측치였다.

이러한 상황에서 넷플릭스는 다음과 같이 처리했다.

> i번째 고객이 본 영화 세트는 다른 고객이 본 영화 세트와 겹칠 것이다.
>
> 또한, 다른 일부분의 고객들은 i번째 고객과 비슷한 영화 취향을 가질 것이다.
>
> 그러므로, i번째 고객이 보지 못한 영화를, i번째 고객과 비슷한 고객들의 평점을 이용해 예측할 수 있을 것이다.

따라서 i번째 고객과 j번째 영화라면,

$\hat x_{ij} = \sum^M_{m=1} \hat a_{im} \hat b_{jm}$을 계산해서 풀 수 있다.

이때, ixj인 matrix M은 '클릭'과 '장르'의 관점에서 해석할 수 있다.<br>
cf. 클릭 (clique)는 그 영화를 재밌게 본 고객들의 집합

* $\hat a_{im}$ 은 i번째 유저가 m번째 장르에 얼마나 속하는지 (포함되는지) 정도라고 해석한다. 
* $\hat b_{jm}$ 은 j번째 영화가 m번째 장르에 얼마나 속하는지 (포함되는지) 정도로 해석한다.



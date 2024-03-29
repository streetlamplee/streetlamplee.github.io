---
title: ML. 카테고리와 기타 변수 다루기
layout: post
date: 2024-01-11 10:00 +0900
last_modified_at: 2024-01-11 10:00:00 +0900
tag: [ML, data postprocessing]
toc: true
---

## 카테고리와 기타 변수 다루기

### 1. 연속형 변수 다루기

#### 1.1 함수 변환

**로그 변환** (log transform)

$x \rightarrow \log x$

비대칭된 임의의 분포를 정규 분포에 가깝게 전환시키는데에 도움이 된다.

데이터의 스케일을 작게 만들어 데이터간 편차를 줄이는 효과 (큰 값은 많이 줄이고, 작은 값은 적게 줄인다)

특징
* 비대칭 분포가 제거되는 것을 확인
* 이상치 완화에도 효율적이지만, 0이나 음수에 적용할 수 없음

---

**제곱근 변환** (square root transform)

변수에 제곱근을 취해서 변환

로그 변환과 효과가 비슷하지만, 로그 변화과의 차이점은 데이터 편차의 강도

$\therefore$ 비대칭(Right-skewed)이 강한 경우 로그 변환, 약한 경우 제곱근 변환이 유리

단, 왼쪽으로 치우쳐진 데이터 분포에 활용 시 더 치우치는 부작용 발생

---

**거듭제곱 변환** (power transform)

변수에 제곱을 취해서 변환

---

**Box-Cox 변환** (Box-Cox Transform)

임의의 하이퍼파라미터 $\lambda$를 이용해서 변환

$\tilde{x} = \displaystyle \begin{cases} \displaystyle \frac{x^{\lambda} - 1}{\lambda} & if \lambda \neq 0, \newline ln(x) & if \lambda = 0 \end{cases}$

목적에 맞게 최적 $\lambda$를 찾아야 한다.

---

#### 1.2 스케일링

동일한 수치 범위로 변경하는 방법

ex. 키는 100cm ~ 190cm, 몸무게는 40kg ~ 110kg으로 각 변수의 수치 범위가 다른 경우를 맞춰줌

수치 범위가 다르게 존재하면 종속 변수에 각기 다르게 영향을 미친다.

특히 수치 범위가 큰 변수일수록 다른 변수에 비해 더 크게 종속 변수에 영향을 준다.

*필요한 이유*

KNN은 벡터간 거리를 측정해서 데이터를 분류하는 방식

변수들이 동일한 범위로 스케일링이 안되어 있다면, 결과가 올바르지 않은 리스크가 존재할 것이다.

* 수치 범위가 다르게 존재하면 종속 변수에 다르게 영향을 미침
* 수치 범위가 큰 변수일수록 다른 변수에 비해 더 중요하게 인식될 수 있음

---

**Min-Max 스케일링**

연속형 변수의 수치 범위를 0~1 사이로 변환하는 방법

$\tilde{x} = \displaystyle \frac{x - min(x)}{max(x) - min(x)}$

다만 min과 maz가 무엇이냐에 따라 이상치에 취약한 부분이 존재한다.

---

**표준화**

변수의 수치 범위를 평균이 0, 표준 편차가 1이 되도록 변경 (Z-score)

$\tilde{x} = \displaystyle \frac{x - mean(x)}{sqrt(var(x))}$

---

**로버스트 스케일링** (robust scaling)

IQR을 기준으로 변환

$\tilde{x} = \displaystyle \frac{x - IQR_2}{IQR_3 - IQR_1}$

평균 대신 중앙값을 활용해서 이상치에 강건한 효과

#### 1.3 구간화

수치형 변수를 범주형 변수로 전환시키는 방법

데이터가 범주화되기 때문에 학습 모델의 복잡도가 줄어드는 장점이 있다.

ex. 나이 데이터 $\rightarrow$ 10대, 20대, 30대

특징
* 등간격(동일 길이), 등빈도(동일 개수)로 나누어 구간화를 진행
* 범주로 통일되어서 이상치를 완화
* 데이터의 구분이 가능해져 데이터 및 모델 해석에 용이

### 2. 범주형 변수 다루기

#### 2.1 One-hot Encoding

범주 변수를 0과 1로 구성된 이진 벡터 형태로 변환하는 방법 (binary)

ex.<br>
|종목 번호|종목|
|:------:|:------:|
|32|네이버|
|95|카카오|
|356|라인플러스|

One-Hot Encoding<br>
|종목 번호|네이버|카카오|라인플러스|
|:---:|:---:|:---:|:---:|
|32|1|0|0|
|95|0|1|0|
|356|0|0|1|

* 장점1 : 변수의 이진화를 통해 컴퓨터가 인식하는 것에 적합
* 장점2 : 알고리즘 모델이 변수의 의미를 정확하게 파악가능

* 단점1 : 고유 범주 변수의 크기가 늘어날 때마다 희소 벡터 차원이 늘어나는 문제점이 존재
* 단점2 : 벡터의 차원이 늘어나면 메모리 및 연산에 악영향
* 단점3 : 차원의 저주 발생

#### 2.2 레이블 인코딩

각 범주를 정수로 표현한다.

ex.<br>
|종목 번호|종목|
|:------:|:------:|
|32|네이버|
|95|카카오|
|356|라인플러스|

Label Encoding<br>
|종목 번호|종목|
|:------:|:------:|
|32|0|
|95|1|
|356|2|

하나의 칼럼으로 모든 변수를 표현가능하지만, 순차적 데이터에서 더 효과적이다.

* 장점1 : 범주 당 정수로 간단하게 변환 가능
* 장점2 : 하나의 변수로 표현 가능해서 메모리 관리 측면에서 효율적

* 단점1 : 순서가 아닌 값을 순서로 인식할 수 있는 문제 발생

#### 2.3 빈도 인코딩

고유 범주의 빈도 값을 인코딩 (= Count Encoding)

빈도가 높을수록 높은 정수값을, 빈도가 낮을수록 낮은 정수값을 부여받는 형태

ex.<br>
|날짜|종목|
|:------:|:------:|
|01-01|네이버|
|01-01|카카오|
|01-02|라인플러스|
|01-02|라인플러스|
|01-02|네이버|

Frequency Encoding<br>
|날짜|종목|
|:------:|:------:|
|01-01|2|
|01-01|1|
|01-02|2|
|01-02|2|
|01-02|2|

* 장점1 : 빈도라는 수치적인 의미를 변수에 부여가능
* 장점2 : 하나의 변수로 표현이 가능해서 메모리 관리 측면에서 효율적

* 단점1 : 빈도가 같은 경우, 다른 변수도 같은 변수로 처리할 수 있다.<br>
이러한 문제를 해결하기 위해, feature1에 label encoding을 하고, feature2로 frequency encoding을 할 수 있다.

#### 2.4 타겟 인코딩

특정 타겟 변수를 통계량(평균)으로 인코딩하는 방식(=Mean Encoding)

ex.<br>
|날짜|종목|시가|
|:------:|:------:|------:|
|01-01|네이버|64,000|
|01-01|카카오|24,000|
|01-02|라인플러스|14,000|
|01-02|라인플러스|16,000|
|01-02|네이버|65,000|

Target Encoding<br>
|날짜|종목|
|:------:|:------:|
|01-01|64,500|
|01-01|24,000|
|01-02|15,000|
|01-02|15,000|
|01-02|64,500|

* 장점1 : 범주간 수치적인 의미를 변수에 부여
* 장점2 : target 변수의 추가적인 정보를 가진 변수에 의존해서 추가된 정보를 알고리즘에 입력 가능
* 장점3 : 하나의 변수로 표현 가능해서 메모리 관리 측면에서 효율적

* 단점1 : target 변수에 이상치가 존재하거나 범주의 종류가 소수라면 과적합 가능성이 있다.
* 단점2 : 학습과 검증 데이터를 분할하면, 타겟 변수 특성이 이미 학습 데이터 셋에서 노출되어서 Data-Leakage 문제가 발생<br>
$\because$ Data Split 이전에 평균을 구해 인코딩이 선행되므로

---

타겟 인코딩의 과적합 방지

* smoothing
* K-Fold
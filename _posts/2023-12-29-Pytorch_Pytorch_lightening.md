---
layout: post
title: Pytorch 파이토치 라이트닝
date: 2023-12-29 12:29 +0900
last_modified_at: 2023-12-29 12:29 +0900
tag: [deeplearning, pytorch, pytorch_lightening]
toc: true
---

# 파이토치 라이트닝

## 1. Pytorch Lightening 소개

### 1.1 Pytorch Lightening

#### 배경

구현하는 코드의 양이 늘어나며, 코드의 복잡성이 증가
<br>다양한 요소들이 복잡하게 얽힘

- 데이터 전처리
- 모델 구조
- 학습 및 평가 루프
- 결과 시각화

서로 강하게 관계성을 가지고, 한 부분을 변경하면 다른 부분에도 영향을 줌

---

**Pytorch Lightening** : Pytorch에 대한 높은 수준의 UI를 제공 (Opensource lib)
<br>딥러닝 구축의 코드 템플릿

### 1.2 주요 특징

#### 코드의 추상화 및 하드웨어 호출 자동화

*코드의 추상화* : 복잡한 로직을 간단한 UI 뒤에 숨기는 것을 의미

> 기존 Pytorch
>
> model, optimizer, training loop를 따로 구현

**in Pytorch Lightening**<br>
Lightening Module 클래스 안에 모든 것을 한 번에 구현하게 되어 있음

---

#### 다양한 콜백 함수와 로깅

다양한 내장 콜백 함수를 지원

> ex. 초기 lr를 자동으로 찾아주거나, early stop의 기등을 코드 한줄로 구현

또한, 다양한 로깅 도구를 지원하여 로깅해야할 값을 편리하게 기록,<br>
Tensorboard, WandB 등 모니터링 툴을 쉽게 사용 가능

---

#### 16-bit precision

딥러닝 모델의 크기가 대체로 큰 경향 $\Rightarrow$ 모델 전체를 GPU에 로드하기에 제한적

일반적으로 DL모델에서 float형은 32bit인데, 이를 16bit로 줄여 속도 향상 및 메모리 사용량을 줄임

pytorch lightening에서는 16-bit precision과 같은 복잡한 기능 또한 옵션으로 추가할 수 있음

## 2. LighteningModule

### 2.1 LighteningModule 소개

pytorch lightening을 사용하기 위한 class

lightening module 클래스를 상속 받아,
- 모델의 구조
- loss function
- 학습 및 평가 방법
- 최적화 알고리즘

을 클래스에 선언해야함

$\Rightarrow$ 모델 구조와 학습 로직을 함께 class로 선언해서, 코드 구조가 명확하고, 코드의 재 사용성을 향상함

---

**구성**

- `__init__` : <br>
초기화를 담당<br>
모델의 레이어를 **초기화**한다.<br>
$+$ 학습 및 평가 과정에서 사용되는 **loss function**, **metric**을 선언 가능
- `forward` : <br>
모델을 통해 데이터가 연산 되는 과정 의미
- `configure_optimizers` : <br>
**최적화 알고리즘**과 **학습률 스케쥴러**를 정의 및 반환<br>
반환할 때에는, `return [optimizer], [scheduler]`<br>
순서를 맞춰야함<br>
*학습률 스케쥴러* : DL모델 학습 동안, 학습률을 동적으로 조정하는 역할

- `training_step` : <br>
**미니 배치에 대해 손실을 반환하는 과정 정의**<br>
*`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` 안 적어도 됨*

- `validation_step` :<br>
validation set의 **미니 배치에 대한 모델의 성능**을 확인

- `test_step` : <br>
test set의 **미니 배치에 대한 모델의 성능**을 확인

- `predict_step` : <br>
추론해야하는 데이터 셋의 **미니 배치에 대한 예측 과정을 정의**<br>
ex. 입력에 대한 모델의 예측값 반환, 확률값 반환

## 3. Trainer

### 3.1 Trainer 소개

**LighteningModule의 method를 이용해 모델학습을 실행**하는 class

```python
# Basic use
model = Classifier(num_classes = 10, dropout_ratio = 0.2) # lightningmodule

trainer = Trainer(  max_epochs = 100,
                    accelerator = 'auto',
                    callbacks = [callbacks.EarlyStopping(monitor = 'valid_loss', mode = 'min')],
                    logger = CSVLogger(save_dir = '(your path)', name = 'test'))

trainer.fit(model, train_dataloader, valide_dataloader)
```

복잡한 환경에서도 학습 환경을 자동으로 관리해준다.

### 3.2 Trainer의 mothod

- `.fit()` : <br>
LighteningModule (model), train_dataloader, valide_dataloader를 인자로 받음<br>
이때, LighteningModule의 `training_step`, `validation_step`, `configure_optimizer`를 이용

- `.validate()` : <br>
내부적으로 `validation_step` 호출<br>
완료되면, validation set에 대한 metric 출력

- `.test()` : <br>
내부적으로 `test_step` 호출<br>
완료되면, test set에 대한 metric 출력

- `.predict()` : <br>
모델의 결괏값을 반환받음<br>
내부적으로 `predict_step`을 호출
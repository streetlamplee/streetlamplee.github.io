---
layout: post
title: Pytorch 딥러닝을 위한 pytorch의 동작 방법
date: 2023-12-27 18:00 +0900
last_modified_at: 2023-12-27 18:00:00 +0900
tags: [deeplearning, Pytorch]
toc: True
---

# 딥러닝을 위한 pytorch가 어떻게 동작하는가

## 1. Pytorch 작동구조

### 1.1 Deep Learning Key Components

*딥러닝의 학습 단계

1. Data
2. Model
3. Output
4. Loss
5. Optimizer

---

필요한 것은 총 4가지

1. Data<br>
$\Rightarrow$ torch.utils.data.Dataset<br>
$\Rightarrow$ torch.utils.data.DataLoader
2. Model<br>
$\Rightarrow$ torch.nn.Module
3. Loss Function<br>
$\Rightarrow$ torch.nn<br>
$\Rightarrow$ torch.nn.functional
4. Optimizer<br>
$\Rightarrow$ torch.optim

위와 같이 딥 러닝 모델 구현에 필요한 요소들을 pytorch를 통해 편리하게 사용할 수 있다.

### 1.2 Pytorch 클래스 간 관계

![Alt text](\..\img\DL4-15.png)

## 2. 데이터

### 2.1 Dataset & DataLoader의 결과

Dataset과 DataLoader를 사용하면 데이터를 로드할 수 있다.

데이터 셋에서 **미니 배치** 크기의 데이터를 반환할 수 있다.

    미니 배치 : 전체 데이터 셋을 더 작은 부분집합으로 분할한 일부 데이터

### 2.2 Dataset

단일 데이터를 모델의 입력으로 사용할 수 있는 형태(tensor)로 변환하는 작업을 수행한다.

MNIST같은 기본 데이터도 제공한다.

### 2.3 Custom Dataset

pytorch에서 제공하는 데이터는 제한적이니까, 자신의 데이터를 사용한 Dataset을 만들 수 있다.

Custom Dataset 구현을 위해서는 Dataset 클래스를 상속하여 만들어야함

아래의 3개의 method가 꼭 작성되어야 함

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        # 'Dataset' 객체가 실행될 때 한번만 실행된다.
        # 주로 사용할 데이터 셋을 불러오고,
        # 필요한 변수를 선언한다.
        pass
    
    def __getitem__(self, idx):
        # **주어진 인덱스**에 해당하는 단일 데이터를 불러오고 반환함
        # 
        # Pytorch에서는 'Dataset'이 'DataLoader'와 함께 사용됨
        # 'DataLoader'에서는 dataset의 데이터를 로드하는 순서를 의미하는 idx를
        # 'Dataset'에 인자로 주고, dataset의 위치에 해당하는 데이터를 가져옴
        pass
    
    def __len__(self):
        # 데이터 셋의 데이터 개수를 반환함
        pass
```

---
<mark>주의사항</mark>

- 데이터 타입<br>
pytorch는 torch.tensor 객체로 다루기 때문에, 데이터는 tensor로 변환되어야한다.<br>
**'__getitem__'메서드에서 반환하는 데이터도 tensor**여야 한다.
- 데이터 차원<br>
'Dataset'은 'DataLoader'와 함께 사용된다.<br>
'DataLoader'에서는 데이터를 미니 배치로 묶어주는 역할을 하는데<br>
**이때 반환되는 모든 데이터의 차원의 크기는 같아야한다.**

### 2.4 DataLoader

데이터를 미니 배치로 묶어서 반환하는 역할

따라서 인자로 'Dataset'은 필수.

*추가 인자*

1. batch_size : 미니 배치의 크기
2. shuffle : epoch마다 데이터의 순서가 섞이는 여부
3. num_workers : 데이터 로딩에 사용하는 서브 프로세스 개수
4. drop_last : 마지막 미니 배치의 데이터 수가 미니 배치 크기보다 작은 경우,<br>그 데이터를 버릴지 말지 결정하는 인자

## 3. 모델

### 3.1 Pytorch에서 제공하는 모델

- Torchvision<br>
이미지분석에 특화된 다양한 모델 제공<br>
ResNet, VGG, AlexNet, EfficientNet, ViT 등등<br>
<a href='https://pytorch.org/vision/stable/models.html'>관련 문서</a>
- PyTorch Hub
CV, audio, generative, NLP 도메인 모델이 공개되어있음<br>
<a href='https://pytorch.org/hub/'>관련 문서</a>

---
#### 모델 불러오기

- Torchvision<br>

```python
import torchvision

# model = torchvision.models.[모델 이름]()
model = torchvision.models.resnet50()
```

- PyTorch Hub<br>
<br>단, Pytorch Hub는 모델마다 전달하는 인자가 다르므로,
<br><a href='https://pytorch.org/hub/'>Pytorch Hub</a>에서 불러오는 코드를 확인해야함

```python
import torch

# model = torch.hub.load() 사용
model = torch.hub.load('pytorch/vision', 'resnet50')
```

### 3.2 Custom model의 필요성

pytorch에 공개된 모델은 제한적임

안그래도 빠르게 발전하는 DL분야에서는 모든 것을 공개되지 않는다.
<br>*공개 되더라도 보통 Github*

그러므로 pytorch에서 custom model을 정의할 줄 알아야한다.

### 3.3 Custom model

#### 기본 구조

일반적으로 `torch.nn.Module` 클래스를 상속받아 정의

아래의 2가지 method가 반드시 작성되어야 함

```python
class CustomModel(nn.Module):
    def __init__(self):
        # 부모 클래스 (nn.Module)를 초기화 (모델의 레이어와 parameter 초기화)
        super().__init__()
        self.encoder = nn.Linear(10,2)
        self.decoder = nn.Linear(2,10)

    def forward(self, x):
        # 입력데이터에 대한 연산을 정의
        out = self.encoder(x)
        out = self.decoder(out)
        return out

model = CustomModel()
```

## 4. 역전파와 최적화

### 4.1 학습의 기본 구조

```python
for epoch in range(num_epochs):
    for data, label in train_dataloader:
        optimizer.zero_grad()
        # pytorch는 기본적으로 gradient를 누적해서 사용함
        # 따라서 이를 초기화해주는 함수

        output = model(data)
        # 데이터를 모델을 통해 연산함

        loss = loss_function(output, label)
        # loss값 계산

        loss.backward()
        # loss에 대한 gradient 계산
        # 이때 'AutoGrad'를 통해 자동으로 연산한다.

        optimizer.step()
        # 계산된 gradient를 사용해서 각 파라미터를 업데이트
```

### 4.2 AutoGrad

tensor의 연산에 대한 미분을 자동으로 계산한다.<br>
loss.backward()의 기반이기도 하다

내부적으로 <ins>computational graph</ins>를 생성한다.

*computational graph* : 수학적 계산을 node와 edge의 그래프로 표현한 것

이를 통해 tensor간 연산의 중간 결과가 남게 된다.

이러한 computational graph와 chain rule을 이용해서 gradient 계산을 자동으로 해준다.

## 5. 추론과 평가

### 5.1 Inference

학습한 모델을 이용해서, 입력 데이터에 대한 예측 결과를 내놓는 과정

---

`model.eval()` : 모델을 evaluation 모드로 전환한다.<br>
모델의 특정 레이어들이 학습 과정과 추론 과정에서 다르게 동작해야하기 때문에,<br>
이 부분을 처리해준다.

`torch.no_grad()` : AutoGrad 기능을 비활성화<br>
추론에서는 gradient 계산이 필요하지 않아, 메모리 사용량을 줄이고 계산 속도를 향상할 수 있다.

```python
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        pred = model(data)
```

### 5.2 Evaluation

모델의 성능을 평가하는 과정

inference 과정에서의 예측 결과와 실제 라벨을 비교해서 성능 평가

task에 맞는 평가 지수를 선택하거나, pytorch에서 제공하는 것을 사용하거나 할 수 있다.
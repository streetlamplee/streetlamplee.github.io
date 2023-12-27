---
layout: post
title: Pytorch_DNN 실습 (2)
date: 2023-12-27 17:00 +0900
last_modified_at: 2023-12-27 17:00:00 +0900
tags: [deeplearning, Pytorch, DNN]
toc:  true
---

# DNN 구현(2)

### 실습 목차
* 1. 학습(training)
  * 1-1. 손실 함수와 최적화 알고리즘
  * 1-2. 학습 과정
  * 1-3. 활성화 함수와 가중치 초기화의 중요성

* 2. 추론과 평가(inference & evaluation)
  * 2-1. 학습한 딥러닝 모델로 테스트 데이터 추론
  * 2-2. 학습한 딥러닝 모델의 평가

### 환경 설정

- 패키지 설치 및 임포트


```python
!pip install scikit-learn==1.3.0 -q
!pip install torch==2.0.1 -q
!pip install torchvision==0.15.2 -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.8/10.8 MB[0m [31m82.5 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np # 기본적인 연산을 위한 라이브러리
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리
from tqdm.notebook import tqdm # 상태 바를 나타내기 위한 라이브러리

import torch # PyTorch 라이브러리
import torch.nn as nn # 모델 구성을 위한 라이브러리
import torch.optim as optim # optimizer 설정을 위한 라이브러리
from torch.utils.data import Dataset, DataLoader # 데이터셋 설정을 위한 라이브러리

import torchvision # PyTorch의 컴퓨터 비전 라이브러리
import torchvision.transforms as T # 이미지 변환을 위한 모듈

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score # 성능지표 측정
```


```python
# seed 고정
import random
import torch.backends.cudnn as cudnn

def random_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_num)

random_seed(42)
```

###  데이터 셋 개요 </b>

* 데이터 셋: MNIST 데이터베이스(Modified National Institute of Standards and Technology database)
* 데이터 셋 개요: MNIST는 숫자 0부터 9까지의 이미지로 구성된 손글씨 데이터셋입니다. 총 6만 개의 학습 데이터와 1만 개의 숫자 데이터로 이루어져 있으며 [이미지]와 [숫자에 대한 라벨]로 구성됩니다.
* 데이터 셋 저작권: CC BY-SA 3.0
* [MNIST - 위키피디아](https://ko.wikipedia.org/wiki/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4)




```python
# 데이터를 불러올 때, 필요한 변환(transform)을 정의합니다.
mnist_transform = T.Compose([
    T.ToTensor(), # 텐서 형식으로 변환
])
```


```python
# torchvision 라이브러리를 사용하여 MNIST 데이터 셋을 불러옵니다.
download_root = './MNIST_DATASET'

train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True) # train dataset 다운로드
test_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True) # test dataset 다운로드
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 9912422/9912422 [00:00<00:00, 352246981.65it/s]
    

    Extracting ./MNIST_DATASET/MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 93253036.05it/s]
    

    Extracting ./MNIST_DATASET/MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:00<00:00, 158592262.81it/s]

    Extracting ./MNIST_DATASET/MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    

    
    

    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 4542/4542 [00:00<00:00, 4011482.16it/s]

    Extracting ./MNIST_DATASET/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    

    
    


```python
# 데이터 셋을 학습 데이터 셋과 검증 데이터 셋으로 분리합니다.
total_size = len(train_dataset)
train_num, valid_num = int(total_size * 0.8), int(total_size * 0.2) # 8 : 2 = train : valid
print("Train dataset 개수 : ",train_num)
print("Validation dataset 개수 : ",valid_num)
train_dataset,valid_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num]) # train - valid set 나누기
```

    Train dataset 개수 :  48000
    Validation dataset 개수 :  12000
    


```python
# 앞서 선언한 Dataset을 인자로 주어 DataLoader를 선언합니다.
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
```


```python
# 최종 모델 코드
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()

    for i in range(len(self.hidden_dims) - 1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      if apply_activation:
        self.layers.append(nn.ReLU())

      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim = 1)

  def forward(self, x):
    """
    Input and Output Summary

    Input:
      x: [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]

    """
    x = x.view(x.shape[0], -1)  # [batch_size, 784]

    for layer in self.layers:
      x = layer(x)

    x = self.classifier(x) # [batch_size, 10]
    output = self.softmax(x) # [batch_size, 10]
    return output

  def weight_initialization(self, weight_init_method):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        if weight_init_method == 'gaussian':
          nn.init.normal_(m.weight)
        elif weight_init_method == 'xavier':
          nn.init.xavier_normal_(m.weight)
        elif weight_init_method == 'kaiming':
          nn.init.kaiming_normal_(m.weight)
        elif weight_init_method == 'zeros':
          nn.init.zeros_(m.weight)

        nn.init.zeros_(m.bias)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

## 1. 학습(training)

```
💡 목차 개요: 앞서 구현한 Dataset, DataLoader 그리고 custom model을 이용하여 딥러닝 모델을 학습합니다.
```

- 1-1. 손실 함수와 최적화 알고리즘
- 1-2. 학습 과정
- 1-3. 활성화 함수와 가중치 초기화의 중요성


### 1-1 손실 함수와 최적화 알고리즘

> `torch.nn`, `torch.optim`를 사용하여 편리하게 손실 함수와 최적화 알고리즘을 구현할 수 있습니다.


#### 📝 설명: `torch.nn`을 이용하여 손실 함수 구현
`torch.nn` 모듈은 다양한 손실 함수들을 제공합니다. 원하는 손실 함수가 없을 경우, 손실을 계산하는 코드를 직접 구현하여 스칼라 tensor를 반환하는 손실 함수를 직접 구현할 수도 있습니다. 0 ~ 9까지의 클래스를 갖는 MNIST 숫자 이미지 데이터를 분류하기 위해 `NLLLoss`를 불러와 실습해보도록 하겠습니다.

- `torch.nn.NLLLoss`
- `torch.nn.MSELoss`
- `torch.nn.L1Loss`
- `torch.nn.BCELoss`
- `torch.nn.CrossEntropyLoss`
- ..


📚 참고할만한 자료:
* [Loss Functions - PyTorch 공식 문서](https://pytorch.org/docs/stable/nn.html#loss-functions): 소개한 손실 함수 이 외에도 다양한 손실 함수가 존재합니다. 홈페이지에서 불러올 수 있는 손실 함수를 확인할 수 있습니다.


```python
criterion = nn.NLLLoss()
```

#### 📝 설명: `torch.optim`에 구현된 최적화 알고리즘 사용
`torch.optim` 모듈은 다양한 최적화 알고리즘을 제공합니다. 실습에서는 Adam 알고리즘을 사용해 실습해보도록 하겠습니다.

- `torch.optim.SGD`
- `torch.optim.Adam`
- `torch.optim.Adagrad`
- `torch.optim.RMSprop`
- ..

`torch.optim`을 통해 최적화 알고리즘을 구현할 경우, 선언한 모델의 가중치를 필수적으로 선언해야 합니다. 이 외에도 PyTorch의 optimizer에는 주요 인자들이 존재합니다.

- `lr`: 학습률(learning rate) 하이퍼 파라미터입니다.
- `weight_decay`: L2 regularization에 사용되는 하이퍼 파라미터입니다.


📚 참고할만한 자료:
* [torch.optim - PyTorch 공식 문서](https://pytorch.org/docs/stable/optim.html): 소개한 최적화 알고리즘 이 외에도 다양한 최적화 알고리즘이 존재합니다. 홈페이지에서 불러올 수 있는 최적화 알고리즘를 확인할 수 있습니다.
* [torch.optim.Adam - PyTorch 공식 문서](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam): Adam documentation


```python
lr = 0.001
hidden_dim = 128
hidden_dims = [784, hidden_dim * 4, hidden_dim * 2, hidden_dim]
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
optimizer = optim.Adam(model.parameters(), lr = lr)
```

### 1-2 학습 과정

> PyTorch를 사용하여 딥러닝 모델을 학습합니다. 학습을 진행하며, 검증 데이터 셋에 대해 loss가 감소하지 않고 `patience`만큼 계속 증가한다면 학습을 중단합니다.



```python
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train()  # 모델을 학습 모드로 설정
  train_loss = 0.0
  train_accuracy = 0

  tbar = tqdm(dataloader)
  for images, labels in tbar:
      images = images.to(device)
      labels = labels.to(device)

      # 순전파
      outputs = model(images)
      loss = criterion(outputs, labels)

      # 역전파 및 weights 업데이트
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 손실과 정확도 계산
      train_loss += loss.item()
      # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs, 1)
      train_accuracy += (predicted == labels).sum().item()

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}")

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)

  return model, train_loss, train_accuracy

def evaluation(model, dataloader, valid_dataset, criterion, device, epoch, num_epochs):
  model.eval()  # 모델을 평가 모드로 설정
  valid_loss = 0.0
  valid_accuracy = 0

  with torch.no_grad(): # model의 업데이트 막기
      tbar = tqdm(dataloader)
      for images, labels in tbar:
          images = images.to(device)
          labels = labels.to(device)

          # 순전파
          outputs = model(images)
          loss = criterion(outputs, labels)

          # 손실과 정확도 계산
          valid_loss += loss.item()
          # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
          _, predicted = torch.max(outputs, 1)
          valid_accuracy += (predicted == labels).sum().item()

          # tqdm의 진행바에 표시될 설명 텍스트를 설정
          tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}")

  valid_loss = valid_loss / len(dataloader)
  valid_accuracy = valid_accuracy / len(valid_dataset)

  return model, valid_loss, valid_accuracy


def training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name):
    best_valid_loss = float('inf')  # 가장 좋은 validation loss를 저장
    early_stop_counter = 0  # 카운터
    valid_max_accuracy = -1

    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
        model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, valid_dataset, criterion, device, epoch, num_epochs)

        if valid_accuracy > valid_max_accuracy:
          valid_max_accuracy = valid_accuracy

        # validation loss가 감소하면 모델 저장 및 카운터 리셋
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"./model_{model_name}.pt")
            early_stop_counter = 0

        # validation loss가 증가하거나 같으면 카운터 증가
        else:
            early_stop_counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        # 조기 종료 카운터가 설정한 patience를 초과하면 학습 종료
        if early_stop_counter >= patience:
            print("Early stopping")
            break

    return model, valid_max_accuracy
```


```python
num_epochs = 100
patience = 3
scores = dict()
device = 'cuda:0' # gpu 설정
model_name = 'exp1'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.3167, Train Accuracy: 0.9047 Valid Loss: 0.1187, Valid Accuracy: 0.9624
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1651, Train Accuracy: 0.9492 Valid Loss: 0.1065, Valid Accuracy: 0.9676
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.1317, Train Accuracy: 0.9603 Valid Loss: 0.0841, Valid Accuracy: 0.9755
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.1094, Train Accuracy: 0.9650 Valid Loss: 0.0752, Valid Accuracy: 0.9779
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0967, Train Accuracy: 0.9691 Valid Loss: 0.0757, Valid Accuracy: 0.9772
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0833, Train Accuracy: 0.9734 Valid Loss: 0.0633, Valid Accuracy: 0.9802
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0742, Train Accuracy: 0.9764 Valid Loss: 0.0741, Valid Accuracy: 0.9781
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.0677, Train Accuracy: 0.9786 Valid Loss: 0.0685, Valid Accuracy: 0.9794
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.0613, Train Accuracy: 0.9805 Valid Loss: 0.0705, Valid Accuracy: 0.9787
    Early stopping
    


```python
# Batch normalization을 제외하고 학습을 진행합니다.
model_name = 'exp2'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = False, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.2841, Train Accuracy: 0.9133 Valid Loss: 0.1649, Valid Accuracy: 0.9492
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1352, Train Accuracy: 0.9594 Valid Loss: 0.1144, Valid Accuracy: 0.9667
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.1040, Train Accuracy: 0.9689 Valid Loss: 0.1040, Valid Accuracy: 0.9702
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.0860, Train Accuracy: 0.9734 Valid Loss: 0.0874, Valid Accuracy: 0.9752
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0732, Train Accuracy: 0.9791 Valid Loss: 0.0930, Valid Accuracy: 0.9737
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0622, Train Accuracy: 0.9809 Valid Loss: 0.0856, Valid Accuracy: 0.9777
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0581, Train Accuracy: 0.9824 Valid Loss: 0.1063, Valid Accuracy: 0.9726
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.0543, Train Accuracy: 0.9832 Valid Loss: 0.0846, Valid Accuracy: 0.9768
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.0489, Train Accuracy: 0.9847 Valid Loss: 0.0873, Valid Accuracy: 0.9780
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [10/100], Train Loss: 0.0412, Train Accuracy: 0.9878 Valid Loss: 0.0890, Valid Accuracy: 0.9766
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [11/100], Train Loss: 0.0398, Train Accuracy: 0.9875 Valid Loss: 0.1178, Valid Accuracy: 0.9717
    Early stopping
    


```python
# Dropout을 제외하고 학습을 진행합니다.
# dropout_ratio는 하이퍼 파라미터입니다. 최적의 dropout_ratio에 따라 모델의 성능이 달라질 수 있습니다.
model_name = 'exp3'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = False, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.2232, Train Accuracy: 0.9326 Valid Loss: 0.1130, Valid Accuracy: 0.9651
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1074, Train Accuracy: 0.9664 Valid Loss: 0.0851, Valid Accuracy: 0.9732
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.0787, Train Accuracy: 0.9744 Valid Loss: 0.0864, Valid Accuracy: 0.9725
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.0622, Train Accuracy: 0.9800 Valid Loss: 0.0773, Valid Accuracy: 0.9770
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0536, Train Accuracy: 0.9829 Valid Loss: 0.0746, Valid Accuracy: 0.9764
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0446, Train Accuracy: 0.9851 Valid Loss: 0.0719, Valid Accuracy: 0.9788
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0388, Train Accuracy: 0.9869 Valid Loss: 0.0631, Valid Accuracy: 0.9807
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.0329, Train Accuracy: 0.9892 Valid Loss: 0.0686, Valid Accuracy: 0.9794
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.0303, Train Accuracy: 0.9900 Valid Loss: 0.0648, Valid Accuracy: 0.9819
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [10/100], Train Loss: 0.0286, Train Accuracy: 0.9905 Valid Loss: 0.0647, Valid Accuracy: 0.9810
    Early stopping
    

### 1-3 활성화 함수와 가중치 초기화의 중요성

> 활성화 함수와 가중치 초기화가 딥러닝 모델에 끼치는 영향을 실습을 통해 알아봅니다.


#### 📝 설명: 활성화 함수가 없다면 딥러닝 모델은 단순히 입력 데이터와 가중치의 선형 변환(linear transform)일 뿐
활성화 함수가 없는 딥러닝 모델은 기본적으로 선형 분류기와 다르지 않으며, 복잡한 패턴을 학습하는 데에 제한적일 수 있습니다.

활성화 함수가 없다면 각각의 레이어는 선형 변환만을 수행하며, 여러 개의 선형 변환을 결합한 것은 결국 하나의 선형 변환과 다를 바 없습니다.
활성화 함수는 딥러닝 모델에 비선형성(non-linearity)를 도입하여, 모델이 더 복잡한 패턴을 학습할 수 있게 합니다. 비선형 활성화 함수로 인해 딥러닝 모델은 선형 변환 이상의 복잡한 함수를 표현하고 학습할 수 있게 됩니다.

따라서, 딥러닝 모델에서는 활성화 함수가 필요합니다.

📚 참고할만한 자료:
* [What happens if you do not use any activation function in a neural network’s hidden layer(s)?](https://medium.com/data-science-365/what-happens-if-you-do-not-use-any-activation-function-in-a-neural-networks-hidden-layer-s-f3ce089e4508)




```python
# 활성화 함수(activation function)를 제외하고 학습을 진행합니다.
model_name = 'exp4'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = False, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.4630, Train Accuracy: 0.8637 Valid Loss: 0.3509, Valid Accuracy: 0.9004
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.3793, Train Accuracy: 0.8895 Valid Loss: 0.3289, Valid Accuracy: 0.9067
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.3622, Train Accuracy: 0.8942 Valid Loss: 0.3360, Valid Accuracy: 0.9044
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.3492, Train Accuracy: 0.8986 Valid Loss: 0.3289, Valid Accuracy: 0.9070
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.3429, Train Accuracy: 0.8988 Valid Loss: 0.3186, Valid Accuracy: 0.9136
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.3375, Train Accuracy: 0.9021 Valid Loss: 0.3203, Valid Accuracy: 0.9107
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.3290, Train Accuracy: 0.9050 Valid Loss: 0.3236, Valid Accuracy: 0.9113
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.3284, Train Accuracy: 0.9042 Valid Loss: 0.3165, Valid Accuracy: 0.9123
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.3233, Train Accuracy: 0.9063 Valid Loss: 0.3102, Valid Accuracy: 0.9147
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [10/100], Train Loss: 0.3206, Train Accuracy: 0.9070 Valid Loss: 0.3112, Valid Accuracy: 0.9143
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [11/100], Train Loss: 0.3139, Train Accuracy: 0.9088 Valid Loss: 0.3161, Valid Accuracy: 0.9122
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [12/100], Train Loss: 0.3108, Train Accuracy: 0.9101 Valid Loss: 0.3173, Valid Accuracy: 0.9129
    Early stopping
    

#### 📝 설명: 딥러닝 모델의 모든 가중치를 '0'으로 초기화하면 학습이 불가합니다.

딥러닝 모델의 가중치를 '0'으로 초기화하면, 모든 뉴런의 출력값은 동일하게 '0'으로 계산됩니다. 이 경우, 역전파 단계에서 chain rule이 계산되는 값에 '0'이 곱해지므로 gradient가 0으로 계산돼 학습이 불가능합니다.

수식적으로 살펴보겠습니다.

$o_{1} = w_{1} \cdot x$,

$o_{2} = w_{2} \cdot x$

$O = w_{o1} \cdot o_{1} + w_{o2} \cdot o_{2}$

위와 같은 연산이 있다고 가정하겠습니다. 이 경우, 가중치 $w_{1}$에 대한 손실 함수 $L$의 편미분은 다음과 같이 계산됩니다.

$\frac{\partial L}{\partial w_{1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial o_{1}} \frac{\partial o_{1}}{\partial w_{1}}$


이 때, 모든 가중치가 '0'으로 초기화된다면, $\frac{\partial L}{\partial w_{1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial o_{1}} \frac{\partial o_{1}}{\partial w_{1}}$과 $\frac{\partial L}{\partial w_{2}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial o_{2}} \frac{\partial o_{2}}{\partial w_{2}}$은 모두 '0'의 값을 가지게 되어 경사 하강법에서 사용되는 gradient가 '0'이 됩니다. 이에 따라 가중치의 변화가 존재하지 않으므로, 학습이 불가능합니다.


📚 참고할만한 자료:
* [가중치 초기화(weight initialization)](https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/)



```python
model_name = 'exp5'
init_method = 'zeros' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 2.3017, Train Accuracy: 0.1116 Valid Loss: 2.3008, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3006, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
    Early stopping
    

![검수내역 figure_crop.png](data:image/png;base64,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)


```python
# 실험별 validation accuracy 최댓값
print(f"BatchNorm, Dropout 사용 모델: {scores['exp1']:.4f}")
print(f"BatchNorm 제거 모델: {scores['exp2']:.4f}")
print(f"Dropout 제거 모델: {scores['exp3']:.4f}")
print(f"Activation Function 제거 모델: {scores['exp4']:.4f}")
print(f"가중치를 0으로 초기화한 모델: {scores['exp5']:.4f}")
```

    BatchNorm, Dropout 사용 모델: 0.9802
    BatchNorm 제거 모델: 0.9780
    Dropout 제거 모델: 0.9819
    Activation Function 제거 모델: 0.9147
    가중치를 0으로 초기화한 모델: 0.1133
    

## 2. 추론과 평가(inference & evaluation)

```
💡 목차 개요: 딥러닝 모델의 학습이 완료되었다면 모델의 성능을 평가해야 합니다. 학습한 모델을 통해 테스트 데이터에 대해 추론하고 다양한 메트릭으로 성능을 측정합니다.
```

- 2-1. 학습한 딥러닝 모델로 테스트 데이터 추론
- 2-2. 학습한 딥러닝 모델의 평가


```python
# BatchNorm, Dropout을 사용한 모델을 로드합니다.
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.load_state_dict(torch.load("./model_exp1.pt"))
model = model.to(device)
```

### 2-1 학습한 딥러닝 모델로 테스트 데이터 추론

> 앞서 선언한 테스트 MNIST 데이터 셋에 대해 추론을 진행합니다.



```python
model.eval()
total_labels = []
total_preds = []
total_probs = []
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels

        outputs = model(images)
        # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
        _, predicted = torch.max(outputs.data, 1)

        total_preds.extend(predicted.detach().cpu().tolist())
        total_labels.extend(labels.tolist())
        total_probs.append(outputs.detach().cpu().numpy())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
total_probs = np.concatenate(total_probs, axis= 0)
```

### 2-2 학습한 딥러닝 모델의 평가

> 모델이 예측한 결과값과 라벨을 사용하여 모델의 성능을 평가합니다.



```python
# precision, recall, f1를 계산합니다.
precision = precision_score(total_labels, total_preds, average='macro')
recall = recall_score(total_labels, total_preds, average='macro')
f1 = f1_score(total_labels, total_preds, average='macro')

# AUC를 계산합니다.
# 모델의 출력으로 nn.LogSoftmax 함수가 적용되어 결과물이 출력되게 됩니다. sklearn의 roc_auc_score 메서드를 사용하기 위해서는 단일 데이터의 클래스들의 확률 합은 1이 되어야 합니다. 이를 위해 확률 matrix에 지수 함수를 적용합니다.
total_probs = np.exp(total_probs)
auc = roc_auc_score(total_labels, total_probs, average='macro', multi_class = 'ovr')

print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')
```

    Precision: 0.9815567035682374, Recall: 0.9815391552489962, F1 Score: 0.9815275156151371, AUC: 0.9997172022898146
    

#Reference
- [early stopping 구현 코드](https://github.com/zhouhaoyi/Informer2020/blob/main/utils/tools.py)




## Required Package

> torch == 2.0.1

> torchvision == 0.15.2

> sklearn == 1.3.0


```python

```

---
layout: post
title: Pytorch_텐서 실습 1
date: 2023-12-26 17:00 +0900
last_modified_at: 2023-12-26 17:00:00 +0900
tags: [deeplearning, Pytorch, tensor]
toc:  true
---

# 텐서 조작 (1)

### 환경 설정
> PyTorch 설치 및 불러오기


- 패키지 설치 및 임포트


```python
!pip install torch==2.0.1 # PyTorch 를 가장 최근 버전으로 설치
```

    Collecting torch==2.0.1
      Downloading torch-2.0.1-cp310-cp310-manylinux1_x86_64.whl (619.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.1.2)
    Collecting nvidia-cuda-nvrtc-cu11==11.7.99 (from torch==2.0.1)
      Downloading nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl (21.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m41.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-runtime-cu11==11.7.99 (from torch==2.0.1)
      Downloading nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl (849 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m45.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu11==11.7.101 (from torch==2.0.1)
      Downloading nvidia_cuda_cupti_cu11-11.7.101-py3-none-manylinux1_x86_64.whl (11.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m55.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu11==8.5.0.96 (from torch==2.0.1)
      Downloading nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl (557.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cublas-cu11==11.10.3.66 (from torch==2.0.1)
      Downloading nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl (317.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu11==10.9.0.58 (from torch==2.0.1)
      Downloading nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux1_x86_64.whl (168.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m7.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu11==10.2.10.91 (from torch==2.0.1)
      Downloading nvidia_curand_cu11-10.2.10.91-py3-none-manylinux1_x86_64.whl (54.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m13.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusolver-cu11==11.4.0.1 (from torch==2.0.1)
      Downloading nvidia_cusolver_cu11-11.4.0.1-2-py3-none-manylinux1_x86_64.whl (102.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusparse-cu11==11.7.4.91 (from torch==2.0.1)
      Downloading nvidia_cusparse_cu11-11.7.4.91-py3-none-manylinux1_x86_64.whl (173.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nccl-cu11==2.14.3 (from torch==2.0.1)
      Downloading nvidia_nccl_cu11-2.14.3-py3-none-manylinux1_x86_64.whl (177.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvtx-cu11==11.7.91 (from torch==2.0.1)
      Downloading nvidia_nvtx_cu11-11.7.91-py3-none-manylinux1_x86_64.whl (98 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting triton==2.0.0 (from torch==2.0.1)
      Downloading triton-2.0.0-1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (63.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (67.7.2)
    Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (0.42.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch==2.0.1) (3.27.9)
    Collecting lit (from triton==2.0.0->torch==2.0.1)
      Downloading lit-17.0.6.tar.gz (153 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m153.0/153.0 kB[0m [31m16.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch==2.0.1) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch==2.0.1) (1.3.0)
    Building wheels for collected packages: lit
      Building wheel for lit (pyproject.toml) ... [?25l[?25hdone
      Created wheel for lit: filename=lit-17.0.6-py3-none-any.whl size=93255 sha256=a2db675c407969af1bb77b5921c595377dd31dd22a83d4ac2f8ce6da2d296823
      Stored in directory: /root/.cache/pip/wheels/30/dd/04/47d42976a6a86dc2ab66d7518621ae96f43452c8841d74758a
    Successfully built lit
    Installing collected packages: lit, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, nvidia-cusolver-cu11, nvidia-cudnn-cu11, triton, torch
      Attempting uninstall: triton
        Found existing installation: triton 2.1.0
        Uninstalling triton-2.1.0:
          Successfully uninstalled triton-2.1.0
      Attempting uninstall: torch
        Found existing installation: torch 2.1.0+cu121
        Uninstalling torch-2.1.0+cu121:
          Successfully uninstalled torch-2.1.0+cu121
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.1.0+cu121 requires torch==2.1.0, but you have torch 2.0.1 which is incompatible.
    torchdata 0.7.0 requires torch==2.1.0, but you have torch 2.0.1 which is incompatible.
    torchtext 0.16.0 requires torch==2.1.0, but you have torch 2.0.1 which is incompatible.
    torchvision 0.16.0+cu121 requires torch==2.1.0, but you have torch 2.0.1 which is incompatible.[0m[31m
    [0mSuccessfully installed lit-17.0.6 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 torch-2.0.1 triton-2.0.0
    


```python
import torch # PyTorch 불러오기
import numpy as np # numpy 불러오기
import warnings # 경고 문구 제거
warnings.filterwarnings('ignore')
```

## 1. 텐서 이해하기

- 1-1. 텐서를 생성하고 텐서로 변환하는 방법을 이해 및 실습
- 1-2. 텐서에서의 indexing 이해 및 실습


### 1-1 텐서를 생성하고 텐서로 변환하는 방법을 이해 및 실습

> Random 한 값을 가지는 텐서를 생성하고, list 나 numpy array 같은 다양한 형태의 배열들을 PyTorch 를 이용하여 텐서로 변환하는 과정을 알아봅니다.


#### 📝 설명 : 텐서의 값을 무작위로 생성하는 방법들
* rand :  0과 1 사이의 균일한 분포 (Uniform Distribution) 에서 무작위로 생성된 텐서를 반환

📚 참고할만한 자료:
* [rand] https://pytorch.org/docs/stable/generated/torch.rand.html


```python
# 0부터 1 사이의 값을 랜덤하게 NxM 텐서로 반환
torch.rand(2, 3) # torch.rand(NxM) NxM은 텐서의 크기를 말합니다.
```




    tensor([[0.4832, 0.6079, 0.0965],
            [0.9239, 0.1382, 0.7564]])



#### 📝 설명 : Tensor 의 값을 무작위로 생성하는 방법들
* randn : 평균이 0이고 표준 편차가 1인 정규 분포(가우시안 분포)에서 무작위로 생성된 텐서를 반환

📚 참고할만한 자료:
* [randn] https://pytorch.org/docs/stable/generated/torch.randn.html


```python
# 가우시안 분포에서 렌덤하게 값을 추출 후, NxM 텐서로 반환
torch.randn(2, 3) # torch.randn(NxM) NxM은 텐서의 크기를 말합니다.
```




    tensor([[ 0.9027, -0.4474, -0.5633],
            [ 1.6880,  0.0257,  0.3113]])



#### 📝 설명 : 텐서의 값을 무작위로 생성하는 방법들

* randint : 주어진 범위 내에서 정수값을 무작위로 선택하여 텐서를 생성 (단, 최솟값을 포함하고, 최댓값은 포함하지 않음)

📚 참고할만한 자료:

* [randint] https://pytorch.org/docs/stable/generated/torch.randint.html


```python
# 범위 내의 정수를 N x M 텐서로 반환
torch.randint(1, 10, (5, 5)) # 생성 가능한 최솟값 : 1, 최댓값 : 9, (5x5) Tensor 크기
```




    tensor([[9, 1, 3, 9, 2],
            [6, 9, 7, 2, 5],
            [4, 9, 4, 7, 1],
            [3, 4, 1, 7, 7],
            [3, 8, 1, 3, 1]])



#### 📝 설명 : 텐서의 값을 지정해서 생성하는 방법들
* zeros : 모든 요소가 0인 텐서 반환

📚 참고할만한 자료:
* [zeros] https://pytorch.org/docs/stable/generated/torch.zeros.html


```python
torch.zeros(3, 3) # torch.zeros(*size) 여기서 size 는 ","로 구분하며 차원을 여러개로 늘릴 수 있습니다.
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])



#### 📝 설명 : 텐서의 값을 지정해서 생성하는 방법들
* ones : 모든 요소가 1인 텐서 반환

📚 참고할만한 자료:
* [ones] https://pytorch.org/docs/stable/generated/torch.ones.html


```python
torch.ones(2, 2, 2) # torch.ones(*size) 여기서 size 는 ","로 구분하며 채널을 여러개로 늘릴 수 있습니다.
```




    tensor([[[1., 1.],
             [1., 1.]],
    
            [[1., 1.],
             [1., 1.]]])



#### 📝 설명 : 텐서의 값을 지정해서 생성하는 방법들
* full: 모든 요소가 지정된 값인 텐서 반환

📚 참고할만한 자료:
* [full] https://pytorch.org/docs/stable/generated/torch.full.html


```python
torch.full((2, 3), 5) # torch.full((size),value) => 괄호로 텐서의 크기 (2,3) 를 입력하고, 지정한 값 value (5) 로 모든 요소가 설정됩니다.
```




    tensor([[5, 5, 5],
            [5, 5, 5]])



#### 📝 설명 : 텐서의 값을 지정해서 생성하는 방법들
* eye : 단위 행렬 반환 (※ 단위 행렬이란? 대각선 요소가 1이고, 나머지 요소가 0인 행렬)

📚 참고할만한 자료:
* [eye] https://pytorch.org/docs/stable/generated/torch.eye.html


```python
torch.eye(3) # torch.eye(n) (nxn) 크기를 가지는 단위 행렬 반환, 단위행렬 특성 상 정사각행렬 (square matrix)만 가능
```




    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])



#### 📝 설명 : 다양한 데이터를 텐서 형식으로 변환하기
* tensor : 주어진 데이터를 텐서로 변환. 데이터는 list, tuple, numpy array 등의 형태일 수 있음.

📚 참고할만한 자료:
* [tensor] https://pytorch.org/docs/stable/generated/torch.tensor.html


```python
# list, tuple, numpy array를 텐서로 바꾸기
ls = [[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]] # sample list 생성
tup = tuple([1, 2, 3]) # sample tuple 생성
arr = np.array([[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]]) # sample numpy array 생성

print(torch.tensor(ls))
print('\n')
print(torch.tensor(tup))
print('\n')
print(torch.tensor(arr))
```

    tensor([[ 1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10]])
    
    
    tensor([1, 2, 3])
    
    
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
    
            [[ 7,  8,  9],
             [10, 11, 12]]])
    

#### 📝 설명 : 다양한 형태를 텐서 형식으로 변환하기
* from_numpy : numpy array 를 텐서로 변환

📚 참고할만한 자료:
* [from_numpy] https://pytorch.org/docs/stable/generated/torch.from_numpy.html


```python
torch.from_numpy(arr) # array 를 tensor로 바꾸기 (2)
```




    tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
    
            [[ 7,  8,  9],
             [10, 11, 12]]])



#### 📝 설명: 다양한 형식의 텐서 변환
* as_tensor: 변환 전 데이터와의 메모리 공유(memory sharing)를 사용하므로, 변환 전 데이터 변경 시 변환되어 있는 텐서에도 반영됨

📚 참고할만한 자료:
* [as_tensor] https://pytorch.org/docs/stable/generated/torch.as_tensor.html


```python
# torch.tensor 와 torch.as_tensor 의 차이점 알아보기
print("torch.tensor")
data1 = np.array([1, 2, 3, 4, 5]) # 샘플 데이터 리스트 생성
tensor1 = torch.tensor(data1) # memory 공유 X
data1[0] = 10  # 원본 데이터 변경
print(tensor1)  # 원본 데이터의 값 변경에 영향을 받지 않음

print('-------'*10)

print("torch.as_tensor")
data2 = np.array([1, 2, 3, 4, 5])
tensor2 = torch.as_tensor(data2) # memory 공유 O
data2[0] = 10  # 원본 데이터 변경
print(tensor2)  # 원본 데이터의 값 변경에 영향을 받음
```

    torch.tensor
    tensor([1, 2, 3, 4, 5])
    ----------------------------------------------------------------------
    torch.as_tensor
    tensor([10,  2,  3,  4,  5])
    

#### 📝 설명 : 다양한 형식의 텐서 변환
* Tensor : **float32** type으로 텐서 변환

📚 참고할만한 자료:
* [Tensor] https://pytorch.org/docs/stable/tensors.html


```python
data = [1, 2, 3, 4, 5]
tensor1 = torch.tensor(data) # list 에서 Tensor 변환
print("torch.tensor")
print("Output:", tensor1)
print("Type", tensor1.dtype) # dtype : Tensor 안의 원소들의 자료형, torch.tensor 는 원본의 데이터 타입을 그대로 따라감

print('-------'*3)

tensor2 = torch.Tensor(data) # list 에서 Tensor 변환
print("torch.Tensor")
print("Output:", tensor2)
print("Type", tensor2.dtype) # torch.tensor 는 float32 타입으로 Tensor 변환
```

    torch.tensor
    Output: tensor([1, 2, 3, 4, 5])
    Type torch.int64
    ---------------------
    torch.Tensor
    Output: tensor([1., 2., 3., 4., 5.])
    Type torch.float32
    

### 1-2 텐서에서의 Indexing 을 이해 및 실습

> Indexing 개념과 Indexing 을 통해 값을 변경하는 방법에 대해 이해하고 실습합니다.

#### 📝 설명 : Indexing 이란?
Indexing 은 텐서 내의 특정 **요소**를 index를 통해 접근할 수 있는 방법을 의미합니다.
* Indexing 기본 : **대괄호("[ ]")**를 통해 이뤄지며, **":"** 는 특정 범위의 접근을 의미합니다.

📚 참고할만한 자료:
* [Tensor indexing] : https://pytorch.org/cppdocs/notes/tensor_indexing.html


```python
# 1차원 텐서에서 Indexing 하기
tmp_1dim = torch.tensor([i for i in range(10)]) # 0부터 9 까지의 값을 가지는 1차원 텐서 생성

print(tmp_1dim[0]) # 첫번째 원소 값 추출
print(tmp_1dim[5]) # 6번째 원소 값 추출
print(tmp_1dim[-1]) # -1 번째 원소 값 (뒤에서 첫번째) 추출
```

    tensor(0)
    tensor(5)
    tensor(9)
    


```python
# 3차원 텐서에서 Indexing 하기
tmp_3dim = torch.randn(4, 3, 2) # 4채널, 3행, 2열
print("Shape : ", tmp_3dim.shape)
print(tmp_3dim)

print('-------'*8)

print(tmp_3dim[:,:,0].shape)
print(tmp_3dim[:,:,0]) # 전체 채널과 전체 행에서 0번째 열만 추출

print('\n') # 줄 띄움

print(tmp_3dim[0,:,1].shape)
print(tmp_3dim[0,:,1])  # 0번째 채널의 전체 행에서 1번째 열만 추출
```

    Shape :  torch.Size([4, 3, 2])
    tensor([[[ 0.9144,  0.3184],
             [ 1.0496,  0.8238],
             [-1.2094,  1.1296]],
    
            [[ 0.8414, -0.5814],
             [ 1.1207, -0.7272],
             [-2.3303, -0.8834]],
    
            [[ 0.5461,  1.8506],
             [ 0.2818, -1.4328],
             [-1.5978, -2.4369]],
    
            [[ 1.6518,  1.2504],
             [-0.0586, -0.0322],
             [-0.1793,  0.2690]]])
    --------------------------------------------------------
    torch.Size([4, 3])
    tensor([[ 0.9144,  1.0496, -1.2094],
            [ 0.8414,  1.1207, -2.3303],
            [ 0.5461,  0.2818, -1.5978],
            [ 1.6518, -0.0586, -0.1793]])
    
    
    torch.Size([3])
    tensor([0.3184, 0.8238, 1.1296])
    

#### 📝 설명 : Indexing 이란?
* index_select : 선택한 차원에서 인덱스에 해당하는 요소만 추출하는 함수

📚 참고할만한 자료:
* [index_select] : https://pytorch.org/docs/stable/generated/torch.index_select.html


```python
# index_select
tmp_2dim = torch.tensor([[i for i in range(10)],[i for i in range(10, 20)]])
print(tmp_2dim)

print('\n')

my_index = torch.tensor([0, 2]) # 선택하고자 하는 index 는 텐서 형태이어야 함.
torch.index_select(tmp_2dim, dim=1, index=my_index) # 열을 기준으로 0열과 2열을 추출
```

    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    
    
    




    tensor([[ 0,  2],
            [10, 12]])



#### 📝 설명 : Indexing 이란?
* Masking 을 이용한 Indexing : 조건에 따른 텐서의 요소를 사용하기 위한 방법으로 조건에 맞는 요소들만 반환하는 방법입니다.


```python
# mask 를 이용한 텐서 Indexing (조건에 맞는 값만 추출)
mask = tmp_2dim >= 5 # 5보다 큰 텐서만 추출
tmp_2dim[mask] # 1차원 Tensor 로 반환
```




    tensor([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])



#### 📝 설명 : Indexing 이란?
* masked_select : 주어진 mask에 해당하는 요소들을 추출하여 1차원으로 펼친 새로운 텐서를 반환하는 함수

📚 참고할만한 자료:
* [masked_select] : https://pytorch.org/docs/stable/generated/torch.masked_select.html


```python
torch.masked_select(tmp_2dim, mask = mask) # tmp_2dim[tmp_2dim >= 5] 와 동일
```




    tensor([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])



#### 📝 설명 : Indexing 이란?
* take : 주어진 인덱스를 사용하여 텐서에서 요소를 선택하는 함수. 인덱스 번호는 텐서를 1차원으로 늘려졌을 때 기준으로 접근해야합니다.

📚 참고할만한 자료:
* [take] : https://pytorch.org/docs/stable/generated/torch.take.html


```python
tmp_2dim = torch.tensor([[i for i in range(10)], [i for i in range(10, 20)]])
print(tmp_2dim)

print('\n')

my_index = torch.tensor([0, 15])
torch.take(tmp_2dim, index = my_index) # Tensor가 1차원으로 늘려졌을 때 기준으로 index 번호로 접근
```

    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    
    
    




    tensor([ 0, 15])



#### 📝 설명 : Indexing 이란?
* gather : 주어진 차원에서 인덱스에 해당하는 요소들을 선택하여 새로운 텐서를 반환

📚 참고할만한 자료:
* [gather] : https://pytorch.org/docs/stable/generated/torch.gather.html


```python
tmp_2dim = torch.tensor([[i for i in range(10)],[i for i in range(10,20)]])
print(tmp_2dim)

print('\n')

recon_index =  torch.tensor([[0 ,1],[9, 8]]) # 0번째 값, 1번 째 값을 0번째 행으로 설정하고, 9번째 값, 8번째 값을 1번째 행으로 설정한다.
dim = 1 # 열 기준
print(recon_index)
print('\n')

torch.gather(tmp_2dim, dim = 1, index = recon_index) # dim =1 이므로 열 기준, 0행 0열, 0행 1열 선택, 1행 9열, 1행 8열
```

    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    
    
    tensor([[0, 1],
            [9, 8]])
    
    
    




    tensor([[ 0,  1],
            [19, 18]])



## 2. 텐서의 모양 바꾸기



### 2-1 텐서의 shape을 바꾸는 여러가지 함수 이해 및 실습
> 텐서의 모양을 자유자재로 바꾸는 방법에 대해 알아보고 실습합니다.



#### 📝 설명 : 텐서의 shape 변경
텐서에 대한 모양을 변경하기 위해 명심해야 할 점은 텐서의 크기 (요소의 개수)는 유지되어야 한다는 점입니다.
* size : 텐서의 모양을 확인합니다.

📚 참고할만한 자료:
* [size] : https://pytorch.org/docs/stable/generated/torch.Tensor.size.html


```python
a = torch.randn(2, 3, 5) # random 한 값을 가진 (1,3,5) 텐서 생성
a.size() # 차원 크기 확인
```




    torch.Size([2, 3, 5])




```python
a.shape # a.size() 와 동일
```




    torch.Size([2, 3, 5])



#### 📝 설명 : 텐서의 shape 변경
* reshape : 텐서의 모양을 변경합니다. 메모리를 공유하지 않습니다.

📚 참고할만한 자료:
* [reshape] : https://pytorch.org/docs/stable/generated/torch.reshape.html


```python
# 모양 변경
a = torch.randn(2, 3, 5) # (2,3,5) 크기를 가지는 텐서 생성
print(a)
print("Shape : ", a.size()) # 텐서 모양 반환
print('\n')

reshape_a = a.reshape(5, 6) # 3차원 텐서를 2차원 텐서로 크기 변경 (2,3,5) -> (5,6)
print(reshape_a)
print("Shape : ", reshape_a.size()) # 변경한 텐서 모양 반환
```

    tensor([[[ 1.4395,  0.5985, -1.1691,  1.5279, -0.0985],
             [-2.0897,  2.5121,  0.0553,  0.2532,  1.1086],
             [-1.2554, -1.1751, -0.0270,  0.0227, -0.6094]],
    
            [[ 0.8175, -0.7804,  1.2114, -2.4871,  0.7790],
             [-0.4992, -0.0488,  1.5461, -0.1898, -1.0895],
             [ 0.0334,  2.3150, -1.0674, -0.8454, -0.4906]]])
    Shape :  torch.Size([2, 3, 5])
    
    
    tensor([[ 1.4395,  0.5985, -1.1691,  1.5279, -0.0985, -2.0897],
            [ 2.5121,  0.0553,  0.2532,  1.1086, -1.2554, -1.1751],
            [-0.0270,  0.0227, -0.6094,  0.8175, -0.7804,  1.2114],
            [-2.4871,  0.7790, -0.4992, -0.0488,  1.5461, -0.1898],
            [-1.0895,  0.0334,  2.3150, -1.0674, -0.8454, -0.4906]])
    Shape :  torch.Size([5, 6])
    


```python
# -1 로 모양 자동 설정
reshape_auto_a = a.reshape(3, -1) # (2,3,5) 크기를 가지는 Tensor를 (3,n)의 모양으로 변경, "-1" 로 크기 자동 계산
print(reshape_auto_a.size()) # 2x3x5 = 3 x n 의 방정식을 푸는 문제로 n 이 자동설정
```

    torch.Size([3, 10])
    


```python
a.reshape(7, -1) #  2x3x5 = 3 x n 의 방정식의 해가 정수가 아니면 오류 발생
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-26-7a4b40636089> in <cell line: 1>()
    ----> 1 a.reshape(7, -1) #  2x3x5 = 3 x n 의 방정식의 해가 정수가 아니면 오류 발생
    

    RuntimeError: shape '[7, -1]' is invalid for input of size 30


#### 📝 설명 : 텐서의 shape 변경
* view : 텐서의 모양을 변경합니다.

📚 참고할만한 자료:
* [view] : https://pytorch.org/docs/stable/generated/torch.Tensor.view.html


```python
print(a)
print("Shape : ", a.size()) # 텐서 모양 반환
print('\n')

view_a = a.view(5, 6) # reshape 과 동일하게 (2,3,5) 크기를 (5,6) 크기로 변경
print(view_a)
print("Shape : ", view_a.size())
```

    tensor([[[ 1.4395,  0.5985, -1.1691,  1.5279, -0.0985],
             [-2.0897,  2.5121,  0.0553,  0.2532,  1.1086],
             [-1.2554, -1.1751, -0.0270,  0.0227, -0.6094]],
    
            [[ 0.8175, -0.7804,  1.2114, -2.4871,  0.7790],
             [-0.4992, -0.0488,  1.5461, -0.1898, -1.0895],
             [ 0.0334,  2.3150, -1.0674, -0.8454, -0.4906]]])
    Shape :  torch.Size([2, 3, 5])
    
    
    tensor([[ 1.4395,  0.5985, -1.1691,  1.5279, -0.0985, -2.0897],
            [ 2.5121,  0.0553,  0.2532,  1.1086, -1.2554, -1.1751],
            [-0.0270,  0.0227, -0.6094,  0.8175, -0.7804,  1.2114],
            [-2.4871,  0.7790, -0.4992, -0.0488,  1.5461, -0.1898],
            [-1.0895,  0.0334,  2.3150, -1.0674, -0.8454, -0.4906]])
    Shape :  torch.Size([5, 6])
    


```python
view_auto_a = a.view(3, -1) # reshape 과 동일하게 (3,n)의 모양으로 변경. "-1" 로 크기 자동 계산
print(view_auto_a.size())
```

    torch.Size([3, 10])
    

#### 📝 설명 : 텐서의 shape 변경
* transpose : 텐서의 차원을 전치합니다.

📚 참고할만한 자료:
* [transpose] : https://pytorch.org/docs/stable/generated/torch.transpose.html


```python
tensor_a = torch.randint(1, 10, (3, 2, 5)) # 1 ~ 9의 값을 가지는 (3,2,5) 사이즈의 Tensor 생성
print(tensor_a)
print("Shape : ", tensor_a.size())
print('\n')

# (3,2,5) 를 (2,3,5) 의 크기로 변경
trans_a = tensor_a.transpose(1, 2) # 행과 열을 서로 전치, 서로 전치할 차원 2개를 지정
print(trans_a)
print("Shape : ", trans_a.size())
```

    tensor([[[2, 9, 8, 1, 5],
             [1, 4, 1, 7, 4]],
    
            [[3, 5, 2, 9, 9],
             [7, 8, 9, 1, 4]],
    
            [[4, 4, 6, 3, 4],
             [6, 9, 6, 2, 3]]])
    Shape :  torch.Size([3, 2, 5])
    
    
    tensor([[[2, 1],
             [9, 4],
             [8, 1],
             [1, 7],
             [5, 4]],
    
            [[3, 7],
             [5, 8],
             [2, 9],
             [9, 1],
             [9, 4]],
    
            [[4, 6],
             [4, 9],
             [6, 6],
             [3, 2],
             [4, 3]]])
    Shape :  torch.Size([3, 5, 2])
    

#### 📝 설명 : 텐서의 shape 변경
* permute : 텐서 차원의 순서를 재배열합니다.

📚 참고할만한 자료:
* [permute] : https://pytorch.org/docs/stable/generated/torch.permute.html


```python
print(tensor_a)
print("Shape : ", tensor_a.size())
print('\n')

permute_a = tensor_a.permute(0, 2, 1) # (3,2,5)의 모양을 (3,5,2)의 모양으로 변경
print(permute_a)
print("Shape : ", permute_a.size())
```

    tensor([[[2, 9, 8, 1, 5],
             [1, 4, 1, 7, 4]],
    
            [[3, 5, 2, 9, 9],
             [7, 8, 9, 1, 4]],
    
            [[4, 4, 6, 3, 4],
             [6, 9, 6, 2, 3]]])
    Shape :  torch.Size([3, 2, 5])
    
    
    tensor([[[2, 1],
             [9, 4],
             [8, 1],
             [1, 7],
             [5, 4]],
    
            [[3, 7],
             [5, 8],
             [2, 9],
             [9, 1],
             [9, 4]],
    
            [[4, 6],
             [4, 9],
             [6, 6],
             [3, 2],
             [4, 3]]])
    Shape :  torch.Size([3, 5, 2])
    

### 2-2 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습

#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* unsqueeze : 텐서에 특정 차원에 크기가 1인 차원을 추가합니다.

📚 참고할만한 자료:
* [unsqueeze] : https://pytorch.org/docs/stable/generated/torch.unsqueeze.html


```python
tensor_a = torch.tensor([i for i in range(10)]).reshape(5, 2) # 0부터 9까지의 숫자들을 (5,2) 크기로 변경
print(tensor_a)
print('Shape : ', tensor_a.size())
print('\n')

unsqu_a = tensor_a.unsqueeze(0) # 0번째 차원 하나 추가 (5,2) => (1,5,2)
print(unsqu_a)
print('Shape : ', unsqu_a.size())
```

    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
    Shape :  torch.Size([5, 2])
    
    
    tensor([[[0, 1],
             [2, 3],
             [4, 5],
             [6, 7],
             [8, 9]]])
    Shape :  torch.Size([1, 5, 2])
    


```python
unsqu_a2 = tensor_a.unsqueeze(-1) # 마지막번째에 차원 하나 추가 (5,2) => (5,2,1)
print(unsqu_a2)
print('Shape : ', unsqu_a2.size())
```

    tensor([[[0],
             [1]],
    
            [[2],
             [3]],
    
            [[4],
             [5]],
    
            [[6],
             [7]],
    
            [[8],
             [9]]])
    Shape :  torch.Size([5, 2, 1])
    

#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* squeeze : 텐서에 차원의 크기가 1인 차원을 제거합니다.

📚 참고할만한 자료:
* [squeeze] : https://pytorch.org/docs/stable/generated/torch.squeeze.html


```python
print(unsqu_a)
print("Shape : ", unsqu_a.size())
print('\n')

squ = unsqu_a.squeeze() # 차원이 1인 차원을 제거
print(squ)
print("Shape : ", squ.size())
```

    tensor([[[0, 1],
             [2, 3],
             [4, 5],
             [6, 7],
             [8, 9]]])
    Shape :  torch.Size([1, 5, 2])
    
    
    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
    Shape :  torch.Size([5, 2])
    


```python
x = torch.zeros(2, 1, 2, 1, 2) # 모든 원소가 0인 (2,1,2,1,2) 크기를 가지는 텐서
print("Shape (original) : ", x.size()) # 원래 텐서 크기
print('\n')

print("Shape (squeeze()) :", x.squeeze().size()) # 차원이 1인 차원이 여러개일 때, 모든 차원이 1인 차원 제거
print('\n')

print("Shape (squeeze(0)) :", x.squeeze(0).size()) # 0번째 차원은 차원의 크기가 1이 아니므로, 변화 없음
print('\n')

print("Shape (squeeze(1)) :", x.squeeze(1).size()) # 1번째 차원은 차원의 크기가 1이므로 제거
print('\n')

print("Shape (squeeze(0,1,3)) :", x.squeeze((0, 1, 3)).size()) # 여러 차원 제거 가능 (0번째 차원은 차원의 크기가 1이 아니기 때문에 무시)
```

    Shape (original) :  torch.Size([2, 1, 2, 1, 2])
    
    
    Shape (squeeze()) : torch.Size([2, 2, 2])
    
    
    Shape (squeeze(0)) : torch.Size([2, 1, 2, 1, 2])
    
    
    Shape (squeeze(1)) : torch.Size([2, 2, 1, 2])
    
    
    Shape (squeeze(0,1,3)) : torch.Size([2, 2, 2])
    

#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* expand : 텐서의 값을 반복하여 크기를 확장합니다.
  * A 텐서가 1차원일 경우 : A 텐서의 크기가 (m,) 이면 m은 고정하고 (x,m)의 크기로만 확장 가능
  * A 텐서가 2차원 이상일 경우 : 크기가 1인 차원에 대해서만 적용 가능. A 텐서의 크기가 (1,m) 이면 (x,m) , (m,1) 이면 (m,y) 로만 확장 가능.

📚 참고할만한 자료:
* [expand] : https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html



```python
tensor_1dim = torch.tensor([1, 2, 3, 4])
print(tensor_1dim)
print("Shape : ", tensor_1dim.size())
print('\n')

expand_tensor = tensor_1dim.expand(3, 4) # (,4) 를 (3,4) 의 크기로 확장 (값을 반복)
print(expand_tensor)
print("Shape : ", expand_tensor.size())
```

    tensor([1, 2, 3, 4])
    Shape :  torch.Size([4])
    
    
    tensor([[1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]])
    Shape :  torch.Size([3, 4])
    


```python
tensor_2dim = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]) # (2,4) 크기를 가진 Tensor
print(tensor_2dim)
print("Shape : ", tensor_2dim.size())
print('\n')

expand_tensor = tensor_2dim.expand(4,4) # (2,4) 를 (4,8) 의 크기로 확장 (값을 반복)
print(expand_tensor) # 에러 발생
print("Shape : ", expand_tensor.size()) # 에러 발생
```

    tensor([[1, 2, 3, 4],
            [1, 2, 3, 4]])
    Shape :  torch.Size([2, 4])
    
    
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-36-ca0aa7093d8c> in <cell line: 6>()
          4 print('\n')
          5 
    ----> 6 expand_tensor = tensor_2dim.expand(4,4) # (2,4) 를 (4,8) 의 크기로 확장 (값을 반복)
          7 print(expand_tensor) # 에러 발생
          8 print("Shape : ", expand_tensor.size()) # 에러 발생
    

    RuntimeError: The expanded size of the tensor (4) must match the existing size (2) at non-singleton dimension 0.  Target sizes: [4, 4].  Tensor sizes: [2, 4]


#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* repeat : 텐서를 반복하여 크기를 확장합니다.
  * ex) A 텐서가 (m,n) 크기를 가진다하고, A 텐서를 repeat(i,j) 를 하면 결과값으로 (m x i, n x j)의 크기의 텐서가 생성됩니다.

📚 참고할만한 자료:
* [repeat] : https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html


```python
tensor_1dim = torch.tensor([1, 2, 3, 4])
print(tensor_1dim)
print("Shape : ", tensor_1dim.size())
print('\n')

repeat_tensor = tensor_1dim.repeat(3, 4) # tensor_1dim 자체를 행으로 3번 반복, 열로 4번 반복
print(repeat_tensor)
print("Shape : ", repeat_tensor.size())
```

    tensor([1, 2, 3, 4])
    Shape :  torch.Size([4])
    
    
    tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])
    Shape :  torch.Size([3, 16])
    

#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* flatten : 다차원 텐서를 1차원 텐서로 변경합니다.

📚 참고할만한 자료:
* [flatten] : https://pytorch.org/docs/stable/generated/torch.flatten.html


```python
t = torch.tensor([i for i in range(20)]).reshape(2, 5, 2) # 0부터 19까지의 숫자를 4행 5열 Tensor로 구성
print(t)
print("Shape : ", t.size())
print('\n')

flat_tensor = t.flatten() # (2, 5, 2) 의 Tensor를 (20,)로 모양 변경, 1차원으로 변경
print(flat_tensor)
print("Shape : ", flat_tensor.size())
```

    tensor([[[ 0,  1],
             [ 2,  3],
             [ 4,  5],
             [ 6,  7],
             [ 8,  9]],
    
            [[10, 11],
             [12, 13],
             [14, 15],
             [16, 17],
             [18, 19]]])
    Shape :  torch.Size([2, 5, 2])
    
    
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19])
    Shape :  torch.Size([20])
    


```python
flat_tensor2 = t.flatten(start_dim=1) # flatten을 시작할 차원을 지정할 수 있음. 지정한 차원 이후의 모든 차원을 하나의 차원으로 평면화, 기본값 = 0 (1차원)
print(flat_tensor2)
print(flat_tensor2.size())
```

    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    torch.Size([2, 10])
    

#### 📝 설명 : 텐서의 차원을 추가하거나 변경하는 방법에 대한 이해 및 실습
* ravel : 다차원 텐서를 1차원 텐서로 변경합니다.

📚 참고할만한 자료:
* [ravel] : https://pytorch.org/docs/stable/generated/torch.ravel.html


```python
t = torch.tensor([i for i in range(20)]).reshape(2, 5, 2) # 0부터 19까지의 숫자를 (2, 5, 2) 크기 Tensor로 구성
print(t)
print("Shape : ", t.size())
print('\n')

ravel_tensor = t.ravel() # flatten 과 동일하게 (2,5,2) 의 텐서를 (20,)로 모양 변경, 1차원으로 변경
print(ravel_tensor)
print("Shape : ", ravel_tensor.size())
```

    tensor([[[ 0,  1],
             [ 2,  3],
             [ 4,  5],
             [ 6,  7],
             [ 8,  9]],
    
            [[10, 11],
             [12, 13],
             [14, 15],
             [16, 17],
             [18, 19]]])
    Shape :  torch.Size([2, 5, 2])
    
    
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19])
    Shape :  torch.Size([20])
    


```python
t.ravel(1) # 에러 발생, ravel 은 flatten 과 달리 어떠한 축을 기준으로 평탄화 하는 작업이 없음
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-41-4ad2f534a783> in <cell line: 1>()
    ----> 1 t.ravel(1) # 에러 발생, ravel 은 flatten 과 달리 어떠한 축을 기준으로 평탄화 하는 작업이 없음
    

    TypeError: _TensorBase.ravel() takes no arguments (1 given)


### 2-3 역할이 비슷한 함수들의 차이 이해 및 실습
> 역할이 비슷한 함수들의 공통점과 차이점을 이해하고 활용할 수 있습니다.

#### 📝 설명 : 역할이 비슷한 함수들의 차이 이해 및 실습
* 모양 변경 : view vs. reshape vs. unsqueeze
  * ※ contiguous 란?
    * 텐서의 메모리 상에 연속적인 데이터 배치를 갖는 것
    * 텐서를 처음 생성 후 정의하면 기본적으로 contiguous 하지만, 이에 대해 차원의 순서를 변경하는 과정을 거치면 contiguous 하지 않습니다.
    * 텐서의 contiguous 함을 확인하기 위해선 is_contiguous() 를 사용합니다.
  * view 는 contiguous 하지 않은 텐서에 대해서 동작하지 않습니다.
  * reshape 는 contiguous 하지 않은 텐서를 contiguous 하게 만들어주고, 크기를 변경합니다.
  * unsqueeze 는 차원의 크기가 1인 차원을 추가하지만, 차원의 크기가 1이 아니면 차원의 모양을 변경할 수 없습니다.

📚 참고할만한 자료:
* [what is contiguous?] : https://titania7777.tistory.com/3
* [view vs reshape] :  https://inmoonlight.github.io/2021/03/03/PyTorch-view-transpose-reshape/
* [view, reshape, transpose, permute 비교] : https://sanghyu.tistory.com/3


```python
# view vs reshape
tmp = torch.tensor([[[0, 1], [2, 3], [4, 5]], \
                 [[6, 7], [8, 9], [10, 11]], \
                 [[12, 13], [14, 15], [16, 17]], \
                 [[18, 19], [20, 21], [22, 23]]])
tmp_t = tmp.transpose(0,1) # contiguous 를 False 로 만들기 위한 작업
print(tmp_t.is_contiguous()) # contiguous 한지 검사
print(tmp_t.view(-1)) # view는 contiguous 하지 않은 텐서에 대해선 동작이 되지 않음
```

    False
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-42-58c86dac39d7> in <cell line: 8>()
          6 tmp_t = tmp.transpose(0,1) # contiguous 를 False 로 만들기 위한 작업
          7 print(tmp_t.is_contiguous()) # contiguous 한지 검사
    ----> 8 print(tmp_t.view(-1)) # view는 contiguous 하지 않은 텐서에 대해선 동작이 되지 않음
    

    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.



```python
reshape_tmp = tmp_t.reshape(-1) # reshape은 contiguous 하지 않아도 동작이 됨
print(reshape_tmp)
print(reshape_tmp.is_contiguous()) # contiguous 하지 않았던 Tensor를 contiguous 하게 변경해 줌
```

    tensor([ 0,  1,  6,  7, 12, 13, 18, 19,  2,  3,  8,  9, 14, 15, 20, 21,  4,  5,
            10, 11, 16, 17, 22, 23])
    True
    


```python
# (view , reshape) vs unsqueeze
tensor_a = torch.randn(2, 3)
# (2, 3) 의 텐서를 (2, 3, 1)의 크기로 변경
view_tensor = tensor_a.view(2, 3, 1) # view 를 이용하여 (2,3,1) 의 크기로 변경
reshape_tensor = tensor_a.reshape(2, 3, 1) # reshape 를 이용하여 (2,3,1) 의 크기로 변경
unsqueeze_tensor = tensor_a.unsqueeze(-1) # unsqueeze 를 이용하여 (2,3,1) 의 크기로 변경

print("View output size : ",view_tensor.size())
print("Reshape output size : ",reshape_tensor.size())
print("Unsqueeze output size : ",unsqueeze_tensor.size())
```

    View output size :  torch.Size([2, 3, 1])
    Reshape output size :  torch.Size([2, 3, 1])
    Unsqueeze output size :  torch.Size([2, 3, 1])
    

#### 📝 설명 : 역할이 비슷한 함수들의 차이 이해 및 실습
* 차원 변경 : transpose vs. permute
  * transpose : 두 차원에 대해서만 변경이 가능
    * 인자가 총 2개여야함.
  * permute : 모든 차원에 대해서 변경이 가능
    * 인자가 차원의 개수와 동일해야 함.

📚 참고할만한 자료:
* [view, reshape, transpose, permute 비교] : https://sanghyu.tistory.com/3


```python
import torch
tensor_a = torch.randn(2, 3, 2)
transpose_tensor = tensor_a.transpose(2, 1) # 행과 열을 전치
permute_tensor = tensor_a.permute(0, 2, 1) # 행과 열을 바꿈.

print("Transpose tensor shape : ", transpose_tensor.size())
print("Permute tensor shape : ", permute_tensor.size())
```

    Transpose tensor shape :  torch.Size([2, 2, 3])
    Permute tensor shape :  torch.Size([2, 2, 3])
    

#### 📝 설명 : 역할이 비슷한 함수들의 차이 이해 및 실습
* 반복을 통한 텐서 크기 확장 : expand vs. repeat
  * expand
    * 원본 텐서와 메모리를 공유한다.
  * repeat
    * 원본 텐서와 메모리를 공유하지 않는다.

📚 참고할만한 자료:
* [expand vs repeat] : https://seducinghyeok.tistory.com/9


```python
import torch

# 원본 텐서 생성
tensor_a = torch.rand(1, 1, 3)
print('Original Tensor Size')
print(tensor_a.size())
print(tensor_a)

print('\n')

# expand 사용하여 (1,1,3) => (4, 1, 3)
expand_tensor = tensor_a.expand(4, 1, -1)
print("Shape of expanded tensor:", expand_tensor.size())

print('\n')

# repeat 사용하여 (1,1,3) => (4, 1, 3)
repeat_tensor = tensor_a.repeat(4, 1, 1)
print("Shape of repeated tensor:", repeat_tensor.size())

print('\n')

# 평면화된 뷰 수정 후 원본 텐서 확인
tensor_a[:] = 0

print("Expanded Tensor")
print(expand_tensor) # 값 변경이 됨

print('\n')

print("Repeated Tensor")
print(repeat_tensor) # 깂 변경 안됨
```

    Original Tensor Size
    torch.Size([1, 1, 3])
    tensor([[[0.8880, 0.6426, 0.6744]]])
    
    
    Shape of expanded tensor: torch.Size([4, 1, 3])
    
    
    Shape of repeated tensor: torch.Size([4, 1, 3])
    
    
    Expanded Tensor
    tensor([[[0., 0., 0.]],
    
            [[0., 0., 0.]],
    
            [[0., 0., 0.]],
    
            [[0., 0., 0.]]])
    
    
    Repeated Tensor
    tensor([[[0.8880, 0.6426, 0.6744]],
    
            [[0.8880, 0.6426, 0.6744]],
    
            [[0.8880, 0.6426, 0.6744]],
    
            [[0.8880, 0.6426, 0.6744]]])
    

## 3. 텐서 합치기 나누기


- 3-1. 여러 텐서를 합치는 방법에 대한 이해 및 실습
- 3-2. 하나의 텐서를 여러 텐서로 나누는 방법에 대한 이해 및 실습


### 3-1 여러 텐서를 합치는 방법에 대한 이해 및 실습

> 여러 텐서를 하나의 텐서로 합쳐서 새로운 텐서를 생성하는 과정을 알아봅니다.



#### 📝 설명 : 여러 텐서 합치기
* cat : 주어진 차원을 따라 텐서들을 연결합니다. (주어진 차원 외의 다른 차원의 크기가 같아야합니다.)

📚 참고할만한 자료:
* [cat] : https://pytorch.org/docs/stable/generated/torch.cat.html


```python
tensor_a = torch.randint(1, 10, (2, 3)) # 1부터 9까지의 무작위 정수가 있는 (2,3) Tensor
tensor_b = torch.rand(5, 3) # 0부터 1까지의 균등분포를 따르는 (5,3) Tensor

print("Tensor A shape : ", tensor_a.size())
print(tensor_a)

print('\n')

print("Tensor B shape : ", tensor_b.size())
print(tensor_b)

print('\n')

a_cat_b_row = torch.cat((tensor_a, tensor_b), dim=0) # dim = 0 (행), Tensor A 와 Tensor B 를 행 기준으로 합친다.
print("Concat Tensor A and B (by row) Shape : ", a_cat_b_row.shape) # (Tensor A 행 개수 + Tensor B 행 개수, Tensor A/B 열 개수)
print(a_cat_b_row)
```

    Tensor A shape :  torch.Size([2, 3])
    tensor([[3, 1, 1],
            [9, 6, 7]])
    
    
    Tensor B shape :  torch.Size([5, 3])
    tensor([[0.7098, 0.1477, 0.1453],
            [0.0746, 0.5604, 0.2862],
            [0.4602, 0.4730, 0.4643],
            [0.4764, 0.1533, 0.3704],
            [0.5958, 0.8291, 0.7434]])
    
    
    Concat Tensor A and B (by row) Shape :  torch.Size([7, 3])
    tensor([[3.0000, 1.0000, 1.0000],
            [9.0000, 6.0000, 7.0000],
            [0.7098, 0.1477, 0.1453],
            [0.0746, 0.5604, 0.2862],
            [0.4602, 0.4730, 0.4643],
            [0.4764, 0.1533, 0.3704],
            [0.5958, 0.8291, 0.7434]])
    

#### 📝 설명 : 여러 텐서 합치기
* stack : 주어진 차원을 새로운 차원으로 추가하여 텐서들을 쌓습니다.
  * 합쳐질 텐서들의 크기는 모두 같아야합니다.

📚 참고할만한 자료:
* [stack] : https://pytorch.org/docs/stable/generated/torch.stack.html


```python
tensor_a = torch.randint(1, 10, (3, 2))  # 1부터 9까지의 무작위 정수가 있는 (3,2) Tensor
tensor_b = torch.rand(3, 2)  # 0부터 1까지의 균등분포를 따르는 (3,2) Tensor

print("Tensor A shape : ", tensor_a.size())
print(tensor_a)

print('\n')

print("Tensor B shape : ", tensor_b.size())
print(tensor_b)

print('\n')

stack_tensor_row = torch.stack([tensor_a, tensor_b], dim=0)  # dim = 0, 행을 기준으로 Tensor A 에 Tensor B 를 쌓기
print("Stack A and B (by row): ", stack_tensor_row.size()) # (쌓은 Tensor 개수, Tensor A/B 행 개수, Tensor A/B 열 개수)
print(stack_tensor_row)
```

    Tensor A shape :  torch.Size([3, 2])
    tensor([[9, 9],
            [1, 6],
            [2, 2]])
    
    
    Tensor B shape :  torch.Size([3, 2])
    tensor([[0.5477, 0.6201],
            [0.8358, 0.7233],
            [0.4310, 0.2094]])
    
    
    Stack A and B (by row):  torch.Size([2, 3, 2])
    tensor([[[9.0000, 9.0000],
             [1.0000, 6.0000],
             [2.0000, 2.0000]],
    
            [[0.5477, 0.6201],
             [0.8358, 0.7233],
             [0.4310, 0.2094]]])
    

### 3-2. 하나의 텐서를 여러 텐서로 나누는 방법에 대한 이해 및 실습

> 하나의 텐서를 다양한 방법을 통해 여러 텐서로 나누는 과정을 알아봅니다.

#### 📝 설명 : 텐서 나누기
* chunk : 나누고자 하는 **텐서의 개수**를 지정하여 원래의 텐서를 개수에 맞게 분리합니다.
  * chunks 인자
    * 몇 **개**의 텐서로 나눌 것인지

📚 참고할만한 자료:
* [chunk] : https://pytorch.org/docs/stable/generated/torch.chunk.html


```python
tensor_a = torch.randint(1, 10, (6, 4))  # (6,4) 텐서
print("Original : ", tensor_a)

print('\n')

chunk_num = 3
chunk_tensor = torch.chunk(tensor_a, chunks = chunk_num, dim=0)  # dim = 0 (행), 6개의 행이 3개로 나누어 떨어지므로 3개의 텐서로 분리
print(f'{len(chunk_tensor)} 개의 Tensor로 분리')

print('\n')

for idx,a in enumerate(chunk_tensor):
    print(f'{idx} 번째 Tensor \n{a}')
    print(f'{idx} 번째 Tensor 크기', a.size())
    print('---'*10)
```

    Original :  tensor([[9, 2, 4, 5],
            [8, 8, 9, 1],
            [5, 6, 6, 3],
            [7, 8, 3, 2],
            [1, 9, 6, 7],
            [9, 7, 4, 2]])
    
    
    3 개의 Tensor로 분리
    
    
    0 번째 Tensor 
    tensor([[9, 2, 4, 5],
            [8, 8, 9, 1]])
    0 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    1 번째 Tensor 
    tensor([[5, 6, 6, 3],
            [7, 8, 3, 2]])
    1 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    2 번째 Tensor 
    tensor([[1, 9, 6, 7],
            [9, 7, 4, 2]])
    2 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    

#### 📝 설명 : 텐서 나누기
* split : 입력한 **크기**로 여러 개의 작은 텐서로 나눕니다.
  * split_size_or_sections 인자
    * split_size (int): 얼마만큼의 크기로 자를 것인지
    * sections (list): 얼마만큼의 크기로 **각각** 자를 것인지 (리스트 형태로 각 텐서의 크기를 각각 지정해 줄 수 있음)

📚 참고할만한 자료:
* [split] : https://pytorch.org/docs/stable/generated/torch.split.html


```python
tensor_a = torch.randint(1, 10, (6, 4))  # (6,4) 텐서
print(tensor_a)

print('\n')

split_size = 2
split_tensor = torch.split(tensor_a , split_size_or_sections = split_size, dim=0)  # dim = 0 (행), 텐서 A 를 행의 길이가 2 (split_size)인 텐서로 나눔
print(f'{len(split_tensor)} 개의 Tensor로 분리')

print('\n')

for idx,a in enumerate(split_tensor):
    print(f'{idx} 번째 Tensor \n{a}')
    print(f'{idx} 번째 Tensor 크기', a.size())
    print('---'*10)
```

    tensor([[3, 3, 3, 4],
            [3, 9, 4, 2],
            [7, 4, 8, 2],
            [4, 8, 9, 8],
            [4, 9, 8, 5],
            [1, 7, 7, 8]])
    
    
    3 개의 Tensor로 분리
    
    
    0 번째 Tensor 
    tensor([[3, 3, 3, 4],
            [3, 9, 4, 2]])
    0 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    1 번째 Tensor 
    tensor([[7, 4, 8, 2],
            [4, 8, 9, 8]])
    1 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    2 번째 Tensor 
    tensor([[4, 9, 8, 5],
            [1, 7, 7, 8]])
    2 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    


```python
tensor_a = torch.randint(1, 10, (6, 4))  # (6,4) 텐서
print("Original : ", tensor_a)

print('\n')

split_num = [2, 4]
split_tensor = torch.split(tensor_a, split_size_or_sections = split_num, dim=0)  # dim = 0 (행), 텐서 A 를 행의 길이가 (2개인 텐서와 4개인 텐서)로 나눔
print(f'{len(split_tensor)} 개의 Tensor로 분리')

print('\n')

for idx,a in enumerate(split_tensor):
    print(f'{idx} 번째 Tensor \n{a}')
    print(f'{idx} 번째 Tensor 크기', a.size())
    print('---'*10)
```

    Original :  tensor([[5, 9, 4, 3],
            [8, 4, 4, 6],
            [3, 6, 3, 3],
            [3, 7, 7, 9],
            [2, 8, 8, 8],
            [5, 6, 1, 5]])
    
    
    2 개의 Tensor로 분리
    
    
    0 번째 Tensor 
    tensor([[5, 9, 4, 3],
            [8, 4, 4, 6]])
    0 번째 Tensor 크기 torch.Size([2, 4])
    ------------------------------
    1 번째 Tensor 
    tensor([[3, 6, 3, 3],
            [3, 7, 7, 9],
            [2, 8, 8, 8],
            [5, 6, 1, 5]])
    1 번째 Tensor 크기 torch.Size([4, 4])
    ------------------------------
    

#Reference
> <b><font color = green>(📒가이드)
- <a href='https://pytorch.org/docs/stable/index.html'>PyTorch 공식 문서</a>
- <a href='https://inmoonlight.github.io/2021/03/03/PyTorch-view-transpose-reshape/'>view, transpose, reshape 비교</a>

## Required Package

> torch == 2.0.1

## 콘텐츠 라이선스

저작권 : <font color='blue'> <b> ©2023 by Upstage X fastcampus Co., Ltd. All rights reserved.</font></b>

<font color='red'><b>WARNING</font> : 본 교육 콘텐츠의 지식재산권은 업스테이지 및 패스트캠퍼스에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다. </b>


```python

```

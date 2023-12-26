---
layout: post
title: Pytorch_텐서 실습 2
date: 2023-12-26 17:00 +0900
last_modified_at: 2023-12-26 17:00:00 +0900
tags: [deeplearning, Pytorch, tensor]
toc:  true
---

# 텐서 조작 (2)

### 환경 설정
> PyTorch 설치 및 불러오기

<font color = blue><b>
- 패키지 설치 및 임포트
</font><b>


```python
!pip install torch==2.0.1 # PyTorch 를 가장 최근 버전으로 설치
```

    Requirement already satisfied: torch==2.0.1 in /usr/local/lib/python3.10/dist-packages (2.0.1+cu118)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.12.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (4.7.1)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (1.11.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (3.1.2)
    Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch==2.0.1) (2.0.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch==2.0.1) (3.25.2)
    Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch==2.0.1) (16.0.6)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch==2.0.1) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch==2.0.1) (1.3.0)
    


```python
import torch # PyTorch 불러오기
import numpy as np # numpy 불러오기
import warnings # 경고 문구 제거
warnings.filterwarnings('ignore')
```

## 1. 텐서 연산 및 조작

- 1-1. 텐서 간의 계산 실습
- 1-2. Broadcasting 을 이용한 텐서 값 변경
- 1-3. Broadcasting 을 이용한 차원이 다른 텐서 간의 계산 실습


### 1-1 텐서 간의 계산 실습
> 텐서 간의 계산과 텐서 내의 계산 과정을 알아봅니다.



#### 📝 설명 : 텐서 간의 사칙연산
* add : 텐서 간의 덧셈을 수행합니다. (+)
    * torch.add(a, b)
    * a.add(b)
    * a + b
* sub : 텐서 간의 뺄셈을 수행합니다. (-)
    * torch.sub(a, b)
    * a.sub(b)
    * a - b
* mul : 텐서 간의 곱셈을 수행합니다. (*)
    * torch.mul(a, b)
    * a.mul(b)
    * a * b
* div : 텐서 간의 나눗셈을 수행합니다. (/)
    * torch.div(a, b)
    * a.div(b)
    * a / b

📚 참고할만한 자료:
* [add] https://pytorch.org/docs/stable/generated/torch.add.html
* [sub] https://pytorch.org/docs/stable/generated/torch.add.html
* [mul] https://pytorch.org/docs/stable/generated/torch.add.html
* [div] https://pytorch.org/docs/stable/generated/torch.add.html



```python
tensor_a = torch.tensor([[1, -1], [2, 3]])
tensor_b = torch.tensor([[2, -2] ,[3, 1]])

print('덧셈')
print("a+b : \n", tensor_a + tensor_b)
print('\n')
print("torch.add(a,b) : \n", torch.add(tensor_a, tensor_b))

print('---'*10)

print('뺄셈')
print("a-b : \n", tensor_a - tensor_b)
print('\n')
print("torch.sub(a,b) : \n", torch.sub(tensor_a, tensor_b))

print('---'*10)

print('곱셈')
print("a*b : \n", tensor_a * tensor_b)
print('\n')
print("torch.mul(a,b) : \n", torch.mul(tensor_a, tensor_b))

print('---'*10)

print('나눗셈')
print("a/b : \n", tensor_a / tensor_b)
print('\n')
print("torch.div(a,b) : \n", torch.div(tensor_a, tensor_b))
```

    덧셈
    a+b : 
     tensor([[ 3, -3],
            [ 5,  4]])
    
    
    torch.add(a,b) : 
     tensor([[ 3, -3],
            [ 5,  4]])
    ------------------------------
    뺄셈
    a-b : 
     tensor([[-1,  1],
            [-1,  2]])
    
    
    torch.sub(a,b) : 
     tensor([[-1,  1],
            [-1,  2]])
    ------------------------------
    곱셈
    a*b : 
     tensor([[2, 2],
            [6, 3]])
    
    
    torch.mul(a,b) : 
     tensor([[2, 2],
            [6, 3]])
    ------------------------------
    나눗셈
    a/b : 
     tensor([[0.5000, 0.5000],
            [0.6667, 3.0000]])
    
    
    torch.div(a,b) : 
     tensor([[0.5000, 0.5000],
            [0.6667, 3.0000]])
    

#### 📝 설명 : 텐서의 통계치
함수의 dim 파라미터 값에 따라 결과가 달라지는 것을 유의하세요❗
* sum : 텐서의 원소들의 합을 반환

📚 참고할만한 자료:
* [sum] : https://pytorch.org/docs/stable/generated/torch.sum.html


```python
tensor_a = torch.tensor([[1, 2], [3, 4]])
print(tensor_a)
print("Shape : ", tensor_a.size())

print('\n')

print("dimension 지정 안했을 때 : ", torch.sum(tensor_a))  # 모든 원소의 합을 반환 함
print("dim = 0 일 때 : ", torch.sum(tensor_a, dim=0))  # 행을 기준 (행 인덱스 변화)으로 합함 (0행 0열 + 1행 0열, 0행 1열 + 1행 1열)
print("dim = 1 일 때 : ", torch.sum(tensor_a, dim=1)) # 열을 기준 (열 인덱스 변화)으로 합함 (0행 0열 + 0행 1열, 1행 0열 + 1행 1열)
```

    tensor([[1, 2],
            [3, 4]])
    Shape :  torch.Size([2, 2])
    
    
    dimension 지정 안했을 때 :  tensor(10)
    dim = 0 일 때 :  tensor([4, 6])
    dim = 1 일 때 :  tensor([3, 7])
    

#### 📝 설명 : 텐서의 통계치
함수의 dim 파라미터 값에 따라 결과가 달라지는 것을 유의하세요❗
* mean : 텐서의 원소들의 평균을 반환
📚 참고할만한 자료:
* [mean] : https://pytorch.org/docs/stable/generated/torch.mean.html


```python
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32) # mean 은 실수가 나올 수 있으므로 float 로 지정해주어야 함.
print(tensor_a)
print("Shape : ", tensor_a.size())

print('\n')

print("dimension 지정 안했을 때 : ", torch.mean(tensor_a))  # 모든 원소의 평균을 반환 함
print("dim = 0 일 때 : ", torch.mean(tensor_a, dim=0))  # 행을 기준 (행 인덱스 변화)으로 평균 구함 ((0행 0열 + 1행 0열)/2, (0행 1열 + 1행 1열)/2)
print("dim = 1 일 때 : ", torch.mean(tensor_a, dim=1)) # 열을 기준 (열 인덱스 변화)으로 평균 구함 ((0행 0열 + 0행 1열)/2, (1행 0열 + 1행 1열)/2)
```

    tensor([[1., 2.],
            [3., 4.]])
    Shape :  torch.Size([2, 2])
    
    
    dimension 지정 안했을 때 :  tensor(2.5000)
    dim = 0 일 때 :  tensor([2., 3.])
    dim = 1 일 때 :  tensor([1.5000, 3.5000])
    

#### 📝 설명 : 텐서의 통계치
함수의 dim 파라미터 값에 따라 결과가 달라지는 것을 유의하세요❗
* max : 텐서의 원소들의 가장 큰 값을 반환
* min : 텐서의 원소들의 가장 작은 값을 반환

📚 참고할만한 자료:
* [max] : https://pytorch.org/docs/stable/generated/torch.max.html
* [min] : https://pytorch.org/docs/stable/generated/torch.min.html


```python
import torch
tensor_a = torch.tensor([[1, 2], [3, 4]])
print(tensor_a)
print("Shape : ", tensor_a.size())
print('\n')

print("dimension 지정 안했을 때 : ", torch.max(tensor_a))  # 모든 원소 중 최댓값 반환
print("dim = 0 일 때 : ", torch.max(tensor_a, dim=0).values)  # 행을 기준 (행 인덱스 변화)으로 max 비교 (max(0행 0열 , 1행 0열), max(0행 1열 , 1행 1열))
print("dim = 1 일 때 : ", torch.max(tensor_a, dim=1).values) # 열을 기준 (열 인덱스 변화)으로 max 비교 (max(0행 0열 , 0행 1열), max(1행 0열 , 1행 1열))
print('\n')

print("dimension 지정 안했을 때 : ", torch.min(tensor_a))  # 모든 원소의 최솟값 반환 함
print("dim = 0 일 때 : ", torch.min(tensor_a, dim=0).values)  # 행을 기준 (행 인덱스 변화)으로 min 비교 (min(0행 0열 , 1행 0열), min(0행 1열 , 1행 1열))
print("dim = 1 일 때 : ", torch.min(tensor_a, dim=1).values) # 열을 기준 (열 인덱스 변화)으로 min 비교 (min(0행 0열 , 0행 1열), min(1행 0열 , 1행 1열))
```

    tensor([[1, 2],
            [3, 4]])
    Shape :  torch.Size([2, 2])
    
    
    dimension 지정 안했을 때 :  tensor(4)
    dim = 0 일 때 :  tensor([3, 4])
    dim = 1 일 때 :  tensor([2, 4])
    
    
    dimension 지정 안했을 때 :  tensor(1)
    dim = 0 일 때 :  tensor([1, 2])
    dim = 1 일 때 :  tensor([1, 3])
    

#### 📝 설명 : 텐서의 통계치
함수의 dim 파라미터 값에 따라 결과가 달라지는 것을 유의하세요❗
* argmax : 텐서의 원소들의 가장 큰 값의 **위치** 반환
* argmin : 텐서의 원소들의 가장 작은 값의 **위치** 반환

📚 참고할만한 자료:
* [argmax] : https://pytorch.org/docs/stable/generated/torch.argmax.html
* [argmin] : https://pytorch.org/docs/stable/generated/torch.argmin.html


```python
tensor_a = torch.tensor([[1, 2], [3, 4]])
print(tensor_a)
print("Shape : ",tensor_a.size())
print('\n')

print("dimension 지정 안했을 때 : ", torch.argmax(tensor_a))  # 모든 원소 중 최댓값 위치 반환함
print("dim = 0 일 때 : ", torch.argmax(tensor_a, dim=0))  # 행을 기준 (행 인덱스 변화)으로 max 비교 (max(0행 0열 , 1행 0열), max(0행 1열 , 1행 1열)) => 위치 반환
print("dim = 1 일 때 : ", torch.argmax(tensor_a, dim=1)) # 열을 기준 (열 인덱스 변화)으로 max 비교 (max(0행 0열 , 0행 1열), max(1행 0열 , 1행 1열)) => 위치 반환

print('\n')

print("dimension 지정 안했을 때 : ", torch.argmin(tensor_a))  # 모든 원소의 최솟값 위치 반환 함
print("dim = 0 일 때 : ", torch.argmin(tensor_a, dim=0))  # 행을 기준 (행 인덱스 변화)으로 min 비교 (min(0행 0열 , 1행 0열), min(0행 1열 , 1행 1열)) => 위치 반환
print("dim = 1 일 때 : ", torch.argmin(tensor_a, dim=1)) # 열을 기준 (열 인덱스 변화)으로 min 비교 (min(0행 0열 , 0행 1열), min(1행 0열 , 1행 1열)) => 위치 반환
```

    tensor([[1, 2],
            [3, 4]])
    Shape :  torch.Size([2, 2])
    
    
    dimension 지정 안했을 때 :  tensor(3)
    dim = 0 일 때 :  tensor([1, 1])
    dim = 1 일 때 :  tensor([1, 1])
    
    
    dimension 지정 안했을 때 :  tensor(0)
    dim = 0 일 때 :  tensor([0, 0])
    dim = 1 일 때 :  tensor([0, 0])
    

#### 📝 설명 : 행렬 및 벡터 계산
* dot : **벡터**의 내적 (inner product) 반환
  * torch.dot(a,b)
  * a.dot(b)

📚 참고할만한 자료:
* [dot] : https://pytorch.org/docs/stable/generated/torch.dot.html


```python
v1 = torch.tensor([1, 2])
u1 = torch.tensor([3, 4])

print("v1.dot(u1) : ", v1.dot(u1)) # v1 과 u1 내적 (torch.tensor 에도 dot 함수 존재)
print("torch.dot(v1, u1) : ", torch.dot(v1, u1)) # v1 과 u1 내적
```

    v1.dot(u1) :  tensor(11)
    torch.dot(v1, u1) :  tensor(11)
    

#### 📝 설명 : 행렬 및 벡터 계산
* matmul : 두 텐서 간의 행렬곱 반환 ***※ 원소 곱과 다름 주의❗***
  * torch.matmul(a,b)
  * a.matmul(b)

📚 참고할만한 자료:
* [matmul] : https://pytorch.org/docs/stable/generated/torch.matmul.html


```python
A = torch.tensor([[1, 2], [3, 4]])  # (2,2) Tensor
B = torch.tensor([[-1, 2], [1, 0]])  # (2,2) Tensor
print("A: ", A)
print("B: ", B)

print('\n')

print("AB : ", torch.matmul(A, B)) # A에서 B를 행렬곱
print("BA : ", B.matmul(A))  # B에서 A를 행렬곱
```

    A:  tensor([[1, 2],
            [3, 4]])
    B:  tensor([[-1,  2],
            [ 1,  0]])
    
    
    AB :  tensor([[1, 2],
            [1, 6]])
    AB :  tensor([[1, 2],
            [1, 6]])
    BA :  tensor([[5, 6],
            [1, 2]])
    

### 1-2. Broadcasting 을 이용한 텐서 값 변경
> Broadcasting 을 이용하여 텐서의 원소를 변경하는 방법에 대해 이해하고 실습합니다.

#### 📝 설명 : Broadcasting 을 이용한 텐서 원소 변경
* scalar 값으로 텐서 원소 변경하기
  * Indexing으로 텐서 원소에 접근 후 scalar 값으로 원소 변경

📚 참고할만한 자료:
* [Broadcasing semantics] : https://pytorch.org/docs/stable/notes/broadcasting.html


```python
tensor_a = torch.randn(3, 2)
print("Original : \n", tensor_a)

print('\n')

## 0 행의 모든 열을 10 으로 변경하기
tensor_a[0, :] = 10 # 0행의 모든 열에 broadcasting 을 통한 scalar 값 대입
print("변경된 텐서 : \n", tensor_a)
```

    Original : 
     tensor([[ 0.8310, -0.0577],
            [ 1.3267,  0.9531],
            [ 0.4545, -0.8515]])
    
    
    변경된 텐서 : 
     tensor([[10.0000, 10.0000],
            [ 1.3267,  0.9531],
            [ 0.4545, -0.8515]])
    

#### 📝 설명 : Broadcasting 을 이용한 텐서 원소 변경
* 텐서 값으로 텐서 원소 변경하기
  * Indexing으로 텐서 원소에 접근 후 텐서 값으로 원소 변경

📚 참고할만한 자료:
* [Broadcasing semantics] : https://pytorch.org/docs/stable/notes/broadcasting.html


```python
tensor_a = torch.randn(3, 2)
print("Original : \n", tensor_a)

print('\n')

## 모든 값을 tensor [0,1]로 변경하기
tensor_a[:, :] = torch.tensor([0, 1]) # 모든 값에 접근하여 [0,1] 로 변경
print("변경된 Tensor : \n", tensor_a)
```

    Original : 
     tensor([[ 0.2746,  1.4863],
            [ 0.4195,  1.0571],
            [-1.6873,  2.0483]])
    
    
    변경된 Tensor : 
     tensor([[0., 1.],
            [0., 1.],
            [0., 1.]])
    

### 1-3. Broadcasting 을 이용한 차원이 다른 텐서 간의 계산 실습
> Broadcasting 을 이용하여 차원이 다른 텐서 간의 계산 방식에 대해 이해하고 실습합니다.

#### 📝 설명 : Broadcasting 을 이용한 계산
* 차원이 서로 다른 텐서 간의 계산을 broadcasting 을 통해 할 수 있습니다.

📚 참고할만한 자료:
* [Broadcasing semantics] : https://pytorch.org/docs/stable/notes/broadcasting.html


```python
tensor_a = torch.eye(3)
print("Tensor A : \n",tensor_a)

print('\n')

tensor_b = torch.tensor([1, 2, 3])
print("Tensor B : \n", tensor_b)

print('\n')

print('A + B : \n', tensor_a + tensor_b) # broadcasting을 통해 (3,) 인 B가 (3,3)으로 변환되어 계산 (행의 확장)
```

    Tensor A : 
     tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    
    
    Tensor B : 
     tensor([1, 2, 3])
    
    
    A + B : 
     tensor([[2., 2., 3.],
            [1., 3., 3.],
            [1., 2., 4.]])
    


```python
tensor_a = torch.eye(3)
print("Tensor A : \n", tensor_a)

print('\n')

tensor_b = torch.tensor([1, 2, 3]).reshape(3, 1) # 행 벡터로 형식으로 변환
print("Tensor B : \n", tensor_b)

print('\n')

print('A + B : \n', tensor_a + tensor_b) # broadcasting을 통해 (3,1) 인 B가 (3,3)으로 변환되어 계산 (열의 확장)
```

    Tensor A : 
     tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    
    
    Tensor B : 
     tensor([[1],
            [2],
            [3]])
    
    
    A + B : 
     tensor([[2., 1., 1.],
            [2., 3., 2.],
            [3., 3., 4.]])
    

#### 📝 설명 : Broadcasting 을 이용한 계산
* 차원의 맞지 않는 경우, 차원을 추가하여 broadcasting 으로 텐서 간의 계산을 할 수 있습니다.

📚 참고할만한 자료:
* [Broadcasing semantics] : https://pytorch.org/docs/stable/notes/broadcasting.html


```python
tensor_a = torch.randn(3, 2, 5)
mean_a = tensor_a.mean(2) # 열 기준 평균값
print(f"Tensor size : {tensor_a.size()}, mean size : {mean_a.size()}")

print('\n')

print(tensor_a - mean_a)  # 에러 발생! 차원이 달라서 계산이 되지 않음
```

    Tensor size : torch.Size([3, 2, 5]), mean size : torch.Size([3, 2])
    
    
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-15-1f36eac833e5> in <cell line: 7>()
          5 print('\n')
          6 
    ----> 7 print(tensor_a - mean_a )  # 에러 발생! 차원이 달라서 계산이 되지 않음
    

    RuntimeError: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 2



```python
# 차원 생성 후 broadcasting
unseq_mean = mean_a.unsqueeze(-1) # 마지막 축 추가
print(unseq_mean.size())

print('\n')

print(tensor_a - unseq_mean)
```

    torch.Size([3, 2, 1])
    
    
    tensor([[[-1.0898, -0.3991, -0.3112,  1.4851,  0.3149],
             [-1.0677, -0.1642, -0.2781, -0.3280,  1.8380]],
    
            [[-0.8618, -0.0473,  0.9947,  0.6992, -0.7848],
             [ 0.2205, -1.5032,  1.0109,  1.1358, -0.8640]],
    
            [[-0.9111, -0.7767,  0.6328,  0.3092,  0.7459],
             [-0.5014, -1.0947,  1.4581, -0.8344,  0.9724]]])
    

## 2. Sparse Tensor 조작 및 실습

- 2-1. COO Tensor 에 대한 이해 및 실습
- 2-2. CSC/CSR Tensor 에 대한 이해 및 실습
- 2-3. Sparse Tensor의 필요성 이해 및 실습
- 2-4. Sparse Tensor 의 조작 예시


### 2-1 COO Sparse Tensor에 대한 실습

> Sparse tensor 로 변환하는 방법 중 COO 방식에 대해 알아보고 실습합니다.


```python
a = torch.tensor([[0, 2.], [3, 0]])

a.to_sparse() # COO Sparse tensor 로 변환
```




    tensor(indices=tensor([[0, 1],
                           [1, 0]]),
           values=tensor([2., 3.]),
           size=(2, 2), nnz=2, layout=torch.sparse_coo)



#### 📝 설명 : COO Sparse Tensor 생성
* sparse_coo_tensor : COO 형식의 sparse tensor 를 생성하는 함수
  * indices : 0 이 아닌 값을 가진 행,열의 위치
  * values : 0 이 아닌 값
  * nnz : 0 이 아닌 값의 개수


📚 참고할만한 자료:
* [sparse_coo_tensor] : https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html


```python
indices = torch.tensor([[0, 1, 1],[2, 0, 1]]) # 0이 아닌 값의 (index, column) pairs
values = torch.tensor([4, 5, 6]) # 0 이 아닌 값의 values, values의 사이
sparse_tensor = torch.sparse_coo_tensor(indices = indices, values = values, size=(2, 3)) # (2,3)의 sparse tensor

print(sparse_tensor)
print('\n')
print(sparse_tensor.to_dense())
```

    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 1]]),
           values=tensor([4, 5, 6]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)
    
    
    tensor([[0, 0, 4],
            [5, 6, 0]])
    

### 2-2 CSR/CSC Sparse Tensor에 대한 실습

> Sparse tensor 로 변환하는 방법 중 CSR/CSC 방식에 대해 알아보고 실습합니다.

#### 📝 설명 : CSR Sparse Tensor 로 변환
* to_sparse_csr : Dense tensor를 CSR 형식의 Sparse tensor로 변환하는 함수
  * crow_indices : 0 이 아닌 값을 가진 행의 위치 (첫번째는 무조건 0)
  * col_indices : 0 이 아닌 값을 가진 열의 위치
  * values : 0 이 아닌 값
  * nnz : 0 이 아닌 값의 개수
  
📚 참고할만한 자료:
* [csr] : https://pytorch.org/docs/stable/sparse.html#sparse-csr-docs



```python
t = torch.tensor([[0, 0, 4, 3], [5, 6, 0, 0]])
print("Shape : ", t.size())
print(t)

print('\n')

t.to_sparse_csr()  # Dense Tensor를 CSR Sparse Tensor 형식으로 변환
```

    Shape :  torch.Size([2, 4])
    tensor([[0, 0, 4, 3],
            [5, 6, 0, 0]])
    
    
    




    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([2, 3, 0, 1]),
           values=tensor([4, 3, 5, 6]), size=(2, 4), nnz=4,
           layout=torch.sparse_csr)



#### 📝 설명 : CSC Sparse Tensor 로 변환
* to_sparse_csc : Dense tensor를 CSC 형식의 Sparse tensor로 변환하는 함수
  * ccol_indices : 0 이 아닌 값의 열 위치 (첫번째 원소는 무조건 0)
  * row_indices : 0 이 아닌 값의 행 위치
  * values : 0 이 아닌 값들
  * nnz : 0 이 아닌 값의 개수
  
📚 참고할만한 자료:
* [csc] : https://pytorch.org/docs/stable/sparse.html#sparse-csc-docs


```python
t = torch.tensor([[0, 0, 4, 3], [5, 6, 0, 0]])
print("Shape : ", t.size())
print(t)

print('\n')

t.to_sparse_csc()  # Dense Tensor 를 CSC Spare tensor 형식으로 변환
```

    Shape :  torch.Size([2, 4])
    tensor([[0, 0, 4, 3],
            [5, 6, 0, 0]])
    
    
    




    tensor(ccol_indices=tensor([0, 1, 2, 3, 4]),
           row_indices=tensor([1, 1, 0, 0]),
           values=tensor([5, 6, 4, 3]), size=(2, 4), nnz=4,
           layout=torch.sparse_csc)



#### 📝 설명 : CSR Sparse Tensor 생성
* sparse_csr_tensor : CSR 형식의 Sparse tensor 를 생성하는 함수

📚 참고할만한 자료:
* [sparse_csr_tensor] : https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html


```python
crow_indices = torch.tensor([0, 2, 2]) # 0이 아닌 행의 위치 (첫번쨰는 무조건 0), 즉 row_pointer
col_indices = torch.tensor([0, 1]) # 0이 아닌 열의 위치
values = torch.tensor([1, 2]) # 0이 아닌 값
csr = torch.sparse_csr_tensor(crow_indices = crow_indices, col_indices = col_indices, values = values)

print(csr)
print('\n')
print(csr.to_dense())
```

    tensor(crow_indices=tensor([0, 2, 2]),
           col_indices=tensor([0, 1]),
           values=tensor([1, 2]), size=(2, 2), nnz=2, layout=torch.sparse_csr)
    
    
    tensor([[1, 2],
            [0, 0]])
    

#### 📝 설명 : CSC Sparse Tensor 생성
* sparse_csc_tensor : CSC 형식의 Sparse tensor 를 생성하는 함수

📚 참고할만한 자료:
* [sparse_csc_tensor] : https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html


```python
ccol_indices = torch.tensor([0, 2, 2]) # 0이 아닌 열의 위치 (첫번쨰는 무조건 0), 즉 column_pointer
row_indices = torch.tensor([0, 1]) # 0이 아닌 행의 위치
values = torch.tensor([1, 2]) # 0이 아닌 값
csc = torch.sparse_csc_tensor(ccol_indices = ccol_indices, row_indices = row_indices, values = values)

print(csc)
print('\n')
print(csc.to_dense())
```

    tensor(ccol_indices=tensor([0, 2, 2]),
           row_indices=tensor([0, 1]),
           values=tensor([1, 2]), size=(2, 2), nnz=2, layout=torch.sparse_csc)
    
    
    tensor([[1, 0],
            [2, 0]])
    

### 2-3 Sparse Tensor의 필요성 이해 및 실습

> Dense tensor가 가지는 한계점에 대해 이해하고, Sparse tensor 가 필요한 이유에 대해 알아봅니다.

#### 📝 설명 : Sparse Tensor 의 필요성
* 아주 큰 크기의 matrix 를 구성할 때, 일반적인 dense tensor 는 메모리 아웃 현상이 발생하지만, sparse tensor 는 메모리 아웃현상이 발생하지 않습니다.
  * to_dense() : sparse tensor 를 dense tensor 로 만드는 함수


```python
i = torch.randint(0, 100000, (200000,)).reshape(2, -1)
v = torch.rand(100000)
coo_sparse_tensor = torch.sparse_coo_tensor(indices = i, values = v, size = [100000, 100000]) # COO Sparse Tensor (100000 x 100000)
```


```python
crow = torch.randint(0, 100000, (100000,))
col = torch.randint(0, 100000, (100000,))
v = torch.rand(100000)
csr_sparse_tensor = torch.sparse_csr_tensor(crow_indices = crow, col_indices = col, values = v) # CSR Sparse Tensor (100000 x 100000)
```

    <ipython-input-2-f025b39a270c>:4: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at ../aten/src/ATen/SparseCsrTensorImpl.cpp:54.)
      csr_sparse_tensor = torch.sparse_csr_tensor(crow_indices = crow, col_indices = col, values = v) # CSR Sparse Tensor (100000 x 100000)
    


```python
coo_sparse_tensor.to_dense() # COO 형식으로 만들어진 Sparse Tensor 를 Dense Tensor 로 변환 , 메모리 아웃
```


```python
import torch # 커널 재시작 하기에 다시 torch library 로드
```


```python
csr_sparse_tensor.to_dense() # CSR 형식으로 만들어진 Sparse Tensor 를 Dense Tensor 로 변환 , 메모리 아웃
```


```python
import torch # 커널 재시작 하기에 다시 torch library 로드
```

### 2-4 Sparse Tensor의 조작 방법

> Sparse tensor 의 Indexing 과 연산 방법에 대해 알아봅니다.

#### 📝 설명 : Sparse Tensor 의 연산 (=2차원)
* 2차원 sparse tensor 간에는 일반 텐서와 동일하게 사칙연산 함수들과 행렬곱을 사용할 수 있습니다.

📚 참고할만한 자료:
* [sparse] : https://pytorch.org/docs/stable/sparse.html


```python
# Sparse 와 Sparse Tensor 간의 연산 (2차원)
a = torch.tensor([[0, 1], [0, 2]], dtype=torch.float)
b = torch.tensor([[1, 0],[0, 0]], dtype=torch.float)

sparse_a = a.to_sparse()
sparse_b = b.to_sparse()

print('덧셈')
print(torch.add(a, b).to_dense() == torch.add(sparse_a, sparse_b).to_dense())

print('\n')
print('곱셈')
print(torch.mul(a, b).to_dense() == torch.mul(sparse_a, sparse_b).to_dense())

print('\n')

print('행렬곱')
print(torch.matmul(a, b).to_dense() == torch.matmul(sparse_a, sparse_b).to_dense())
```

    덧셈
    tensor([[True, True],
            [True, True]])
    
    
    곱셈
    tensor([[True, True],
            [True, True]])
    
    
    행렬곱
    tensor([[True, True],
            [True, True]])
    

#### 📝 설명 : Sparse Tensor 의 연산 (=3차원)
* 3차원 sparse tensor 에는 일반 텐서와 동일하게 사칙연산 함수들은 사용 가능하지만 행렬곱을 사용할 수 없습니다.
  * CSR/CSC 형식에서는 곱셈도 3차원에선 불가능합니다.
* 이는 sparse tensor 와 sparse tensor 간에도 적용이 되고, sparse tensor 와 dense tensor 간의 연산에도 적용이 됩니다.


```python
# Sparse 와 Sparse Tensor 간의 연산 (3차원)
a = torch.tensor([[[0, 1], [0, 2]], [[0, 1], [0, 2]]], dtype=torch.float)
b = torch.tensor([[[1, 0],[0, 0]], [[1, 0], [0, 0]]], dtype=torch.float)

sparse_a = a.to_sparse()
sparse_b = b.to_sparse()

print('덧셈')
print(torch.add(a, b).to_dense() == torch.add(sparse_a, sparse_b).to_dense())

print('\n')
print('곱셈')
print(torch.mul(a, b).to_dense() == torch.mul(sparse_a, sparse_b).to_dense())

print('\n')

print('행렬곱')
print(torch.matmul(a, b).to_dense() == torch.matmul(sparse_a, sparse_b).to_dense()) # 에러 발생
```

    덧셈
    tensor([[[True, True],
             [True, True]],
    
            [[True, True],
             [True, True]]])
    
    
    곱셈
    tensor([[[True, True],
             [True, True]],
    
            [[True, True],
             [True, True]]])
    
    
    행렬곱
    


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-3-55a4bf6b86d8> in <cell line: 18>()
         16 
         17 print('행렬곱')
    ---> 18 print(torch.matmul(a, b).to_dense() == torch.matmul(sparse_a, sparse_b).to_dense()) # 에러 발생
    

    NotImplementedError: Could not run 'aten::as_strided' with arguments from the 'SparseCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::as_strided' is only available for these backends: [CPU, CUDA, Meta, QuantizedCPU, QuantizedCUDA, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMeta, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].
    
    CPU: registered at aten/src/ATen/RegisterCPU.cpp:31034 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:43986 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26824 [kernel]
    QuantizedCPU: registered at aten/src/ATen/RegisterQuantizedCPU.cpp:929 [kernel]
    QuantizedCUDA: registered at aten/src/ATen/RegisterQuantizedCUDA.cpp:459 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:144 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:491 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:20475 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: fallthrough registered at ../aten/src/ATen/ConjugateFallback.cpp:21 [kernel]
    Negative: fallthrough registered at ../aten/src/ATen/native/NegateFallback.cpp:23 [kernel]
    ZeroTensor: registered at aten/src/ATen/RegisterZeroTensor.cpp:161 [kernel]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4733 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_0.cpp:16728 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:487 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:354 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:819 [kernel]
    FuncTorchVmapMode: fallthrough registered at ../aten/src/ATen/functorch/VmapModeRegistrations.cpp:28 [backend fallback]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1077 [kernel]
    VmapMode: fallthrough registered at ../aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:152 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:487 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:148 [backend fallback]
    



```python
# Dense 와 Sparse Tensor 간의 연산 (2차원)
a = torch.tensor([[0,1],[0,2]],dtype=torch.float)
b = torch.tensor([[1,0],[0,0]],dtype=torch.float).to_sparse()

sparse_b = b.to_sparse()

print('덧셈')
print(torch.add(a, b).to_dense() == torch.add(a, sparse_b).to_dense())

print('\n')
print('곱셈')
print(torch.mul(a, b).to_dense() == torch.mul(a, sparse_b).to_dense())

print('\n')

print('행렬곱')
print(torch.matmul(a, b).to_dense() == torch.matmul(a, sparse_b).to_dense())
```

    덧셈
    tensor([[True, True],
            [True, True]])
    
    
    곱셈
    tensor([[True, True],
            [True, True]])
    
    
    행렬곱
    tensor([[True, True],
            [True, True]])
    


```python
a = torch.tensor([[[0, 1], [0, 2]], [[0, 1], [0, 2]]],dtype=torch.float)
b = torch.tensor([[[1, 0], [0, 0]], [[1, 0], [0, 0]]],dtype=torch.float)

sparse_b = b.to_sparse()

print('덧셈')
print(torch.add(a, b).to_dense() == torch.add(a, sparse_b).to_dense())

print('\n')
print('곱셈')
print(torch.mul(a, b).to_dense() == torch.mul(a, sparse_b).to_dense())

print('\n')

print('행렬곱')
print(torch.matmul(a, b).to_dense() == torch.matmul(a, sparse_b).to_dense()) # 에러 발생
```

    덧셈
    tensor([[[True, True],
             [True, True]],
    
            [[True, True],
             [True, True]]])
    
    
    곱셈
    tensor([[[True, True],
             [True, True]],
    
            [[True, True],
             [True, True]]])
    
    
    행렬곱
    


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-6-89ab68b0fa2a> in <cell line: 16>()
         14 
         15 print('행렬곱')
    ---> 16 print(torch.matmul(a, b).to_dense() == torch.matmul(a, sparse_b).to_dense())
    

    NotImplementedError: Could not run 'aten::as_strided' with arguments from the 'SparseCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::as_strided' is only available for these backends: [CPU, CUDA, Meta, QuantizedCPU, QuantizedCUDA, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMeta, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].
    
    CPU: registered at aten/src/ATen/RegisterCPU.cpp:31034 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:43986 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26824 [kernel]
    QuantizedCPU: registered at aten/src/ATen/RegisterQuantizedCPU.cpp:929 [kernel]
    QuantizedCUDA: registered at aten/src/ATen/RegisterQuantizedCUDA.cpp:459 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:144 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:491 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:20475 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: fallthrough registered at ../aten/src/ATen/ConjugateFallback.cpp:21 [kernel]
    Negative: fallthrough registered at ../aten/src/ATen/native/NegateFallback.cpp:23 [kernel]
    ZeroTensor: registered at aten/src/ATen/RegisterZeroTensor.cpp:161 [kernel]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4733 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_0.cpp:16728 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:487 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:354 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:819 [kernel]
    FuncTorchVmapMode: fallthrough registered at ../aten/src/ATen/functorch/VmapModeRegistrations.cpp:28 [backend fallback]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1077 [kernel]
    VmapMode: fallthrough registered at ../aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:152 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:487 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:148 [backend fallback]
    


#### 📝 설명 : Sparse Tensor 의 Indexing
* 일반 텐서와 동일하게 indexing 이 가능합니다.
  * slicing (":" 을 사용)은 불가능 합니다.
  
📚 참고할만한 자료:
* [sparse] : https://pytorch.org/docs/stable/sparse.html


```python
a = torch.tensor([[0,1 ], [0, 2]], dtype=torch.float)
b = torch.tensor([[[1, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=torch.float)

sparse_a = a.to_sparse()
sparse_b = b.to_sparse()

print('2차원 Sparse Tensor 인덱싱')
print(a[0] == sparse_a[0].to_dense())

print('\n')

print('3차원 Sprase Tensor 인덱싱')
print(b[0] == sparse_b[0].to_dense())
```

    2차원 Sparse Tensor 인덱싱
    tensor([True, True])
    
    
    3차원 Sprase Tensor 인덱싱
    tensor([[True, True],
            [True, True]])
    


```python
a = torch.tensor([[0, 1], [0, 2]], dtype=torch.float).to_sparse_csr() # 2dim Sparse Tensor (CSR)
a[0,:] # 0행의 모든 원소 추출 => 에러 발생
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-9-d2d97aa5e6da> in <cell line: 2>()
          1 a = torch.tensor([[0, 1], [0, 2]], dtype=torch.float).to_sparse_csr() # 2dim Sparse Tensor (CSR)
    ----> 2 a[0,:] # 0행의 모든 원소 추출 => 에러 발생
    

    NotImplementedError: Could not run 'aten::as_strided' with arguments from the 'SparseCPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::as_strided' is only available for these backends: [CPU, CUDA, Meta, QuantizedCPU, QuantizedCUDA, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMeta, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].
    
    CPU: registered at aten/src/ATen/RegisterCPU.cpp:31034 [kernel]
    CUDA: registered at aten/src/ATen/RegisterCUDA.cpp:43986 [kernel]
    Meta: registered at aten/src/ATen/RegisterMeta.cpp:26824 [kernel]
    QuantizedCPU: registered at aten/src/ATen/RegisterQuantizedCPU.cpp:929 [kernel]
    QuantizedCUDA: registered at aten/src/ATen/RegisterQuantizedCUDA.cpp:459 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Python: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:144 [backend fallback]
    FuncTorchDynamicLayerBackMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:491 [backend fallback]
    Functionalize: registered at aten/src/ATen/RegisterFunctionalization_0.cpp:20475 [kernel]
    Named: fallthrough registered at ../aten/src/ATen/core/NamedRegistrations.cpp:11 [kernel]
    Conjugate: fallthrough registered at ../aten/src/ATen/ConjugateFallback.cpp:21 [kernel]
    Negative: fallthrough registered at ../aten/src/ATen/native/NegateFallback.cpp:23 [kernel]
    ZeroTensor: registered at aten/src/ATen/RegisterZeroTensor.cpp:161 [kernel]
    ADInplaceOrView: registered at ../torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp:4733 [kernel]
    AutogradOther: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradCUDA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHIP: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXLA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMPS: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradIPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradXPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradHPU: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradVE: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradLazy: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMeta: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradMTIA: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse1: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse2: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradPrivateUse3: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    AutogradNestedTensor: registered at ../torch/csrc/autograd/generated/VariableType_0.cpp:15232 [autograd kernel]
    Tracer: registered at ../torch/csrc/autograd/generated/TraceType_0.cpp:16728 [kernel]
    AutocastCPU: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:487 [backend fallback]
    AutocastCUDA: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:354 [backend fallback]
    FuncTorchBatched: registered at ../aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:819 [kernel]
    FuncTorchVmapMode: fallthrough registered at ../aten/src/ATen/functorch/VmapModeRegistrations.cpp:28 [backend fallback]
    Batched: registered at ../aten/src/ATen/LegacyBatchingRegistrations.cpp:1077 [kernel]
    VmapMode: fallthrough registered at ../aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
    FuncTorchGradWrapper: registered at ../aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
    PythonTLSSnapshot: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:152 [backend fallback]
    FuncTorchDynamicLayerFrontMode: registered at ../aten/src/ATen/functorch/DynamicLayer.cpp:487 [backend fallback]
    PythonDispatcher: registered at ../aten/src/ATen/core/PythonFallbackKernel.cpp:148 [backend fallback]
    


#Reference
> <b><font color = green>(📒가이드)
- <a href='https://pytorch.org/docs/stable/index.html'>PyTorch 공식 문서</a>
- <a href='https://bkshin.tistory.com/entry/NLP-7-%ED%9D%AC%EC%86%8C-%ED%96%89%EB%A0%AC-Sparse-Matrix-COO-%ED%98%95%EC%8B%9D-CSR-%ED%98%95%EC%8B%9D'>COO 와 CSR/CSC</a>

## Required Package

> torch == 2.0.1


```python

```

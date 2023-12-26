---
layout: post
title: Pytorch_텐서 조작의 개념
date: 2023-12-26 12:00 +0900
last_modified_at: 2023-12-26 12:00:00 +0900
tags: [deeplearning, Pytorch, tensor]
toc:  true
---

# 텐서 조작의 개념

## 1. 텐서란?

텐서는 데이터 배열 (array)를 의미한다.

![Alt text](\..\img\DL4-10.png)

각각 1차원, 2차원을 1D 텐서, 2D 텐서라고 표현한다.

## 2. 수학적 연산

### 2.1 텐서의 사칙연산

#### 원소 단위로 연산

말그대로, 같은 위치의 원소 단위로 덧셈, 뺄셈, 곱셈, 나누기를 진행할 수 있다.

### 2.2 내적 (inner Product)

내적은 1D 텐서 단위에서만 가능하다.

$C = A \cdot B \newline = \begin{bmatrix} a_1 & a_2 \end{bmatrix} \cdot \begin{bmatrix} b_1 \newline b_2 \end{bmatrix} = a_1 \times b_1 + a_2 \times b_2$

이때 $C$는 Scalar

### 2.3 행렬곱 연산

두 행렬의 대응하는 행과 열의 원소들을 곱한 뒤 더하는 방식<br>
$\Rightarrow$ 행과 열 단위로 <mark>내적</mark>

계산 상 앞의 **행렬의 열의 길이**와 **뒤의 행렬의 행의 길이**가 같아야 함

## 3. Broadcasting

### 3.1 Broadcasting이란?

**차원이 다른** 두 텐서 혹은 텐서와 스칼라 간의 연산을 가능하게 해주는 기능<br>(불가능한 경우도 존재한다.)

### 3.2 Broadcasting을 이용한 연산

*2D 텐서와 Scalar 연산*<br>
$\begin{aligned} & \begin{pmatrix} -1&0&-1 \newline 0&1&2 \newline 1&-1&1\end{pmatrix} + 3 \newline = &\begin{pmatrix}-1&0&-1 \newline 0&1&2 \newline 1&-1&1\end{pmatrix} + \begin{pmatrix} 3&3&3 \newline 3&3&3 \newline 3&3&3 \end{pmatrix} \newline = & \begin{pmatrix}-2&0&2 \newline 2&2&3 \newline 4&2&3 \end{pmatrix} \end{aligned}$

---

*2D 텐서와 1D 텐서의 연산*<br>
$\begin{aligned} & \begin{pmatrix} -1&0&-1 \newline 0&1&2 \newline 1&-1&1\end{pmatrix} + \begin{pmatrix} 1&2&3 \end{pmatrix} \newline = &\begin{pmatrix}-1&0&-1 \newline 0&1&2 \newline 1&-1&1\end{pmatrix} + \begin{pmatrix} 1&2&3 \newline 1&2&3 \newline 1&2&3 \end{pmatrix} \newline = & \begin{pmatrix}2&2&2 \newline 1&3&5 \newline 2&1&4 \end{pmatrix} \end{aligned}$


## 4. Sparse Tensor



### 4.1 Dense Tensor와 Sparse Tensor

**Dense tensor** : 배열의 모든 위치에 값을 가지는 텐서

**Sparse tensor** : 0이 아닌 원소와 그 위치를 저장하는 텐서

![Alt text](\..\img\DL4-11.png)

### 4.2 Dense Tensor의 문제점 및<br>Sparse Tensor의 장점

Dense Tensor<br>
>생략 가능한 원소까지 모두 저장하여 memory 사용량이 큼
>
> **Out of memory 문제 발생가능**
>
> 또한 계산 시간이 증가한다.

Sparse Tensor<br>
>tensor에 0이 많으면 메모리를 효율적으로 사용한다.
>
>COO (**Coo**rdinate list) 방식과<br> CSR/CSC (**C**ompressed **S**parse **R**ow/**C**olumn) 방식이 있다.

### 4.3 Sparse COO Tensor

(*row_index*, *column_index*, *value*)의 형태로 저장하는 방식

- 장점 : 직관적이다.
- 단점 :<br>
>1. 원소가 위치한 행과 열 인덱스를 별도로 저장해서 반복해서 저장되는 값 발생<br>메모리를 비효율적으로 사용함
>2. 원소에 접근할 때마다 행-열 인덱스와 값을 찾아야함<br>원소를 반복적으로 접근할 때, 연산 성능이 저하된다.

### 4.4 Sparse SCR/CSC Tensor

CSR : (*row_pointer*, *column_index*, *value*)

CSC : (*column_pointer*, *row_index*, *value*)

여기서 row_pointer와 column_pointer는 경계를 표시하는 배열

column/row_index 는 0이 아닌 값을 가지는 행/열 원소 **순서대로** 열/행 인덱스를 정렬한 배열

pointer는 index 배열 중에서, 몇 번째까지가 같은 행 or 열에 있는지를 확인시켜주는 역할이고, index는 말 그대로 index의 역할을 한다.

따라서, value 배열의 크기와, index 배열의 크기는 같다.<br>(index는 말 index이므로)

pointer는 해당 row/column의 크기 + 1의 크기를 가진다.

- 장점 1 : 원소를 순회하는 방식으로 접근 가능<br>장점 2 : 중복 저장이 줄어들어 메모리가 효율적임
- 단점 : sparse tensor의 구조가 복잡해서 직관적이지는 않음
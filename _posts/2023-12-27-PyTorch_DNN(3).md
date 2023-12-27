---
layout: post
title: Pytorch_DNN 실습 (2)
date: 2023-12-27 17:00 +0900
last_modified_at: 2023-12-27 17:00:00 +0900
tags: [deeplearning, Pytorch, DNN]
toc:  true
---

# DNN 구현(3)

### 실습 목차
* 1. Custom Dataset 구축하기
  * 1-1. 자연어 데이터의 전처리
  * 1-2. 데이터셋 클래스 구축하기

* 2. Next word prediction 모델 구축
  * 2-1. Next word prediction을 위한 DNN 모델 구축
  * 2-2. 모델 학습 및 추론


### 환경 설정

- 패키지 설치 및 임포트


```python
!pip install scikit-learn==1.3.0 -q
!pip install torch==2.0.1 -q
!pip install torchvision==0.15.2 -q
!pip install torchtext==0.15.2 -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.8/10.8 MB[0m [31m85.9 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np # 기본적인 연산을 위한 라이브러리
import matplotlib.pyplot as plt # 그림이나 그래프를 그리기 위한 라이브러리
from tqdm.notebook import tqdm # 상태 바를 나타내기 위한 라이브러리
import pandas as pd # 데이터프레임을 조작하기 위한 라이브러리

import torch # PyTorch 라이브러리
import torch.nn as nn # 모델 구성을 위한 라이브러리
import torch.optim as optim # optimizer 설정을 위한 라이브러리
from torch.utils.data import Dataset, DataLoader # 데이터셋 설정을 위한 라이브러리

from torchtext.data import get_tokenizer # torch에서 tokenizer를 얻기 위한 라이브러리
import torchtext # torch에서 text를 더 잘 처리하기 위한 라이브러리

from sklearn.metrics import accuracy_score # 성능지표 측정
from sklearn.model_selection import train_test_split # train-validation-test set 나누는 라이브러리

import re # text 전처리를 위한 라이브러리
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


```python
device = 'cuda:0'
```

###  데이터 셋 개요 </b>

* 데이터셋: <a href='https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset'>Medium Dataset</a>
* 데이터셋 개요: "Towards Data Science", "UX Collective", "The Startup", "The Writing Cooperative", "Data Driven Investor", "Better Humans", "Better Marketing" 의 7개의 주제를 가지는 publication 에 대해서 크롤링을 한 데이터입니다. 원본 데이터는 총 6,508개의 블로그 이미지와 메타 데이터(.csv)로 구성됩니다. 실습에서는 메타데이터를 사용하여 CustomDataset을 구현합니다.
  * [How to collect ths dataset?](https://dorianlazar.medium.com/scraping-medium-with-python-beautiful-soup-3314f898bbf5)
- 메타 데이터 스키마: 메타 데이터는 총 **10**개의 column으로 구성됩니다.
  - id: 아이디
  - url: 포스팅 링크
  - title: 제목
  - subtitle: 부제목
  - image: 포스팅 이미지의 파일 이름
  - claps: 추천 수
  - reponses: 댓글 수
  - reading_time: 읽는데 걸리는 시간
  - publication: 주제 카테고리(e.g. Towards Data Science..)
  - date: 작성 날짜
- 데이터 셋 저작권: CC0: Public Domain

## 1. Custom Dataset 구축하기



- 1-1. 자연어 데이터의 전처리
- 1-2. Custom Dataset class 구축하기


### 1-1 자연어 데이터 전처리

> text로 된 데이터를 어떻게 숫자 형식으로 바꾸고, 모델에 넣는 구조로 바꾸는지 직접 실습해봅니다.


#### 📝 설명: Next word prediction
* 글의 일부가 주어졌을 때, 다음 단어를 예측 (next word prediction)하는 모델을 구축하는 것을 목표로 합니다.
* 예를 들어, "나는 학교를 가서 밥을 먹었다." 라는 문장이 주어진다고 해봅시다.

|input|label|
|------|---|
|나는|학교를|
|나는 학교를|가서|
|나는 학교를 가서|밥을|
|나는 학교를 가서 밥을|먹었다.|

* 이와 같이 데이터셋을 구축하고, DNN을 통해 다음 단어를 예측해봅니다.

📚 참고할만한 자료:
* [Next word prediction](https://wikidocs.net/45101)


```python
data_csv = pd.read_csv('medium_data.csv')
data_csv.head()
```





  <div id="df-74176281-52a0-4fcd-99ba-4ffd46dddc3f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>subtitle</th>
      <th>image</th>
      <th>claps</th>
      <th>responses</th>
      <th>reading_time</th>
      <th>publication</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://towardsdatascience.com/a-beginners-gui...</td>
      <td>A Beginner’s Guide to Word Embedding with Gens...</td>
      <td>NaN</td>
      <td>1.png</td>
      <td>850</td>
      <td>8</td>
      <td>8</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://towardsdatascience.com/hands-on-graph-...</td>
      <td>Hands-on Graph Neural Networks with PyTorch &amp; ...</td>
      <td>NaN</td>
      <td>2.png</td>
      <td>1100</td>
      <td>11</td>
      <td>9</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>https://towardsdatascience.com/how-to-use-ggpl...</td>
      <td>How to Use ggplot2 in Python</td>
      <td>A Grammar of Graphics for Python</td>
      <td>3.png</td>
      <td>767</td>
      <td>1</td>
      <td>5</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>https://towardsdatascience.com/databricks-how-...</td>
      <td>Databricks: How to Save Files in CSV on Your L...</td>
      <td>When I work on Python projects dealing…</td>
      <td>4.jpeg</td>
      <td>354</td>
      <td>0</td>
      <td>4</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>https://towardsdatascience.com/a-step-by-step-...</td>
      <td>A Step-by-Step Implementation of Gradient Desc...</td>
      <td>One example of building neural…</td>
      <td>5.jpeg</td>
      <td>211</td>
      <td>3</td>
      <td>4</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-74176281-52a0-4fcd-99ba-4ffd46dddc3f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-74176281-52a0-4fcd-99ba-4ffd46dddc3f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-74176281-52a0-4fcd-99ba-4ffd46dddc3f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-333e7f20-b2a1-4d96-82dc-8434ffaf6b47">
  <button class="colab-df-quickchart" onclick="quickchart('df-333e7f20-b2a1-4d96-82dc-8434ffaf6b47')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-333e7f20-b2a1-4d96-82dc-8434ffaf6b47 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
print(data_csv.shape)
```

    (6508, 10)
    


```python
# 각각의 title만 추출합니다.
# 우리는 title의 첫 단어가 주어졌을 때, 다음 단어를 예측하는 것을 수행할 것입니다.
data = data_csv['title'].values
```

#### 📝 설명: 텍스트 데이터 전처리하기
* 해당 데이터셋은 크롤링(인터넷에 있는 정보를 수집하는 기법)을 통해 구축되었기 때문에 no-break space가 종종 발생합니다. 이러한 no-break space를 제거하는 전처리를 진행합니다.
  * No-Break Space란? 웹 페이지나 문서 등에서 단어나 문장 사이의 공백이 있는 경우, 해당 공백이 줄 바꿈으로 인해 분리되지 않고 한 단어나 문장으로 인식되도록 하는데 사용되는 공백
  * 예시 (no-break-space 사용 X)
    ```
    Hello
    World~
    ```
    
    (no-break-space 사용)
    ```
    Hello,⎵world!
    ```
* no-break space를 제거하기 위해선 unicode 형식으로 제거를 해야합니다.
  * unicode란? 전 세계의 모든 문자와 기호를 일관성 있게 표현하기 위한 표준 문자 인코딩 체계

* `re` 라이브러리를 이용하여 텍스트 데이터를 쉽게 처리할 수 있습니다.

📚 참고할만한 자료:
* <a href='https://www.compart.com/en/unicode'>unicode 검색 사이트</a>
* [re 라이브러리를 이용한 텍스트 데이터 사용법](https://velog.io/@hoegon02/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC-12-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%A0%95%EA%B7%9C-%ED%91%9C%ED%98%84%EC%8B%9D-3qmtwryf)


```python
def cleaning_text(text):
    cleaned_text = re.sub( r"[^a-zA-Z0-9.,@#!\s']+", "", text) # 특수문자 를 모두 지우는 작업을 수행합니다.
    cleaned_text = cleaned_text.replace(u'\xa0',u' ') # No-break space를 unicode 빈칸으로 변환
    cleaned_text = cleaned_text.replace('\u200a',' ') # unicode 빈칸을 빈칸으로 변환
    return cleaned_text

cleaned_data = list(map(cleaning_text, data)) # 모든 특수문자와 공백을 지움
print('Before preprocessing')
print(data[:5])
print('After preprocessing')
print(cleaned_data[:5])
```

    Before preprocessing
    ['A Beginner’s Guide to Word Embedding with Gensim Word2Vec\xa0Model'
     'Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric'
     'How to Use ggplot2 in\xa0Python'
     'Databricks: How to Save Files in CSV on Your Local\xa0Computer'
     'A Step-by-Step Implementation of Gradient Descent and Backpropagation']
    After preprocessing
    ['A Beginners Guide to Word Embedding with Gensim Word2Vec Model', 'Handson Graph Neural Networks with PyTorch  PyTorch Geometric', 'How to Use ggplot2 in Python', 'Databricks How to Save Files in CSV on Your Local Computer', 'A StepbyStep Implementation of Gradient Descent and Backpropagation']
    

#### 📝 설명: Tokenizer
* Tokenizer는 텍스트 데이터를 작은 단위로 분리해주는 도구입니다.
* 텍스트 데이터를 머신 러닝 모델에 입력으로 사용하거나 자연어 처리 작업을 수행할 때, 문장을 단어 또는 하위 단위(subword)로 분리하는 역할을 수행하기 위한 도구입니다.
  * 텍스트를 단어 또는 하위 단위로 분리 (토큰 분리): 텍스트를 띄어쓰기 단위로 나누거나, 보다 작은 단위로 분리합니다.
    * 예를 들어, "I love PyTorch"이라는 문장을 단어 단위로 분리하면 ["I", "love", "PyTorch"]과 같이 됩니다.
    * 하위 단위 토크나이저는 언어의 특성에 따라 단어를 더 작은 단위로 분리하여 처리할 수 있습니다. 예를 들어, "playing"이라는 단어를 "play"와 "ing"으로 분리하는 것입니다.

  * 토큰을 숫자로 매핑: 머신 러닝 모델은 텍스트를 숫자로 처리해야 합니다. 따라서 모델이 텍스트를 처리할 수 있게 단어나 하위 단위를 고유한 숫자 ID로 매핑하는 작업을 수행합니다.
    * 예를 들어, ["I", "love", "PyTorch"] 이라는 단어들이 있을 때, 이를 이용하여 {"I":0, "love":1, "PyTorch":2}와 같은 단어 사전을 만들고, 이를 통해 [0, 1, 2]로 변환합니다.
  * 특수 토큰 추가: 텍스트를 모델에 입력으로 사용할 때, 특별한 의미를 가진 토큰을 추가할 수 있습니다.
    * 예를 들어 문장의 시작(<sos> 토큰)과 끝(<eox> 토큰)을 나타내는데 사용되거나, 미리 정의된 사전에 없는 단어를 대체하는데 사용될 수 있습니다.
    
* 자연어 처리를 위한 라이브러리인 `torchtext.vocab.build_vocab_from_iterator`를 이용하여 위 과정을 모두 쉽게 처리할 수 있습니다.
  
📚 참고할만한 자료:
* [torchtext getTokenizer](https://pytorch.org/text/stable/data_utils.html#get-tokenizer)
* [Vocab tokenize 설명](https://velog.io/@nkw011/nlp-vocab)

#### 📝 설명: build_vocab_from_iterator
`torchtext.vocab.build_vocab_from_iterator`는 iterator를 이용하여 Vocab 클래스(단어사전)를 만드는 함수입니다.
* 주요 parameter
  * iterator: 단어 사전을 만들 때 사용되는 iterator
  * min_freq: 단어 사전에 포함되기 위한 최소 빈도 수
* output
  * torchtext.vocab.Vocab 클래스를 반환합니다.
  * 이로써 Vocab class에 있는 함수들을 모두 사용할 수 있습니다.

📚 참고할만한 자료:
* [build_vocab_from_iterator](https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator)
* [Vocab class의 함수들](https://pytorch.org/text/stable/vocab.html)


```python
# 토크나이저를 통해 단어 단위의 토큰을 생성합니다.
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(cleaned_data[0])
print("Original text : ", cleaned_data[0])
print("Token: ", tokens)
```

    Original text :  A Beginners Guide to Word Embedding with Gensim Word2Vec Model
    Token:  ['a', 'beginners', 'guide', 'to', 'word', 'embedding', 'with', 'gensim', 'word2vec', 'model']
    


```python
# 단어 사전을 생성한 후, 시작과 끝 표시를 해줍니다.
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, cleaned_data)) # 단어 사전을 생성합니다.
vocab.insert_token('<pad>', 0)
```


```python
id2token = vocab.get_itos() # id to string
id2token[:10]
```




    ['<pad>', 'to', 'the', 'a', 'of', 'and', 'how', 'in', 'your', 'for']




```python
token2id = vocab.get_stoi() # string to id
token2id = dict(sorted(token2id.items(), key=lambda item: item[1]))
for idx, (k,v) in enumerate(token2id.items()):
    print(k,v)
    if idx == 5:
        break
```

    <pad> 0
    to 1
    the 2
    a 3
    of 4
    and 5
    


```python
vocab.lookup_indices(tokenizer(cleaned_data[0])) # 문장을 토큰화 후 id로 변환합니다.
```




    [3, 273, 66, 1, 467, 1582, 12, 2884, 8549, 99]



#### 📝 설명: 데이터 전처리

  
* input에 들어가는 단어 수가 모두 다르므로 이를 바로 모델에 넣기에는 어렵습니다. 이를 위해, \<pad\> (0)을 넣어서 길이를 맞춰주는 과정을 padding 이라고 합니다.
<!-- * label 값은 OneHotEncoding을 해야합니다.
  * torch.nn.functional.one_hot 함수를 이용하여 onehot encoding을 쉽게 할 수 있습니다.
  * OneHotEncoding이란? : 카테고리 형태의 데이터를 벡터로 변환하는 방법으로, 해당하는 카테고리에 해당하는 인덱스만 1이고 나머지는 모두 0인 이진 벡터로 표현하는 것을 의미합니다.
  * 왜 OneHotEncodingd을 해야할까? : multi-class(개, 고양이, 토끼 분류와 같은) 문제로 풀기 위함입니다.   -->
  
📚 참고할만한 자료:
* [Padding 설명](https://wikidocs.net/83544)


```python
seq = []
for i in cleaned_data:
    token_id = vocab.lookup_indices(tokenizer(i))
    for j in range(1, len(token_id)):
        sequence = token_id[:j+1]
        seq.append(sequence)
```


```python
seq[:5]
```




    [[3, 273],
     [3, 273, 66],
     [3, 273, 66, 1],
     [3, 273, 66, 1, 467],
     [3, 273, 66, 1, 467, 1582]]




```python
max_len = max(len(sublist) for sublist in seq) # seq에 저장된 최대 토큰 길이 찾기
print(max_len)
```

    24
    


```python
def pre_zeropadding(seq, max_len): # max_len 길이에 맞춰서 0 으로 padding 처리 (앞부분에 padding 처리)
    return np.array([i[:max_len] if len(i) >= max_len else [0] * (max_len - len(i)) + i for i in seq])
zero_padding_data = pre_zeropadding(seq, max_len)
zero_padding_data[0]
```




    array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 273])




```python
input_x = zero_padding_data[:,:-1]
label = zero_padding_data[:,-1]
```


```python
input_x[:5] # input 값 확인
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   3],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   3, 273],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   3, 273,  66],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   3, 273,  66,   1],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   3, 273,  66,   1, 467]])




```python
label[:5] # label 값 확인
```




    array([ 273,   66,    1,  467, 1582])



### 1-2 Custom Dataset 구현

> 1-1에서 진행한 전처리 진행을 모듈화 시켜서 하나의 class로 구현합니다.


#### 📝 설명: Custom Dataset 정의하기
* 1-1에서 진행한 전처리 과정을 모두 함수화 시켜서 하나의 class로 구축합니다.
* 이로 인해, 손쉬운 모듈화가 가능합니다.
* 데이터를 변환하는 과정은 되도록이면 getitem 이 아닌 init 부분에 하여, 전처리하는 시간을 줄이도록 합니다.
  * init 부분에 한 번에 하게 되면 dataset을 정의할 때만 변환 시간이 소요되고, 그 이후로는 데이터 전처리 시간이 소요되지 않습니다.

📚 참고할만한 자료:
* [Custom Dataset 구축 - Pytorch 공식 튜토리얼](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)


```python
class CustomDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        seq = self.make_sequence(self.data, self.vocab, self.tokenizer) # next word prediction을 하기 위한 형태로 변환
        self.seq = self.pre_zeropadding(seq, self.max_len) # zero padding으로 채워줌
        self.X = torch.tensor(self.seq[:,:-1])
        self.label = torch.tensor(self.seq[:,-1])

    def make_sequence(self, data, vocab, tokenizer):
        seq = []
        for i in data:
            token_id = vocab.lookup_indices(tokenizer(i))
            for j in range(1, len(token_id)):
                sequence = token_id[:j+1]
                seq.append(sequence)
        return seq

    def pre_zeropadding(self, seq, max_len): # max_len 길이에 맞춰서 0 으로 padding 처리 (앞부분에 padding 처리)
        return np.array([i[:max_len] if len(i) >= max_len else [0] * (max_len - len(i)) + i for i in seq])

    def __len__(self): # dataset의 전체 길이 반환
        return len(self.X)

    def __getitem__(self, idx): # dataset 접근
        X = self.X[idx]
        label = self.label[idx]
        return X, label
```


```python
def cleaning_text(text):
    cleaned_text = re.sub( r"[^a-zA-Z0-9.,@#!\s']+", "", text) # 특수문자 를 모두 지우는 작업을 수행합니다.
    cleaned_text = cleaned_text.replace(u'\xa0',u' ') # No-break space를 unicode 빈칸으로 변환
    cleaned_text = cleaned_text.replace('\u200a',' ') # unicode 빈칸을 빈칸으로 변환
    return cleaned_text

data = list(map(cleaning_text, data))
tokenizer = get_tokenizer("basic_english")
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, data))
vocab.insert_token('<pad>',0)
max_len = 20
```


```python
# train set, validation set, test set으로 data set을 나눕니다. 8 : 1 : 1 의 비율로 나눕니다.
train, test = train_test_split(data, test_size = .2, random_state = 42)
val, test = train_test_split(test, test_size = .5, random_state = 42)
```


```python
print("Train 개수: ", len(train))
print("Validation 개수: ", len(val))
print("Test 개수: ", len(test))
```

    Train 개수:  5206
    Validation 개수:  651
    Test 개수:  651
    


```python
train_dataset = CustomDataset(train, vocab, tokenizer, max_len)
valid_dataset = CustomDataset(val, vocab, tokenizer, max_len)
test_dataset = CustomDataset(test, vocab, tokenizer, max_len)
```


```python
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
```

## 2. Next word prediction 모델 구현


- 2-1. Next word prediction을 위한 DNN 모델 구현
- 2-2. 모델 학습 및 추론

### 2-1 Next word prediction을 위한 DNN 모델 구축

> Next word prediction을 위한 DNN 모델을 직접 구축해봅니다.


#### 📝 설명: Next word prediction을 위한 DNN 모델 구축
* DNN 구현 (2)에서 학습하였던, DNN 모델을 기반에 `nn.Embedding`을 추가하여 next word prediction을 하기 위한 DNN 모델을 구축합니다.
* Embedding이란?
  * 텍스트나 범주형 데이터와 같이 모델이 처리하기 어려운 형태의 데이터를 수치 형태로 변환하는 기술입니다.
  * 주어진 데이터를 저차원의 벡터 공간에 표현하는 방법으로, 단어, 문장, 범주형 변수 등을 고정된 길이의 실수 벡터로 매핑하여 표현합니다.
* `nn.Embedding`
  * num_embedding : embedding할 input값의 수를 의미합니다. 자연어처리에선 단어 사전의 크기와 동일합니다.
  * embedding_dim : embedding 벡터의 차원을 의미합니다.
  
📚 참고할만한 자료:
* [torch.nn.Embedding - Pytorch 공식 튜토리얼](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* [Embedding 설명](https://wikidocs.net/64779)


```python
class NextWordPredictionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_dims, num_classes, dropout_ratio, set_super):
        if set_super:
            super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dims, padding_idx = 0) # padding index 설정 => gradient 계산에서 제외
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

            self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

            self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_ratio))

        self.classifier = nn.Linear(self.hidden_dims[-1], self.num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        '''
        INPUT:
            x: [batch_size, sequence_len] # padding 제외
        OUTPUT:
            output : [batch_size, vocab_size]
        '''
        x = self.embedding(x) # [batch_size, sequence_len, embedding_dim]
        x = torch.sum(x, dim=1) # [batch_size, embedding_dim] 각 문장에 대해 임베딩된 단어들을 합쳐서, 해당 문장에 대한 임베딩 벡터로 만들어줍니다.
        for layer in self.layers:
            x = layer(x)

        output = self.classifier(x) # [batch_size, num_classes]
        output = self.softmax(output) # [batch_size, num_classes]
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 2-2 모델 학습 및 추론

> Next word prediction 모델을 직접 학습하고, text를 직접 넣어 next word prediction을 직접 수행해봅니다.


#### 📝 설명: Next word prediction 학습하기
* DNN 모델을 학습하기 위해 모델의 파라미터를 정해줍니다.
* embedding layer와 fully connected layer의 연산이 가능하게 하기 위해 hidden dimension 리스트 구성 시, embedding dimension을 첫번째 값으로 설정합니다.
* 예측하려는 label의 개수는 단어 사전에 있는 단어의 개수와 동일합니다.


```python
# training 코드, evaluation 코드, training loop 코드
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    train_loss = 0.0
    train_accuracy = 0

    tbar = tqdm(dataloader)
    for texts, labels in tbar:
        texts = texts.to(device)
        labels = labels.to(device)

        # 순전파
        outputs = model(texts)

        loss = criterion(outputs, labels)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실과 정확도 계산
        train_loss += loss.item()
        # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
        _, predicted = torch.max(outputs, dim=1)


        train_accuracy += (predicted == labels).sum().item()

        # tqdm의 진행바에 표시될 설명 텍스트를 설정
        tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}")

    # 에폭별 학습 결과 출력
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(train_dataset)

    return model, train_loss, train_accuracy

def evaluation(model, dataloader, val_dataset, criterion, device, epoch, num_epochs):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    valid_accuracy = 0

    with torch.no_grad(): # model의 업데이트 막기
        tbar = tqdm(dataloader)
        for texts, labels in tbar:
            texts = texts.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # 손실과 정확도 계산
            valid_loss += loss.item()
            # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
            _, predicted = torch.max(outputs, 1)
            # _, true_labels = torch.max(labels, dim=1)
            valid_accuracy += (predicted == labels).sum().item()


            # tqdm의 진행바에 표시될 설명 텍스트를 설정
            tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}")

    valid_loss = valid_loss / len(dataloader)
    valid_accuracy = valid_accuracy / len(val_dataset)

    return model, valid_loss, valid_accuracy


def training_loop(model, train_dataloader, valid_dataloader, train_dataset, val_dataset, criterion, optimizer, device, num_epochs, patience, model_name):
    best_valid_loss = float('inf')  # 가장 좋은 validation loss를 저장
    early_stop_counter = 0  # 카운터
    valid_max_accuracy = -1

    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
        model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, val_dataset, criterion, device, epoch, num_epochs)

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
lr = 1e-3
vocab_size = len(vocab.get_stoi())
embedding_dims = 512
hidden_dims = [embedding_dims, embedding_dims*4, embedding_dims*2, embedding_dims]
model = NextWordPredictionModel(vocab_size = vocab_size, embedding_dims = embedding_dims, hidden_dims = hidden_dims, num_classes = vocab_size, \
            dropout_ratio = 0.2, set_super = True).to(device)

num_epochs = 100
patience = 3
model_name = 'next'

optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.NLLLoss(ignore_index=0) # padding 한 부분 제외
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, criterion, optimizer, device, num_epochs, patience, model_name)
print('Valid max accuracy : ', valid_max_accuracy)
```


      0%|          | 0/1159 [00:00<?, ?it/s]



      0%|          | 0/149 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 7.3844, Train Accuracy: 0.0641 Valid Loss: 7.2101, Valid Accuracy: 0.0680
    


      0%|          | 0/1159 [00:00<?, ?it/s]



      0%|          | 0/149 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 6.7291, Train Accuracy: 0.0776 Valid Loss: 7.2159, Valid Accuracy: 0.0779
    


      0%|          | 0/1159 [00:00<?, ?it/s]



      0%|          | 0/149 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 6.3590, Train Accuracy: 0.0871 Valid Loss: 7.3250, Valid Accuracy: 0.0842
    


      0%|          | 0/1159 [00:00<?, ?it/s]



      0%|          | 0/149 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 6.0647, Train Accuracy: 0.0950 Valid Loss: 7.4306, Valid Accuracy: 0.0865
    Early stopping
    Valid max accuracy :  0.08653440270156185
    

#### 📝 설명: Next word prediction 평가하기
* 학습한 DNN 모델을 accuracy score로 평가합니다.


```python
model.load_state_dict(torch.load("./model_next.pt")) # 모델 불러오기
model = model.to(device)
model.eval()
total_labels = []
total_preds = []
with torch.no_grad():
    for texts, labels in tqdm(test_dataloader):
        texts = texts.to(device)
        labels = labels

        outputs = model(texts)
        # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
        _, predicted = torch.max(outputs.data, 1)

        total_preds.extend(predicted.detach().cpu().tolist())
        total_labels.extend(labels.tolist())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
nwp_dnn_acc = accuracy_score(total_labels, total_preds) # 정확도 계산
print("Next word prediction DNN model accuracy : ", nwp_dnn_acc)
```


      0%|          | 0/143 [00:00<?, ?it/s]


    Next word prediction DNN model accuracy :  0.07239720034995625
    


```python
print(vocab_size)
```

    8618
    

## Required Package

> torch == 2.0.1

> torchvision == 0.15.2

> sklearn == 1.3.0

> torchtext == 0.15.2


```python

```

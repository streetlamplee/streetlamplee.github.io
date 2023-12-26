---
layout: post
title: Pytorch_huggingface tutorial 실습 (1)
date: 2023-12-26 18:00 +0900
last_modified_at: 2023-12-26 18:00:00 +0900
tags: [deeplearning, Pytorch, huggingface]
toc:  true
---

jupyter나 colab 환경에서 huggingface hub를 연결하는 코드

```python
# from huggingface_hub import notebook_login
# notebook_login()
```

---

필요 lib 설치<br>
각각 **transformers**, **datasets**, **evaluate**, **accelerate**

```python
!pip install transformers datasets evaluate numpy accelerate
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.35.2)
    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (2.16.0)
    Requirement already satisfied: evaluate in /usr/local/lib/python3.10/dist-packages (0.4.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (1.23.5)
    Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (0.25.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.6.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.0)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.1)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.1)
    Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (10.0.1)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Requirement already satisfied: dill<0.3.8,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.7)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (1.5.3)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.15)
    Requirement already satisfied: fsspec[http]<=2023.10.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.9.1)
    Requirement already satisfied: responses<0.19 in /usr/local/lib/python3.10/dist-packages (from evaluate) (0.18.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)
    Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.1.0+cu121)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->transformers) (4.5.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2023.11.17)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.2)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.1.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.3.post1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
    
---

imdb 예제 데이터를 불러와보자

```python
from datasets import load_dataset

imdb = load_dataset("imdb")
```

---

잘 불러와졌는지 확인을 한번 해보자

```python
imdb['test'][0]
```




    {'text': 'I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn\'t match the background, and painfully one-dimensional characters cannot be overcome with a \'sci-fi\' setting. (I\'m sure there are those of you out there who think Babylon 5 is good sci-fi TV. It\'s not. It\'s clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It\'s really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it\'s rubbish as they have to always say "Gene Roddenberry\'s Earth..." otherwise people would not continue watching. Roddenberry\'s ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.',
     'label': 0}


---

fine-tuning을 하기 위해, pre-trained 모델을 가져오자

해당 실습에서는 distilbert를 불러왔다.

가져오면서, tokenizing을 하기 위해 tokenizer를 선언하고,<br>
tokenizing을 해주는 함수를 선언했다.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def preprocess_function(ex):
    return tokenizer(ex['text'], truncation=True)
```

---

선언해준 tokenizer 함수를 모든 데이터에 진행해주었다.<br>
*datasets에서 불러온 데이터는, `map` method를 이용해 한번에 해줄 수 있더라*

```python
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```


    Map:   0%|          | 0/25000 [00:00<?, ? examples/s]

---

DataCollator를 이용해서, input에 들어갈 dataset element를 만든다.

batch를 만드는 과정이라고 생각하면된다.

보통 이 과정에서, padding을 진행하기 때문에, `DataCollatorWithPadding`을 import한다고 이해했다.

자세한 설명은 <a href='https://huggingface.co/docs/transformers/main_classes/data_collator'>여기</a>를 참조해보자

batch를 만들면서 필요한 tokenizer는 선언해준 그 녀석으로 넣어준다.

```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
```

---

모델 지표를 확인할 평가 지수를 선언한다.

이 실습에서는 accurracy를 선언했다.

```python
import evaluate
acc = evaluate.load('accuracy')
```

---

모델의 output을 받아, 이를 acc로 계산해주는 함수를 선언했다.

```python
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return acc.compute(predictions = predictions, references = labels)
```

---

trainer에 들어갈 모델의 arguments인 id2label과 label2id를 선언했다.

해당 모델의 경우, 감성 분석이므로, 각각 0일 때 negative, 1일 때 positive인 label을 각각 선언한다.

```python
id2label = {0: 'neg', 1: 'pos'}
label2id = {'neg': 0, 'pos': 1}
```

---

fine-tuning에 필요한 함수 3개를 불러온다.

1. AutoModelForSequenceClassification
2. TrainingArguments
3. Trainer

그리고, model을 선언한다.

이 때 모델은 AutoModelForSequenceClassification을 이용했으며,<br>
from_pretrained method를 통해 pretrained 모델을 이용할 수 있다고 한다.

에러 메시지가 출력되는데, 아직 weight와 bias가 전부 정해지지 않은 모델이라 학습을 하라는 경고 메시지이므로, 문제가 되지는 않는다.

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2, id2label = id2label, label2id = label2id)
```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.weight', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    
---

training_args에 trainer에 들어갈 training_args에 들어갈 내용을 선언해주고,

trainer에는 모델과 데이터 등을 넣어 train method를 통해 학습해준다.

```python
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=10000,
    per_device_eval_batch_size=10000,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

테스트해볼 텍스트 선언

```python
text = "This was a masterpiece."
```

#### pipeline을 이용해 간단하게 보기


```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model = '/content/my_awesome_model')
classifier(text)
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-15-484fe9213a1d> in <cell line: 3>()
          1 from transformers import pipeline
          2 
    ----> 3 classifier = pipeline('sentiment-analysis', model = model)
          4 classifier(text)
    

    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/__init__.py in pipeline(task, model, config, tokenizer, feature_extractor, image_processor, framework, revision, use_fast, token, device, device_map, torch_dtype, trust_remote_code, model_kwargs, pipeline_class, **kwargs)
        948             else:
        949                 # Impossible to guess what is the right tokenizer here
    --> 950                 raise Exception(
        951                     "Impossible to guess which tokenizer to use. "
        952                     "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
    

    Exception: Impossible to guess which tokenizer to use. Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer.


#### 혹은 직접 수동으로 확인하기


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/content/my_awesome_model')
inputs = tokenizer(text, return_tensors='pt')
```


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('/content/my_awesome_model')
with torch.no_grad():
    logit = model(**inputs).logits
```


```python
predicted_class_id = logit.argmax().item()
model.config.id2label[predicted_class_id]
```

---
title: 일상 대화 요약 대회
layout: post
date: 2024-03-25 11:00 +0900
last_modified_at: 2024-03-25 12:00:00 +0900
tag: [DL,  Upstage, AI ,데이터분석 ,데이터사이언스 , NLP ,자연어처리 ,경진대회]
toc: true
---

# 일상 대화 요약 대회

## 1. Abstract

- **Goal of the Competition**
    - 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 중 하는 대화들에 대해 요약합니다.

- **Timeline**
    - 2024.03.08 : 대회 시작
    - 2024.03.20 : 대회 종료

## 2. Process : Competition Model

- 사용 library<br>
```python
# in python
import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb
```


- 결과(개인)<br>
final_result = ROUGE-1, ROUGE-2, ROUGE-L의 평균<br>
public final_result : 41.7972<br>
private final_result : 38.9605

- 결과(팀)<br>
public final_result : 41.9246<br>
private final_result : 39.1958

## 3. Process : Issues

**Describe the issue that your team faced during the project.**

1. 데이터의 수가 작아 데이터가 더 필요했음
2. 관련한 다양한 모델을 사용하기에 batch size 문제가 발생했음

**Describe the possible solution to imporve your project.**

1. back translation을 이용해서 데이터의 수를 증강해볼 수 있었음<br>
데이터는 파파고 홈페이지에 데이터를 주고 받는 코드를 자동화하여 생성할 수 있었다
2. device에 올리는 batch size 크기를 줄이고, gradient를 더 자주 업데이트하게 만들면서 batch size 문제를 해결해보았음

    

## 4. Role

**Describe your role with task in your team.**
* Data Engineering
* Modeling
* EDA

(notion. 각자 EDA와 모델링을 진행하고 그 insight를 공유하는 식으로 대회를 진행하여, 서로 다른 모델을 만들었음)

**Describe papers or book chapeters you found relevant to the problem, references to see.**

<a href = 'https://arxiv.org/pdf/1910.13461.pdf'> BART : Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</a>

<a href = 'https://dacon.io/competitions/official/235673/data'>Dacon 한국어 문서 생성요약 AI 경진대회</a>

<a href = 'https://aiconnect.kr/competition/detail/223/task/272/community'>AIconnection 한국어 문서 생성 요약</a>

<a href = 'https://velog.io/@cjkangme/DL-%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%AC%B8%EC%84%9C-%EC%83%9D%EC%84%B1-%EC%9A%94%EC%95%BD-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C-%ED%9B%84%EA%B8%B0-with-%EC%97%90%EC%9D%B4%EB%B8%94%EB%9F%AC'>한국어 문서 생성 요약 경진대회 후기 포스팅</a>

**Explain which are relevant for your Project.**

사용 모델인 BART 논문과, Dacon 한국어 문서 생성 요약 경진대회의 커뮤니티가 도움이 되었음
    

## 5. Results

**Write the main result of Competition**

- 결과(개인)<br>
final_result = ROUGE-1, ROUGE-2, ROUGE-L의 평균<br>
public final_result : 41.7972<br>
private final_result : 38.9605

- 결과(팀)<br>
public final_result : 41.9246<br>
private final_result : 39.1958

**Final standings of the Leaderboard**

**7등 기록**
    

## 6. Conclusion

**Describe your running code with its own advantages and disadvantages, in relation to other groups in the course.**

## ⚙️ 데이터 및 환경설정

### 1) 필요한 라이브러리 설치

- 필요한 라이브러리를 설치한 후 불러옵니다.


```python
import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge 

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb 




```python
# config 설정에 tokenizer 모듈이 사용되므로 미리 tokenizer를 정의
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
```


```python
config_data = {
    "general": {
        "data_path": "../data/", 
        "model_name": "digit82/kobart-summarization", 
        "output_dir": "./"
    },
    "tokenizer": {
        "encoder_max_len": 512,
        "decoder_max_len": 100,
        "bos_token": f"{tokenizer.bos_token}",
        "eos_token": f"{tokenizer.eos_token}",
        # 특정 단어들이 분해되어 tokenization이 수행되지 않도록 special_tokens 지정
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 50,
        "per_device_eval_batch_size": 32,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": 'cosine',
        "optim": 'adamw_torch',
        "gradient_accumulation_steps": 1,
        "evaluation_strategy": 'epoch',
        "save_strategy": 'epoch',
        "save_total_limit": 5,
        "fp16": True,
        "load_best_model_at_end": True,
        "seed": 42,
        "logging_dir": "./logs",
        "logging_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 100,
        "do_train": True,
        "do_eval": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
        "report_to": "wandb" 
    },

    "wandb": {
        "entity": "wandb_repo",
        "project": "project_name",
        "name": "run_name"
    },
    "inference": {
        "ckt_path": "model ckt path", # 사전 학습이 진행된 모델의 checkpoint를 저장할 경로 설정
        "result_path": "./prediction/",
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "generate_max_length": 100,
        "num_beams": 4,
        "batch_size" : 32,
        # 정확한 모델 평가를 위해 제거할 불필요한 생성 토큰 정의
        "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
    }
}
```



```python
# 모델의 구성 정보를 YAML 파일로 저장
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)
```

### 3) Configuration 불러오기


```python
# 저장된 config 파일을 불러오기
config_path = "./config.yaml"

with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

# 불러온 config 파일의 전체 내용 확인
pprint(loaded_config)
```


```python
# 실험에 쓰일 데이터의 경로, 사용될 모델, 모델의 최종 출력 결과를 저장할 경로에 대해 확인
loaded_config['general']
```


```python
# 이곳에 사용자가 저장한 데이터 dir 설정
# loaded_config['general']['data_path'] = "data_path"
```


```python
# 데이터 전처리를 하기 위해 tokenization 과정에서 필요한 정보 확인
loaded_config['tokenizer']
```


```python
# 모델이 훈련 시 적용될 매개변수 확인
loaded_config['training']
```


```python
# 모델 학습 과정에 대한 정보를 제공해주는 wandb 설정 내용 확인
loaded_config['wandb']
```


```python
# (선택) 이곳에 사용자가 사용할 wandb config 설정
loaded_config['wandb']['entity'] = "jinintg"
loaded_config['wandb']['name'] = "1"
loaded_config['wandb']['project'] = "Upstage_NLP_Project1"
```


```python
# 모델이 최종 결과를 출력하기 위한 매개변수 정보 확인
loaded_config['inference']
```

### 4) 데이터 확인


```python
# config에 저장된 데이터 경로를 통해 train과 validation data를 불러오기
data_path = loaded_config['general']['data_path']

# train data의 구조와 내용 확인
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
train_df.tail()
```


```python
# validation data의 구조와 내용 확인
val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
val_df.tail()
```

## 1. 데이터 가공 및 데이터셋 클래스 구축
- csv file 을 불러와서 encoder 와 decoder의 입력형태로 가공
- 가공된 데이터를 torch dataset class 로 구축하여 모델에 입력가능한 형태로 변환


```python
# 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력생성
class Preprocess:
    def __init__(self,
            bos_token: str,
            eos_token: str,
        ) -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    # 실험에 필요한 컬럼 가져오기
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리 진행
    def make_input(self, dataset,is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # Ground truth를 디코더의 input으로 사용하여 학습
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

```


```python
# Train에 사용되는 Dataset 클래스 정의
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Validation에 사용되는 Dataset 클래스 정의
class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Test에 사용되는 Dataset 클래스 정의
class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len

```


```python
# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터 출력
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'train.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # train, validation에 대해 각각 데이터프레임 구축
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset
```

## 2. Trainer 및 Trainingargs 구축하기
- Huggingface 의 Trainer 와 Training arguments를 활용하여 모델 학습을 일괄적으로 처리해주는 클래스를 정의


```python
# 모델 성능에 대한 평가 지표를 정의
def compute_metrics(config,tokenizer,pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수 계산
    results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

    # ROUGE 점수 중 F-1 score를 통해 평가
    result = {key: value["f"] for key, value in results.items()}
    return result
```


```python
# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    # set training args
    training_args = Seq2SeqTrainingArguments(
                output_dir=config['general']['output_dir'], # model output directory
                overwrite_output_dir=config['training']['overwrite_output_dir'],
                num_train_epochs=config['training']['num_train_epochs'],  # total number of training epochs
                learning_rate=config['training']['learning_rate'], # learning_rate
                per_device_train_batch_size=config['training']['per_device_train_batch_size'], # batch size per device during training
                per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],# batch size for evaluation
                warmup_ratio=config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
                weight_decay=config['training']['weight_decay'],  # strength of weight decay
                lr_scheduler_type=config['training']['lr_scheduler_type'],
                optim =config['training']['optim'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                evaluation_strategy=config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
                save_strategy =config['training']['save_strategy'],
                save_total_limit=config['training']['save_total_limit'], # number of total save model.
                fp16=config['training']['fp16'],
                load_best_model_at_end=config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
                seed=config['training']['seed'],
                logging_dir=config['training']['logging_dir'], # directory for storing logs
                logging_strategy=config['training']['logging_strategy'],
                predict_with_generate=config['training']['predict_with_generate'], #To use BLEU or ROUGE score
                generation_max_length=config['training']['generation_max_length'],
                do_train=config['training']['do_train'],
                do_eval=config['training']['do_eval'],
                report_to=config['training']['report_to'] # (선택) wandb를 사용할 때 설정합니다.
            )

    # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )

    # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수 설정
    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    # Trainer 클래스를 정의
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델 입력
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config,tokenizer, pred),
        callbacks = [MyCallback]
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer
```


```python
# 학습을 위한 tokenizer와 사전 학습된 모델을 불러오기
def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) #special token 재구성
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer
```

## 3. 모델 학습하기

- 앞에서 구축한 클래스 및 함수를 활용하여 학습 진행합니다.


```python
def main(config):
    # 사용할 device 정의
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # 사용할 모델과 tokenizer
    generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

    # 학습에 사용할 데이터셋
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token']) # decoder_start_token: str, eos_token: str
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)

    # Trainer 클래스
    trainer = load_trainer_for_train(config, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
    trainer.train()   # 모델 학습

    # (선택) 모델 학습이 완료된 후 wandb를 종료
    wandb.finish()
```


```python
if __name__ == "__main__":
    main(loaded_config)
```

## 4. 모델 추론하기


```python
# 이곳에 내가 사용할 wandb config 설정
loaded_config['inference']['ckt_path'] = "추론에 사용할 ckt 경로 설정"
```

- test data를 사용하여 모델의 성능을 확인


```python
# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력
def prepare_test_dataset(config,preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset
```


```python
# 추론 : tokenizer와 학습시킨 모델을 불러오기
def load_tokenizer_and_model_for_test(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model , tokenizer
```


```python
# 학습된 모델이 생성한 요약문의 출력 결과
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    #노이즈에 해당되는 스페셜 토큰을 제거
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output
```


```python

if __name__ == "__main__":
    output = inference(loaded_config)
```


```python
output 
```

---

<mark>데이터 증강에 사용된 코드</mark>

```python
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.common.by import By
import time



parsing_dict = {'한국어' : 2,
                '영어' : 3,
                '일본어' : 4,
                '중국어(간체)' : 5,
                '중국어(번체)' : 6,
                '스페인어' : 7,
                '프랑스어' : 8,
                '독일어' : 9,
                '러시아어' : 10,
                '포르투갈어' : 11,
                '이탈리아어' : 12,
                '베트남어' : 13,
                '태국어' : 14,
                '인도네시아어' : 15,
                '힌디어' : 16,
                '아랍어' : 17}
with open("translation.txt", 'r', encoding = 'utf-8') as f:
    inp = f.readlines()

src = inp[0].split('->')[0]
dest = inp[0].split('->')[1].rstrip()
output_li = []
browser = webdriver.Chrome()
browser.get("https://papago.naver.com/")
time.sleep(2)
browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[2]/div[1]/div/div[1]/button[2]/span').click()
browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[2]/div[1]/div/div[2]/ul/li[{parsing_dict[src]}]').click()
browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[3]/div/div[1]/div/div/div[1]/button[2]/span').click()
browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[3]/div/div[1]/div/div/div[2]/ul/li[{parsing_dict[dest] - 1}]').click()
for idx, value in enumerate(inp):
    if idx == 0:
        continue
    browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[3]/label/textarea').click()
    browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[3]/label/textarea').send_keys(value.strip())
    time.sleep(3)
    output = browser.find_element(By.XPATH, f'/html/body/div/div/div[1]/section/div/div/div[3]/div/div[5]/div').text
    output_li.append(output)
    browser.find_element(By.XPATH, '/html/body/div/div/div[1]/section/div/div/div[2]/div/div[3]/button').click()
with open('result.txt', 'w') as f:
    for i in range(len(output_li)):
        f.write(output_li[i] + "\n")

```


**Sum up your project and suggest future extensions and improvements.**

BART를 이용해서 문서 요약을 도전해보았다.

NLP 대회를 제대로 참가해보는 것이 처음이었고, 배울 수 있는 것이 참 많은 대회였다.<br>
하지만 개인 사정으로 인해 대회 간 4일 정도 대회에 집중을 하지 못해 더 많은 실험을 못 해본 점이 매우 아쉽다.

다음에 기회가 된다면 해당 코드를 한번 더 발전시켜 블로그에 기록해보고 싶다.
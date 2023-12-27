---
layout: post
title: huggingface 사전 학습 모델 사용해보기
date: 2023-12-27 20:00 +0900
last_modified_at: 2023-12-27 20:00:00 +0900
tag: [deeplearning, huggingface]
doc: True
---

`AutoTokenizer`와 `AutoModelForSequenceClassification`을 불러온다.

각각 토크나이저와 모델을 사용하기 위해 불러옴

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
```

사용모델은, 한국어 감성분석을 fine-tuning해둔 모델이다.

모델에 대한 자세한 설명은 아래 링크를 참조

<a href='https://huggingface.co/WhitePeak/bert-base-cased-Korean-sentiment'>링크</a>

각각 토크나이저 (*tk*)와 모델 (*model*)로 선언했다.

```python
tk = AutoTokenizer.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")
```

예측하는 pipeline을 함수로 선언

'LABEL_0'은 부정적<br>
'LABEL_1'은 긍정적이므로<br>
`return`을 그에 맞게 수정해준다.


```python

def predict(text):
    inputs = tk(text, return_tensors = 'pt')
    with torch.no_grad():
        logit = model(**inputs).logits
    
    predicted_class_id = logit.argmax().item()

    res = model.config.id2label[predicted_class_id]

    if res == 'LABEL_0':
        return 'negative'
    else:
        return 'positive'
```

활용해보기로 한 데이터인 리뷰 데이터 중 하나를 테스트 용으로 가지고 왔다.

아주 부정적인 리뷰임에 틀림이 없다.

```python
text = '''
디자인이 너무 마음에 들어서 구매한 워치~
가격도 저렴하게 나왔길래 고민 없이 바로 구매했습니다.
왜 고민을 그 당시에 안 했는지 의문이지만.. -_-.. 
내 자신 반성해일단 받자마자 언박싱을 했지요... 
그런데 스트랩 상태 어쩔꺼여...
누가 스트랩 밟고 상자에 넣어 주신건가여.. 
왜 이렇게 상태가 엉망이고 더러운건지아시는 분.,,? 
이거 리퍼 제품 아니죠,,,,? 
거기다가 방수 기능도 없어서.. 
손 씻을때 완전 조심.. 조심해야한다는거..-_- 
심박수 재는 기능도 있는데 심박수도 안 맞고 아주 가지가지 
그냥 사진으로만 이뻐 보인 게 다임...
'''
predict(text)
```


    'negative'


실제로 부정적이라고 잘 예측을 한 모습이다.

이렇게 이미 fine tuning이 된 모델을 실제로 inference하는 방법을 익혀보았다.



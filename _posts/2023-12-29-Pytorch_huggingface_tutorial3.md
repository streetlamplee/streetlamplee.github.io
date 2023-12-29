---
layout: post
title: Pytorch_huggingface tutorial 실습 (3)
date: 2023-12-29 18:00 +0900
last_modified_at: 2023-12-29 18:00:00 +0900
tags: [deeplearning, Pytorch, huggingface]
toc:  true
---

# huggingface 실습

*이전 포스팅*<br>
<a href = 'https://streetlamplee.github.io/2023/12/26/Pytorch_huggingface_tutorial/'>Pytorch_huggingface tutorial 실습 (1)</a><br>
<a href = 'https://streetlamplee.github.io/2023/12/27/Pytorch_huggingface_tutorial2/'>Pytorch_huggingface tutorial 실습 (2)</a>

이전 포스팅에서는, GPU OOM 문제를 해결하고, fine-tuning까지 해봤다.

이번 포스팅에서는, fine-tuning한 모델을 통해 inference를 해볼 수 있도록 하자

사실 pre-trained 모델에 대한 inference는 다른 포스팅에서 다룬 적이 있다.

해당 포스팅은 <a href = 'https://streetlamplee.github.io/2023/12/27/Pytorch_huggingface_pretrained/'>여기</a>에서 확인할 수 있다.

위의 포스팅에서는 <a href='https://huggingface.co/WhitePeak/bert-base-cased-Korean-sentiment'>감성 분석을 위해 fine-tuning된 모델</a>을 사용했다.

이번에는 실습이라서 간단하게, 그냥 huggingface tutorial에 있는 예제를 그대로 fine-tuning 한거라 성능은 기대하지 않고, 진행했다.

![Alt text](\..\img\DL4-18.png)

pytorch lightening을 이용한 trainer 객체를 이용한 예제이다.
<br>나중에 native pytorch로 수정도 해볼 생각이다.

보면 output_dir을 'my_awesome_model'이라는 폴더에 집어 넣도록 설정해두었다.

![Alt text](\..\img\DL4-19.png)

간단하게 추론을 해볼 수 있게 하는 pipeline이라는 함수를 꺼내와서 써볼려했는데<br>
에러가 났다.

```
# 보기 쉽게 문장별로 잘랐다.
OSError: Can't load the configuration of '\my_awesome_model'.
If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name.
Otherwise, make sure '\my_awesome_model' is the correct path to a directory containing a config.json file
```

**pipeline에 인자로 넣은 모델을 읽지 못하는 것이었다!**

에러 메시지를 잘 보면,<br> <mark>https://huggingface.co/models에 porting된 모델</mark>이면<br>
**local 환경에 같은 이름의 디렉토리가 없어야 한다** 라고 하고<br>
<mark>local 환경에 있는 모델</mark>이면<br>
**`config.json`이 있는 폴더를 잘 지정해달라** 고 한다.

나의 경우는 후자의 경우이므로, 폴더 내에서 `config.json`파일을 찾아야 했다.

![Alt text](\..\img\DL4-20.png)

작고 소중한 DL test 폴더이다.

확인을 해보니, fine-tuning을 진행하면서, epoch나 iter에 따라 checkpoint가 나뉘어 폴더에 저장되는 것을 확인할 수 있었다.

![Alt text](\..\img\DL4-21.png)

그리고 각각, `config.json` 파일이 저장되어 있음을 볼 수 있었다.

그럼 `config.json` 파일이 들어있는 폴더를 잘 지정해 달라고 했으니, 모델 경로를 수정해보자

![Alt text](\..\img\DL4-22.png)

폴더를 잘 수정했음에도 불구하고, 마찬가지로 모델을 불러오지 못했다.

---

여기에서 많이 해멨는데, 해결 방법부터 말하자면,

**pipeline의 'model' 인자는, local path를 입력할려면, <mark>.</mark> 을 붙여줘야한다.**

상대 경로를 표시할 때 사용하는 .은 '현재 경로'를 의미한다.

이게 없으면 자꾸, https://huggingface.co/ 에서 모델을 찾을지, local에서 찾을지 모르는 모양이다.

아무튼 수정을 해주면

![잘 추론한 모습](\..\img\DL4-23.png)

local에 저장되어있는 모델도 잘 불러와서 추론하는 것을 볼 수 있다.

물론 제대로 된 fine-tuning이 아니였기에, 그냥 *이렇게 할 수 있었다* 라는 점을 확인해볼 수 있었다.

다음 포스팅에서는 실제 AI hub 데이터를 가지고 와서, 모델에 fine-tuning을 시켜보자
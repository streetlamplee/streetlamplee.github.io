---
title: 아파트 거래금액 예측 대회
layout: post
date: 2024-01-29 11:00 +0900
last_modified_at: 2024-01-29 12:00:00 +0900
tag: [ML,  Upstage, AI]
toc: true
---

# 아파트 거래금액 예측 대회

## 1. Abstract

- **Goal of the Competition**
    - 주어진 데이터를 활용하여 서울의 아파트 실거래가를 효과적으로 예측하는 모델을 개발
- **Timeline**
    - 2024.01.08 : <a href = 'https://dacon.io/competitions/official/21265/overview/description'>참고 대회</a>의 우승자 코드를 확인하며 대회 준비
    - 2024.01.10 : 팀원 공부 내용 정리 및 공유
    - 2024.01.12 : 팀원 곰부 내용 정리 및 공유2
    - 2024.01.15 : Competition 시작 (10:00)
    - 2024.01.16 : EDA 및 insight 정리 및 공유
    - 2024.01.17 : 각자 baseline model 생성 및 제출
    - 2024.01.18 : baseline에서 insight를 바탕으로 feature engineering 시작
    - 2024.01.19 : 1주차 중간 결산<br>
    개인 최고 성능 : local RMSE 3000대 / public RMSE 98000대<br>
    팀 최고 성능 : local RMSE 60000대 / public RMSE 95000대<br>
    - 2024.01.22 : 추가적인 EDA를 통해 insight 발굴
    - 2024.01.23 : insight를 바탕으로 성능 개선 작업
    - 2024.01.24 : 팀 회의를 통해 제출물 정리 및 마무리 작업
    - 2024.01.25 : 최종 제출 (19:00)
    - 2024.01.26 : 팀 결과 공유
    

## 2. Process : Competition Model

- 사용 library<br>
```python
# in python
numpy
pandas
matplotlib
seaborn
sklearn
tqdm
optuna
lightgbm
xgboost
interpret
```

- Data feature<br>
RangeIndex: 1118822 entries, 0 to 1118821
Data columns (total 52 columns):
```python
#   Column                  Non-Null Count    Dtype  
---  ------                  --------------    -----  
0   시군구                     1118822 non-null  object 
1   번지                      1118597 non-null  object 
2   본번                      1118747 non-null  float64
3   부번                      1118747 non-null  float64
4   아파트명                    1116696 non-null  object 
5   전용면적(㎡)                 1118822 non-null  float64
6   계약년월                    1118822 non-null  int64  
7   계약일                     1118822 non-null  int64  
8   층                       1118822 non-null  int64  
9   건축년도                    1118822 non-null  int64  
10  도로명                     1118822 non-null  object 
11  해제사유발생일                 5983 non-null     float64
12  등기신청일자                  1118822 non-null  object 
13  거래유형                    1118822 non-null  object 
14  중개사소재지                  1118822 non-null  object 
15  k-단지분류(아파트,주상복합등등)      248131 non-null   object 
16  k-전화번호                  248548 non-null   object 
17  k-팩스번호                  246080 non-null   object 
18  단지소개기존clob              68582 non-null    float64
19  k-세대타입(분양형태)            249259 non-null   object 
20  k-관리방식                  249259 non-null   object 
21  k-복도유형                  248932 non-null   object 
22  k-난방방식                  249259 non-null   object 
23  k-전체동수                  248192 non-null   float64
24  k-전체세대수                 249259 non-null   float64
25  k-건설사(시공사)              247764 non-null   object 
26  k-시행사                   247568 non-null   object 
27  k-사용검사일-사용승인일           249126 non-null   object 
28  k-연면적                   249259 non-null   float64
29  k-주거전용면적                249214 non-null   float64
30  k-관리비부과면적               249259 non-null   float64
31  k-전용면적별세대현황(60㎡이하)      249214 non-null   float64
32  k-전용면적별세대현황(60㎡~85㎡이하)  249214 non-null   float64
33  k-85㎡~135㎡이하            249214 non-null   float64
34  k-135㎡초과                327 non-null      float64
35  k-홈페이지                  113175 non-null   object 
36  k-등록일자                  10990 non-null    object 
37  k-수정일자                  249214 non-null   object 
38  고용보험관리번호                205518 non-null   object 
39  경비비관리형태                 247834 non-null   object 
40  세대전기계약방법                240075 non-null   object 
41  청소비관리형태                 247644 non-null   object 
42  건축면적                    249108 non-null   float64
43  주차대수                    249108 non-null   float64
44  기타/의무/임대/임의=1/2/3/4     249259 non-null   object 
45  단지승인일                   248536 non-null   object 
46  사용허가여부                  249259 non-null   object 
47  관리비 업로드                 249259 non-null   object 
48  좌표X                     249152 non-null   float64
49  좌표Y                     249152 non-null   float64
50  단지신청일                   249197 non-null   object 
51  target                  1118822 non-null  int64
```

- feature engineering<br>
기타 feature 제거,<br>
이상치 처리,<br>
결측치 선형보간,<br>
아파트명 target encoding,<br>
시군구 target ranking encoding,<br>
역세권 feature 추가,<br>
신축 feature 추가<br>

- 사용 모델<br>
```python
LGMBRegressor(  'n_estimators': 2457,
                'learning_rate': 0.0883515664394438,
                'num_leaves': 2047,
                'colsample_bytree': 0.45526849857171015,
                'reg_lambda': 93.54910084071389,
                'min_child_samples': 16,
                'max_depth': 11,
                'min_split_gain': 0.017382582855317637
            )
```

- 결과(개인)<br>
local RMSE : 3068.63<br>
public RMSE : 103990.5716 (재채점 전)<br>
private RMSE : - (채점 전)

- 결과(팀)<br>
local RMSE : - (정보 누락)<br>
public RMSE : 18200.0390<br>
private RMSE : 16173.2293

## 3. Process : Issues

**Describe the issue that your team faced during the project.**

1. 결측치가 다수 존재<br>
![Alt text](\..\img\house1.png)<br>
대부분의 column에서 결측 비율이 77% 이상을 보여, 이를 보간하는데에 어려움이 있었음<br>
특히, 좌표X와 좌표Y 또한 결측비율이 77%로 나타나서, 같이 주어진 subway_feature나 bus_feature를 사용할 수 없었음<br>
해당 모델에서는 수치형 변수에 대해서 선형보간으로 처리하고, 범주형 변수에 대해서는 null로 일괄 변환
2. public leader board와의 성능 차이가 너무 컸다.<br>
대회 당시에는, local과 public의 성능의 차이가 너무 극명하게 나타나서 올바르게 성능이 개선되고 있는지를 파악할 수 있는 방법이 없었음.<br>
leader board의 성능을 보지않고 local로만 대회를 제출했고, 결과적으로 local의 성능이 높은 모델이 더 성능이 좋았다는 점이 밝혀짐
대회 후에 test data에서 인덱싱 오류로 인해 leaderboard의 점수 자체에 문제가 있었음
3. feature selection의 기준 문제<br>
보통 모델의 성능이 개선되는 방향으로 feature selection을 하지만, local 점수 외에 leaderboard 점수가 문제가 있어 feature selection에서도 큰 혼란이 있었음.<br> 후에 leaderboard RMSE를 생각하지 않고, feature selection을 진행했지만, 대회 준비 기간 자체가 짧아 제대로 된 성능을 가진 모델을 만들지 못해 아쉬움

**Describe the possible solution to imporve your project.**
1. 결측치에 대한 다양한 보간법 확인<br>
해당 모델에서는 모든 결측치에 대한 선형 보간을 실시했다. 하지만 각 feature의 특성을 고려하지 않고 보간을 해 feature가 가지는 정보가 부정확해졌을 가능성이 있다.<br>
따라서 각 feature의 특성에 맞는 다양한 보간 방법이나, 외부 데이터를 이용해서 데이터를 보간하는 방법을 이용해서 데이터의 특성을 잘 살리고 모델링을 진행했으면 성능이 더 좋아지지 않았을까 싶다.<br>
ex. 좌표X 좌표Y를 선형보간 $\rightarrow$ 카카오맵 API를 이용한 데이터 보간<br>
ex. 전용면적을 선형보간 $\rightarrow$ 해당 건물의 전용면적의 최빈값으로 데이터 보간

2. leaderboard 채점의 오류 문제

3. feature selection 시, leaderboard보다 local 점수 우선<br>
대회 중 leaderboard는 public test set에 국한되므로, local에서 일반화를 잘 시킬 수 있다면 (KFold나 train-test-split을 잘한다면) local 점수를 우선해서 feature selection의 기준을 잡는 것이 feature selection에서 혼란을 방지할 수 있을 것 같다.<br>
이번 대회와 같이 leaderboard의 오류도 있을 가능성이 있으며, 또한 public test set에 국한된 데이터이기에, private test set을 검증하는 것에는 일반적으로 더 잘 예측하는 경우가 나을 것이라고 판단했다.

    

## 4. Role

**Describe your role with task in your team.**
* LGBMRegressor model with subway feature, ranking encoder<br>
EDA, Data Clensing, Feature Selection, Modeling
* XGBRegressor model with interest rate, is_Gangnam, subway feature, target meaning<br>
EDA, Data Clensing, Feature Selection, Modeling

(notion. 각자 EDA와 모델링을 진행하고 그 insight를 공유하는 식으로 대회를 진행하여, 서로 다른 모델을 만들었음)

**Describe papers or book chapeters you found relevant to the problem, references to see.**

<a href = 'https://manuscriptlink-society-file.s3-ap-northeast-1.amazonaws.com/kips/conference/2020fall/presentation/KIPS_C2020B0235.pdf'>주정민,강선미,최지웅,한영우 (2020). "기계학습을 이용한 아파트 매매가격 예측 연구 : 한국 아파트의 내·외적 데이터 수집과 가격 예측 중심으로." \<2020 온라인 추계학술발표대회 논문집\>, 27권 2호, 956-959.</a>

<a href = 'https://m.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=853585e06f6a6d46ffe0bdc3ef48d419#redirect'>박성훈. "머신러닝을 이용한 서울특별시 부동산 지수 예측 모델 비교." 국내석사학위논문 한양대학교 공학대학원, 2020. 서울</a>

**Explain which are relevant for your Project.**

위의 두 참조문헌 모두 연관이 있었음.
    

## 5. Results

**Write the main result of Competition**

- 결과(팀)<br>
local RMSE : 6068.63<br>
public RMSE : 17386.8731<br>
private RMSE : 18414.4742

- 결과(개인)<br>
local RMSE : 5481.3258<br>
public RMSE : 18881.3815<br>
private RMSE : 12353.6405

![alt text](..\img\result.png)

**Final standings of the Leaderboard**

**3등 기록**
    

## 6. Conclusion

**Describe your running code with its own advantages and disadvantages, in relation to other groups in the course.**

### 0. 라이브러리 임포트


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')

tqdm.pandas()
```

### 1. 데이터 불러오기


```python
with open('../data/train.csv') as f:
    train = pd.read_csv(f)
with open('../data/test.csv') as f:
    test = pd.read_csv(f)
```

    /tmp/ipykernel_625408/287244562.py:2: DtypeWarning: Columns (16,17,36) have mixed types. Specify dtype option on import or set low_memory=False.
      train = pd.read_csv(f)
    

한번에 전처리하기 위해 train, test를 묶음


```python
test['target'] = 0
train['is_test'] = 0
test['is_test'] = 1
train = pd.concat([train, test])
```


```python
train.reset_index(inplace = True)
train.drop(columns = 'index', inplace = True)
```

columns 이름을 보기 편하게 변경


```python
train.columns = [col.replace('㎡', 'm').replace('k-','').replace('기타/의무/임대/임의=1/2/3/4','기타의무임대임의').replace('(아파트,주상복합등등)','').replace('(','_').replace(')','') for col in train.columns]

```

x, y 데이터 추가


```python
with open('../data/include_xy.csv') as f:
    xydata = pd.read_csv(f)
```


```python
train['addr'] = train['시군구'] + ' ' + train['번지']
```


```python
train = pd.merge(train, xydata, how = 'left', left_on = 'addr', right_on = 'addr')
train.drop(columns = ['좌표X','좌표Y'], inplace = True)
```



```python
train.to_csv('../data/processed_df.csv', index = False)
```

### 2. EDA

#### 결측비율 확인

결측치 표현을 실제 결측치로 변경


```python
train['거래유형'] = train['거래유형'].replace('-', np.nan)
train['등기신청일자'] = train['등기신청일자'].replace(' ', np.nan)
train['중개사소재지'] = train['중개사소재지'].replace('-', np.nan)
```

그래프로 그려보기


```python
missingnum = []
for i, v in enumerate(train.columns):
    missingnum.append(train[v].isnull().sum())

plt.figure(figsize = (14,10))
plots = sns.barplot(x = missingnum, y = train.columns)
for bar in plots.patches:
      plots.annotate(format(bar.get_width() / len(train), '.3f'), 
                   (bar.get_width(), bar.get_y() + bar.get_height()/2),
                    ha='left', va='center',
                   size=12, xytext=(8, 0),
                   textcoords='offset points')
plt.show()
```


    
![png](\..\img\final_19_0.png)
    


#### 상관관계 확인

계약년월을 계약년, 계약월로 나누어서 상관관계 확인


```python
def year_month_parser(x):
    year = int(str(x)[0:4])
    month = int(str(x)[4:6])
    return [year, month]

train['계약년'] = train['계약년월'].apply(lambda x : year_month_parser(x)[0])
train['계약월'] = train['계약년월'].apply(lambda x : year_month_parser(x)[1])
train.drop(columns = '계약년월', inplace = True)
plt.figure(figsize=(15,15))
sns.heatmap(data = train.corr(numeric_only=True), annot = True, fmt = '.2f', cmap='RdYlGn_r')
```

![png](\..\img\final_23_3.png)

1. '135m초과'는 어떠한 column과도 상관관계가 존재하지 않는다.
2. 아파트의 크기와 관련된 feature들은 서로 강한 상관관계가 있다.<br>
$\rightarrow$'전체 동수','전체 세대수','연면적','주거전용면적','관리비부과면적','전용면적별세대현황','주차면적'
3. target과 강한 상관관계가 있는 feature는 '전용면적_m','좌표Y','계약년'와 면적관련 3가지 feature ('연면적','주거전용면적','관리비부과면적')이다.


```python
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (24, 16))
for i, v in enumerate(['전용면적_m','주거전용면적','연면적','주차대수','y']):
    sns.scatterplot(data = train, x = v, y = 'target', ax = ax[i//3][i%3])
```


    
![png](\..\img\final_25_0.png)
    


1. 연면적의 경우, outlier가 있어 이를 처리하고도 상관관계가 있는지 확인해볼 필요가 있음
2. 좌표Y의 경우, 특정 좌표에서 값이 크게 오르는 것을 확인할 수 있음
3. '좌표Y','연면적','주거전용면적','관리비부과면적' 은 결측치가 높아 이를 보간할 방법이 필요 or 아예 drop<br><br>
면적의 경우, 아예 보간을 할 방법이 마땅치 않음 $\rightarrow$ **drop**<br>
관리비도 어렵다는 판단으로 drop

### 3. feature engineering


```python
df = train.drop(columns = ['전화번호','팩스번호'])
df.reset_index(inplace=True)
```


```python
df.columns
```

    Index(['index', '시군구', '번지', '본번', '부번', '아파트명', '전용면적_m', '계약일', '층', '건축년도',
           '도로명', '해제사유발생일', '등기신청일자', '거래유형', '중개사소재지', '단지분류', '단지소개기존clob',
           '세대타입_분양형태', '관리방식', '복도유형', '난방방식', '전체동수', '전체세대수', '건설사_시공사', '시행사',
           '사용검사일-사용승인일', '연면적', '주거전용면적', '관리비부과면적', '전용면적별세대현황_60m이하',
           '전용면적별세대현황_60m~85m이하', '85m~135m이하', '135m초과', '홈페이지', '등록일자', '수정일자',
           '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수',
           '기타의무임대임의', '단지승인일', '사용허가여부', '관리비 업로드', '단지신청일', 'target', 'is_test',
           'addr', 'x', 'y', '계약년', '계약월'],
          dtype='object')


#### 기준 금리 데이터 추가


```python
with open('../data/interest_rate.csv') as f:
    interest = pd.read_csv(f)

```
출처 : <a href = 'https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643'>https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643</a>


```python
t = interest
t = t.astype('str')
interest.loc[:,'datetime'] = t['year'] + t['month'].apply(lambda x: '0'+x if len(x) == 1 else x) + t['date'].apply(lambda x: '0'+x if len(x) == 1 else x)
interest.sort_values(by = ['year','month','date'], ascending=True, inplace = True)
interest.reset_index(inplace=True)
interest.drop(columns = 'index', inplace = True)

```


```python
# import datetime
# df['interest_rate'] = [-1] * len(df)
# for i in range(len(df)):
#     contract_date = str(df.loc[i,'계약년']) + '-' + str(df.loc[i, '계약월'])+ '-'+ str(df.loc[i,'계약일'])
#     contract_date = datetime.datetime.strptime(contract_date, '%Y-%m-%d')
#     for j in range(len(interest)-1):
#         compare_date1 = datetime.datetime.strptime(interest.loc[j,'datetime'], '%Y%m%d')
#         compare_date2 = datetime.datetime.strptime(interest.loc[j+1,'datetime'], '%Y%m%d')
#         if (compare_date1<=contract_date) and (contract_date < compare_date2):
#             df.loc[i, 'interest_rate'] =  interest.loc[j, 'rate']
#             break

```

기준 금리 데이터를 dataframe에 feature로 추가하는 코드

오래 걸려서 따로 csv로 저장후 관리

```python
# df.loc[:,'interest_rate'] = df['interest_rate'].apply(lambda x : 3.5 if x == -1 else x)
```


```python
df.to_csv('../data/dataframe.csv', index = False)
```



```python
with open('../data/dataframe.csv') as f:
    df = pd.read_csv(f)
```

    /tmp/ipykernel_625408/2153709074.py:2: DtypeWarning: Columns (13,14,34) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv(f)
    

큰 상관관계가 없던 계약 월, 계약 일 삭제


```python
# df.drop(columns=['계약월','계약일', 'index'], inplace = True)
# df.head(5)
```

#### 강남여부 확인


```python
def gangnam_parser(x):
    gu_li = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
    if x.split(' ')[1] in gu_li:
        return 1
    else:
        return 0

df.loc[:,'is_gangnam'] = df['시군구'].apply(gangnam_parser)
```


```python
df.drop(columns = ['시군구','아파트명'], inplace = True)
```

#### 역세권 찾기


```python
# with open('../data/subway_feature.csv') as f:
#     subway_df = pd.read_csv(f)

# def subway_distance(x, y):
#     y_building = y
#     x_building = x
#     for i in range(len(subway_df)):
#         x_subway = subway_df.loc[i, '경도']
#         y_subway = subway_df.loc[i, '위도']

#         x_distance = abs(x_building - x_subway)
#         y_distance = abs(y_building - y_subway)

#         #위도 경도 변환
#         x_distance = 88000 * x_distance
#         y_distance = 110000 * y_distance

#         distance = np.sqrt(x_distance ** 2 + y_distance ** 2)
#         if distance <= 500:
#             return 1

#     return 0

# tmp = train.progress_apply(lambda row : subway_distance(row['x'], row['y']), axis = 1)
```

지하철 역을 기준으로 500m이내의 건물여부를 dataframe에 feature로 추가하는 코드

오래 걸려서 따로 csv로 저장후 관리

```python
# tmp.to_csv('../data/is_subway.csv', index = False)
```


```python
with open('../data/is_subway.csv') as f:
    tmp = pd.read_csv(f)

df['is_subway'] = tmp
```

#### target을 평균을 이용해서 처리


```python
df['price'] = df[df['is_test'] == 0].groupby(['도로명'])['target'].transform('mean')

```

### label encoding


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1128094 entries, 0 to 1128093
    Data columns (total 55 columns):
     #   Column               Non-Null Count    Dtype  
    ---  ------               --------------    -----  
     0   index                1128094 non-null  int64  
     1   번지                   1127867 non-null  object 
     2   본번                   1128019 non-null  float64
     3   부번                   1128019 non-null  float64
     4   전용면적_m               1128094 non-null  float64
     5   계약일                  1128094 non-null  int64  
     6   층                    1128094 non-null  int64  
     7   건축년도                 1128094 non-null  int64  
     8   도로명                  1128094 non-null  object 
     9   해제사유발생일              6195 non-null     float64
     10  등기신청일자               16823 non-null    float64
     11  거래유형                 41643 non-null    object 
     12  중개사소재지               38081 non-null    object 
     13  단지분류                 250821 non-null   object 
     14  단지소개기존clob           69136 non-null    float64
     15  세대타입_분양형태            251969 non-null   object 
     16  관리방식                 251969 non-null   object 
     17  복도유형                 251640 non-null   object 
     18  난방방식                 251969 non-null   object 
     19  전체동수                 250887 non-null   float64
     20  전체세대수                251969 non-null   float64
     21  건설사_시공사              250457 non-null   object 
     22  시행사                  250260 non-null   object 
     23  사용검사일-사용승인일          251835 non-null   object 
     24  연면적                  251969 non-null   float64
     25  주거전용면적               251924 non-null   float64
     26  관리비부과면적              251969 non-null   float64
     27  전용면적별세대현황_60m이하      251924 non-null   float64
     28  전용면적별세대현황_60m~85m이하  251924 non-null   float64
     29  85m~135m이하           251924 non-null   float64
     30  135m초과               329 non-null      float64
     31  홈페이지                 114571 non-null   object 
     32  등록일자                 11708 non-null    object 
     33  수정일자                 251924 non-null   object 
     34  고용보험관리번호             207337 non-null   object 
     35  경비비관리형태              250533 non-null   object 
     36  세대전기계약방법             242705 non-null   object 
     37  청소비관리형태              250343 non-null   object 
     38  건축면적                 251815 non-null   float64
     39  주차대수                 251817 non-null   float64
     40  기타의무임대임의             251969 non-null   object 
     41  단지승인일                251240 non-null   object 
     42  사용허가여부               251969 non-null   object 
     43  관리비 업로드              251969 non-null   object 
     44  단지신청일                251907 non-null   object 
     45  target               1128094 non-null  int64  
     46  is_test              1128094 non-null  int64  
     47  addr                 1127867 non-null  object 
     48  x                    1111831 non-null  float64
     49  y                    1111831 non-null  float64
     50  계약년                  1128094 non-null  int64  
     51  계약월                  1128094 non-null  int64  
     52  is_gangnam           1128094 non-null  int64  
     53  is_subway            1128094 non-null  int64  
     54  price                1118822 non-null  float64
    dtypes: float64(20), int64(10), object(25)
    memory usage: 473.4+ MB
    


```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for v in df.columns:
    if pd.api.types.is_object_dtype(df[v]):
        print(v)
        df[v] = encoder.fit_transform(df[v])

```

    

### 3. Modeling


```python
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from functools import partial
import optuna
from sklearn.metrics import mean_squared_error

```





```python
test = df[df['is_test'] == 1]
test.drop(columns = ['target','is_test','price'], inplace = True)
train = df[df['is_test'] == 0]
train.drop(columns = 'is_test',inplace = True)

X_train = train[train['계약년'] <= 2020]
X_train = X_train.drop(columns=['target'])
y_train = X_train['price']
X_train = X_train.drop(columns=['price'])

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"test shape : {test.shape}")

```

    /tmp/ipykernel_625408/751170959.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test.drop(columns = ['target','is_test','price'], inplace = True)
    /tmp/ipykernel_625408/751170959.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train.drop(columns = 'is_test',inplace = True)
    

    X_train shape : (1045943, 52)
    y_train shape : (1045943,)
    test shape : (9272, 52)
    

사용안하는 feature 제거


```python
X_train = X_train[['전용면적_m','건축년도','y']]

```


```python
test = test[['전용면적_m','건축년도','y']]
```


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1045943 entries, 0 to 1118821
    Data columns (total 3 columns):
     #   Column  Non-Null Count    Dtype  
    ---  ------  --------------    -----  
     0   전용면적_m  1045943 non-null  float64
     1   건축년도    1045943 non-null  int64  
     2   y       1030220 non-null  float64
    dtypes: float64(2), int64(1)
    memory usage: 31.9 MB
    


```python
model = LGMBRegressor(  'n_estimators': 2457,
                        'learning_rate': 0.0883515664394438,
                        'num_leaves': 2047,
                        'colsample_bytree': 0.45526849857171015,
                        'reg_lambda': 93.54910084071389,
                        'min_child_samples': 16,
                        'max_depth': 11,
                        'min_split_gain': 0.017382582855317637
            )
    
kf = KFold(n_splits = 5, random_state=42, shuffle=True)
evaluation = []
for idx, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train))):
    X_val = X_train.iloc[val_idx,:]
    X_training = X_train.iloc[train_idx,:]
    y_val = y_train.iloc[val_idx]
    y_training = y_train.iloc[train_idx]
  
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 42)


    model.fit(X_training, y_training)
    prediction = model.predict(X_val)
    evaluation.append(np.sqrt(mean_squared_error(y_val, prediction)))

np.mean(evaluation)
```

    5it [00:06,  1.33s/it]

    
    3068.63


    
### 6. inference

```python
model.fit(X_train, y_train)

pred = model.predict(test)

preds_df = pd.DataFrame(pred.astype(int), columns=["target"])
preds_df.to_csv('output.csv', index=False)

import pickle
with open('saved_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```




```python
final_model = LGBMRegressor(**study.best_params, random_state = 42, verbose  = -1)

final_model.fit(X_train, y_train)
```


```python
final_pred = final_model.predict(test)

preds_df = pd.DataFrame(final_pred.astype(int), columns=["target"])
preds_df.to_csv('output.csv', index=False)
```

### 7. Hyperparameter Tuning

```python
def optimizer1(trial, X, y, K):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_float('learning_rate',0.01, 0.1)
    num_leaves = trial.suggest_categorical('num_leaves',[255,511,1023, 2047, 4095])
    colsample_bytree = trial.suggest_float('colsample_bytree',0.4,0.8)
    reg_lambda = trial.suggest_float('reg_lambda',0.5,200)
    min_child_samples = trial.suggest_int('min_child_samples',4,20)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_split_gain = trial.suggest_float('min_split_gain',0.001,0.1)

    model = LGBMRegressor(n_estimators = n_estimators,
                          learning_rate = learning_rate,
                          num_leaves = num_leaves
                          colsample_bytree = colsample_bytree,
                          reg_lambda = reg_lambda,
                          min_child_samples = min_child_samples
                          max_depth = max_depth,
                          min_split_gain = min_split_gain,
                          random_state = 42,
                          verbose = -1
                          )
    
    kf = KFold(n_splits = K, random_state=42, shuffle=True)
    evaluation = []
    for idx, (train_idx, val_idx) in enumerate(tqdm(kf.split(X,y))):
        X_train = X.iloc[train_idx,:]
        X_val = X.iloc[val_idx,:]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        

        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        evaluation.append(np.sqrt(mean_squared_error(y_val, prediction)))
    
    return np.mean(evaluation)

K = 5
opt_func = partial(optimizer1, X = X_train, y = y_train, K = K)
study = optuna.create_study(direction = 'minimize')
study.optimize(opt_func, n_trials = 5000)
```
(이상 생략)

[I 2024-01-24 17:12:47,029] A new study created in memory with name: no-name-b0109014-70a2-49db-8400-5c2bdfa8b097

5it [00:08,  1.77s/it]
[I 2024-01-24 18:49:10,789] Trial 882 finished with value: 3068.63159665463 and parameters: <br>{'n_estimators': 2457,
                        'learning_rate': 0.0883515664394438,
                        'num_leaves': 2047,
                        'colsample_bytree': 0.45526849857171015,
                        'reg_lambda': 93.54910084071389,
                        'min_child_samples': 16,
                        'max_depth': 11,
                        'min_split_gain': 0.017382582855317637}.
                        <br> Best is trial 882 with value: 3068.63159665463.

(이후 생략)


### 8. feature importance를 확인하기 위해 사용한 코드

```python
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

interpretmodel = ExplainableBoostingRegressor()
interpretmodel.fit(X_train, y_train)

from interpret import set_visualize_provider
from interpret.provider import InlineProvider

set_visualize_provider(InlineProvider())
show(interpretmodel.explain_global())

```

**Sum up your project and suggest future extensions and improvements.**

XGB와 LGBM을 이용해서 정형데이터로 회귀 문제를 해결해보았다.

주최 측의 오류로 인해 대회 중에 leaderboard가 제대로 동작하지 않은 것은 명백히 아쉬운 일이지만, 오히려 local 점수를 믿으면서 대회를 진행해도 어느 정도 대회를 성공적으로 마무리 지을 수 있다는 것을 증명하기도 하였다.

데이터 자체의 결측치가 많아 이를 보간하기 위해서 노력을 많이 했으며, 실제로 위의 코드는 `include_xy.csv`를 이용해서 x와 y좌표를 외부 데이터로 보간하는 코드가 포함되어 있다. 하지만 이외의 feature에 대해서는 결측치를 정리할 수 없었고, 단순한 선형보간을 이용해서 진행했던 점이 너무 아쉬운 점이다.

이후에 기회가 있다면, EDA를 더 명확히 해 대회 준비 시간을 효율적으로 사용하고 싶다. <br>단순히 그래프를 그려보는 EDA와는 다르게, 다양한 외부 데이터와 파생 feature를 도메인 지식에 따라 미리 만들어보고 싶다. 그 다음으로 EDA를 진행해서 좀 더 많은 insight를 빠르게 얻은 다음 feature engineerning이나 modeling으로 발전하는 것이 조금 더 효율적이라는 생각이 들었다.
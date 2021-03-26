---
layout: post
title: "Feature Engineering"
date: 2021-03-25
categories: ml
tags: ml
---

이 과정에서는 훌륭한 기계 학습 모델을 구축하는 가장 중요한 단계 중 하나 인 피쳐(feature) 엔지니어링에 대해 알아 봅니다. 다음 방법을 배우게됩니다.

- 상호의존정보에서 가장 중요한 피쳐를 결정
- 여러 실제 문제 영역에서 새로운 피쳐를 개발
- target encoding을 사용하여 카디널리티가 높은 범주를 인코딩
- k-means 클러스터링으로 세분화 피쳐 생성
- 주성분 분석을 통해 데이터 세트(dataset)의 변형을 피쳐로 분해

피쳐 엔지니어링의 목표는 단순히 데이터를 문제를 해결하기에 더 적합하게 만드는 것입니다.

예를 들어 열지수 및 체감추위와 같은 "체감 온도"를 측정할 때, 우리가 직접 측정 할 수 있는 공기 온도, 습도 및 풍속을 기반으로 사람이 체감하는 온도를 측정합니다. 체감 온도는 일종의 피쳐 엔지니어링의 결과로 생각할 수 있습니다. 관측된 데이터를 우리가 알고싶은 데이터로 가공하는 것을 의미합니다.

피쳐 엔지니어링은 다음과 같은 목적으로 사용됩니다.
- 모델의 예측 성능 향상
- 계산 또는 데이터 요구 감소
- 결과의 해석 가능성 향상

#### 피쳐 엔지니어링의 기본 원리
피쳐가 유용하려면 모델이 학습 할 수 있는 대상과의 관계가 있어야합니다. 예를 들어 선형 모델(linear model)은 선형 관계만 학습할 수 있습니다. 따라서 선형 모델을 사용할 때 목표는 대상과의 관계를 선형으로 만들기 위해 피쳐를 변환하는 것입니다.

여기서 핵심 아이디어는 피쳐에 적용하는 변환(transformation)이 본질적으로 모델 자체의 일부가 된다는 것입니다. 한 면의 길이에서 토지의 정사각형 구획 가격을 예측하려한다고 가정해보십시오. 선형 모델을 길이에 직접 피팅(fitting)하면 결과가 좋지 않습니다. 관계가 선형이 아니기 때문입니다.

![사진](/assets/imgs/posts/ml/feature-engineering-001.png)

그러나 'Area'를 얻기 위해 Length 피쳐를 제곱하면 선형 관계가 생성됩니다. 피쳐 세트에 Area를 추가하면 이 선형 모델이 이제 포물선에 맞을 수 있습니다.

![사진](/assets/imgs/posts/ml/feature-engineering-002.png)

이는 피쳐 엔지니어링에 투자 된 시간에 비해 높은 성과를 내는 모습을 보여줍니다. 모델이 학습 할 수 없는 관계가 무엇이든 변환을 통해 우리가 직접 만들어 제공해 줄 수 있습니다. 피쳐 세트를 개발할 때 최상의 성능을 달성하기 위해 모델이 사용할 수 있는 정보에 대해 생각하십시오.

#### 예제 - 콘크리트 제형
이러한 아이디어를 설명하기 위해 데이터 세트에 몇 가지 합성 피쳐를 추가하여 랜덤 포레스트 모델의 예측 성능을 향상시킬 수 있는 방법을 살펴 보겠습니다.

[*콘크리트 데이터 세트*](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength)에는 다양한 콘크리트 제형과 결과물품의 압축 강도가 포함되어 있습니다. 이는 해당 종류의 콘크리트가 견딜 수 있는 하중의 척도입니다. 이 데이터 세트의 작업은 제형이 주어진 콘크리트의 압축 강도를 예측하는 것입니다.
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("input_data/concrete.csv")
df.head()
```
```text
   Cement  BlastFurnaceSlag  FlyAsh  ...  FineAggregate  Age  CompressiveStrength
0   540.0               0.0     0.0  ...          676.0   28                79.99
1   540.0               0.0     0.0  ...          676.0   28                61.89
2   332.5             142.5     0.0  ...          594.0  270                40.27
3   332.5             142.5     0.0  ...          594.0  365                41.05
4   198.6             132.4     0.0  ...          825.5  360                44.30

[5 rows x 9 columns]
```

여기에서 다양한 콘크리트에 들어가는 다양한 재료를 볼 수 있습니다. 우리는 이들로부터 파생 된 몇 가지 추가 합성 피쳐를 추가하는 것이, 모델이 이들 간의 중요한 관계를 학습하는 데 어떻게 도움이 되는지 잠시 후에 알게 될 것입니다.

먼저 비증강 데이터 세트에서 모델을 학습하여 기준선을 설정합니다. 이것은 우리의 새로운 피쳐가 실제로 유용한 지 판단하는 데 도움이 될 것입니다.

이와 같은 기준을 설정하는 것은 피쳐 엔지니어링 프로세스를 시작할 때 좋은 방법입니다. 기준 점수는 새 피쳐를 유지할 가치가 있는지 또는 삭제하고 다른 작업을 시도해야하는지 여부를 결정하는 데 도움이 될 수 있습니다.
```python
X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="mae", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")
```
```text
MAE Baseline Score: 8.232
```

집에서 요리 한 적이 있다면 레시피의 재료 비율이 일반적으로 절대량보다 레시피의 결과를 더 잘 예측할 수 있다는 것을 알고있을 것입니다. 그런 다음 위 기능의 비율이 CompressiveStrength의 좋은 예측 변수가 될 것이라고 추론 할 수 있습니다.

데이터 세트에 세 가지 새로운 비율 피쳐를 추가해봅시다.
```python
X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="mae", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")
```
```text
MAE Score with Ratio Features: 7.948
```

확실히 성능이 향상되었습니다! 이는 이러한 새로운 비율 피쳐가 이전에 감지하지 못했던 모델에 중요한 정보를 노출했다는 증거입니다.

### 목차

>* [*Kaggle Course - Feature Engineering*](https://www.kaggle.com/learn/feature-engineering)

1. 상호의존정보(Mutual Information)
1. Creating Features
1. Clustering With K-Means
1. Principal Component Analysis
1. Target Encoding

---

### 상호의존정보(Mutual Information)
새 데이터 세트를 처음 만나는 것은 때때로 압도적이라고 느낄 수 있습니다. 설명 없이도 수백 또는 수천 개의 기능이 제공 될 수 있습니다. 어디서부터 시작해야할까요?

첫 번째 단계는 피쳐와 대상 간의 연관성을 측정하는 기능인 feature utility metric을 사용하여 순위를 구성하는 것입니다. 그런 다음 초기에 개발할 가장 유용한 피쳐의 작은 집합을 선택하고 시간을 잘 보낼 것이라는 확신을 가질 수 있습니다.

우리가 사용할 측정 항목을 "상호의존정보(MI, mutual information)"라고합니다. 상호의존정보는 두 수량 간의 관계를 측정한다는 점에서 상관 관계와 매우 유사합니다. 상호의존정보의 장점은 모든 종류의 관계를 감지 할 수 있는 반면 상관 관계는 선형 관계만 감지한다는 것입니다.

상호의존정보는 훌륭한 범용 측정 항목이며 아직 어떤 모델을 사용하고 싶은지 모를 때 기능 개발을 시작할 때 특히 유용합니다.

#### 상호의존정보의 장점
- 사용과 해석이 쉽다.
- 계산 효율성이 좋다.
- 이론적 근거가 충분하다.
- 과적합에 강하다.
- 모든 종류의 관계를 감지할 수 있다.

#### 상호의존정보 및 측정 대상
상호의존정보는 불확실성의 관점에서 관계를 설명합니다. 두 수량 간의 상호의존정보는 한 수량에 대한 지식이 다른 수량에 대한 불확실성을 줄이는 정도의 척도입니다. 만약 우리가 피쳐의 값을 알고 있다면 대상에 대해 얼마나 더 확신 할 수 있습니까?

다음은 Ames Housing 데이터의 예입니다. 그림은 집의 외관 품질과 판매 가격 간의 관계를 보여줍니다. 각 점은 집을 나타냅니다.

![사진](/assets/imgs/posts/ml/feature-engineering-003.png)

그림에서 ExterQual의 값을 알면 해당 SalePrice에 대해 더 확실하게 알 수 있습니다. ExterQual의 각 범주는 SalePrice를 특정 범위 내로 집중시키는 경향이 있습니다. ExterQual이 SalePrice와 함께 가지고 있는 상호 정보는 ExterQual의 4가지 값을 인수한 SalePrice의 평균 불확실성 감소입니다. 예를 들어 Fair는 Typical보다 덜 자주 발생하므로 Fair는 MI 점수에서 가중치가 더 적습니다.

<b>Technical note</b>: What we're calling uncertainty is measured using a quantity from information theory known as "entropy". The entropy of a variable means roughly: "how many yes-or-no questions you would need to describe an occurance of that variable, on average." The more questions you have to ask, the more uncertain you must be about the variable. Mutual information is how many questions you expect the feature to answer about the target.

#### 상호의존정보 점수 해석
수량 간의 가능한 최소 상호의존정보는 0.0입니다. MI가 0일 때 서로의 관계는 완전히 독립적입니다. 반대로 이론상 MI의 상한선은 없습니다. 실제로 2.0 이상의 값은 흔하지 않습니다. (상호의존정보는 로그함수적 지표이므로 매우 느리게 증가합니다.)

다음 그림은 MI값이 피쳐와 대상과의 연관성의 종류 및 정도에 해당하는 방식에 대한 아이디어를 제공합니다.

![사진](/assets/imgs/posts/ml/feature-engineering-004.png)

상호의존정보를 적용 할 때 기억해야 할 사항은 다음과 같습니다.
- MI는 자체적으로 고려되는 대상의 예측 변수로서 피쳐의 상대적 잠재력을 이해하는 데 도움이 될 수 있습니다.
- 피쳐가 다른 피쳐와 상호 작용할 때 매우 유익 할 수 있지만 그다지 유익하지 않을 수도 있습니다. MI는 피쳐 간의 상호 작용을 감지 할 수 없습니다. 일변량 측정 항목입니다.
- 피쳐의 실제 유용성은 사용하는 모델에 따라 다릅니다. 특성은 대상과의 관계가 모델이 학습 할 수 있는 경우에만 유용합니다. 기능의 MI 점수가 높다고해서 모델이 해당 정보로 무엇이든 할 수 있다는 의미는 아닙니다. 연관성을 표시하려면 먼저 피쳐를 변환해야 할 수 있습니다.

#### 예제 - 1985년도 자동차
[*자동차 데이터 세트*](https://www.kaggle.com/toramky/automobile-dataset)는 1985년 모델 연도의 자동차 193대로 구성됩니다. 이 데이터 세트의 목표는 make, body_style, horsepower와 같은 자동차의 23가지 기능에서 자동차의 price(target)를 예측하는 것입니다. 이 예시에서는 상호의존정보로 피쳐의 순위를 매기고 데이터 시각화를 통해 결과를 조사합니다.
```python
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-whitegrid")

data_path = 'input_data/autos.csv'
# Replace '?' with null value
df = pd.read_csv(data_path, na_values='?')

# Exclude data with null price
df = df[df['price'].notna()]
df.head()
```
```text
   symboling  normalized-losses         make  ... city-mpg highway-mpg    price
0          3                NaN  alfa-romero  ...       21          27  13495.0
1          3                NaN  alfa-romero  ...       21          27  16500.0
2          1                NaN  alfa-romero  ...       19          26  16500.0
3          2              164.0         audi  ...       24          30  13950.0
4          2              164.0         audi  ...       18          22  17450.0

[5 rows x 26 columns]
```

MI에 대한 scikit-learn 알고리즘은 이산 피쳐를 연속 피쳐와 다르게 처리합니다. 따라서 어떤 종류의 피쳐인지 알려주어야합니다. 경험상 float dtype을 가져야하는 모든 것은 이산적이지 않습니다. 범주형(객체 또는 범주형 dtype)은 레이블 인코딩을 제공하여 이산형으로 처리 할 수 ​​있습니다.
```python
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int
```

데이터가 기존 튜토리얼과 다르기 때문에 추가적인 처리가 필요합니다. 결측값 대치(Imputation)를 이용해 결측값을 없애줍니다.
```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
# Imputation removed column names; put them back
imputed_X.columns = X.columns
```

Scikit-learn에는 feature_selection 모듈에 두 개의 상호의존정보 메트릭이 있습니다. 하나는 실제값 targets(mutual_info_regression)이고 다른 하나는 범주형 targets(mutual_info_classif)입니다. 우리의 목표 가격은 실제값입니다. 아래 코드는 피쳐에 대한 MI 점수를 계산하고 이를 멋진 데이터 프레임으로 래핑합니다.
```python
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(imputed_X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```
```text
curb-weight          1.448852
horsepower           0.858156
wheel-base           0.590054
fuel-system          0.477124
height               0.336631
symboling            0.229504
normalized-losses    0.171853
body-style           0.069742
num-of-doors         0.007341
Name: MI Scores, dtype: float64
```
이제 쉽게 비교해보기 위해 막대 그래프로 만들어봅시다.
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

![사진](/assets/imgs/posts/ml/feature-engineering-005.png)

데이터 시각화는 유틸리티 순위에 대한 훌륭한 후속 조치입니다. 이 두 가지를 자세히 살펴 보겠습니다.

예상 할 수 있듯이 높은 점수를 받는 curb_weight 피쳐는 목표인 price와 강력한 관계를 나타냅니다.
```python
import seaborn as sns

sns.relplot(x="curb-weight", y="price", data=df)
```

![사진](/assets/imgs/posts/ml/feature-engineering-006.png)

fuel_type 피쳐는 MI 점수가 상당히 낮지만 그림에서 볼 수 있듯이 horsepower 피쳐 내에서 다른 추세를 가진 두 price 인구를 명확하게 구분합니다. 이것은 fuel_type이 상호의존작용 효과에 기여하지만 결과적으론 중요하지 않을 수 있음을 나타냅니다. 피쳐를 결정하기 전에 MI 점수에서 중요하지 않은 상호의존작용 효과를 조사하는 것이 좋습니다. 도메인 지식은 여기에서 많은 지침을 제공 할 수 있습니다.
```python
sns.lmplot(x="horsepower", y="price", hue="fuel-type", data=df)
```

![사진](/assets/imgs/posts/ml/feature-engineering-007.png)

데이터 시각화는 피쳐 엔지니어링 도구 상자에 추가된 훌륭한 기능입니다. 상호의존정보와 같은 유틸리티 메트릭과 함께 이와 같은 시각화는 데이터에서 중요한 관계를 발견하는 데 도움이 될 수 있습니다.

---

### Creating Features
잠재력이 있는 피쳐 세트를 식별했으면 이제 개발을 시작할 때입니다. 여기에서는 Pandas에서 전적으로 수행 할 수 있는 몇 가지 일반적인 변환에 대해 알아봅니다.

먼저 데이터 프레임의 열 이름을 파이썬 친화적이게 바꿔주는 함수를 정의해봅니다.
```python
import pandas as pd
import re

def camel_to_snake(text):
    under_scorer_1 = re.compile(r'(.)([A-Z][a-z]+)')
    under_scorer_2 = re.compile('([a-z0-9])([A-Z])')

    subbed = under_scorer_1.sub(r'\1_\2', text)
    return under_scorer_2.sub(r'\1_\2', subbed).lower()


def get_df(path, na_values=None, nrows=None):
    # Get column names of DF
    column_name_list = list(pd.read_csv(path, nrows=1).columns)
    
    for c_idx in range(len(column_name_list)):
        # Convert camel to snake
        c_name = camel_to_snake(column_name_list[c_idx])

        # remove bracket
        pattern = re.compile(r'\([^)]*\)| +|-+')
        c_name = pattern.sub('_', c_name)

        # Remove duplicated _
        pattern = re.compile(r'_+')
        c_name = pattern.sub('_', c_name)

        # Strip _
        c_name = c_name.strip('_')
        column_name_list[c_idx] = c_name
        
    df = pd.read_csv(path, na_values=na_values, nrows=nrows)
    # Change column names
    df.columns = column_name_list
    return df
``` 

여기에서는 [*미국 교통사고*](https://www.kaggle.com/sobhanmoosavi/us-accidents), [*1985년대 자동차*](https://www.kaggle.com/toramky/automobile-dataset), [*콘크리트 제형*](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength) 및 [*고객 평생 가치*](https://www.kaggle.com/pankajjsh06/ibm-watson-marketing-customer-value-data)와 같은 다양한 피쳐 유형이 있는 4개의 데이터 세트를 사용합니다.
```python
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=14,
    titlepad=10,
)

accidents = get_df('input_data/accidents.csv', nrows=20000, na_values='?')
autos = get_df('input_data/autos.csv', na_values='?')
concrete = get_df('input_data/concrete.csv', na_values='?')
customer = get_df('input_data/customer.csv', na_values='?')
```

#### 새로운 피쳐 발견을 위한 팁
- 피쳐를 이해하십시오. 가능한 경우 데이터 세트의 데이터 문서를 참조하세요.
- 문제 영역을 조사하여 영역 지식을 습득하십시오. 집값을 예측하는 데 문제가 있다면 부동산에 대한 조사를 해보십시오. Wikipedia는 좋은 출발점이 될 수 있지만 책과 저널 기사는 종종 최고의 정보를 가지고 있습니다.
- 이전 작업들을 공부하십시오. 과거 Kaggle 대회의 솔루션 글은 훌륭한 리소스입니다.
- 데이터 시각화를 사용하십시오. 시각화는 단순화 할 수 있는 복잡한 관계 또는 피쳐의 분포에서 병리를 나타낼 수 있습니다. 피쳐 엔지니어링 프로세스를 진행하면서 데이터 세트를 시각화해야합니다.

#### 수학적 변환(Mathematical Transforms)
수치적 특징 간의 관계는 종종 수학적 공식을 통해 표현되며, 도메인 연구의 일부로 자주 접하게됩니다. Pandas에서는 일반 숫자처럼 산술 연산을 열에 적용 할 수 있습니다.

Automobile 데이터 세트에는 자동차의 엔진을 설명하는 피쳐가 있습니다. 연구를 통해 잠재적으로 유용한 새 피쳐를 만들기위한 다양한 공식이 제공됩니다. 예를 들어, "stroke ratio"은 엔진의 효율성과 성능의 척도입니다.
```python
autos["stroke_ratio"] = autos.stroke / autos.bore
autos[["stroke", "bore", "stroke_ratio"]].head()
```
```text
   stroke  bore  stroke-ratio
0    2.68  3.47      0.772334
1    2.68  3.47      0.772334
2    3.47  2.68      1.294776
3    3.40  3.19      1.065831
4    3.40  3.19      1.065831
```

조합이 복잡할수록 엔진의 "displacement"에 대한 공식과 같이 엔진의 힘을 측정하는 공식을 모델이 학습하기가 더 어려워집니다.
```python
import numpy as np

# Change autos.num_of_cylinders feature to int64 type
num_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
autos.num_of_cylinders = autos.num_of_cylinders.map(lambda noc: num_list.index(noc))

# Add displacement feature
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)
autos[["stroke", "bore", "num_of_cylinders", "displacement"]].head()
```
```text
   stroke  bore  num_of_cylinders  displacement
0    2.68  3.47                 4    101.377976
1    2.68  3.47                 4    101.377976
2    3.47  2.68                 6    117.446531
3    3.40  3.19                 4    108.695147
4    3.40  3.19                 5    135.868934
```

데이터 시각화는 변환, 종종 거듭 제곱 또는 로그를 통한 피쳐의 "reshaping"을 제안 할 수 있습니다. 예를 들어 미국 교통사고에서 WindSpeed의 분포는 매우 왜곡되어 있습니다. 이 경우 로그는 정규화에 효과적입니다.
```python
import seaborn as sns

# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents['log_wind_speed'] = accidents.wind_speed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.wind_speed, shade=True, ax=axs[0])
sns.kdeplot(accidents.log_wind_speed, shade=True, ax=axs[1])
```

![사진](/assets/imgs/posts/ml/feature-engineering-008.png)

#### Counts
어떤 것의 존재 또는 부재를 설명하는 특징들은 종종 질병의 위험 요소들의 집합으로 나타납니다. count를 생성하여 이러한 피쳐를 집계 할 수 있습니다.

이러한 기능은 바이너리(1은 존재, 0은 부재) 또는 부울(True 또는 False)입니다. 파이썬에서 부울은 마치 정수인 것처럼 더할 수 있습니다.

교통 사고에는 도로의 지형지물이 사고 장소 근처에 있었는지 여부를 나타내는 몇 가지 피쳐가 있습니다. 이것을 이용해 sum 메소드로 사고 장소 근처의 도로 지형지물의 총 개수를 나타낼 수 있습니다.
```python
roadway_features = [
    "amenity", "bump", "crossing", "give_way", "junction", 
    "no_exit", "railway", "roundabout", "station", "stop",
    "traffic_calming", "traffic_signal"
]
accidents["roadway_features"] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ["roadway_features"]].head(10)
```
```text
   amenity   bump  crossing  ...  traffic_calming  traffic_signal  roadway_features
0    False  False     False  ...            False           False                 0
1    False  False     False  ...            False           False                 0
2    False  False     False  ...            False            True                 1
3    False  False     False  ...            False           False                 0
4    False  False     False  ...            False            True                 1
5    False  False     False  ...            False           False                 0
6    False  False     False  ...            False           False                 0
7    False  False     False  ...            False           False                 0
8    False  False     False  ...            False           False                 0
9    False  False     False  ...            False           False                 0

[10 rows x 13 columns]
```

데이터 프레임의 기본 제공 메서드를 사용하여 부울값을 만들 수도 있습니다. 콘크리트 데이터 세트에는 콘크리트 제형에 포함된 성분의 양이 있습니다. 많은 공식에는 하나 이상의 구성 요소가 포함되지않습니다(즉, 구성 요소의 값이 0 임). 아래 코드는 데이터 프레임에 내장된 gt 메소드를 사용하여 공식에 포함된 구성 요소의 수를 계산합니다.
```python
components = [
    "cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", 
    "coarse_aggregate", "fine_aggregate"
]
concrete["components"] = concrete[components].gt(0).sum(axis=1)
concrete[components + ["components"]].head(10)
```
```text
   cement  blast_furnace_slag  ...  fine_aggregate  components
0   540.0                 0.0  ...           676.0           5
1   540.0                 0.0  ...           676.0           5
2   332.5               142.5  ...           594.0           5
3   332.5               142.5  ...           594.0           5
4   198.6               132.4  ...           825.5           5
5   266.0               114.0  ...           670.0           5
6   380.0                95.0  ...           594.0           5
7   380.0                95.0  ...           594.0           5
8   266.0               114.0  ...           670.0           5
9   475.0                 0.0  ...           594.0           4

[10 rows x 8 columns]
```

#### Building-Up and Breaking-Down Features
종종 다음과 같이 더 간단한 조각으로 나눌 수있는 복잡한 문자열이 있습니다.
- ID numbers: '123-45-6789'
- Phone numbers: '(999) 555-0123'
- Street addresses: '8241 Kaggle Ln., Goose City, NV'
- Internet addresses: 'http://www.kaggle.com'
- Product codes: '0 36000 29145 2'
- Dates and times: 'Mon Sep 30 07:06:05 2013'

이와 같은 피쳐에는 종종 사용할 수 있는 일종의 구조가 있습니다. 예를 들어 미국 전화 번호에는 발신자의 위치를 ​​알려주는 지역 번호('(999)'부분)가 있습니다. 항상 그렇듯이 일부 연구는 여기에서 성과를 거둘 수 있습니다.

str 접근자를 사용하면 split과 같은 문자열 메서드를 열에 직접 적용 할 수 있습니다. 고객 평생 가치 데이터 세트에는 보험 회사의 고객을 설명하는 피쳐가 포함되어 있습니다. Policy 피쳐에서, 우리는 Level과 Type을 분리 할 수 ​​있습니다.
```python
customer[["type", "level"]] = (
    customer["policy"].str.split(" ", expand=True)
)
customer[["policy", "type", "level"]].head(10)
```
```text
         policy       type level
0  Corporate L3  Corporate    L3
1   Personal L3   Personal    L3
2   Personal L3   Personal    L3
3  Corporate L2  Corporate    L2
4   Personal L1   Personal    L1
5   Personal L3   Personal    L3
6  Corporate L3  Corporate    L3
7  Corporate L3  Corporate    L3
8  Corporate L3  Corporate    L3
9    Special L2    Special    L2
```

조합에 상호 작용이 있다고 믿을만한 이유가 있다면 피쳐를 결합 할 수도 있습니다.
```python
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()
```
```text
          make   body_style           make_and_style
0  alfa-romero  convertible  alfa-romero_convertible
1  alfa-romero  convertible  alfa-romero_convertible
2  alfa-romero    hatchback    alfa-romero_hatchback
3         audi        sedan               audi_sedan
4         audi        sedan               audi_sedan
```

#### Group Transforms
마지막으로 일부 범주별로 그룹화 된 여러 행에 걸쳐 정보를 집계하는 그룹 변환(Group Transforms)이 있습니다. 그룹 변환을 사용하여 "사람의 거주 국가 평균 수입" 또는 "장르별 평일에 개봉 된 영화의 비율"과 같은 피쳐를 만들 수 있습니다. 범주 상호 작용을 발견 한 경우 해당 범주에 대한 그룹 변환을 조사하는 것이 좋습니다.

집계 함수를 사용하여 그룹 변환은 그룹화를 제공하는 범주형 피쳐와 값을 집계하려는 다른 피쳐의 두 피쳐를 결합합니다. "주별 평균 소득"의 경우 그룹화 기능에 대해 State를 선택하고 집계 함수에 대해 mean을 선택하고 집계 피쳐에 대해 Income을 선택합니다. Pandas에서 이를 계산하기 위해 groupby 및 transform 메서드를 사용합니다.
```python
customer["average_income"] = (
    customer.groupby("state")["income"].transform("mean")
)
customer[["state", "income", "average_income"]].head(10)
```
```text
        state  income  average_income
0  Washington   56274    38122.733083
1     Arizona       0    37405.402231
2      Nevada   48767    38369.605442
3  California       0    37558.946667
4  Washington   43836    38122.733083
5      Oregon   62902    37557.283353
6      Oregon   55350    37557.283353
7     Arizona       0    37405.402231
8      Oregon   14072    37557.283353
9      Oregon   28812    37557.283353
```

mean 함수는 기본 제공 데이터 프레임 메서드이므로 변환할 문자열로 전달할 수 있습니다. 다른 편리한 방법으로는 max, min, median, var, std 및 count가 있습니다. 데이터 세트에서 각 상태가 발생하는 빈도를 계산하는 방법은 다음과 같습니다.
```python
customer["state_freq"] = (
    customer.groupby("state")["state"].transform("count") / customer.state.count()
)
customer[["state", "state_freq"]].head(10)
```
```text
        state  state_freq
0  Washington    0.087366
1     Arizona    0.186446
2      Nevada    0.096562
3  California    0.344865
4  Washington    0.087366
5      Oregon    0.284760
6      Oregon    0.284760
7     Arizona    0.186446
8      Oregon    0.284760
9      Oregon    0.284760
```

이와 같은 변환을 사용하여 범주형 피쳐에 대한 "frequency encoding"을 만들 수 있습니다.

훈련 및 검증 분할을 사용하는 경우 독립성을 유지하려면 훈련 세트만 사용하여 그룹화 된 기능을 생성한 다음 검증 세트에 조인하는 것이 가장 좋습니다. 훈련 세트에서 drop_duplicates를 사용하여 고유한 값 세트를 생성한 후 검증 세트의 merge 메서드를 사용할 수 있습니다.
```python
# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["average_claim"] = df_train.groupby("coverage")["total_claim_amount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["coverage", "average_claim"]].drop_duplicates(),
    on="coverage",
    how="left",
)

df_valid[["coverage", "average_claim"]].head(10)
```
```text
   coverage  average_claim
0   Premium     643.230579
1     Basic     376.021438
2   Premium     643.230579
3  Extended     481.159342
4     Basic     376.021438
5     Basic     376.021438
6     Basic     376.021438
7     Basic     376.021438
8     Basic     376.021438
9   Premium     643.230579
```

#### Tips on Creating Features
피쳐를 만들 때 모델의 강점과 약점을 염두에 두는 것이 좋습니다. 다음은 몇 가지 지침입니다.
- 선형 모델은 합과 차이를 자연스럽게 학습하지만 더 복잡한 것은 학습 할 수 없습니다.
- 비율(Ratios)은 대부분의 모델이 배우기 어려운 것 같습니다. 비율 조합은 종종 몇 가지 쉬운 성능 향상으로 이어집니다.
- 선형 모델과 신경망은 일반적으로 정규화 된 기능에서 더 잘 수행됩니다. 신경망은 특히 0에서 너무 멀지 않은 값으로 개선된 피쳐가 필요합니다. 트리 기반 모델(예: 랜덤 포레스트 및 XGBoost)은 때때로 정규화의 이점을 얻을 수 있지만 일반적으론 거의 그렇지 못 합니다.
- 트리 모델은 거의 모든 피쳐 조합을 근사화하는 방법을 배울 수 있지만 조합이 특히 중요한 경우, 특히 데이터가 제한된 경우 명시적으로 생성하면 이점을 얻을 수 있습니다.
- 개수(Counts)는 트리 모델에 특히 유용합니다. 이러한 모델에는 한 번에 여러 피쳐에 걸쳐 정보를 집계하는 자연스러운 방법이 없기 때문입니다.

---

### Clustering With K-Means
(계속)

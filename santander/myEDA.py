# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,Rmd
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

trn = pd.read_csv('input/train_ver2.csv')
# -

# 데이터의 크기 / 첫 5줄 확인
#
# - 총 13,647,309개의 고객 데이터
# - 고객마다 48개의 변수가 존재

# trn.shape
trn.head()

# 모든 변수에 대하여 미리 보기

for col in trn.columns:
    print('{}\n'.format(trn[col].head()))

# **주의할 점**
#
# - fecha_dato : 날짜 전용 데이터 타입인 datetime이 아닌 object이다.
# - age        : 역시 데이터 타입이 object로 전처리 과정에서 int 타입으로 변환이 필요하다.
# - renta      : 가구 총 수입을 나타냄. 5번째 열에서 NaN이 보임. 전처리 과정에서 이와 같은 결측값에 대한 변환도 필요하다.

# ### TIP 2-2 탐색적 데이터 분석을 통해 우리가 얻고자 하는 것?
#
# > 새로 접하는 데이터에 대한 첫 분석 과정에서 데이터의 모든 것을 이해하려고 할 필요 없음!
# 랜덤하게 일부 행을 눈으로 살펴보며 단계적으로 데이터에 익숙해지려고 해보자
# 데이터에 대한 간단한 시각화도 큰 도움이 된다!
# >
# > - 아, 이번 경진대회 데이터는 이렇게 생겼구나
# > - 이러한 변수들이 존재하는구나
# > - data type을 보니, 이 변수는 전처리를 수행해야겠다
# >
# >정도의 느낌을 가져보자

trn.info()

# ### 데이터 크기
#
# 앞서 .shape 함수를 통해 확인 했었다.
#
# .info() 결과의 2, 3번째 줄 (RangeIndex, Data columns) 값으로 총 고객 데이터 수와 변수의 개수를 알 수 있다

# ### 변수
#
# 이어지는 48줄은 변수명과 해당 변수의 데이터 타입을 보여준다.
#
# 변수명이 스페인어로 구성되어 있는 것을 확인할 수 있다.
#
# 총 24개의 고객 관련 변수, 24개의 금융 제품 변수로 구성되어 있다.

# ### 데이터 타입
#
# 8개의 64bit float, 23개의 64bit int, 17개의 object 타입이 있다. 
#
# 머신러닝 모델 학습을 위해서는 훈련 데이터의 데이터 타입이 모두 int 혹은 float이어야 한다!
#
# 그러므로 전처리 과정에서 object 타입과 같은 변수를 적절하게 변환해서 모델 학습을 진행해야 한다.

# ### 메모리
#
# .info() 를 통해 훈련 데이터의 메모리량도 확인 가능하다.
#
# train_ver2.csv 파일은 2.2GB의 용량이지만, pandas를 통해 읽어오면 총 4.9GB의 메모리를 사용하게 된다.
#
# 불필요한 변수명 제거, 데이터 타입 변경 등을 통해 메모리를 효율적으로 사용할 수 있다.

# ---
#
# ### 수치형 / 범주형 변수
#
# 24개의 고객 관련 변수에 대해서 자세히 살펴보자.
#
#
# #### 수치형
#
# 수치형 (int64, float64) 타입을 갖는 고객 변수를 "num_cols"로 추출하고 .describe()를 통해 간단한 요약 통계를 확인한다.

# +
num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]

trn[num_cols].describe()
# -

# 24개 변수 중, 7개가 수치형 변수임을 확인 할 수 있다.
#
# **.describe()** 함수는 pandas 데이터 프레임의 기초 통계를 보여준다.
#
# min, max 값 사이의 precentile 값을 확인하는 것만으로도 해당 변수에 대한 이해를 높일 수 있다.

# |번호|변수 이름|설명|비고|
# |---|:---|:---|:---|
# |1|nocodpers|고객 고유 식별 번호|(min) 15,889 ~ (max) 1,553,689 |
# |2|ind_nuevo|신규 고객 지표|75%의 값이 0, 나머지(15%)가 값이 1|
# |3|inderel|고객 등급 변수|75%의 값이 0, 나머지(15%)가 값이 99|
# |4|tipodom|주소 유형 변수|모든 값이 1 (모든 값이 상수일 경우 변수로서 식별력을 가질 수 없음. 즉, 학습에 도움이 되지 않는 변수)|
# |5|cod_prov|지방 코드 변수|(min) 1 ~ (max) 52의 값을 가지며, 수치형이지만 범주형 변수로서 의미를 가짐|
# |6|ind_actividad_cliente|활발성 지표|50% 값이 0, 나머지(50%)가 값 1|
# |7|renta|가구 총 수입|(min) 1,202.73 ~ (max) 28,894,400의 값|

# #### 범주형
#
# 이번에는 (object) 데이터 타입을 갖는 범주형 변수를 "cat_cols"로 추출하고 간단한 요약 통계를 확인한다.

# +
cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]

trn[cat_cols].describe()
# -

# 24개 변수 중, 17개가 범주형 변수이다.
#
# 수치형 변수에 대한 결과값과 조금 다른 것을 볼 수 있다. 각 행은 다음과 같은 의미가 있다.

# |번호|결과행|설명|비고|
# |-----|:---|:------|:---|
# |1| count | 해당 변수의 유요한 데이터 개수 의미 | 'ult_fec_cil_1t'의 count가 24,793 밖에 확인되지 않으므로 결측값이 대부분임을 확인|
# |2| unique | 해당 범주형 변수의 고유값 개수를 의미 | 성별 변수인 'sexo'에는 고유값이 2개임을 확인 |
# |3| top | 가장 빈도가 높은 데이터 표시 | 나이 변수 'age'에서 최빈 데이터는 23세임을 확인 |
# |4| freq | 최빈 데이터의 빈도수를 의미 | 총 데이터 수(count) 대비 최빈값(top)이 어느 정도인지에 따라 분포를 가늠할 수 있음.'ind_empleado'의 5개 고유값 중 가장 빈도가 높은 'N' 데이터가 전체의 99.9% 가량을 차지하며 데이터가 매우 편중되어 있음을 확인|

# **주의해야할 점**
#
# 나이를 의미하는 'age', 은행 누적 거래 기간을 나타내는 'antiguedad'가 변수가 수치형이 아닌 범주형으로 분류되어 있다.
#
# 전처리 과정에서 수치형으로 변환해야 한다.
#
# .
# .
#
# 범주형 변수의 고유값을 직접 눈으로 확인해보자.

for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print(f'# col {col}, n_uniq {len(uniq)}, uniq{uniq}')



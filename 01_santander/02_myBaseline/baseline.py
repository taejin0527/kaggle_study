# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

# 데이터 불러오기
trn = pd.read_csv('../input/train_ver2.csv')
tst = pd.read_csv('../input/train_ver2.csv')

# ########## 데이터 전처리 ###########

# 제품 변수 저장
prods = trn.columns[24:].tolist()

# 1. 제품 변수 결측값 0으로 대체
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 제품 하나도 보유하지 않는 고객 제거
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

# 2. 훈련 데이터와 테스트 데이터 통합
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)

features = [] # 4. 학습에 사용할 변수

# 3.1 범주형 변수 전처리
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)
features += categorical_cols

# 3.2 수치형 변수 전처리(결측값 -99)
df['age'].replace(' NA', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)

df['antiguedad'].replace('     NA', -99, inplace=True)
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace('         NA', -99, inplace=True)
df['renta'].fillna(-99, inplace=True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_1mes'].replace('P', 5, inplace=True)
df['indrel_1mes'].fillna(-99, inplace=True)
df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

# 학습에 사용할 수치형 변수를 features에 추구한다.
features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']

# ########## 피처 엔지니어링 ###########

# 두 날짜 변수에서 연도와 월 정보를 추출한다.
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month', 'fecha_alta_year']

df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']

# 그 외 변수의 결측값은 모두 -99로 대체한다.
df.fillna(-99, inplace=True)

# lag-1 데이터
# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장한다
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
df_lag['int_date'] += 1

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. Lag 데이터의 int_date는 1 밀려 있기 때문에, 저번 달의 제품 정보가 삽입된다.
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다
del df, df_lag

# 저번 달의 제품 정보가 존재하지 않을 경우를 대비하여 0으로 대체한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

# lag-1 변수를 추가한다.
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

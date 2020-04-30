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

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# ### 주의할 점
#
# - fecha_dato : 날짜 전용 데이터 타입인 datetime이 아닌 object이다.
# - age        : 역시 데이터 타입이 object로 전처리 과정에서 int 타입으로 변환이 필요하다.
# - renta      : 가구 총 수입을 나타냄. 5번째 열에서 NaN이 보임. 전처리 과정에서 이와 같은 결측값에 대한 변환도 필요하다.

# ## TIP 2-2 탐색적 데이터 분석을 통해 우리가 얻고자 하는 것?
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

# ### 변수

# ### 데이터 타입

# ### 메모리



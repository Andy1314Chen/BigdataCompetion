# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:52:44 2018

@author: MSIK
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime

train = pd.read_table('../data/zhengqi_train.txt')
test = pd.read_table('../data/zhengqi_test.txt')
print('train :', train.shape,
      'test :', test.shape)

model_lgb = lgb.LGBMRegressor(random_state=2018)

feature_col = [v for v in train.columns if v != 'target']
X_train = train[feature_col].values
y_train = train.target.values

X_test = test[feature_col].values

model_lgb.fit(X_train, y_train)

y_test = model_lgb.predict(X_test)


now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
test['target'] = y_test

# test['target'].to_csv('../result/result_%s.txt' % now, index=False)
print(test['target'].describe())
print('Done.')
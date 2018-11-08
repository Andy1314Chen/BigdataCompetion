# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:52:44 2018

@author: MSIK
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
from sklearn.preprocessing import MinMaxScaler

train = pd.read_table('../data/zhengqi_train.txt')
test = pd.read_table('../data/zhengqi_test.txt')
print('train :', train.shape,
      'test :', test.shape)

"""
   标准化
"""
min_max_scaler = MinMaxScaler()
cols_normalize = train.columns.difference(['target'])
norm_train = pd.DataFrame(min_max_scaler.fit_transform(train[cols_normalize]),
                          columns=cols_normalize,
                          index=train.index)
join_df = train[train.columns.difference(cols_normalize)].join(norm_train)
train = join_df.reindex(columns = train.columns)

norm_test = pd.DataFrame(min_max_scaler.transform(test[cols_normalize]),
                         columns=cols_normalize,
                         index=test.index)
test_join = test[test.columns.difference(cols_normalize)].join(norm_test)
test = test_join.reindex(columns = test.columns)
test = test.reset_index(drop=True)

X_train = np.array(train[cols_normalize])
y_train = np.array(train.target)
X_test = np.array(test[cols_normalize])

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

model_dnn = Sequential()
model_dnn.add(Dense(38, input_dim=X_train.shape[1]))
model_dnn.add(Activation('relu'))
model_dnn.add(Dense(1024))

model_dnn.add(Activation('relu'))
model_dnn.add(Dropout(0.2))
model_dnn.add(Dense(32))
 
model_dnn.add(Dense(1))
model_dnn.add(Activation('linear'))
model_dnn.compile(loss="mean_squared_error",optimizer="rmsprop")

hist = model_dnn.fit(X_train, y_train, batch_size=20, epochs=100, shuffle=True,verbose=1,validation_split=0.2)
# prediction = model.predict(X_test,batch_size=1)
# print(prediction)
import matplotlib.pyplot as plt
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label =  'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.legend()

plt.show()


y_test = model_dnn.predict(X_test, verbose=1)

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
test['target'] = y_test

test['target'].to_csv('../result/result_%s.txt' % now, index=False)
print(test['target'].describe())




"""
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
"""
print('Done.')
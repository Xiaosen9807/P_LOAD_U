#%%

import mat4py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

Start=time.time()
mat_data = mat4py.loadmat(r'D:\PythonProjects\MAT\res.mat')
mat_Data = pd.DataFrame(mat_data)
mat_Data.drop('RESITER_cell', axis=1, inplace=True)
mat_Data

RESLOAD_cell = []


def RESLOAD(index):

    for i in mat_Data['RESLOAD_cell'][index]:
        for s in i[20:]:
            for q in s:
                RESLOAD_cell.append(q)
        #RESLOAD_cell_TOTAL.append(RESLOAD_cell)

    return RESLOAD_cell


for index in range(12):
    RESLOAD(index)

print(len(RESLOAD_cell))

RESU_cell = []


def RESU(index):

    for i in mat_Data['RESU_cell'][index]:
        for s in i[20:]:
            for q in s:
                RESU_cell.append(q)
        #RESLOAD_cell_TOTAL.append(RESLOAD_cell)

    return RESU_cell


for index in range(12):
    RESU(index)

print(len(RESU_cell))

P = []
t = 1
for s in range(12):
    i = 0
    while i < 20:
        P.append(t)
        i += 1
    t += 1

print('P:', len(P))

# %%
Table = pd.DataFrame({'LOAD': RESLOAD_cell, 'P': P, 'U': RESU_cell}, )
Table

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)

x = Table.P
y = Table.LOAD
z = Table.U

ax.scatter(x, y, z, s=100)
plt.xlabel('P', fontsize='30', labelpad=30)
plt.ylabel('LOAD', fontsize='30', labelpad=30)
ax.set_zlabel('U', fontsize='30', labelpad=30)
plt.show()

#%%
from math import pi
from math import sin
from math import log

#print(RESLOAD_cell)
LOAD_Norm = []
for i in RESLOAD_cell:
    LOAD_Norm.append(log(i,30)+0.4)
#print(LOAD_Norm)

#print(RESU_cell)
U_Norm = []
for i in RESU_cell:
    U_Norm.append(-log(i,100)-0.9)
#print(U_Norm)

Table_Norm = pd.DataFrame({'LOAD': LOAD_Norm, 'P': P, 'U': U_Norm}, )

print('||||||||||||||||||||||||||||||')

# %%

import tensorflow as tf

Table_train = Table_Norm[~Table['P'].isin([10])]
Table_test = Table_Norm[Table['P'].isin([10])]

x_train = Table_train.loc[:, ('LOAD', 'P')]
y_train = Table_train.loc[:, ('U')]

x_test = Table_test.loc[:, ('LOAD', 'P')]
y_test = Table_test.loc[:, ('U')]

#x_test_original = Table.loc[:, ('LOAD', 'P')]



model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2, ), activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='linear')
])

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=1000)

plt.plot(history.epoch, history.history.get('loss'), label='loss')


print(model.predict(x_test.iloc[1:2]))
print('y:', y_test.iloc[1:2])

#%%
from sklearn.metrics import r2_score

plt.scatter(model.predict((x_test)), y_test)
plt.plot([0,0.7], [0,0.7])
plt.xlabel('Predict', fontsize=30)
plt.ylabel('Actual', fontsize=30)

r2 = r2_score(model.predict((x_test[:5])), y_test[:5])
plt.title(f"{r2=}")
plt.show()

#print('r2_score: %.2f' % r2_score(model.predict((x_test)), y_test))
End=time.time()
print('Running time is:',End-Start,'s')
# %%

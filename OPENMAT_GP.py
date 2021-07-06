#%%

import mat4py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

Start = time.time()

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

P = []
t = 1
for s in range(12):
    i = 0
    while i < 20:
        P.append(t)
        i += 1
    t += 1

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

df_train = Table[Table.U > 0.0005]
x_label = ['P', 'LOAD']
y_label = ['U']
in_scaler = MinMaxScaler()
x_train = in_scaler.fit_transform(
    df_train[x_label])  #Normalization with Min_Max method
y_train = df_train[y_label]


# %%
import george
from george import kernels

y = y_train.values[:, 0]


kernel = 1 * kernels.ExpSquaredKernel(0.5, ndim=2)
gp = george.GP(kernel)
gp.compute(x_train)

pred, pred_var = gp.predict(y, x_train, return_var=True)
plt.plot(y_train, pred, 'd')
plt.title(f'r2={r2_score(pred,y_train)}')

#%%

End = time.time()
print('Running time is:', End - Start, 's')

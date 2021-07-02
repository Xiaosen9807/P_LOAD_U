#%%

import mat4py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy import interpolate

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
print(len(x))

# ax.scatter(x, y, z, s=100)
# plt.xlabel('P', fontsize='30', labelpad=30)
# plt.ylabel('LOAD', fontsize='30', labelpad=30)
# ax.set_zlabel('U', fontsize='30', labelpad=30)
# plt.show()




#%%
#Func_P = interpolate.interp1d(Table.P, Table.U, kind='kind')



#%%

Func_U = interpolate.interp2d(Table.P, Table.LOAD, Table.U, kind='cubic')


New_U=Func_U(Table.P, Table.LOAD)
New_LOAD=Table.LOAD
New_P=Table.P

print(len(New_U))
print(len(New_P))
print(len(New_LOAD))

#%%

import matplotlib.cm as cm

x = New_P
y = New_LOAD
z = New_U

ax.plot_surface(x,
                y,
                z,
                rstride=2,
                cstride=2,
                cmap=cm.coolwarm,
                linewidth=0.5,
                antialiased=True)

plt.show()
# %%

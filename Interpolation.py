#%%

import mat4py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import plotly.express as px

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
#Table.sort_values(by=['LOAD', 'P'], ascending=True, inplace=True)


#%%

x = Table.P
y = Table.LOAD
z = Table.U

print(len(x))

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)

ax.scatter(x, y, z, s=50)
plt.xlabel('P', fontsize='30', labelpad=30)
plt.ylabel('LOAD', fontsize='30', labelpad=30)
ax.set_zlabel('U', fontsize='30', labelpad=30)
plt.show()

#px.scatter_3d(x=Table.P, y=Table.LOAD, z=Table.U)

#%%

import matplotlib.cm as cm

Func_U = interpolate.interp2d(Table.P, Table.LOAD, Table.U, kind='cubic')


#计算480*480的网格上的插值
Table_test = pd.DataFrame({'LOAD': Table.LOAD, 'P': Table.P}, )
print(Table_test.size)


Table_test_2 = pd.DataFrame({'LOAD': Table.LOAD, 'P': Table.P}, )
Table_New=Table_test.append(Table_test_2, ignore_index=True)
Table_New.sort_values(by=[ 'P'], ascending=True, inplace=True)




New_U = Func_U(Table_New.P, Table_New.LOAD)
New_LOAD = Table_New.LOAD
New_P = Table_New.P

xnew = New_P
ynew = New_LOAD
znew = New_U

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)

# ax.scatter(x, y, z, s=100)
# plt.xlabel('P', fontsize='30', labelpad=30)
# plt.ylabel('LOAD', fontsize='30', labelpad=30)
# ax.set_zlabel('U', fontsize='30', labelpad=30)
# plt.show()

ax2 = plt.subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xnew,
                         ynew,
                         znew,
                         rstride=2,
                         cstride=2,
                         cmap=cm.coolwarm,
                         linewidth=0.5,
                         antialiased=True)
ax2.set_xlabel('xnew')
ax2.set_ylabel('ynew')
ax2.set_zlabel('fnew(x, y)')
#plt.colorbar(surf2, shrink=0.5, aspect=5)  #标注

plt.show()

# %%

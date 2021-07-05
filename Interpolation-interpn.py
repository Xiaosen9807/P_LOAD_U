#%%

import mat4py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import plotly.express as px

Start = time.time()

mat_data = mat4py.loadmat(r'D:\PythonProjects\MAT\res.mat')
mat_Data = pd.DataFrame(mat_data)
mat_Data.drop('RESITER_cell', axis=1, inplace=True)
mat_Data



RESLOAD_cell_TOTAL = []


def RESLOAD(index):
    RESLOAD_cell = []
    for i in mat_Data['RESLOAD_cell'][index]:
        for s in i[20:]:
            for q in s:
                RESLOAD_cell.append(q)
        RESLOAD_cell_TOTAL.append(RESLOAD_cell)
        RESLOAD_cell = []

    return RESLOAD_cell_TOTAL


for index in range(12):
    RESLOAD(index)


RESU_cell_TOTAL = []


def RESU(index):
    RESU_cell = []
    for i in mat_Data['RESU_cell'][index]:
        for s in i[20:]:
            for q in s:
                RESU_cell.append(q)
        RESU_cell_TOTAL.append(RESU_cell)
        RESU_cell = []

    return RESU_cell_TOTAL


for index in range(12):
    RESU(index)

Table = pd.DataFrame(
    {
        'LOAD': RESLOAD_cell_TOTAL,
        'P': mat_Data['p'],
        'U': RESU_cell_TOTAL
    }, )

Table

# %%

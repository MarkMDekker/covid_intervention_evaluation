# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- #
# Calculations
# ----------------------------------------------------------------- #


def draw_fractions(self, r, g):
    if self.UniGroups[g][0] in ['d', 'e', 'f', 'h']:
        lst = np.copy(self.MobMat_freq[r])
    else:
        lst = np.copy(self.MobMat_inc[r])
    # if self.Intervention == 'Scen1':
    #     if self.UniGroups[g][0] in ['b', 'c', 'd']:
    #         lst = np.zeros(len(self.MobMat_inc[r]))
    lst = lst / self.HomePop[r]
    # if self.Intervention == 'Scen2':
    #     lst = lst*0.25
    lst[r] = np.array([1.5, 1.5, 1.5, 1, 1, 1, 1.5, 1, 1.5, 1.5, 1.5])[g]
    lst[lst == 0] = 1e-9
    lst = 2.5*lst / np.sum(lst)
    fs = np.array(np.random.dirichlet(lst))
    return fs


def translate_polymod(g):
    if g == 0:
        return [0], [1]
    if g == 1:
        return [1, 2], [1, 0.5]
    if g == 2:
        return [2, 3], [0.5, 1]
    if g == 3 or g == 4:
        return [4, 5], [1, 0.5]
    if g == 5 or g == 6:
        return [5, 6, 7, 8, 9, 10], [0.5, 1, 1, 1, 1, 1]
    if g == 7 or g == 8:
        return [11, 12, 13], [1, 1, 0.5]
    if g == 9:
        return [13, 14, 15], [0.5, 1, 1]
    if g == 10:
        return [15], [1]


def new_mixmat(matraw):
    mat = np.zeros(shape=(11, 11))
    for i in range(11):
        row, ps = translate_polymod(i)
        ps = np.array(ps)
        matrow = (matraw[row].T*ps)/len(ps)
        for j in range(11):
            col, ps = translate_polymod(j)
            ps = np.array(ps)
            mat[i, j] = np.sum(matrow[col].T*ps)
            # mat[i, j] = matraw[translate_polymod(i)
            #                    ].sum(axis=0)[translate_polymod(j)].sum()
    return mat

# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

from tqdm import tqdm
import pandas as pd
import numpy as np

path = ('/Users/mmdekker/OneDrive - Universiteit Utrecht/Data/Mezuro/'
        'nederland/gemeente niveau/per dag/'
        'Bestemmingen bewoners per dag - van gemeente naar gemeente - '
        '01-03-2019 tm 14-03-2019 - ilionx.csv')

# ----------------------------------------------------------------- #
# Read data
# ----------------------------------------------------------------- #

DF = pd.read_csv(path, delimiter=';')
MunIDs = np.unique(DF.bezoek_gemeente_id)
MunIDs2 = np.unique(DF.woon_gemeente_id)
MunIDs = np.unique(list(MunIDs)+list(MunIDs2))
N = len(MunIDs)
B0 = np.array(DF.incidentele_bezoeker)
B1 = np.array(DF.regelmatige_bezoeker)
B2 = np.array(DF.frequente_bezoeker)
bezoek = np.array(DF.bezoek_gemeente_id)
woon = np.array(DF.woon_gemeente_id)

# ----------------------------------------------------------------- #
# Create matrix
# ----------------------------------------------------------------- #

A = np.zeros(shape=(3, N, N))
for i in tqdm(range(N)):
    for j in range(N):
        if i != j:
            w = np.where((woon == MunIDs[i]) & (bezoek == MunIDs[j]))[0]
            if len(w) > 0:
                b0 = np.nanmean(B0[w])
                b1 = np.nanmean(B1[w])
                b2 = np.nanmean(B2[w])
                A[0, i, j] = b0
                A[1, i, j] = b1
                A[2, i, j] = b2
A = A.astype(int)

# ----------------------------------------------------------------- #
# Normalize matrix
# ----------------------------------------------------------------- #

An = np.zeros(shape=A.shape)
for i in range(3):
    An[i] = (A[i].T/np.nansum(A[i], axis=1)).T

# ----------------------------------------------------------------- #
# Save matrix
# ----------------------------------------------------------------- #

np.save('/Users/mmdekker/OneDrive - Universiteit Utrecht/Data/Mezuro/NormMat', An)
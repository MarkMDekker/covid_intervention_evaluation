# =========================================================================== #
# Additional functions for the Transmission class
# =========================================================================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import scipy.sparse

# =========================================================================== #
# GENERAL FUNCTIONS
# =========================================================================== #

def windowmean(Data, size):
    ''' Function to compute a running window along a time series '''

    if size == 1:
        return Data
    elif size == 0:
        print('Size 0 not possible!')
        sys.exit()
    else:
        Result = np.zeros(len(Data)) + np.nan
        for i in range(np.int(size/2.), np.int(len(Data) - size/2.)):
            Result[i] = np.nanmean(Data[i-np.int(size/2.):i+np.int(size/2.)])
        return np.array(Result)

# =========================================================================== #
# PREPROCESSING FUNCTIONS
# =========================================================================== #

def rivm_to_model(mat):
    newvec = np.zeros(shape=(11, 11))
    for k in range(11):
        for l in range(11):
            newvec[k, l] = np.mean(mat[iconv(k)][:, iconv(l)])
    return newvec

def iconv(i):
    if i == 0:
        j = [0]
    if i == 1:
        j = [1]
    if i == 2:
        j = [2]
    if i == 3 or i == 4:
        j = [3]
    if i == 5 or i == 6:
        j = [4, 5, 6]
    if i == 7 or i == 8:
        j = [6, 7]
    if i == 9:
        j = [7, 8]
    if i == 10:
        j = [9]
    return j

def recalc_positions(self):
    PosMat = np.zeros(shape=(7, 24, len(self.UniLocs), self.N))
    for m in tqdm(range(len(self.UniLocs))):
        PosMat[:, :, m, :][self.Positions == m] = 1
    return PosMat, PosMat

# =========================================================================== #
# GENERAL FORCE OF INFECTION CALCULATORS
# =========================================================================== #

def determine_exposed(self, Stat, day, hour, phase):
    s_t = self.sleepfactor[hour]
    Svec = np.zeros(self.N)
    Svec[Stat == 0] = 1
    Ivec = np.zeros(self.N)
    Ivec[Stat == 2] = 1
    Lvec = np.zeros(self.N)
    Ipos = scipy.sparse.csr_matrix(self.PosMat[day, hour])
    Is = scipy.sparse.csr_matrix((self.GroupsMat_sp.multiply(Ivec)).toarray()).T

    infs = (Ipos.dot(Is)).toarray()
    tots = (Ipos.dot(self.GroupsMat_sp.T)).toarray()
    sucs = (Ipos.multiply(Svec)).toarray()
    
    #infs = Ipos.dot((self.GroupsMat*Ivec).T)
    #tots = Ipos.dot(self.GroupsMat.T)
    #sucs = Ipos*Svec
    #sucs = np.zeros(shape=(380, 11))
    #sucs[np.random.choice(np.arange(380), size=300000), np.random.choice(np.arange(11), size=300000)] = 1
    fracs = infs/(1e-9+tots)
    wh = np.unique(np.where(infs == 1)[0])
    for m in wh:
        fracs2 = fracs[m]
        if np.sum(fracs2) > 0:
            beta = self.Betaf
            if self.Intervention == 'local' or self.Intervention == 'brablim' or self.Intervention == 'G4':
                beta = self.Betas[m]
            people = np.where(sucs[m] == 1)[0]
            for p in people:
                Lvec[p] = force_of_infection2(self, p, fracs2, m, hour, phase)*beta#*500
    Lvec = Lvec*s_t
    En = np.where(np.random.random(self.N) < Lvec)[0]#S[np.random.random(len(S)) < lds]
    del Svec, Ivec, Lvec, Ipos, Is, infs, tots, sucs, fracs
    return En

def force_of_infection2(self, p, fracs, m, hour, phase):
    group = self.GroupsI[p]
    mixvec = get_mixmat(self, hour, m, group, p, self.Homes, phase)[group]
    return np.sum(mixvec*fracs*self.HG[group])

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


def get_mixmat(self, h, r, g, p, Homes, phase):
    factor = 1
    
    if self.Intervention == 'local':
        if phase[r] == 2 or phase[r] == 3:
            factor = self.MixChange_phase2
        if phase[r] == 4:
            factor = self.MixChange_phase4
    if self.Intervention == 'brablim':
        if self.Brablim[r] == 1:
            if phase == 2 or phase == 3:
                factor = self.MixChange_phase2
            if phase == 4:
                factor = self.MixChange_phase4
    if self.Intervention == 'G4':
        if self.G4[r] == 1:
            if phase == 2 or phase == 3:
                factor = self.MixChange_phase2
            if phase == 4:
                factor = self.MixChange_phase4

    if p in self.Homeschoolers:
        if self.Intervention == 'schoolextreme' or self.Intervention == 'schoolparents':
            mixmat = np.zeros(shape=(11, 11))
        else:
            if h < 9 or h > 17:
                mixmat = nighttimemixer(self, h, r, g, p, Homes, phase)
            else:
                mixmat = self.Mix_h
                factor = self.MixChange_phase2
    elif p in self.Homeworkers:
        if h < 9 or h > 17:
            mixmat = nighttimemixer(self, h, r, g, p, Homes, phase)
        else:
            mixmat = self.Mix_h
            factor = self.MixChange_phase2
    else:
        if h < 9 or h > 17:
            mixmat = nighttimemixer(self, h, r, g, p, Homes, phase)
        else:
            mixmat = daytimemixer(self, h, r, g, p, Homes, phase)
    return mixmat*factor


def daytimemixer(self, h, r, g, p, Homes, phase):
    if Homes[p] == self.UniLocs[r]: # Day time, home region
        if g+1 in [1, 7, 9, 10, 11]:
            mixmat = self.Mix_h
        if g+1 in [2, 3]:
            mixmat = self.Mix_s
        if g+1 == 4:
            mixmat = self.Mix_ws
        if g+1 in [5, 6, 8]:
            mixmat = self.Mix_w
    elif Homes[p] != self.UniLocs[r]: # Day time, other region
        if g+1 in [1, 7, 9, 10, 11]:
            mixmat = self.Mix_o
        if g+1 in [2, 3]:
            mixmat = self.Mix_s
        if g+1 == 4:
            mixmat = self.Mix_ws
        if g+1 in [5, 6, 8]:
            mixmat = self.Mix_w
    return mixmat


def nighttimemixer(self, h, r, g, p, Homes, phase):
    if Homes[p] == self.UniLocs[r]:
        mixmat = self.Mix_h
    else:
        mixmat = self.Mix_o
    return mixmat

# =========================================================================== #
# PHASE MANAGEMENT - GENERAL
# =========================================================================== #

def close_all_schools(self):
    #self.Mix_s = self.Mix_h0*self.MixChange_phase2
    
    # Children
    wh = self.Schoolers
    for d in range(7):
        for h in range(8, 18):
            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    self.Positions[:, 8:18, wh] = self.HomesI[wh]
    self.PosMat[:, 8:18, self.HomesI[wh], wh] = 1
    #     for h in range(8, 18):
    #         self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    # self.Positions[:, 8:18, wh] = self.HomesI[wh]
    # self.PosMat[:, 8:18, self.HomesI[wh], wh] = 1
    
    # Parents
    if self.Intervention != 'schoolparents':
        wh2 = self.SchoolParents
        for d in range(7):
            for h in range(8, 18):
                self.PosMat[d, h, self.Positions[d, h, wh2], wh2] = 0
        self.Positions[:, 8:18, wh2] = self.HomesI[wh2]
        self.PosMat[:, 8:18, self.HomesI[wh2], wh2] = 1
    else:
        wh2 = []

    # Set to homeschoolers
    self.Homeschoolers = np.unique(list(wh)+list(wh2))

def close_local_schools(self, mun):
    #self.Mix_s = self.Mix_h0*self.MixChange_phase2
    
    # Children
    wh = self.SchoolersinMun[mun]
    for d in range(7):
        for h in range(8, 18):
            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    self.Positions[:, 8:18, wh] = self.HomesI[wh]
    self.PosMat[:, 8:18, self.HomesI[wh], wh] = 1
    #     for h in range(8, 18):
    #         self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    # self.Positions[:, 8:18, wh] = self.HomesI[wh]
    # self.PosMat[:, 8:18, self.HomesI[wh], wh] = 1
    
    # Parents
    wh2 = self.SchoolparentsinMun[mun]
    for d in range(7):
        for h in range(8, 18):
            self.PosMat[d, h, self.Positions[d, h, wh2], wh2] = 0
    self.Positions[:, 8:18, wh2] = self.HomesI[wh2]
    self.PosMat[:, 8:18, self.HomesI[wh2], wh2] = 1

    # Set to homeschoolers
    self.Homeschoolers_m[mun] = np.unique(list(wh)+list(wh2))

def change_phases(self, phase, IR, t, t0):
    if phase == 1 and IR >= self.Threshold*self.N+self.Inisum:  # 0.0095*self.N:
        print('Entering phase 2')
        enter_phase2(self)
        phase = 2
        t0 = np.float(t)
        self.Timestep12March = t
        if self.Intervention == 'schoolextreme' or self.Intervention == 'schoolparents' or self.Intervention == 'schoolisolation':
            close_all_schools(self)
        if self.Intervention == 'schoolearly':
            close_all_schools(self)
    elif phase == 2 and t == int(t0+4*24):
        if self.Intervention != 'school':
            close_all_schools(self)
    elif phase == 2 and t == int(t0+11*24):
        print('Entering phase 3')
        enter_phase3(self)
        phase = 3
        if self.Intervention != 'school':
            close_all_schools(self)
    elif phase == 3 and t == int(t0+60*24):
        print('Entering phase 4')
        enter_phase4(self)
        phase = 4
        if self.Intervention == 'schoolextreme' or self.Intervention == 'schoolparents' or self.Intervention == 'schoolisolation':
            close_all_schools(self)
        if self.Intervention == 'schoolearly':
            close_all_schools(self)
    return phase, t0

def enter_phase2(self):

    ''' Reduced mobility (1-14 Mar) '''
    working = self.Workers
    wh = np.random.choice(working, size=int((-self.MobChange_phase2/100)*self.N), replace=False)
    for d in range(7):
        for h in range(24):
            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    self.Positions[:, :, wh] = self.HomesI[wh]
    self.Homeworkers = wh
    self.PosMat[:, :, self.HomesI[wh], wh] = 1

    ''' Reduced mixing (April 2020) '''
    self.Mix_h = self.Mix_h0*self.MixChange_phase2
    self.Mix_s = self.Mix_s0*self.MixChange_phase2
    self.Mix_w = self.Mix_w0*self.MixChange_phase2
    self.Mix_o = self.Mix_o0*self.MixChange_phase2
    self.Mix_ws = self.Mix_ws0*self.MixChange_phase2

    ''' Change transmissivity '''
    self.Betaf = self.Beta_f2

def enter_phase3(self):

    ''' Reduced mobility (14 Mar - 31 May) '''
    working = self.Workers
    wh = np.random.choice(working, size=int((-self.MobChange_phase3/100)*self.N), replace=False)
    self.Positions = np.copy(self.Positions0)
    for d in range(7):
        for h in range(24):
            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    self.Positions[:, :, wh] = self.HomesI[wh]
    self.Homeworkers = wh
    self.PosMat[:, :, self.HomesI[wh], wh] = 1
    
    ''' if border intervention: borders should remain closed! '''
    if self.Intervention == 'border':
        closed = np.where(self.ClosedBorders == 1)[0]
        for m in closed:
            # Living now stay there
            wh = np.where(self.HomesI == m)[0]
            for d in range(7):
                for h in range(24):
                    self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
            self.Positions[:, :, wh] = self.HomesI[wh]
            self.PosMat[:, :, self.HomesI[wh], wh] = 1
            
            # Going tos now stay at their respective homes
            wh1 = np.where(self.HomesI != m)[0]
            for d in range(7):
                for h in range(24):
                    wh2 = np.where(self.Positions[d, h] == m)[0]
                    wh = np.intersect1d(wh1, wh2)
                    self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
                    self.Positions[d, h, wh] = self.HomesI[wh]
                    self.PosMat[d, h, self.HomesI[wh], wh] = 1

    ''' Change transmissivity '''
    self.Betaf = self.Beta_f3

def enter_phase4(self):
    
    ''' Reduced mobility (31 May - 31 Jul) '''
    working = self.Workers
    wh = np.random.choice(working, size=int((-self.MobChange_phase4/100)*self.N), replace=False)
    self.Positions = np.copy(self.Positions0)
    for d in range(7):
        for h in range(24):
            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
    self.Positions[:, :, wh] = self.HomesI[wh]
    self.Homeworkers = wh
    self.PosMat[:, :, self.HomesI[wh], wh] = 1
    
    ''' if border intervention: borders should open! -> taken care of via Positions0 '''
    
    ''' Reduced mixing (June 2020) '''
    self.Mix_h = self.Mix_h0*self.MixChange_phase4
    self.Mix_s = self.Mix_s0*self.MixChange_phase4
    self.Mix_w = self.Mix_w0*self.MixChange_phase4
    self.Mix_o = self.Mix_o0*self.MixChange_phase4
    self.Mix_ws = self.Mix_ws0*self.MixChange_phase4

    ''' Change transmissivity '''
    self.Betaf = self.Beta_f4

# =========================================================================== #
# PHASE MANAGEMENT - BRABLIM
# =========================================================================== #

def change_phases_brablim(self, phase, IR, t, t0):
    if phase == 1 and IR >= self.Threshold*self.N+self.Inisum:  # 0.0095*self.N:
        print('Entering phase 2')
        enter_phase2_brablim(self)
        phase = 2
        t0 = np.float(t)
        self.Timestep12March = t
    elif phase == 2 and t == int(t0+4*24):
        brablim = np.where(self.Brablim == 1)[0]
        for m in brablim:
            close_local_schools(self, m)
    elif phase == 2 and t == int(t0+11*24):
        print('Entering phase 3')
        enter_phase3_brablim(self)
        phase = 3
        brablim = np.where(self.Brablim == 1)[0]
        for m in brablim:
            close_local_schools(self, m)
    elif phase == 3 and t == int(t0+60*24):
        print('Entering phase 4')
        enter_phase4_brablim(self)
        phase = 4
    return phase, t0

def enter_phase2_brablim(self):
    brablim = np.where(self.Brablim == 1)[0]
    for m in brablim:
        living = np.where(self.HomesI == m)[0]
        working = self.WorkersinMun[m]
        wh = np.random.choice(working, size=int((-self.MobChange_phase2/100
                                                 )*len(living)), 
                              replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.Homeworkers_m[m] = wh
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Betas[m] = self.Beta_f2

def enter_phase3_brablim(self):
    brablim = np.where(self.Brablim == 1)[0]
    for m in brablim:
        living = np.where(self.HomesI == m)[0]
        self.Positions[:, :, living] = self.Positions0[:, :, living]
        working = self.WorkersinMun[m]
        try:
            wh = np.random.choice(working, size=int((-(self.MobChange_phase3)/100
                                                     )*len(living)), 
                                  replace=False)
        except:
            wh = np.random.choice(working, size=len(working), 
                                  replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Homeworkers_m[m] = wh

        # schooling = self.SchoolersinMun[m]
        # for d in range(7):
        #     for h in range(8, 18):
        #         self.PosMat[d, h, self.Positions[d, h, schooling], schooling] = 0
        # self.Positions[:, 8:18, schooling] = self.HomesI[schooling]
        # self.PosMat[:, 8:18, self.HomesI[schooling], schooling] = 1
        # self.Homeschoolers_m[m] = schooling
        self.Betas[m] = self.Beta_f3

def enter_phase4_brablim(self):
    brablim = np.where(self.Brablim == 1)[0]
    for m in brablim:
        living = np.where(self.HomesI == m)[0]
        self.Positions[:, :, living] = self.Positions0[:, :, living]
        working = self.WorkersinMun[m]
        wh = np.random.choice(working, size=int((-self.MobChange_phase4/100
                                                 )*len(living)), 
                              replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.Homeworkers_m[m] = wh
        self.Homeschoolers_m[m] = []
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Betas[m] = self.Beta_f4

# =========================================================================== #
# PHASE MANAGEMENT - G4
# =========================================================================== #

def change_phases_G4(self, phase, IR, t, t0):
    if phase == 1 and IR >= self.Threshold*self.N+self.Inisum:  # 0.0095*self.N:
        print('Entering phase 2')
        enter_phase2_G4(self)
        phase = 2
        t0 = np.float(t)
        self.Timestep12March = t
    elif phase == 2 and t == int(t0+4*24):
        G4 = np.where(self.G4 == 1)[0]
        for m in G4:
            close_local_schools(self, m)
    elif phase == 2 and t == int(t0+11*24):
        print('Entering phase 3')
        enter_phase3_G4(self)
        phase = 3
        G4 = np.where(self.G4 == 1)[0]
        for m in G4:
            close_local_schools(self, m)
    elif phase == 3 and t == int(t0+60*24):
        print('Entering phase 4')
        enter_phase4_G4(self)
        phase = 4
    return phase, t0

def enter_phase2_G4(self):
    G4 = np.where(self.G4 == 1)[0]
    for m in G4:
        living = np.where(self.HomesI == m)[0]
        working = self.WorkersinMun[m]
        wh = np.random.choice(working, size=int((-self.MobChange_phase2/100
                                                 )*len(living)), 
                              replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.Homeworkers_m[m] = wh
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Betas[m] = self.Beta_f2

def enter_phase3_G4(self):
    G4 = np.where(self.G4 == 1)[0]
    for m in G4:
        living = np.where(self.HomesI == m)[0]
        self.Positions[:, :, living] = self.Positions0[:, :, living]
        working = self.WorkersinMun[m]
        wh = np.random.choice(working, size=int((-self.MobChange_phase3/100
                                                 )*len(living)), 
                              replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Homeworkers_m[m] = wh

        # schooling = self.SchoolersinMun[m]
        # for d in range(7):
        #     for h in range(8, 18):
        #         self.PosMat[d, h, self.Positions[d, h, schooling], schooling] = 0
        # self.Positions[:, 8:18, schooling] = self.HomesI[schooling]
        # self.PosMat[:, 8:18, self.HomesI[schooling], schooling] = 1
        # self.Homeschoolers_m[m] = schooling
        self.Betas[m] = self.Beta_f3

def enter_phase4_G4(self):
    G4 = np.where(self.G4 == 1)[0]
    for m in G4:
        living = np.where(self.HomesI == m)[0]
        self.Positions[:, :, living] = self.Positions0[:, :, living]
        working = self.WorkersinMun[m]
        wh = np.random.choice(working, size=int((-self.MobChange_phase4/100
                                                 )*len(living)), 
                              replace=False)
        for d in range(7):
            for h in range(24):
                self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        self.Positions[:, :, wh] = self.HomesI[wh]
        self.Homeworkers_m[m] = wh
        self.Homeschoolers_m[m] = []
        self.PosMat[:, :, self.HomesI[wh], wh] = 1
        self.Betas[m] = self.Beta_f4

# =========================================================================== #
# PHASE MANAGEMENT - LOCAL
# =========================================================================== #

def national_datesetting(self, t, IR):
    if IR >= self.Threshold*self.N+self.Inisum:  # 0.0095*self.N:
        self.Predated = 1
        self.Timestep12March = t

def interv_local(self, Stat, t, phase, t0):
    IRw = np.where(Stat == 2)[0]
    Muns = self.HomesI[IRw]
    unimuns = np.unique(Muns)
    #oldphase = np.copy(phase)
    self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
    self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]
    CURRENTPHASE = 2
    if t >= self.Timestep12March+11*24:
        CURRENTPHASE = 3
    
    for m in unimuns:#range(len(self.UniLocs)):
        
        ''' If not phased yet, go to either phase 2 or 3 '''
        if self.Phased[m] == 0:
            tot = self.HomePops[m]
            inf = len(np.where(Muns == m)[0])
            frac = inf/tot
            if frac >= self.Thresholdlocal:
                if CURRENTPHASE == 2:
                    ''' Reduced mobility '''
                    living = self.HomeMat[m]
                    self.Positions[:, :, living] = self.Positions0[:, :, living]
                    working = self.WorkersinMun[m]
                    try:
                        wh = np.random.choice(working, size=int((-(self.MobChange_phase2)/100
                                                                 )*len(living)), 
                                              replace=False)
                    except:
                        wh = np.random.choice(working, size=len(working), 
                                              replace=False)
                    for d in range(7):
                        for h in range(24):
                            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
                    self.Positions[:, :, wh] = self.HomesI[wh]
                    self.PosMat[:, :, self.HomesI[wh], wh] = 1
                    self.Homeworkers_m[m] = wh
        
                    ''' Schools close -> Only in phase 3 '''
                    #close_local_schools(self, m)
                    
                    self.Betas[m] = self.Beta_f2
                    phase[m] = 2 # Voor de mixing reductie!
                    self.Phased[m] = 1
                    self.TimePhased[m] = t
                    self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
                    self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]

                if CURRENTPHASE == 3:
                    ''' Reduced mobility '''
                    living = self.HomeMat[m]
                    self.Positions[:, :, living] = self.Positions0[:, :, living]
                    working = self.WorkersinMun[m]
                    try:
                        wh = np.random.choice(working, size=int((-(self.MobChange_phase3)/100
                                                                 )*len(living)), 
                                              replace=False)
                    except:
                        wh = np.random.choice(working, size=len(working), 
                                              replace=False)
                    for d in range(7):
                        for h in range(24):
                            self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
                    self.Positions[:, :, wh] = self.HomesI[wh]
                    self.PosMat[:, :, self.HomesI[wh], wh] = 1
                    self.Homeworkers_m[m] = wh
        
                    ''' Schools close '''
                    close_local_schools(self, m)
                    
                    self.Betas[m] = self.Beta_f3
                    phase[m] = 3 # Voor de mixing reductie!
                    self.Phased[m] = 1
                    self.TimePhased[m] = t
                    self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
                    self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]
        if self.Phased[m] == 1 and t == int(self.Timestep12March+11*24):
            ''' If the date arrives, we have to check that phase-2 countries also become phase-3 '''
            ''' Reduced mobility '''
            living = self.HomeMat[m]
            self.Positions[:, :, living] = self.Positions0[:, :, living]
            working = self.WorkersinMun[m]
            try:
                wh = np.random.choice(working, size=int((-(self.MobChange_phase3)/100
                                                         )*len(living)), 
                                      replace=False)
            except:
                wh = np.random.choice(working, size=len(working), 
                                      replace=False)
            for d in range(7):
                for h in range(24):
                    self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
            self.Positions[:, :, wh] = self.HomesI[wh]
            self.PosMat[:, :, self.HomesI[wh], wh] = 1
            self.Homeworkers_m[m] = wh

            ''' Schools close '''
            close_local_schools(self, m)
            
            self.Betas[m] = self.Beta_f3
            phase[m] = 3 # Voor de mixing reductie!
            self.Phased[m] = 1
            self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
            self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]

    
        ''' No recovery '''
        # if oldphase[m] == 3:
        #     if t == int(t0[m]+60*24):
        #         living = self.HomeMat[m]
        #         self.Positions[:, :, living] = self.Positions0[:, :, living]
        #         working = self.WorkersinMun[m]
        #         wh = np.random.choice(working, size=int((-self.MobChange_phase4/100
        #                                                  )*len(living)), 
        #                               replace=False)
        #         for d in range(7):
        #             for h in range(24):
        #                 self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
        #         self.Positions[:, :, wh] = self.HomesI[wh]
        #         self.Homeworkers_m[m] = wh
        #         self.Homeschoolers_m[m] = []
        #         self.PosMat[:, :, self.HomesI[wh], wh] = 1
        #         self.Betas[m] = self.Beta_f4
        #         phase[m] = 4
    return phase, t0

# =========================================================================== #
# PHASE MANAGEMENT - BORDER
# =========================================================================== #

def interv_border(self, Stat, t):
    IRw = np.where(Stat == 2)[0]
    Muns = self.HomesI[IRw]
    unimuns = np.unique(Muns)
    nonclosed = np.where(self.ClosedBorders == 0)[0]
    for m in nonclosed:
        tot = self.HomePops[m]
        inf = len(np.where(Muns == m)[0])
        frac = inf/tot
        if frac >= self.Thresholdlocal:
            # Living
            wh = np.where(self.HomesI == m)[0]
            for d in range(7):
                for h in range(24):
                    self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
            self.Positions[:, :, wh] = self.HomesI[wh]
            self.PosMat[:, :, self.HomesI[wh], wh] = 1
            
            # Going to
            wh1 = np.where(self.HomesI != m)[0]
            for d in range(7):
                for h in range(24):
                    wh2 = np.where(self.Positions[d, h] == m)[0]
                    wh = np.intersect1d(wh1, wh2)
                    self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
                    self.Positions[d, h, wh] = self.HomesI[wh]
                    self.PosMat[d, h, self.HomesI[wh], wh] = 1
            self.ClosedBorders[m] = 1
            self.TimeClosures[m] = t
            #print('Border closed of', self.UniLocs[m])


# # =========================================================================== #
# # SCHOOL CLOSURE - SCHOOLEXTREME
# # =========================================================================== #

# def close_all_schools_extreme(self):
#     #self.Mix_s = self.Mix_h0*self.MixChange_phase2
#     wh = self.Schoolers
#     for d in range(7):
#         for h in range(8, 18):
#             self.PosMat[d, h, self.Positions[d, h, wh], wh] = 0
#     self.Positions[:, 8:18, wh] = self.HomesI[wh]
#     self.PosMat[:, 8:18, self.HomesI[wh], wh] = 1
#     self.Homeschoolers = wh
#     self.Mix_h[[1, 2, 3], :] = 0
#     self.Mix_h[:, [1, 2, 3]] = 0
#     self.Mix_s[[1, 2, 3], :] = 0
#     self.Mix_s[:, [1, 2, 3]] = 0
#     self.Mix_w[[1, 2, 3], :] = 0
#     self.Mix_w[:, [1, 2, 3]] = 0
#     self.Mix_o[[1, 2, 3], :] = 0
#     self.Mix_o[:, [1, 2, 3]] = 0
#     self.Mix_ws[[1, 2, 3], :] = 0
#     self.Mix_ws[:, [1, 2, 3]] = 0

# =========================================================================== #
# OLD
# =========================================================================== #

# def check_phases(self, PhaseVec, Totcum):
#     ''' check and change the phases of the municipalities '''
    
#     for i in range(len(self.UniLocs)):
#         phase = PhaseVec[i]
#         totcum = Totcum[t-1][i]
#         totpop = self.HomePops[i]
#         if phase == 1 and totcum >= 0.019*totpop:  # 0.0095*self.N:
#             enter_phase2(self)
#             phase = 2
#             t0 = np.float(t)
#             self.WH = recalc_positions(self)
#         elif phase == 2 and t == int(t0+11*24):
#             enter_phase3(self)
#             phase = 3
#             self.WH = recalc_positions(self)
#         elif phase == 3 and t == int(t0+60*24):
#             enter_phase3(self)
#             phase = 4
#             self.WH = recalc_positions(self)


# def recalc_positions(self):
#     ''' To enhance computational speed '''
    
#     posA = []
#     for day in range(7):
#         for hour in range(24):
#             pos = self.Positions[day][hour]
#             poss = []
#             for i in range(len(self.UniLocs)):
#                 poss2 = []
#                 for j in range(len(self.UniGroups)):
#                     poss2.append(np.where((pos == i) & (self.GroupsI == j))[0])
#                 poss.append(poss2)
#             posA.append(poss)
#     return posA


# def recalc_positions2(self):
#     ''' To enhance computational speed '''
    
#     POS = np.zeros(shape=(7, 24, len(self.UniLocs), len(self.UniGroups), self.N))
#     TOTS = np.zeros(shape=(7, 24, len(self.UniLocs), len(self.UniGroups)))
#     for day in tqdm(range(7)):
#         for hour in range(24):
#             pos = self.Positions[day][hour]
#             for p in range(self.N):
#                 m = int(pos[p])
#                 g = int(self.GroupsI[p])
#                 POS[day, hour, m, g, p] = 1
#     #TOTS = POS.sum(axis=4)
#     return POS
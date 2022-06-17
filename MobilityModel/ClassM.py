# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import sys
import scipy.io
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import warnings
from Functions import draw_fractions, translate_polymod, new_mixmat
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class ModelM(object):
    def __init__(self, params_input):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config.ini')

        ''' Variables '''
        self.SaveName = params_input['savename']
        self.Path_RawDataDay = config['PATHS']['RAWDATA_DAY']
        self.Path_RawDataGem = config['PATHS']['RAWDATA_GEM']
        self.Path_DemoMat = config['PATHS']['RAWDATA_DEMO']
        self.Path_Data = config['PATHS']['DATA']
        self.Path_Datasave = config['PATHS']['FIG']
        self.Path_RawDataMix = config['PATHS']['RAWDATA_MIX']
        self.Path_RawDataMix2 = config['PATHS']['RAWDATA_MIX2']
        self.Div = np.float(params_input['division'])
        self.Ndays = int(config['PARAMS']['NDAYS'])

        print('# ---------------------------------------------------------- #')
        print('# Starting Mobility model')
        print('# ------ Resolution:   '+self.SaveName)
        print('# ------ Amount days:  '+str(self.Ndays))
        print('# ---------------------------------------------------------- #')

    def read_data(self):
        ''' Read raw data '''

        ''' Get main municipality list (for version control): 2018 '''
        self.DF_Gem = pd.read_csv(self.Path_RawDataGem, delimiter=';', encoding='latin-1')
        self.UniLocs = np.unique(self.DF_Gem.Gemeentenaam)
        self.UniIDs = [list(self.DF_Gem.Gemeentecode[self.DF_Gem.Gemeentenaam == i])[0] for i in self.UniLocs]

        ''' Mezuro data '''
        self.RawData = pd.read_csv(self.Path_RawDataDay, delimiter=';')

        ''' Demographic data (home pop) '''
        DF_Demo = pd.read_csv(self.Path_DemoMat, delimiter=',')
        DemoIDs = np.array(DF_Demo['Unnamed: 0'])
        DemoMat_unsorted = np.array(DF_Demo[DF_Demo.keys()[1:]])
        DemoMat_sorted = []
        for ID in self.UniIDs:
            DemoMat_sorted.append(DemoMat_unsorted[DemoIDs == ID])
        self.DemoMat = (np.array(DemoMat_sorted)/self.Div).astype(int)[:, 0]
        self.HomePop = self.DemoMat.sum(axis=1)
        self.UniGroups = list(DF_Demo.keys()[1:])
        self.N = np.sum(self.DemoMat)

    def mobility_matrix(self):
        ''' Computes mobility matrix from Mezuro data (averaging over 14 days) '''

        self.Datum = np.array(self.RawData.datum)
        UniDates = np.unique(self.Datum)
        mf = np.zeros(shape=(len(UniDates), len(self.UniLocs), len(self.UniLocs)))
        mi = np.zeros(shape=(len(UniDates), len(self.UniLocs), len(self.UniLocs)))
        for d in tqdm(range(len(UniDates))):
            datpart = self.RawData[self.Datum == UniDates[d]]
            datpart = datpart.reset_index()
            LocID_visit = np.array(datpart.bezoek_gemeente_id)
            LocID_home = np.array(datpart.woon_gemeente_id)
            Visits_freq = np.array(datpart.frequente_bezoeker)+np.array(datpart.regelmatige_bezoeker)
            Visits_inc = np.array(datpart.incidentele_bezoeker)

            TotLocs = len(self.UniLocs)
            MobMat_freq = np.zeros(shape=(TotLocs, TotLocs))
            MobMat_inc = np.zeros(shape=(TotLocs, TotLocs))
            for i in range(TotLocs):
                w1 = np.where(LocID_home == self.UniIDs[i])[0]
                visits = LocID_visit[w1]
                amountsf = Visits_freq[w1]
                amountsi = Visits_inc[w1]
                for v in range(len(visits)):
                    w2f = np.where(self.UniIDs == visits[v])[0]
                    MobMat_freq[i, w2f] = amountsf[v]
                    w2i = np.where(self.UniIDs == visits[v])[0]
                    MobMat_inc[i, w2i] = amountsi[v]
            mf[d] = (MobMat_freq/self.Div).astype(int)
            mi[d] = (MobMat_inc/self.Div).astype(int)
        self.MobMat_freq = np.nanmean(mf, axis=0)
        self.MobMat_inc = np.nanmean(mi, axis=0)

    def create_people_DF(self):
        ''' Create day schedules per person '''

        self.PeopleDFs = []
        for N in tqdm(range(self.Ndays)):
            DF = {}
            DF = pd.DataFrame(DF)
            PeopleMat = np.zeros(shape=(self.N, len(self.UniLocs)))
            a = 0
            for r in range(len(self.UniLocs)):
                for g in range(len(self.UniGroups)):
                    for person in range(self.DemoMat[r][g]):
                        DF_p = {}
                        DF_p['Home'] = [self.UniLocs[r]]
                        DF_p['Group'] = [self.UniGroups[g]]
                        PeopleMat[a] = draw_fractions(self, r, g)
                        a += 1
                        if r == 0 and g == 0 and person == 0:
                            DF = pd.DataFrame(DF_p)
                        else:
                            DF = DF.append(pd.DataFrame(DF_p),
                                           ignore_index=True)
            for f in range(len(self.UniLocs)):
                DF['F_'+str(f)] = PeopleMat[:, f]
            self.PeopleDFs.append(DF)

    def position_people(self):
        ''' Determine actual positioning of people over time '''

        self.Positions_all = []
        for N in tqdm(range(self.Ndays)):
            PeopleDF = self.PeopleDFs[N]
            allpos = []
            hourpos = np.zeros(shape=(len(PeopleDF), 24*1))-10
            for p in range(len(PeopleDF)):
                keys = np.array(PeopleDF[PeopleDF.index == p])[0]
                r = keys[0]
                r = np.where(np.array(self.UniLocs) == r)[0][0]
                fs = np.array(keys)[2:].astype(float)
                hs = np.round(fs*24*1, 0).astype(int)
                hs = hs / np.sum(hs) * 24 * 1
                hs = hs.astype(int)
                a = np.arange(len(self.UniLocs))
                np.random.shuffle(a)
                hourpos[p, :int(np.round(hs[r]/2))] = r
                d = int(np.round(hs[r]/2))
                hs[r] = 0
                for i in a:
                    dnew = d+hs[i]
                    hourpos[p, d:dnew] = i
                    d = dnew
                hourpos[p, dnew:] = r
            allpos = allpos + list(hourpos.T)
            Positions = np.array(allpos)
            self.Positions_all.append(Positions)

    def count_people(self):
        ''' Counts total numbers of people per R and G '''

        choose = 0
        PeopleDF = self.PeopleDFs[choose]
        Positions = self.Positions_all[choose]
        pops = []
        for j in range(len(self.UniGroups)):
            whs = np.where(PeopleDF.Group == self.UniGroups[j])[0]
            totpop = np.zeros(shape=(24, len(self.UniLocs)))
            for i in range(24):
                allp = Positions[i][whs]
                for j in allp.astype(int):
                    totpop[i, j] += 1
            pops.append(totpop)
        self.AggPositions = np.array(pops)

    def save(self, seed):
        ''' Saves per seed '''

        path = self.Path_Data+self.SaveName+'/'+'Seed_'+str(seed)+'/'
        pathg = self.Path_Data+'General/'
        pd.DataFrame(self.PeopleDFs[0]).to_pickle(path+'PeopleDF.pkl')
        pd.DataFrame(self.UniLocs).to_pickle(pathg+'Gemeenten.pkl')
        pd.DataFrame(self.UniIDs).to_pickle(pathg+'GemeentenID.pkl')
        np.save(path+'Positions', self.Positions_all)
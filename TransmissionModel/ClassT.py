# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import configparser
import sys
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.sparse
import warnings 
from Functions import (get_mixmat, translate_polymod, new_mixmat, windowmean,
                       enter_phase2, enter_phase3,
                       enter_phase4, rivm_to_model, iconv,
                       change_phases, determine_exposed,
                       force_of_infection2, recalc_positions,
                       interv_border, interv_local, change_phases_G4,
                       change_phases_brablim, close_all_schools,
                       close_local_schools,
                       daytimemixer, nighttimemixer, national_datesetting)
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class ModelT(object):
    def __init__(self, params_input):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config.ini')

        ''' Variables '''
        self.SaveName = params_input['savename']
        self.Intervention = params_input['intervention']
        self.Seed = int(params_input['seed'])
        self.T = int(params_input['Ndays'])
        self.Path_Data = config['PATHS']['DATA']
        self.Path_Datasave = config['PATHS']['FIG']
        self.Path_RawDataMix = config['PATHS']['RAWDATA_MIX']
        self.Path_RawDataMix2 = config['PATHS']['RAWDATA_MIX2']
        self.Path_InfData = config['PATHS']['INFDATA']
        self.Path_ICData = config['PATHS']['ICDATA']
        self.Path_GoogleData = config['PATHS']['GOOGLEDATA']
        self.Path_PienterData = config['PATHS']['PIENTERDATA']
        #self.Prob_ic = np.float(config['PARAMS']['PROB_IC'])
        #self.Frac_ic = np.float(config['PARAMS']['FRAC_IC'])
        self.Prob_hos = np.float(config['PARAMS']['PROB_HOS'])#self.Prob_ic/self.Frac_ic
        self.Hos_lag_av = np.float(config['PARAMS']['LAG_HOS_MEAN'])
        self.Hos_lag_sh = np.float(config['PARAMS']['LAG_HOS_SHAPE'])
        self.Threshold = np.float(config['PARAMS']['THRESH'])
        self.Thresholdlocal = np.float(config['PARAMS']['THRESHLOCAL'])
        if self.SaveName == 'High':
            self.Div = 100
        elif self.SaveName == 'Med':
            self.Div = 500
        elif self.SaveName == 'Low':
            self.Div = 1000
        elif self.SaveName == 'Verylow':
            self.Div = 5000

        print('# ---------------------------------------------------------- #')
        print('# Starting Transmission model')
        print('# ------ Resolution:    '+self.SaveName)
        print('# ------ Intervention:  '+self.Intervention)
        print('# ------ Mobility seed: '+str(self.Seed))
        print('# ------ Amount days:   '+str(int(self.T/24)))
        print('# ---------------------------------------------------------- #')

    def read_model_data(self):
        ''' Read raw data '''

        pathg = self.Path_Data+'/General/'
        path = self.Path_Data+self.SaveName+'/'+'Seed_'+str(self.Seed)+'/'
        self.PeopleDF = pd.read_pickle(path+'PeopleDF.pkl')
        self.Positions = np.load(path+'Positions.npy')
        self.Positions0 = np.load(path+'Positions.npy').astype(int)
        self.UniLocs = np.array(pd.read_pickle(pathg+'Gemeenten.pkl')).T[0]
        self.UniIDs = np.array(pd.read_pickle(pathg+'GemeentenID.pkl')).T[0]
        self.Homes = np.array(self.PeopleDF.Home)
        self.Groups = np.array(self.PeopleDF.Group)
        self.UniGroups = np.unique(self.Groups)
        self.GroupsI = np.zeros(shape=len(self.Groups))
        for i in range(len(self.UniGroups)):
            self.GroupsI[self.Groups == self.UniGroups[i]] = i
        self.GroupsI = self.GroupsI.astype(int)
        self.HomesI = np.zeros(shape=len(self.Homes))
        for i in range(len(self.UniLocs)):
            self.HomesI[self.Homes == self.UniLocs[i]] = i
        self.HomesI = self.HomesI.astype(int)

        ''' Mixing matrices '''
        self.Mix_h_r = pd.read_excel(self.Path_RawDataMix2 +
                                     'MUestimates_home_2.xlsx',
                                     sheet_name='Netherlands', header=None)
        self.Mix_s_r = pd.read_excel(self.Path_RawDataMix2 +
                                     'MUestimates_school_2.xlsx',
                                     sheet_name='Netherlands', header=None)
        self.Mix_w_r = pd.read_excel(self.Path_RawDataMix2 +
                                     'MUestimates_work_2.xlsx',
                                     sheet_name='Netherlands', header=None)
        self.Mix_o_r = pd.read_excel(self.Path_RawDataMix2 +
                                     'MUestimates_other_locations_2.xlsx',
                                     sheet_name='Netherlands', header=None)
        self.Mix_h = new_mixmat(np.array(self.Mix_h_r))
        self.Mix_s = new_mixmat(np.array(self.Mix_s_r))
        self.Mix_w = new_mixmat(np.array(self.Mix_w_r))
        self.Mix_o = new_mixmat(np.array(self.Mix_o_r))  
        self.Mix_ws = (self.Mix_s+self.Mix_w)/2   

        self.Mix_h0 = new_mixmat(np.array(self.Mix_h_r))
        self.Mix_s0 = new_mixmat(np.array(self.Mix_s_r))
        self.Mix_w0 = new_mixmat(np.array(self.Mix_w_r))
        self.Mix_o0 = new_mixmat(np.array(self.Mix_o_r))  
        self.Mix_ws0 = (self.Mix_s0+self.Mix_w0)/2
        
        self.HomePops = np.zeros(len(self.UniLocs))
        for i in range(len(self.UniLocs)):
            self.HomePops[i] = len(np.where(self.Homes == self.UniLocs[i])[0])

    def read_empirical_data(self):
        ''' Read COVID / IC-incident data from RIVM '''
        
        ''' Inf data '''
        self.InfDF = pd.read_csv(self.Path_InfData, delimiter=';')
        Loc = np.array(self.InfDF.Municipality_code)
        cleanedList = [x for x in range(len(Loc)) if str(Loc[x]) != 'nan']
        self.InfDF = self.InfDF[self.InfDF.index.isin(cleanedList)]
        self.InfDF = self.InfDF.reset_index(drop=True)
        Loc = np.array(self.InfDF.Municipality_name)
        Day = np.array(pd.to_datetime(self.InfDF.Date_of_publication
                                      ).dt.day)
        Month = np.array(pd.to_datetime(self.InfDF.Date_of_publication
                                        ).dt.month)
        Year = np.array(pd.to_datetime(self.InfDF.Date_of_publication
                                       ).dt.year)-2020
        mondays = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30,
                            31, 31, 28, 31, 30, 31, 30, 31])
        cumdays = np.cumsum(mondays)
        Days = Day+np.array(cumdays)[Month-1+12*Year]
        Index_f1 = np.where(Days <= 31+29+0) # up to (excl.) 1th Mar
        Index_f1_adj = np.where(Days <= 31+29+0+self.Hos_lag_av) # up to (excl.) 1th Mar (+ lag)
        I_rep = np.array(self.InfDF.Total_reported)
        I_hos = np.array(self.InfDF.Hospital_admission)
        F1_loc = Loc[Index_f1_adj]
        F1_i = I_hos[Index_f1_adj]/self.Prob_hos
        self.InitialI = np.zeros(len(self.UniLocs))
        for i in range(len(self.UniLocs)):
            l = self.UniLocs[i]
            w = np.where(F1_loc == l)[0]
            if len(w) > 0:
                self.InitialI[i] = np.nansum(F1_i[w])
            else:
                if l in ['Aalburg', 'Werkendam', 'Woudrichem']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Altena'])/3
                elif l in ['Dongeradeel', 'Ferwerderadiel', 'Kollumerland en Nieuwkruisland']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Noardeast-Fryslân'])/3
                elif l in ['Geldermalsen', 'Lingewaal', 'Neerijnen']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'West Betuwe'])/3
                elif l in ['Groningen', 'Haren', 'Ten Boer']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Groningen'])/3
                elif l in ['Bedum', 'De Marne', 'Eemsmond', 'Winsum']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Het Hogeland'])/4
                elif l in ['Grootegast', 'Leek', 'Marum', 'Zuidhorn']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Westerkwartier'])/4
                elif l in ['Nuth', 'Onderbanken', 'Schinnen']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Beekdaelen'])/3
                elif l in ['Haarlemmerliede en Spaarnwoude', 'Haarlemmermeer']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Haarlemmermeer'])/2
                elif l in ['Leerdam', 'Zederik', 'Vianen']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Vijfheerenlande'])/3
                elif l in ['Binnenmaas', 'Cromstrijen', 'Korendijk', 'Oud-Beijerland', 'Strijen']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Hoeksche Waard'])/5
                elif l in ['Giessenlanden', 'Molenwaard']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Molenlanden'])/2
                elif l in ['Noordwijk', 'Noordwijkerhout']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Noordwijk'])/2
                elif l in ['Appingedam', 'Delfzijl', 'Loppersum']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Eemsdelta'])/3
                elif l in ['Sdwest-Frysln']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'S\x9cdwest-Frysl\x89n'])
                elif l in ['Hengelo']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Hengelo (O.)'])
                elif l in ['Haaren', 'Oisterwijk']:
                    self.InitialI[i] = np.nansum(F1_i[F1_loc == 'Oisterwijk'])/2
        self.InitialI = np.round(self.InitialI/self.Div).astype(int)

        ''' Mixing data from PIENTER '''
        self.PienterDF = pd.read_csv(self.Path_PienterData, delimiter='\t')
        groups = np.array(['[0,5)', '[5,10)', '[10,20)', '[20,30)',
                           '[30,40)', '[40,50)', '[50,60)', '[60,70)',
                           '[70,80)', '[80,Inf]'])
        MatRef = np.zeros(shape=(len(groups), len(groups)))
        MatApr = np.zeros(shape=(len(groups), len(groups)))
        MatJun = np.zeros(shape=(len(groups), len(groups)))
        for i in range(len(groups)):
            MatRef[i] = np.array(self.PienterDF[(self.PienterDF.survey == 'baseline') &
                                                (self.PienterDF.contact_type == 'all') &
                                                (self.PienterDF.part_age == groups[i])].m_est)
            MatApr[i] = np.array(self.PienterDF[(self.PienterDF.survey == 'April 2020') &
                                                (self.PienterDF.contact_type == 'all') &
                                                (self.PienterDF.part_age == groups[i])].m_est)
            MatJun[i] = np.array(self.PienterDF[(self.PienterDF.survey == 'June 2020') &
                                                (self.PienterDF.contact_type == 'all') &
                                                (self.PienterDF.part_age == groups[i])].m_est)
        Reference = rivm_to_model(MatRef)
        April2020 = rivm_to_model(MatApr)
        June2020 = rivm_to_model(MatJun)
        self.MixChange_phase2 = April2020/Reference
        self.MixChange_phase4 = June2020/Reference

        ''' Mobility data from Google '''
        self.GoogleDF = pd.read_csv(self.Path_GoogleData)
        dates = pd.to_datetime(self.GoogleDF.date)
        UniDates = np.unique(dates)
        TimePhase2 = np.where((UniDates >= pd.Timestamp('2020-03-12 00:00:00')) & (UniDates <= pd.Timestamp('2020-03-22 23:59:59')))[0]
        TimePhase3 = np.where((UniDates >= pd.Timestamp('2020-03-23 00:00:00')) & (UniDates <= pd.Timestamp('2020-05-10 23:59:59')))[0]
        TimePhase4 = np.where((UniDates >= pd.Timestamp('2020-05-11 00:00:00')) & (UniDates <= pd.Timestamp('2020-07-31 23:59:59')))[0]
        Mob = np.array(self.GoogleDF[['retail_and_recreation_percent_change_from_baseline',
                                      'transit_stations_percent_change_from_baseline',
                                      'workplaces_percent_change_from_baseline']])
        Mob2 = np.zeros(shape=(len(UniDates),  3))
        for i in range(len(UniDates)):
            wh = np.where(dates == UniDates[i])[0]
            Mob2[i] = np.nanmean(Mob[wh], axis=0)
        AvMob = np.nanmean(Mob2[:, :], axis=1)

        self.MobChange_phase2 = np.nanmean(AvMob[TimePhase2]) # -31.7 %
        self.MobChange_phase3 = np.nanmean(AvMob[TimePhase3]) # -42.4 %
        self.MobChange_phase4 = np.nanmean(AvMob[TimePhase4]) # -20.1 %

        if self.Intervention == 'working':
            self.MobChange_phase2 = 0
            self.MobChange_phase3 = 0
            self.MobChange_phase4 = 0
        
        if self.Intervention == 'brablim':
            df = pd.read_csv('/Users/mmdekker/Documents/Werk/Data/CBS/Gemeenten2018.csv', delimiter=';', encoding='latin')
            ids = np.array(df.Gemeentecode).astype(int)
            provs = np.array(df.Provincienaam)
            self.Brablim = np.zeros(len(self.UniLocs))
            for m in range(len(self.UniLocs)):
                idy = self.UniIDs[m]
                w = np.where(ids == idy)[0][0]
                Prov = provs[w]
                if Prov == 'Noord-Brabant' or Prov == 'Limburg':
                    self.Brablim[m] = 1

        self.N = len(self.PeopleDF)
        
        if self.Intervention == 'G4':
            self.G4 = np.zeros(shape=(self.N))
            for m in range(len(self.UniLocs)):
                if self.UniLocs[m] in ['Amsterdam', 'Rotterdam', 'Utrecht', "'s-Gravenhage"]:
                    self.G4[m] = 1

    def set_parameters(self):
        ''' Read raw data '''

        self.N = len(self.PeopleDF)     # Amount of people
        self.T = self.T                 # Amount of hours simulated
        self.EI_l = 4.6                 # was 5.5
        self.EI_k = 20
        self.IR_l = 5                   # was 10
        self.IR_k = 1.0                 # was 0.8
        self.Beta_f1 = 0.135#0.09
        self.Beta_f2 = 0.11#0.06
        self.Beta_f3 = 0.09#0.03
        self.Beta_f4 = 0.11#0.06
        if self.Intervention == 'behavior':
            self.Beta_f2 = 0.135#0.09
            self.Beta_f3 = 0.135#0.09
            self.Beta_f4 = 0.135#0.09

    def initialise(self):
        ''' Initialise transmission model '''
        
        self.Init = np.zeros(len(self.Homes))
        for i in range(len(self.UniLocs)):
            amount = self.InitialI[i]
            if amount > 0:
                wh = np.where(self.Homes == self.UniLocs[i])[0]
                self.Init[np.random.choice(wh, size=amount, replace=False)] = 2
        wh = np.where(self.Init == 2)[0]
        self.Gammas = np.zeros(shape=(self.N, 2))+np.nan
        self.Rhos = np.zeros(shape=(self.N, 2))+np.nan
        self.Rhos[wh, 0] = 24*np.random.weibull(self.EI_k, size=len(wh))*self.EI_l
        self.Rhos[wh, 1] = np.random.choice(np.arange(-5.5*24, 0), size=len(wh))
        
        mixmats = np.array([self.Mix_h, self.Mix_s, self.Mix_w, self.Mix_o])
        mixav = np.mean(mixmats, axis=0)
        lens = np.zeros(11)
        for g in range(len(self.UniGroups)):
            lens[g] = len(np.where(self.GroupsI == g)[0])
        lens = lens/np.sum(lens)
        
        sleeping = [1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1]
        awake = 1-np.array(sleeping)
        sleepfactor = np.array(awake)/np.sum(awake)
        self.sleepfactor = sleepfactor
        self.mixav = mixav
        HG = [1.0, 2.0, 3.051, 5.751, 5.751, 3.6, 3.6, 5.0, 5.0, 5.3, 7.2]
        self.HG = np.array(HG)/sum(HG)*11
        self.Betaf = self.Beta_f1

    # def simulate(self):
    #     ''' Initialise transmission model '''

    #     def force_of_infection(p, Stat, s_t, pos, hour):
    #         group = self.GroupsI[p]
    #         position = int(pos[p])
    #         mixvec = get_mixmat(self, hour, position, group, p, self.Homes)[group]
    #         lda_p = 0
    #         for g in range(len(self.UniGroups)):
    #             stat = Stat[self.WH[np.mod(t, 24*7)][position][g]]
    #             Frac_Igm_Ngm = len(np.where(stat == 2)[0])/(1e-9+len(stat)) 
    #             if Frac_Igm_Ngm > 0:
    #                 n_pg = mixvec[g]
    #                 lda_p += n_pg*Frac_Igm_Ngm
    #         return lda_p*self.HG[group]

    #     self.WH = recalc_positions(self)
    #     mixav=self.mixav
    #     Status = np.zeros(shape=(self.T, self.N))+np.nan
    #     Status[0] = self.Init
    #     totcum = np.zeros(self.T)+np.nan
    #     totcum[0] = len(np.where(self.Init == 2)[0])-np.sum(self.InitialI)
    #     ldm = np.zeros(shape=(self.T))
    #     btm = np.zeros(shape=(self.T))
    #     phase = 1
    #     Phases = [1]
    #     self.Homeworkers = []
    #     for t in tqdm(range(1, self.T)):
            
    #         if phase == 1 and totcum[t-1] >= 0.019*self.N:  # 0.0095*self.N:
    #             enter_phase2(self)
    #             phase = 2
    #             t0 = np.float(t)
    #             self.WH = recalc_positions(self)
    #         elif phase == 2 and t == int(t0+11*24):
    #             enter_phase3(self)
    #             phase = 3
    #             self.WH = recalc_positions(self)
    #         elif phase == 3 and t == int(t0+60*24):
    #             enter_phase4(self)
    #             phase = 4
    #             self.WH = recalc_positions(self)
                
    #         S = np.where(Status[t-1] == 0)[0]
    #         day = np.mod(int(np.floor(t/24)), 7)
    #         hour = np.mod(t, 24)
    #         s_t = self.sleepfactor[hour]
    #         pos = self.Positions[day][hour]
            
    #         # S -> E
    #         lds = np.zeros(len(S))
    #         for p in range(len(S)):
    #             lds[p] = force_of_infection(S[p], Status[t-1], s_t, pos, hour)
    #         lds = lds*self.Betaf*s_t
    #         En = S[np.random.random(len(S)) < lds]
    #         self.Rhos[En, 0] = 24*np.random.weibull(self.EI_k, size=len(En))*self.EI_l
    #         self.Rhos[En, 1] = t

    #         # E -> I
    #         In = np.where(self.Rhos.sum(axis=1) <= t)[0]
    #         self.Rhos[In, 1] = np.nan
    #         self.Gammas[In, 0] = 24*np.random.weibull(self.IR_k, size=len(In))*self.IR_l
    #         self.Gammas[In, 1] = t
            
    #         # I -> R
    #         Rn = np.where(self.Gammas.sum(axis=1) <= t)[0]
    #         self.Gammas[Rn, 1] = np.nan
            
    #         # Change
    #         Status[t] = Status[t-1]
    #         Status[t, En] = 1
    #         Status[t, In] = 2
    #         Status[t, Rn] = 3
            
    #         totcum[t] = len(np.where((Status[t] == 2) | (Status[t] == 3))[0])-np.sum(self.InitialI)
    #         Phases.append(phase)
    #     self.Status = Status
    #     self.Totcum = totcum
    #     self.Phases = Phases

    def simulate_new(self):
        ''' Initialise transmission model '''

        self.Positions = self.Positions.astype(int)
        Status = np.zeros(shape=(self.T, self.N))+np.nan
        Status[0] = self.Init
        self.Inisum = np.sum(self.InitialI)
        phase = 1
        self.Predated = 0
        self.Timestep12March = 999999
        t0 = 0
        self.Homeschoolers = []
        self.Homeworkers = []
        self.Homeschoolers_m = [[]]*len(self.UniLocs)
        self.Homeworkers_m = [[]]*len(self.UniLocs)
        self.Workers = np.where((self.Groups == 'e) Non-studying adolescents') |
                                (self.Groups == 'f) Middle-age working') |
                                (self.Groups == 'h) Higher-age working'))[0]
        self.Schoolers = np.where((self.Groups == 'b) Primary school children') |
                                  (self.Groups == 'c) Secondary school children') |
                                  (self.Groups == 'd) Students'))[0]
        
        GroupsMat = np.zeros(shape=(11, self.N))
        for g in range(11):
            GroupsMat[g, self.GroupsI == g] = 1
        self.GroupsMat = GroupsMat
        self.GroupsMat_sp = scipy.sparse.csr_matrix(GroupsMat)

        if self.Intervention == 'local':
            self.Phased = np.zeros(len(self.UniLocs))
            self.TimePhased = np.zeros(len(self.UniLocs))
            phase = np.ones(380)
            t0 = np.zeros(380)
            self.Betas = np.zeros(len(self.UniLocs))+self.Beta_f1
        if self.Intervention in ['local', 'G4', 'brablim']:
            self.HomeMat = []
            self.WorkersinMun = []
            self.SchoolersinMun = []
            self.SchoolparentsinMun = []
            for i in range(len(self.UniLocs)):
                maw = np.where((self.Homes == self.UniLocs[i]) & (self.Groups == 'f) Middle-age working'))[0]
                self.SchoolparentsinMun.append(np.random.choice(maw, int(0.12*len(maw))))
                self.HomeMat.append(np.where(self.Homes == self.UniLocs[i])[0])
                self.WorkersinMun.append(np.where((self.Homes == self.UniLocs[i]
                                                  ) & (np.isin(self.Groups,
                                                               ['e) Non-studying adolescents',
                                                                'f) Middle-age working',
                                                                'h) Higher-age working'])
                                                               ))[0])
                self.SchoolersinMun.append(np.where((self.Homes == self.UniLocs[i]
                                                     ) & (np.isin(self.Groups,
                                                                  ['b) Primary school children',
                                                                   'c) Secondary school children',
                                                                   'd) Students'])
                                                                  ))[0])
        if self.Intervention == 'brablim':
            self.Betas = np.zeros(len(self.UniLocs))+self.Beta_f1
        if self.Intervention == 'G4':
            self.Betas = np.zeros(len(self.UniLocs))+self.Beta_f1
        if self.Intervention == 'border':
            self.ClosedBorders = np.zeros(len(self.UniLocs))
            self.TimeClosures = np.zeros(len(self.UniLocs))
        Phases = [phase]
        self.PosMat, self.PosMat0 = recalc_positions(self)
        
        # Identify parents schoolers
        maw = np.where(self.Groups == 'f) Middle-age working')[0]
        self.SchoolParents = np.random.choice(maw, int(0.12*len(maw)))
        
        if self.Intervention == 'schoolextreme' or self.Intervention == 'schoolparents' or self.Intervention == 'schoolisolation':
            close_all_schools(self)
        if self.Intervention == 'schoolearly':
            close_all_schools(self)
        
        # SIMULATION PART
        for t in tqdm(range(1, self.T)):
            # NEW IR? <--- let op
            # IR = len(np.where(Status[t-1] >= 2)[0])
            
            # INTERVENTIONS
            if self.Intervention == 'local':
                if self.Predated == 0:
                    IR = len(np.where(Status[t-1] >= 2)[0])
                    national_datesetting(self, t, IR)
                if self.Predated == 1:
                    phase, t0 = interv_local(self, Status[t-1], t, phase, t0)
            elif self.Intervention == 'brablim':
                phase, t0 = change_phases_brablim(self, phase, IR, t, t0)
                self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
                self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]
            elif self.Intervention == 'G4':
                phase, t0 = change_phases_G4(self, phase, IR, t, t0)
                self.Homeworkers = [item for sublist in self.Homeworkers_m for item in sublist]
                self.Homeschoolers = [item for sublist in self.Homeschoolers_m for item in sublist]
            else:
                IR = len(np.where(Status[t-1] >= 2)[0])
                phase, t0 = change_phases(self, phase, IR, t, t0)
            if self.Intervention == 'border': # this is complementary to regular phase changing
                interv_border(self, Status[t-1], t)
            
            # TRANSMISSION
            day = np.mod(int(np.floor(t/24)), 7)
            hour = np.mod(t, 24)
            En = determine_exposed(self, Status[t-1], day, hour, phase)
            In = np.where(self.Rhos.sum(axis=1) <= t)[0]
            Rn = np.where(self.Gammas.sum(axis=1) <= t)[0]
            
            # SAVE NEW STATUS AND RHO/GAMMA TIME SCALES
            Status[t] = Status[t-1]
            
            self.Rhos[En, 0] = 24*np.random.weibull(self.EI_k, size=len(En))*self.EI_l
            self.Rhos[En, 1] = t
            Status[t, En] = 1
            
            self.Rhos[In, 1] = np.nan
            self.Gammas[In, 0] = 24*np.random.weibull(self.IR_k, size=len(In))*self.IR_l
            self.Gammas[In, 1] = t
            Status[t, In] = 2
            
            self.Gammas[Rn, 1] = np.nan
            Status[t, Rn] = 3
            
            Phases.append(phase)
            del En, In, Rn
        self.Status = Status
        self.Phases = Phases

    def save(self, run):
        ''' Saves '''

        add = ''#'_001'
        Status_sparse = scipy.sparse.csr_matrix(self.Status)
        path = self.Path_Data+self.SaveName+'/'+'Seed_'+str(self.Seed)+'/'
        np.savetxt(path+'Runs_'+self.Intervention+add+'/Timestep_'+str(run), np.array([self.Timestep12March]))
        scipy.sparse.save_npz(path+'Runs_'+self.Intervention+add+'/Status_'+str(run)+'.npz', Status_sparse)
        pd.DataFrame(self.Phases).to_pickle(path+'Runs_'+self.Intervention+add+'/Phases_'+str(run)+'.pkl')
        if self.Intervention == 'border':
            pd.DataFrame(self.TimeClosures).to_pickle(path+'Runs_'+self.Intervention+add+'/TimeClosures_'+str(run)+'.pkl')
        if self.Intervention == 'local':
            pd.DataFrame(self.TimePhased).to_pickle(path+'Runs_'+self.Intervention+add+'/TimePhased_'+str(run)+'.pkl')
            
# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
from ClassM import ModelM
from tqdm import tqdm
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from Functions import draw_fractions

# ----------------------------------------------------------------- #
# Initialize Class
# ----------------------------------------------------------------- #

params_input = {'savename': 'High',
                'division': 100 # 5000 - 1000 - 500 - 100
                }
ClassM = ModelM(params_input)
ClassM.read_data()
ClassM.mobility_matrix()
for mc in range(5, 10):
    ClassM.create_people_DF()
    ClassM.position_people()
    ClassM.count_people()
    ClassM.save(mc) 
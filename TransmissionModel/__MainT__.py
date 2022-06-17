# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
from ClassT import ModelT
from tqdm import tqdm
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import scipy.sparse

# ----------------------------------------------------------------- #
# Initialize Class
# ----------------------------------------------------------------- #

# Interventions:
    # 'ref'
    # 'working'
    # 'behavior'

    # 'school' 
    # 'schoolisolation'
    # 'schoolparents'
    # 'schoolextreme'

    # 'G4'
    # 'border'
    # 'local'
    # 'brablim'

for interv in ['schoolearly']:
    for run in [0]:
        for seed in [3, 4, 5]:#range(5, 10):
            params_input = {'savename': 'High',
                            'intervention': interv,
                            'Ndays': 120*24,
                            'seed': seed}
            ClassT = ModelT(params_input)
            ClassT.read_model_data()
            ClassT.read_empirical_data()
            ClassT.set_parameters()
            ClassT.initialise()
            ClassT.simulate_new()
            ClassT.save(run)
            del ClassT
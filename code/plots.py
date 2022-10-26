# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
sessions = np.arange(18)
sessions
# %%
for ses in sessions:
    data = pd.read_csv('../../data/P1/P1_pre_sesion_%d.txt'%ses, index_col=False, header=6)

# %%

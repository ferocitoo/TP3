import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection

from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

module_name = "utils_v2"
file_path = "/workspaces/TP-Chaos/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd

############################
# ------Import datas------ #
############################

# --- Ethanol --- #
# Ethanol 3.5°C
Ethanol__3_5 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 3.5 deg.C 2024-Nov-01 11_43_48.txt", delimiter='\t', decimal = ".",header = 10) 

Ethanol__3_5__freq = Ethanol__3_5.iloc[:,0]
Ethanol__3_5__EpsR = Ethanol__3_5.iloc[:,3]
Ethanol__3_5__EpsI = Ethanol__3_5.iloc[:,4]
Ethanol__3_5__Sigma = Ethanol__3_5.iloc[:,5]
Ethanol__3_5__TanD = Ethanol__3_5.iloc[:,6]


# Ethanol 6.2°C
Ethanol__6_2 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 6.2 deg.C 2024-Nov-01 11_41_04.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__6_2__freq = Ethanol__6_2.iloc[:,0]
Ethanol__6_2__EpsR = Ethanol__6_2.iloc[:,3]
Ethanol__6_2__EpsI = Ethanol__6_2.iloc[:,4]
Ethanol__6_2__Sigma = Ethanol__6_2.iloc[:,5]
Ethanol__6_2__TanD = Ethanol__6_2.iloc[:,6]


# Ethanol 10.5
Ethanol__10_5 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 10.5 deg.C 2024-Nov-01 11_37_29.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__10_5__freq = Ethanol__10_5.iloc[:,0]
Ethanol__10_5__EpsR = Ethanol__10_5.iloc[:,3]
Ethanol__10_5__EpsI = Ethanol__10_5.iloc[:,4]
Ethanol__10_5__Sigma = Ethanol__10_5.iloc[:,5]
Ethanol__10_5__TanD = Ethanol__10_5.iloc[:,6]


# Ethanol 13.6
Ethanol__13_6 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 13.6 deg.C 2024-Nov-01 11_35_28.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__13_6__freq = Ethanol__13_6.iloc[:,0]
Ethanol__13_6__EpsR = Ethanol__13_6.iloc[:,3]
Ethanol__13_6__EpsI = Ethanol__13_6.iloc[:,4]
Ethanol__13_6__Sigma = Ethanol__13_6.iloc[:,5]
Ethanol__13_6__TanD = Ethanol__13_6.iloc[:,6]


# Ethanol 21.2
Ethanol__21_2 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 21.2 deg.C 2024-Nov-01 11_16_58.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__21_2__freq = Ethanol__21_2.iloc[:,0]
Ethanol__21_2__EpsR = Ethanol__21_2.iloc[:,3]
Ethanol__21_2__EpsI = Ethanol__21_2.iloc[:,4]
Ethanol__21_2__Sigma = Ethanol__21_2.iloc[:,5]
Ethanol__21_2__TanD = Ethanol__21_2.iloc[:,6]


# All ethanol at different temperature in the same array
Temperatures = [3.5,6.2,10.5,13.6,21.2]
Ethanol_freq = [Ethanol__3_5__freq,Ethanol__6_2__freq,Ethanol__10_5__freq,Ethanol__13_6__freq,Ethanol__21_2__freq]
Ethanol_EpsR = [Ethanol__3_5__EpsR,Ethanol__6_2__EpsR,Ethanol__10_5__EpsR,Ethanol__13_6__EpsR,Ethanol__21_2__EpsR]
Ethanol_EpsI = [Ethanol__3_5__EpsI,Ethanol__6_2__EpsI,Ethanol__10_5__EpsI,Ethanol__13_6__EpsI,Ethanol__21_2__EpsI]
Ethanol_Sigma = [Ethanol__3_5__Sigma,Ethanol__6_2__Sigma,Ethanol__10_5__Sigma,Ethanol__13_6__Sigma,Ethanol__21_2__Sigma]
Ethanol_TanD = [Ethanol__3_5__TanD,Ethanol__6_2__TanD,Ethanol__10_5__TanD,Ethanol__13_6__TanD,Ethanol__21_2__TanD]




#####################
# ------Plots------ #
#####################

# --- Ethanol --- #

#freq vs EpsR and EpsI


xlabel = "Frequency [MHz]"
ylabel1 = r"$\epsilon_r^'$"
ylabel2 = r"$\epsilon_r^{''}$"

ax1,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel1)
ax2 = ax1.twinx()
ax2.set_ylabel(ylabel2,fontsize=25)
ax2.tick_params(axis='y', labelsize=18)

i=0
labels = []
lines = []
for T in Temperatures:
    
    freq = Ethanol_freq[i].to_numpy()
    EpsR = Ethanol_EpsR[i].to_numpy()
    EpsI = Ethanol_EpsI[i].to_numpy()

    line1, = ax1.plot(freq,EpsR,label=r"T = "+str(T)+"°C")
    line2, = ax2.plot(freq,EpsI,label=r"T = "+str(T)+"°C",linestyle='dashed')

    lines.append(line1)
    lines.append(line2)
    labels.append(line1.get_label())
    labels.append(line2.get_label())

    

    i+=1
    
ax2.legend(lines, labels, loc='upper right',fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol__3_5_freq_vs_EpsR_and_EpsI.pdf")





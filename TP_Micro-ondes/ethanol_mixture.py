import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection

import scipy.optimize as opt
from scipy.optimize import curve_fit

from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

module_name = "utils_v2"
file_path = "/workspaces/TP3/utils_v2.py"

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

ethanol_ml = 300
# --- Water + Ethanol --- #
# Ethanol 300 ml, water 0 ml
mixt_300_0 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 0 2024-Nov-08 12_09_47.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_0__freq = mixt_300_0.iloc[:,0]
mixt_300_0__EpsR = mixt_300_0.iloc[:,3]
mixt_300_0__EpsI = mixt_300_0.iloc[:,4]
mixt_300_0__Sigma = mixt_300_0.iloc[:,5]
mixt_300_0__TanD = mixt_300_0.iloc[:,6]

# Ethanol 300 ml, water 15 ml
mixt_300_15 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 15 2024-Nov-08 12_18_41.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_15__freq = mixt_300_15.iloc[:,0]
mixt_300_15__EpsR = mixt_300_15.iloc[:,3]
mixt_300_15__EpsI = mixt_300_15.iloc[:,4]
mixt_300_15__Sigma = mixt_300_15.iloc[:,5]
mixt_300_15__TanD = mixt_300_15.iloc[:,6]

# Ethanol 300 ml, water 30 ml
mixt_300_30 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 30 2024-Nov-08 12_22_34.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_30__freq = mixt_300_30.iloc[:,0]
mixt_300_30__EpsR = mixt_300_30.iloc[:,3]
mixt_300_30__EpsI = mixt_300_30.iloc[:,4]
mixt_300_30__Sigma = mixt_300_30.iloc[:,5]
mixt_300_30__TanD = mixt_300_30.iloc[:,6]

# Ethanol 300 ml, water 45 ml
mixt_300_45 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 45 2024-Nov-08 12_31_02.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_45__freq = mixt_300_45.iloc[:,0]
mixt_300_45__EpsR = mixt_300_45.iloc[:,3]
mixt_300_45__EpsI = mixt_300_45.iloc[:,4]
mixt_300_45__Sigma = mixt_300_45.iloc[:,5]
mixt_300_45__TanD = mixt_300_45.iloc[:,6]

# Ethanol 300 ml, water 60 ml
mixt_300_60 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 60 2024-Nov-08 12_34_29.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_60__freq = mixt_300_60.iloc[:,0]
mixt_300_60__EpsR = mixt_300_60.iloc[:,3]
mixt_300_60__EpsI = mixt_300_60.iloc[:,4]
mixt_300_60__Sigma = mixt_300_60.iloc[:,5]
mixt_300_60__TanD = mixt_300_60.iloc[:,6]

# Ethanol 300 ml, water 75 ml
mixt_300_75 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 75 2024-Nov-08 12_37_47.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_75__freq = mixt_300_75.iloc[:,0]
mixt_300_75__EpsR = mixt_300_75.iloc[:,3]
mixt_300_75__EpsI = mixt_300_75.iloc[:,4]
mixt_300_75__Sigma = mixt_300_75.iloc[:,5]
mixt_300_75__TanD = mixt_300_75.iloc[:,6]

# Ethanol 300 ml, water 90 ml
mixt_300_90 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 90 2024-Nov-08 12_42_39.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_90__freq = mixt_300_90.iloc[:,0]
mixt_300_90__EpsR = mixt_300_90.iloc[:,3]
mixt_300_90__EpsI = mixt_300_90.iloc[:,4]
mixt_300_90__Sigma = mixt_300_90.iloc[:,5]
mixt_300_90__TanD = mixt_300_90.iloc[:,6]

# Ethanol 300 ml, water 105 ml
mixt_300_105 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 105 2024-Nov-08 12_44_40.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_105__freq = mixt_300_105.iloc[:,0]
mixt_300_105__EpsR = mixt_300_105.iloc[:,3]
mixt_300_105__EpsI = mixt_300_105.iloc[:,4]
mixt_300_105__Sigma = mixt_300_105.iloc[:,5]
mixt_300_105__TanD = mixt_300_105.iloc[:,6]

# Ethanol 300 ml, water 120 ml
mixt_300_120 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 120 2024-Nov-08 12_46_40.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_120__freq = mixt_300_120.iloc[:,0]
mixt_300_120__EpsR = mixt_300_120.iloc[:,3]
mixt_300_120__EpsI = mixt_300_120.iloc[:,4]
mixt_300_120__Sigma = mixt_300_120.iloc[:,5]
mixt_300_120__TanD = mixt_300_120.iloc[:,6]

# Ethanol 300 ml, water 135 ml
mixt_300_135 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 135 2024-Nov-08 12_48_49.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_135__freq = mixt_300_135.iloc[:,0]
mixt_300_135__EpsR = mixt_300_135.iloc[:,3]
mixt_300_135__EpsI = mixt_300_135.iloc[:,4]
mixt_300_135__Sigma = mixt_300_135.iloc[:,5]
mixt_300_135__TanD = mixt_300_135.iloc[:,6]

# Ethanol 300 ml, water 150 ml
mixt_300_150 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 150 2024-Nov-08 12_53_08.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_150__freq = mixt_300_150.iloc[:,0]
mixt_300_150__EpsR = mixt_300_150.iloc[:,3]
mixt_300_150__EpsI = mixt_300_150.iloc[:,4]
mixt_300_150__Sigma = mixt_300_150.iloc[:,5]
mixt_300_150__TanD = mixt_300_150.iloc[:,6]


water_ml = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
water_concentrations = water_ml/(ethanol_ml + water_ml)

mixt_freq = np.array([mixt_300_0__freq, mixt_300_15__freq, mixt_300_30__freq, mixt_300_45__freq, mixt_300_60__freq, mixt_300_75__freq, mixt_300_90__freq, mixt_300_105__freq, mixt_300_120__freq, mixt_300_135__freq, mixt_300_150__freq])
mixt_EpsI = np.array([mixt_300_0__EpsI, mixt_300_15__EpsI, mixt_300_30__EpsI, mixt_300_45__EpsI, mixt_300_60__EpsI, mixt_300_75__EpsI, mixt_300_90__EpsI, mixt_300_105__EpsI, mixt_300_120__EpsI, mixt_300_135__EpsI, mixt_300_150__EpsI])
mixt_EpsR = np.array([mixt_300_0__EpsR, mixt_300_15__EpsR, mixt_300_30__EpsR, mixt_300_45__EpsR, mixt_300_60__EpsR, mixt_300_75__EpsR, mixt_300_90__EpsR, mixt_300_105__EpsR, mixt_300_120__EpsR, mixt_300_135__EpsR, mixt_300_150__EpsR])


#freq vs EpsR

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

maxfreqs = []

i=0
for conc in water_concentrations:
    
    freq = mixt_freq[i]
    EpsR = mixt_EpsR[i]
    EpsI = mixt_EpsI[i]
    
    maxfreq = freq[np.argmax(EpsI)]
    maxfreqs.append(maxfreq)
    
    # ax.scatter(freq,EpsR,label=f"{conc*100:.1f}% Water, {(1-conc)*100:.1f}% Ethanol",s=2)
    ax.scatter(freq,EpsI,s=2)
    i+=1
    
u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/mixtures_EpsR.pdf")
print(np.shape(mixt_EpsR))
#take all the 100-th values of epsR and epsI 
EpsRs = mixt_EpsR[:,50]
EpsIs = mixt_EpsI[:,50]



Eps = EpsRs + 1j*EpsIs

#Eps water at 24.2°C
Eps_water = 78 + 10j

#Eps ethanol at 24.2°C
Eps_ethanol = 19.28 + 8.8j

#Eps mixtures
Eps_mixt = pow(Eps_water,water_concentrations)*pow(Eps_ethanol,1-water_concentrations)

print(Eps)
print(Eps_mixt)
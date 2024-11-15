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

from scipy.optimize import curve_fit

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

# Propanol

# Propanol -23°C
Propanol__n25 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol -23 deg.C 2024-Nov-01 16_45_26.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__n25__freq = Propanol__n25.iloc[:,0]
Propanol__n25__EpsR = Propanol__n25.iloc[:,3]
Propanol__n25__EpsI = Propanol__n25.iloc[:,4]
Propanol__n25__Sigma = Propanol__n25.iloc[:,5]
Propanol__n25__TanD = Propanol__n25.iloc[:,6]

# Propanol -18°C
Propanol__n18 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol -18 deg.C 2024-Nov-01 16_38_30.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__n18__freq = Propanol__n18.iloc[:,0]
Propanol__n18__EpsR = Propanol__n18.iloc[:,3]
Propanol__n18__EpsI = Propanol__n18.iloc[:,4]
Propanol__n18__Sigma = Propanol__n18.iloc[:,5]
Propanol__n18__TanD = Propanol__n18.iloc[:,6]

# Propanol -10°C
Propanol__n10 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol -10 deg.C 2024-Nov-01 16_32_59.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__n10__freq = Propanol__n10.iloc[:,0]
Propanol__n10__EpsR = Propanol__n10.iloc[:,3]
Propanol__n10__EpsI = Propanol__n10.iloc[:,4]
Propanol__n10__Sigma = Propanol__n10.iloc[:,5]
Propanol__n10__TanD = Propanol__n10.iloc[:,6]

# Propanol -6°C
Propanol__n6 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol -6 deg.C 2024-Nov-01 16_12_32.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__n6__freq = Propanol__n6.iloc[:,0]
Propanol__n6__EpsR = Propanol__n6.iloc[:,3]
Propanol__n6__EpsI = Propanol__n6.iloc[:,4]
Propanol__n6__Sigma = Propanol__n6.iloc[:,5]
Propanol__n6__TanD = Propanol__n6.iloc[:,6]

# Propanol -1°C
Propanol__n1 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol -1 deg.C 2024-Nov-01 16_19_57.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__n1__freq = Propanol__n1.iloc[:,0]
Propanol__n1__EpsR = Propanol__n1.iloc[:,3]
Propanol__n1__EpsI = Propanol__n1.iloc[:,4]
Propanol__n1__Sigma = Propanol__n1.iloc[:,5]
Propanol__n1__TanD = Propanol__n1.iloc[:,6]

# Propanol 1°C
Propanol__1 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 1 deg.C 2024-Nov-01 16_17_32.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__1__freq = Propanol__1.iloc[:,0]
Propanol__1__EpsR = Propanol__1.iloc[:,3]
Propanol__1__EpsI = Propanol__1.iloc[:,4]
Propanol__1__Sigma = Propanol__1.iloc[:,5]
Propanol__1__TanD = Propanol__1.iloc[:,6]

# Propanol 4°C
Propanol__4 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 4 deg.C 2024-Nov-01 16_02_48.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__4__freq = Propanol__4.iloc[:,0]
Propanol__4__EpsR = Propanol__4.iloc[:,3]
Propanol__4__EpsI = Propanol__4.iloc[:,4]
Propanol__4__Sigma = Propanol__4.iloc[:,5]
Propanol__4__TanD = Propanol__4.iloc[:,6]

# Propanol 11.2°C
Propanol__11_2 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 11.2 deg.C 2024-Oct-18 15_10_48.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__11_2__freq = Propanol__11_2.iloc[:,0]
Propanol__11_2__EpsR = Propanol__11_2.iloc[:,3]
Propanol__11_2__EpsI = Propanol__11_2.iloc[:,4]
Propanol__11_2__Sigma = Propanol__11_2.iloc[:,5]
Propanol__11_2__TanD = Propanol__11_2.iloc[:,6]

# Propanol 13.5°C
Propanol__13_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 13.5 deg.C 2024-Oct-18 15_09_22.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__13_5__freq = Propanol__13_5.iloc[:,0]
Propanol__13_5__EpsR = Propanol__13_5.iloc[:,3]
Propanol__13_5__EpsI = Propanol__13_5.iloc[:,4]
Propanol__13_5__Sigma = Propanol__13_5.iloc[:,5]
Propanol__13_5__TanD = Propanol__13_5.iloc[:,6]

# Propanol 15.5°C
Propanol__15_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 15.5 deg.C 2024-Oct-18 15_08_22.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__15_5__freq = Propanol__15_5.iloc[:,0]
Propanol__15_5__EpsR = Propanol__15_5.iloc[:,3]
Propanol__15_5__EpsI = Propanol__15_5.iloc[:,4]
Propanol__15_5__Sigma = Propanol__15_5.iloc[:,5]
Propanol__15_5__TanD = Propanol__15_5.iloc[:,6]

# Propanol 16.5°C
Propanol__16_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 16.5 deg.C 2024-Oct-18 15_07_30.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__16_5__freq = Propanol__16_5.iloc[:,0]
Propanol__16_5__EpsR = Propanol__16_5.iloc[:,3]
Propanol__16_5__EpsI = Propanol__16_5.iloc[:,4]
Propanol__16_5__Sigma = Propanol__16_5.iloc[:,5]
Propanol__16_5__TanD = Propanol__16_5.iloc[:,6]

# Propanol 19.5°C
Propanol__19_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 19.5 deg.C 2024-Oct-18 15_06_12.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__19_5__freq = Propanol__19_5.iloc[:,0]
Propanol__19_5__EpsR = Propanol__19_5.iloc[:,3]
Propanol__19_5__EpsI = Propanol__19_5.iloc[:,4]
Propanol__19_5__Sigma = Propanol__19_5.iloc[:,5]
Propanol__19_5__TanD = Propanol__19_5.iloc[:,6]

# Prpanol 21.5°C
Propanol__21_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 21.5 deg.C 2024-Oct-18 15_04_29.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__21_5__freq = Propanol__21_5.iloc[:,0]
Propanol__21_5__EpsR = Propanol__21_5.iloc[:,3]
Propanol__21_5__EpsI = Propanol__21_5.iloc[:,4]
Propanol__21_5__Sigma = Propanol__21_5.iloc[:,5]
Propanol__21_5__TanD = Propanol__21_5.iloc[:,6]

# Propanol 23.3°C
Propanol__23_3 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 23.3 deg.C 2024-Oct-18 10_10_07.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__23_3__freq = Propanol__23_3.iloc[:,0]
Propanol__23_3__EpsR = Propanol__23_3.iloc[:,3]
Propanol__23_3__EpsI = Propanol__23_3.iloc[:,4]
Propanol__23_3__Sigma = Propanol__23_3.iloc[:,5]
Propanol__23_3__TanD = Propanol__23_3.iloc[:,6]

# Propanol 28.5°C
Propanol__28_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 28.5 deg.C 2024-Nov-08 15_57_29.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__28_5__freq = Propanol__28_5.iloc[:,0]
Propanol__28_5__EpsR = Propanol__28_5.iloc[:,3]
Propanol__28_5__EpsI = Propanol__28_5.iloc[:,4]
Propanol__28_5__Sigma = Propanol__28_5.iloc[:,5]
Propanol__28_5__TanD = Propanol__28_5.iloc[:,6]

# Propanol 33.2°C
Propanol__33_2 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 33.2 deg.C 2024-Nov-08 16_02_05.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__33_2__freq = Propanol__33_2.iloc[:,0]
Propanol__33_2__EpsR = Propanol__33_2.iloc[:,3]
Propanol__33_2__EpsI = Propanol__33_2.iloc[:,4]
Propanol__33_2__Sigma = Propanol__33_2.iloc[:,5]
Propanol__33_2__TanD = Propanol__33_2.iloc[:,6]

# Propanol 36.5°C
Propanol__36_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 36.5 deg.C 2024-Nov-08 16_04_49.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__36_5__freq = Propanol__36_5.iloc[:,0]
Propanol__36_5__EpsR = Propanol__36_5.iloc[:,3]
Propanol__36_5__EpsI = Propanol__36_5.iloc[:,4]
Propanol__36_5__Sigma = Propanol__36_5.iloc[:,5]
Propanol__36_5__TanD = Propanol__36_5.iloc[:,6]

# Propanol 41.5°C
Propanol__41_5 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 41.5 deg.C 2024-Nov-08 16_08_33.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__41_5__freq = Propanol__41_5.iloc[:,0]
Propanol__41_5__EpsR = Propanol__41_5.iloc[:,3]
Propanol__41_5__EpsI = Propanol__41_5.iloc[:,4]
Propanol__41_5__Sigma = Propanol__41_5.iloc[:,5]
Propanol__41_5__TanD = Propanol__41_5.iloc[:,6]

# Propanol 44.3°C
Propanol__44_3 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 44.3 deg.C 2024-Nov-08 16_11_17.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__44_3__freq = Propanol__44_3.iloc[:,0]
Propanol__44_3__EpsR = Propanol__44_3.iloc[:,3]
Propanol__44_3__EpsI = Propanol__44_3.iloc[:,4]
Propanol__44_3__Sigma = Propanol__44_3.iloc[:,5]
Propanol__44_3__TanD = Propanol__44_3.iloc[:,6]

# Propanol 48°C
Propanol__48 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 48 deg.C 2024-Nov-08 16_14_27.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__48__freq = Propanol__48.iloc[:,0]
Propanol__48__EpsR = Propanol__48.iloc[:,3]
Propanol__48__EpsI = Propanol__48.iloc[:,4]
Propanol__48__Sigma = Propanol__48.iloc[:,5]
Propanol__48__TanD = Propanol__48.iloc[:,6]

# Propanol 50°C
Propanol__50 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 50 deg.C 2024-Nov-08 16_17_45.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__50__freq = Propanol__50.iloc[:,0]
Propanol__50__EpsR = Propanol__50.iloc[:,3]
Propanol__50__EpsI = Propanol__50.iloc[:,4]
Propanol__50__Sigma = Propanol__50.iloc[:,5]
Propanol__50__TanD = Propanol__50.iloc[:,6]


# All propanol at different temperature in the same array
Temperatures = [-25,-18,-10,-6,-1,1,4,11.2,13.5,15.5,16.5,19.5,21.5,23.3,28.5,33.2,36.5,41.5,44.3,48,50]
Propanol_freq = [Propanol__n25__freq,Propanol__n18__freq,Propanol__n10__freq,Propanol__n6__freq,Propanol__n1__freq,Propanol__1__freq,Propanol__4__freq,Propanol__11_2__freq,Propanol__13_5__freq,Propanol__15_5__freq,Propanol__16_5__freq,Propanol__19_5__freq,Propanol__21_5__freq,Propanol__23_3__freq,Propanol__28_5__freq,Propanol__33_2__freq,Propanol__36_5__freq,Propanol__41_5__freq,Propanol__44_3__freq,Propanol__48__freq,Propanol__50__freq]
Propanol_EpsR = [Propanol__n25__EpsR,Propanol__n18__EpsR,Propanol__n10__EpsR,Propanol__n6__EpsR,Propanol__n1__EpsR,Propanol__1__EpsR,Propanol__4__EpsR,Propanol__11_2__EpsR,Propanol__13_5__EpsR,Propanol__15_5__EpsR,Propanol__16_5__EpsR,Propanol__19_5__EpsR,Propanol__21_5__EpsR,Propanol__23_3__EpsR,Propanol__28_5__EpsR,Propanol__33_2__EpsR,Propanol__36_5__EpsR,Propanol__41_5__EpsR,Propanol__44_3__EpsR,Propanol__48__EpsR,Propanol__50__EpsR]
Propanol_EpsI = [Propanol__n25__EpsI,Propanol__n18__EpsI,Propanol__n10__EpsI,Propanol__n6__EpsI,Propanol__n1__EpsI,Propanol__1__EpsI,Propanol__4__EpsI,Propanol__11_2__EpsI,Propanol__13_5__EpsI,Propanol__15_5__EpsI,Propanol__16_5__EpsI,Propanol__19_5__EpsI,Propanol__21_5__EpsI,Propanol__23_3__EpsI,Propanol__28_5__EpsI,Propanol__33_2__EpsI,Propanol__36_5__EpsI,Propanol__41_5__EpsI,Propanol__44_3__EpsI,Propanol__48__EpsI,Propanol__50__EpsI]
Propanol_Sigma = [Propanol__n25__Sigma,Propanol__n18__Sigma,Propanol__n10__Sigma,Propanol__n6__Sigma,Propanol__n1__Sigma,Propanol__1__Sigma,Propanol__4__Sigma,Propanol__11_2__Sigma,Propanol__13_5__Sigma,Propanol__15_5__Sigma,Propanol__16_5__Sigma,Propanol__19_5__Sigma,Propanol__21_5__Sigma,Propanol__23_3__Sigma,Propanol__28_5__Sigma,Propanol__33_2__Sigma,Propanol__36_5__Sigma,Propanol__41_5__Sigma,Propanol__44_3__Sigma,Propanol__48__Sigma,Propanol__50__Sigma]
Propanol_TanD = [Propanol__n25__TanD,Propanol__n18__TanD,Propanol__n10__TanD,Propanol__n6__TanD,Propanol__n1__TanD,Propanol__1__TanD,Propanol__4__TanD,Propanol__11_2__TanD,Propanol__13_5__TanD,Propanol__15_5__TanD,Propanol__16_5__TanD,Propanol__19_5__TanD,Propanol__21_5__TanD,Propanol__23_3__TanD,Propanol__28_5__TanD,Propanol__33_2__TanD,Propanol__36_5__TanD,Propanol__41_5__TanD,Propanol__44_3__TanD,Propanol__48__TanD,Propanol__50__TanD]

# All propanol at different temperature in the same array
Temperatures_pos = [11.2,13.5,15.5,16.5,19.5,21.5,23.3,28.5,33.2,36.5,41.5,44.3,48,50]
Propanol_freq_pos = [Propanol__11_2__freq,Propanol__13_5__freq,Propanol__15_5__freq,Propanol__16_5__freq,Propanol__19_5__freq,Propanol__21_5__freq,Propanol__23_3__freq,Propanol__28_5__freq,Propanol__33_2__freq,Propanol__36_5__freq,Propanol__41_5__freq,Propanol__44_3__freq,Propanol__48__freq,Propanol__50__freq]
Propanol_EpsR_pos = [Propanol__11_2__EpsR,Propanol__13_5__EpsR,Propanol__15_5__EpsR,Propanol__16_5__EpsR,Propanol__19_5__EpsR,Propanol__21_5__EpsR,Propanol__23_3__EpsR,Propanol__28_5__EpsR,Propanol__33_2__EpsR,Propanol__36_5__EpsR,Propanol__41_5__EpsR,Propanol__44_3__EpsR,Propanol__48__EpsR,Propanol__50__EpsR]
Propanol_EpsI_pos = [Propanol__11_2__EpsI,Propanol__13_5__EpsI,Propanol__15_5__EpsI,Propanol__16_5__EpsI,Propanol__19_5__EpsI,Propanol__21_5__EpsI,Propanol__23_3__EpsI,Propanol__28_5__EpsI,Propanol__33_2__EpsI,Propanol__36_5__EpsI,Propanol__41_5__EpsI,Propanol__44_3__EpsI,Propanol__48__EpsI,Propanol__50__EpsI]
Propanol_Sigma_pos = [Propanol__11_2__Sigma,Propanol__13_5__Sigma,Propanol__15_5__Sigma,Propanol__16_5__Sigma,Propanol__19_5__Sigma,Propanol__21_5__Sigma,Propanol__23_3__Sigma,Propanol__28_5__Sigma,Propanol__33_2__Sigma,Propanol__36_5__Sigma,Propanol__41_5__Sigma,Propanol__44_3__Sigma,Propanol__48__Sigma,Propanol__50__Sigma]
Propanol_TanD_pos = [Propanol__11_2__TanD,Propanol__13_5__TanD,Propanol__15_5__TanD,Propanol__16_5__TanD,Propanol__19_5__TanD,Propanol__21_5__TanD,Propanol__23_3__TanD,Propanol__28_5__TanD,Propanol__33_2__TanD,Propanol__36_5__TanD,Propanol__41_5__TanD,Propanol__44_3__TanD,Propanol__48__TanD,Propanol__50__TanD]

# All propanol at different temperature in the same array
Temperatures_neg = [-25,-18,-10,-6,-1,1,4]
Propanol_freq_neg = [Propanol__n25__freq,Propanol__n18__freq,Propanol__n10__freq,Propanol__n6__freq,Propanol__n1__freq,Propanol__1__freq,Propanol__4__freq]
Propanol_EpsR_neg = [Propanol__n25__EpsR,Propanol__n18__EpsR,Propanol__n10__EpsR,Propanol__n6__EpsR,Propanol__n1__EpsR,Propanol__1__EpsR,Propanol__4__EpsR]
Propanol_EpsI_neg = [Propanol__n25__EpsI,Propanol__n18__EpsI,Propanol__n10__EpsI,Propanol__n6__EpsI,Propanol__n1__EpsI,Propanol__1__EpsI,Propanol__4__EpsI]
Propanol_Sigma_neg = [Propanol__n25__Sigma,Propanol__n18__Sigma,Propanol__n10__Sigma,Propanol__n6__Sigma,Propanol__n1__Sigma,Propanol__1__Sigma,Propanol__4__Sigma]
Propanol_TanD_neg = [Propanol__n25__TanD,Propanol__n18__TanD,Propanol__n10__TanD,Propanol__n6__TanD,Propanol__n1__TanD,Propanol__1__TanD,Propanol__4__TanD]

#####################
# ------Plots------ #
#####################

#define a linear function for the curve fit 
def linear(x,a,b):
    return a*x + b

def epsR(f, eps_inf, eps_s, tao) : 
    return eps_inf + (eps_s - eps_inf)/(1 + (2 * np.pi * f*tao)**2)

def epsI(f, eps_diff, tao) : 
    return eps_diff*2 * np.pi * f*tao/(1 + (2 * np.pi * f*tao)**2)

def cole_cole(eps_R, R, eps_R_0) :
    return np.sqrt(R**2 - (eps_R-eps_R_0)**2)

maxfreqs = []

# for i in range(len(Temperatures)):
#     freq = Propanol_freq[i].to_numpy()
#     EpsI = Propanol_EpsI[i].to_numpy()

#     maxfreq = freq[np.argmax(EpsI)]
#     maxfreqs.append(maxfreq)

for T in Temperatures:
    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[15, 1], bounds=([14, 0], [30, np.inf]))
    eps_diff, tao = popt

    maxfreq = freq[np.argmax(epsI(freq, eps_diff, tao))]
    maxfreqs.append(maxfreq)

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,8),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsI, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_freq_vs_EpsI.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax,fig = u.create_figure_and_apply_format((8,8),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Propanol_EpsR[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsR, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_freq_vs_EpsR.pdf")

#freq vs EpsR and EpsI with fit

taos_R = []
taos_I = []

for T in Temperatures:

    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Propanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    popt_R, pcov_R = curve_fit(epsR, freq, EpsR, p0=[3, 19, 1], bounds=([2, 18, 0], [7, 30, np.inf]))
    eps_inf_R, eps_s_R, tao_R = popt_R
    f_R = np.linspace(50, 3000, 1000)
    y_R = epsR(f_R, eps_inf_R, eps_s_R, tao_R)
    taos_R.append(tao_R)

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[10, 1], bounds=([9, 0], [21, np.inf]))
    eps_diff_I, tao_I = popt
    f_I = np.linspace(50, 3000, 1000)
    y_I = epsI(f_I, eps_diff_I, tao_I)
    taos_I.append(tao_I)

    # Plot EpsR and EpsI on the same graph
    xlabel = "Frequency [MHz]"
    ylabel = r"$\epsilon_r$"
    ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(freq, EpsR, label=f"{T}°C $\epsilon_r'$ Data", marker='x', s=10)
    ax.plot(f_R, y_R, label=f"{T}°C $\epsilon_r'$ Fit", color = 'red', linestyle="--")
    ax.scatter(freq, EpsI, label=f"{T}°C $\epsilon_r''$ Data", marker='x', s=10)
    ax.plot(f_I, y_I, label=f"{T}°C $\epsilon_r''$ Fit", color = 'black', linestyle="--")
    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Propanol_freq_vs_Eps_{T}.pdf")

    # Plot Cole-Cole

    xlabel = r"$\epsilon_r^{'}$"
    ylabel = r"$\epsilon_r^{''}$"
    ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(EpsR, EpsI, label=r"$\frac{\epsilon_r''}{\epsilon_r'}$", marker='x', s=10)
    ax.plot(epsR(f_R, eps_inf_R, eps_s_R, tao_R), epsI(f_R, eps_diff_I, tao_I),color = 'black', label=r"Fit $\frac{\epsilon_r''}{\epsilon_r'}$", linestyle='--')
    
    eps_r_max = eps_diff_I / 2

    x_max = (eps_s_R + eps_inf_R) / 2

    ax.plot([x_max, x_max], [0, eps_r_max], color='red', linestyle='--', label=r'$\frac{\epsilon_s - \epsilon_\infty}{2} = $' + f"{eps_r_max:.2f}")

    ax.set_xlim(0, 30)
    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Propanol_Cole_Cole_{T}.pdf")

# Plot tao vs T
xlabel = "Temperature [°C]"
ylabel = r"$\tau$ [s]"
ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
ax.scatter(Temperatures, taos_R, label=r"$\tau$ with $\epsilon_r'$", marker='x', s=10)
# ax.scatter(Temperatures, taos_I, label=r"$\tau$ with $\epsilon_r''$", marker='x', s=10)
u.set_legend_properties(ax, fontsize=20)
plt.tight_layout()
fig.savefig(f"TP_Micro-ondes/Figures/Propanol_tao_vs_T.pdf")

#Energy of activation

#we have the relation : ln(w) = ln(w0) - E_a/(k*T), with w 2pi times the max frequencies. w0 is a constant. We are looking for E_a by plotting ln(w) = f(1/T), and making a fit to find the slope, which is -E_a/k. ln(w0) is the intercept.    

#blotzmann k constantt
k = 1.38064852e-23

dt = 1

T = np.array(Temperatures) + 273.15
w = 2*np.pi*np.array(maxfreqs)
dts = dt*1/(np.array(T)**2)

T_pos = np.array(Temperatures_pos) + 273.15
w_pos = 2*np.pi*np.array(maxfreqs[7:])
dts_pos = dt*1/(np.array(T_pos)**2)

T_neg = np.array(Temperatures_neg) + 273.15
w_neg = 2*np.pi*np.array(maxfreqs[:7])
dts_neg = dt*1/(np.array(T_neg)**2)

#T

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T,np.log(w),xerr=dts,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T, np.log(w), )
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3e-3,4.2e-3)
ax.set_ylim(6,9.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_Energy_of_activation.pdf")

# T_pos

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T_pos,np.log(w_pos),xerr=dts_pos,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T_pos, np.log(w_pos))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3e-3,3.6e-3)
ax.set_ylim(6,9.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_Energy_of_activation_pos.pdf")

# T_neg

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T_neg,np.log(w_neg),xerr=dts_neg,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T_neg, np.log(w_neg))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3.5e-3,4.2e-3)
ax.set_ylim(6,9.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_Energy_of_activation_neg.pdf")
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

# Propanol 33.7°C
Propanol__33_7 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 33.7 deg.C 2024-Oct-18 10_22_20.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__33_7__freq = Propanol__33_7.iloc[:,0]
Propanol__33_7__EpsR = Propanol__33_7.iloc[:,3]
Propanol__33_7__EpsI = Propanol__33_7.iloc[:,4]
Propanol__33_7__Sigma = Propanol__33_7.iloc[:,5]
Propanol__33_7__TanD = Propanol__33_7.iloc[:,6]

# Propanol 34°C
Propanol__34 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 34 deg.C 2024-Oct-18 10_23_20.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__34__freq = Propanol__34.iloc[:,0]
Propanol__34__EpsR = Propanol__34.iloc[:,3]
Propanol__34__EpsI = Propanol__34.iloc[:,4]
Propanol__34__Sigma = Propanol__34.iloc[:,5]
Propanol__34__TanD = Propanol__34.iloc[:,6]

# Propanol 40.6°C
Propanol__40_6 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 40.6 deg.C 2024-Oct-18 10_29_31.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__40_6__freq = Propanol__40_6.iloc[:,0]
Propanol__40_6__EpsR = Propanol__40_6.iloc[:,3]
Propanol__40_6__EpsI = Propanol__40_6.iloc[:,4]
Propanol__40_6__Sigma = Propanol__40_6.iloc[:,5]
Propanol__40_6__TanD = Propanol__40_6.iloc[:,6]

# Propanol 44.4°C
Propanol__44_4 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 44.4 deg.C 2024-Oct-18 10_38_26.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__44_4__freq = Propanol__44_4.iloc[:,0]
Propanol__44_4__EpsR = Propanol__44_4.iloc[:,3]
Propanol__44_4__EpsI = Propanol__44_4.iloc[:,4]
Propanol__44_4__Sigma = Propanol__44_4.iloc[:,5]
Propanol__44_4__TanD = Propanol__44_4.iloc[:,6]

# Propanol 48.7°C
Propanol__48_7 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 48.7 deg.C 2024-Oct-18 10_41_53.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__48_7__freq = Propanol__48_7.iloc[:,0]
Propanol__48_7__EpsR = Propanol__48_7.iloc[:,3]
Propanol__48_7__EpsI = Propanol__48_7.iloc[:,4]
Propanol__48_7__Sigma = Propanol__48_7.iloc[:,5]
Propanol__48_7__TanD = Propanol__48_7.iloc[:,6]

# Propanol 50°C
Propanol__50 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 50 deg.C 2024-Oct-18 10_50_05.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__50__freq = Propanol__50.iloc[:,0]
Propanol__50__EpsR = Propanol__50.iloc[:,3]
Propanol__50__EpsI = Propanol__50.iloc[:,4]
Propanol__50__Sigma = Propanol__50.iloc[:,5]
Propanol__50__TanD = Propanol__50.iloc[:,6]


# All propanol at different temperature in the same array
Temperatures = [-25,-18,-10,-6,-1,1,4,11.2,13.5,15.5,16.5,19.5,21.5,23.3,33.7,34,40.6,44.4,48.7,50]
Propanol_freq = [Propanol__n25__freq,Propanol__n18__freq,Propanol__n10__freq,Propanol__n6__freq,Propanol__n1__freq,Propanol__1__freq,Propanol__4__freq,Propanol__11_2__freq,Propanol__13_5__freq,Propanol__15_5__freq,Propanol__16_5__freq,Propanol__19_5__freq,Propanol__21_5__freq,Propanol__23_3__freq,Propanol__33_7__freq,Propanol__34__freq,Propanol__40_6__freq,Propanol__44_4__freq,Propanol__48_7__freq,Propanol__50__freq]
Propanol_EpsR = [Propanol__n25__EpsR,Propanol__n18__EpsR,Propanol__n10__EpsR,Propanol__n6__EpsR,Propanol__n1__EpsR,Propanol__1__EpsR,Propanol__4__EpsR,Propanol__11_2__EpsR,Propanol__13_5__EpsR,Propanol__15_5__EpsR,Propanol__16_5__EpsR,Propanol__19_5__EpsR,Propanol__21_5__EpsR,Propanol__23_3__EpsR,Propanol__33_7__EpsR,Propanol__34__EpsR,Propanol__40_6__EpsR,Propanol__44_4__EpsR,Propanol__48_7__EpsR,Propanol__50__EpsR]
Propanol_EpsI = [Propanol__n25__EpsI,Propanol__n18__EpsI,Propanol__n10__EpsI,Propanol__n6__EpsI,Propanol__n1__EpsI,Propanol__1__EpsI,Propanol__4__EpsI,Propanol__11_2__EpsI,Propanol__13_5__EpsI,Propanol__15_5__EpsI,Propanol__16_5__EpsI,Propanol__19_5__EpsI,Propanol__21_5__EpsI,Propanol__23_3__EpsI,Propanol__33_7__EpsI,Propanol__34__EpsI,Propanol__40_6__EpsI,Propanol__44_4__EpsI,Propanol__48_7__EpsI,Propanol__50__EpsI]
Propanol_Sigma = [Propanol__n25__Sigma,Propanol__n18__Sigma,Propanol__n10__Sigma,Propanol__n6__Sigma,Propanol__n1__Sigma,Propanol__1__Sigma,Propanol__4__Sigma,Propanol__11_2__Sigma,Propanol__13_5__Sigma,Propanol__15_5__Sigma,Propanol__16_5__Sigma,Propanol__19_5__Sigma,Propanol__21_5__Sigma,Propanol__23_3__Sigma,Propanol__33_7__Sigma,Propanol__34__Sigma,Propanol__40_6__Sigma,Propanol__44_4__Sigma,Propanol__48_7__Sigma,Propanol__50__Sigma]
Propanol_TanD = [Propanol__n25__TanD,Propanol__n18__TanD,Propanol__n10__TanD,Propanol__n6__TanD,Propanol__n1__TanD,Propanol__1__TanD,Propanol__4__TanD,Propanol__11_2__TanD,Propanol__13_5__TanD,Propanol__15_5__TanD,Propanol__16_5__TanD,Propanol__19_5__TanD,Propanol__21_5__TanD,Propanol__23_3__TanD,Propanol__33_7__TanD,Propanol__34__TanD,Propanol__40_6__TanD,Propanol__44_4__TanD,Propanol__48_7__TanD,Propanol__50__TanD]

# All propanol at different temperature in the same array
Temperatures1 = [-25,-18,-10,-6,-1,1,4]
Propanol_freq1 = [Propanol__n25__freq,Propanol__n18__freq,Propanol__n10__freq,Propanol__n6__freq,Propanol__n1__freq,Propanol__1__freq,Propanol__4__freq]
Propanol_EpsR1 = [Propanol__n25__EpsR,Propanol__n18__EpsR,Propanol__n10__EpsR,Propanol__n6__EpsR,Propanol__n1__EpsR,Propanol__1__EpsR,Propanol__4__EpsR]
Propanol_EpsI1 = [Propanol__n25__EpsI,Propanol__n18__EpsI,Propanol__n10__EpsI,Propanol__n6__EpsI,Propanol__n1__EpsI,Propanol__1__EpsI,Propanol__4__EpsI]
Propanol_Sigma1 = [Propanol__n25__Sigma,Propanol__n18__Sigma,Propanol__n10__Sigma,Propanol__n6__Sigma,Propanol__n1__Sigma,Propanol__1__Sigma,Propanol__4__Sigma]
Propanol_TanD1 = [Propanol__n25__TanD,Propanol__n18__TanD,Propanol__n10__TanD,Propanol__n6__TanD,Propanol__n1__TanD,Propanol__1__TanD,Propanol__4__TanD]

# All propanol at different temperature in the same array
Temperatures2 = [11.2,13.5,15.5,16.5,19.5,21.5]
Propanol_freq2 = [Propanol__11_2__freq,Propanol__13_5__freq,Propanol__15_5__freq,Propanol__16_5__freq,Propanol__19_5__freq,Propanol__21_5__freq]
Propanol_EpsR2 = [Propanol__11_2__EpsR,Propanol__13_5__EpsR,Propanol__15_5__EpsR,Propanol__16_5__EpsR,Propanol__19_5__EpsR,Propanol__21_5__EpsR]
Propanol_EpsI2 = [Propanol__11_2__EpsI,Propanol__13_5__EpsI,Propanol__15_5__EpsI,Propanol__16_5__EpsI,Propanol__19_5__EpsI,Propanol__21_5__EpsI]
Propanol_Sigma2 = [Propanol__11_2__Sigma,Propanol__13_5__Sigma,Propanol__15_5__Sigma,Propanol__16_5__Sigma,Propanol__19_5__Sigma,Propanol__21_5__Sigma]
Propanol_TanD2 = [Propanol__11_2__TanD,Propanol__13_5__TanD,Propanol__15_5__TanD,Propanol__16_5__TanD,Propanol__19_5__TanD,Propanol__21_5__TanD]

# All propanol at different temperature in the same array
Temperatures3 = [23.3,33.7,34,40.6,44.4,48.7,50]
Propanol_freq3 = [Propanol__23_3__freq,Propanol__33_7__freq,Propanol__34__freq,Propanol__40_6__freq,Propanol__44_4__freq,Propanol__48_7__freq,Propanol__50__freq]
Propanol_EpsR3 = [Propanol__23_3__EpsR,Propanol__33_7__EpsR,Propanol__34__EpsR,Propanol__40_6__EpsR,Propanol__44_4__EpsR,Propanol__48_7__EpsR,Propanol__50__EpsR]
Propanol_EpsI3 = [Propanol__23_3__EpsI,Propanol__33_7__EpsI,Propanol__34__EpsI,Propanol__40_6__EpsI,Propanol__44_4__EpsI,Propanol__48_7__EpsI,Propanol__50__EpsI]
Propanol_Sigma3 = [Propanol__23_3__Sigma,Propanol__33_7__Sigma,Propanol__34__Sigma,Propanol__40_6__Sigma,Propanol__44_4__Sigma,Propanol__48_7__Sigma,Propanol__50__Sigma]
Propanol_TanD3 = [Propanol__23_3__TanD,Propanol__33_7__TanD,Propanol__34__TanD,Propanol__40_6__TanD,Propanol__44_4__TanD,Propanol__48_7__TanD,Propanol__50__TanD]

#####################
# ------Plots------ #
#####################

#define a linear function for the curve fit 
def linear(x,a,b):
    return a*x + b

def epsR(f, eps_inf, eps_s, tao) : 
    return eps_inf + (eps_s - eps_inf)/(1 + (2 * np.pi * f*tao)**2)

def epsI(f, eps_inf, eps_s, tao) : 
    return (eps_s - eps_inf)*2 * np.pi * f*tao/(1 + (2 * np.pi * f*tao)**2)

maxfreqs = []

# for i in range(len(Temperatures)):
#     freq = Propanol_freq[i].to_numpy()
#     EpsI = Propanol_EpsI[i].to_numpy()

#     maxfreq = freq[np.argmax(EpsI)]
#     maxfreqs.append(maxfreq)

for T in Temperatures:
    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[1, 1, 1])
    eps_inf, eps_s, tao = popt

    maxfreq = freq[np.argmax(epsI(freq, eps_inf, eps_s, tao))]
    maxfreqs.append(maxfreq)

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

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
for T in Temperatures:

    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Propanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    popt_R, pcov_R = curve_fit(epsR, freq, EpsR, p0=[1, 1, 1])
    eps_inf_R, eps_s_R, tao_R = popt_R
    f_R = np.linspace(50, 3000, 1000)
    y_R = epsR(f_R, eps_inf_R, eps_s_R, tao_R)

    popt_I, pcov_I = curve_fit(epsI, freq, EpsI, p0=[1, 1, 1])
    eps_inf_I, eps_s_I, tao_I = popt_I
    f_I = np.linspace(50, 3000, 1000)
    y_I = epsI(f_I, eps_inf_I, eps_s_I, tao_I)

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

#Energy of activation

#we have the relation : ln(w) = ln(w0) - E_a/(k*T), with w 2pi times the max frequencies. w0 is a constant. We are looking for E_a by plotting ln(w) = f(1/T), and making a fit to find the slope, which is -E_a/k. ln(w0) is the intercept.    

#blotzmann k constantt
k = 1.38064852e-23

dt = 1

T = np.array(Temperatures) + 273.15
w = 2*np.pi*np.array(maxfreqs)
dts = dt*1/(np.array(T)**2)

T1 = np.array(Temperatures1) + 273.15
w1 = 2*np.pi*np.array(maxfreqs[:7])
dts1 = dt*1/(np.array(T1)**2)

T2 = np.array(Temperatures2) + 273.15
w2 = 2*np.pi*np.array(maxfreqs[7:13])
dts2 = dt*1/(np.array(T2)**2)

T3 = np.array(Temperatures3) + 273.15
w3 = 2*np.pi*np.array(maxfreqs[13:])
dts3 = dt*1/(np.array(T3)**2)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T,np.log(w),xerr=dts,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T, np.log(w))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#Ea in eV
Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3e-3,4.2e-3)
ax.set_ylim(6,9.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_Energy_of_activation.pdf")

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T1,np.log(w1),xerr=dts1,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T1, np.log(w1))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#Ea in eV
Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3.5e-3,4.10e-3)
ax.set_ylim(6.25,7.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_1_Energy_of_activation.pdf")

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T2,np.log(w2),xerr=dts2,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T2, np.log(w2))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#Ea in eV
Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3.35e-3,3.55e-3)
ax.set_ylim(7.4,8)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_2_Energy_of_activation.pdf")

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T3,np.log(w3),xerr=dts3,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T3, np.log(w3))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#Ea in eV
Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

ax.set_xlim(3.05e-3,3.4e-3)
ax.set_ylim(7.5, 9.5)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Propanol_3_Energy_of_activation.pdf")

# Cole Cole plot

for T in Temperatures: 

    freq = Propanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Propanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Propanol_EpsI[Temperatures.index(T)].to_numpy()

    xlabel = r"$\epsilon_r^{'}$"
    ylabel = r"$\epsilon_r^{''}$"
    ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(EpsR, EpsI, label=f"{T}°C", marker='x', s=10)
    popt, _ = curve_fit(epsR, freq, EpsR, p0=[1, 1, 1])
    eps_inf, eps_s, tao = popt
    eps_r_max = (eps_s - eps_inf) / 2

    x_max = (eps_s + eps_inf) / 2

    ax.plot([x_max, x_max], [0, eps_r_max], color='red', linestyle='--', label=r'$\frac{\epsilon_s - \epsilon_\infty}{2} = $' + f"{x_max:.2f}")

    ax.set_xlim(0, 30)
    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Propanol_Cole_Cole_{T}.pdf")
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

# --- Ethanol --- #

# -47.5
Ethanol_n47_5 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -47.6 deg.C 2024-Nov-15 11_49_06.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n47_5_freq = Ethanol_n47_5.iloc[:,0]
Ethanol_n47_5_EpsR = Ethanol_n47_5.iloc[:,3]
Ethanol_n47_5_EpsI = Ethanol_n47_5.iloc[:,4]
Ethanol_n47_5_Sigma = Ethanol_n47_5.iloc[:,5]
Ethanol_n47_5_TanD = Ethanol_n47_5.iloc[:,6]

# -44.4
Ethanol_n44_4 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -44.4 deg.C 2024-Nov-15 11_41_38.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n44_4_freq = Ethanol_n44_4.iloc[:,0]
Ethanol_n44_4_EpsR = Ethanol_n44_4.iloc[:,3]
Ethanol_n44_4_EpsI = Ethanol_n44_4.iloc[:,4]
Ethanol_n44_4_Sigma = Ethanol_n44_4.iloc[:,5]
Ethanol_n44_4_TanD = Ethanol_n44_4.iloc[:,6]

# -39
Ethanol_n39 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -39 deg.C 2024-Nov-15 11_33_19.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n39_freq = Ethanol_n39.iloc[:,0]
Ethanol_n39_EpsR = Ethanol_n39.iloc[:,3]
Ethanol_n39_EpsI = Ethanol_n39.iloc[:,4]
Ethanol_n39_Sigma = Ethanol_n39.iloc[:,5]
Ethanol_n39_TanD = Ethanol_n39.iloc[:,6]

# -36.2
Ethanol_n36_2 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -36.2 deg.C 2024-Nov-15 11_29_35.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n36_2_freq = Ethanol_n36_2.iloc[:,0]
Ethanol_n36_2_EpsR = Ethanol_n36_2.iloc[:,3]
Ethanol_n36_2_EpsI = Ethanol_n36_2.iloc[:,4]
Ethanol_n36_2_Sigma = Ethanol_n36_2.iloc[:,5]
Ethanol_n36_2_TanD = Ethanol_n36_2.iloc[:,6]

# -31.8
Ethanol_n31_8 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -31.8 deg.C 2024-Nov-15 11_25_51.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n31_8_freq = Ethanol_n31_8.iloc[:,0]
Ethanol_n31_8_EpsR = Ethanol_n31_8.iloc[:,3]
Ethanol_n31_8_EpsI = Ethanol_n31_8.iloc[:,4]
Ethanol_n31_8_Sigma = Ethanol_n31_8.iloc[:,5]
Ethanol_n31_8_TanD = Ethanol_n31_8.iloc[:,6]

# -25.4
Ethanol_n25_4 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -25.4 deg.C 2024-Nov-15 11_21_23.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n25_4_freq = Ethanol_n25_4.iloc[:,0]
Ethanol_n25_4_EpsR = Ethanol_n25_4.iloc[:,3]
Ethanol_n25_4_EpsI = Ethanol_n25_4.iloc[:,4]
Ethanol_n25_4_Sigma = Ethanol_n25_4.iloc[:,5]
Ethanol_n25_4_TanD = Ethanol_n25_4.iloc[:,6]

# -22.4
Ethanol_n22_4 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -22.4 deg.C 2024-Nov-15 11_19_14.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n22_4_freq = Ethanol_n22_4.iloc[:,0]
Ethanol_n22_4_EpsR = Ethanol_n22_4.iloc[:,3]
Ethanol_n22_4_EpsI = Ethanol_n22_4.iloc[:,4]
Ethanol_n22_4_Sigma = Ethanol_n22_4.iloc[:,5]
Ethanol_n22_4_TanD = Ethanol_n22_4.iloc[:,6]

# -16.1
Ethanol_n16_1 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -16.1 deg.C 2024-Nov-15 11_15_47.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n16_1_freq = Ethanol_n16_1.iloc[:,0]
Ethanol_n16_1_EpsR = Ethanol_n16_1.iloc[:,3]
Ethanol_n16_1_EpsI = Ethanol_n16_1.iloc[:,4]
Ethanol_n16_1_Sigma = Ethanol_n16_1.iloc[:,5]
Ethanol_n16_1_TanD = Ethanol_n16_1.iloc[:,6]

# -13.2
Ethanol_n13_2 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -13.2 deg.C 2024-Nov-15 11_13_37.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n13_2_freq = Ethanol_n13_2.iloc[:,0]
Ethanol_n13_2_EpsR = Ethanol_n13_2.iloc[:,3]
Ethanol_n13_2_EpsI = Ethanol_n13_2.iloc[:,4]
Ethanol_n13_2_Sigma = Ethanol_n13_2.iloc[:,5]
Ethanol_n13_2_TanD = Ethanol_n13_2.iloc[:,6]

# -9.2
Ethanol_n9_2 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -9.2 deg.C 2024-Nov-15 11_11_11.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_n9_2_freq = Ethanol_n9_2.iloc[:,0]
Ethanol_n9_2_EpsR = Ethanol_n9_2.iloc[:,3]
Ethanol_n9_2_EpsI = Ethanol_n9_2.iloc[:,4]
Ethanol_n9_2_Sigma = Ethanol_n9_2.iloc[:,5]
Ethanol_n9_2_TanD = Ethanol_n9_2.iloc[:,6]

# 3.5
Ethanol_3_5 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 3.5 deg.C 2024-Nov-01 11_43_48.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_3_5_freq = Ethanol_3_5.iloc[:,0]
Ethanol_3_5_EpsR = Ethanol_3_5.iloc[:,3]
Ethanol_3_5_EpsI = Ethanol_3_5.iloc[:,4]
Ethanol_3_5_Sigma = Ethanol_3_5.iloc[:,5]
Ethanol_3_5_TanD = Ethanol_3_5.iloc[:,6]

# 6.2
Ethanol_6_2 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 6.2 deg.C 2024-Nov-01 11_41_04.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_6_2_freq = Ethanol_6_2.iloc[:,0]
Ethanol_6_2_EpsR = Ethanol_6_2.iloc[:,3]
Ethanol_6_2_EpsI = Ethanol_6_2.iloc[:,4]
Ethanol_6_2_Sigma = Ethanol_6_2.iloc[:,5]
Ethanol_6_2_TanD = Ethanol_6_2.iloc[:,6]

# 13.6
Ethanol_13_6 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 13.6 deg.C 2024-Nov-01 11_35_28.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_13_6_freq = Ethanol_13_6.iloc[:,0]
Ethanol_13_6_EpsR = Ethanol_13_6.iloc[:,3]
Ethanol_13_6_EpsI = Ethanol_13_6.iloc[:,4]
Ethanol_13_6_Sigma = Ethanol_13_6.iloc[:,5]
Ethanol_13_6_TanD = Ethanol_13_6.iloc[:,6]

# 21.3
Ethanol_21_3 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 21.3 deg.C 2024-Nov-15 09_43_29.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_21_3_freq = Ethanol_21_3.iloc[:,0]
Ethanol_21_3_EpsR = Ethanol_21_3.iloc[:,3]
Ethanol_21_3_EpsI = Ethanol_21_3.iloc[:,4]
Ethanol_21_3_Sigma = Ethanol_21_3.iloc[:,5]
Ethanol_21_3_TanD = Ethanol_21_3.iloc[:,6]

# 25.1
Ethanol_25_1 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 25.1 deg.C 2024-Nov-15 09_46_56.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_25_1_freq = Ethanol_25_1.iloc[:,0]
Ethanol_25_1_EpsR = Ethanol_25_1.iloc[:,3]
Ethanol_25_1_EpsI = Ethanol_25_1.iloc[:,4]
Ethanol_25_1_Sigma = Ethanol_25_1.iloc[:,5]
Ethanol_25_1_TanD = Ethanol_25_1.iloc[:,6]

# 30.6
Ethanol_30_6 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 30.6 deg.C 2024-Nov-08 10_28_06.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_30_6_freq = Ethanol_30_6.iloc[:,0]
Ethanol_30_6_EpsR = Ethanol_30_6.iloc[:,3]
Ethanol_30_6_EpsI = Ethanol_30_6.iloc[:,4]
Ethanol_30_6_Sigma = Ethanol_30_6.iloc[:,5]
Ethanol_30_6_TanD = Ethanol_30_6.iloc[:,6]

# 35.6
Ethanol_35_6 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 35.6 deg.C 2024-Nov-15 09_53_42.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_35_6_freq = Ethanol_35_6.iloc[:,0]
Ethanol_35_6_EpsR = Ethanol_35_6.iloc[:,3]
Ethanol_35_6_EpsI = Ethanol_35_6.iloc[:,4]
Ethanol_35_6_Sigma = Ethanol_35_6.iloc[:,5]
Ethanol_35_6_TanD = Ethanol_35_6.iloc[:,6]

# 40
Ethanol_40 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 40 deg.C 2024-Nov-15 09_57_09.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_40_freq = Ethanol_40.iloc[:,0]
Ethanol_40_EpsR = Ethanol_40.iloc[:,3]
Ethanol_40_EpsI = Ethanol_40.iloc[:,4]
Ethanol_40_Sigma = Ethanol_40.iloc[:,5]
Ethanol_40_TanD = Ethanol_40.iloc[:,6]

# 45.4
Ethanol_45_4 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 45.4 deg.C 2024-Nov-15 10_01_46.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_45_4_freq = Ethanol_45_4.iloc[:,0]
Ethanol_45_4_EpsR = Ethanol_45_4.iloc[:,3]
Ethanol_45_4_EpsI = Ethanol_45_4.iloc[:,4]
Ethanol_45_4_Sigma = Ethanol_45_4.iloc[:,5]
Ethanol_45_4_TanD = Ethanol_45_4.iloc[:,6]

# 52.8
Ethanol_52_8 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 52.8 deg.C 2024-Nov-15 10_08_18.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_52_8_freq = Ethanol_52_8.iloc[:,0]
Ethanol_52_8_EpsR = Ethanol_52_8.iloc[:,3]
Ethanol_52_8_EpsI = Ethanol_52_8.iloc[:,4]
Ethanol_52_8_Sigma = Ethanol_52_8.iloc[:,5]
Ethanol_52_8_TanD = Ethanol_52_8.iloc[:,6]

# 61.5
Ethanol_61_5 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 61.5 deg.C 2024-Nov-15 10_18_06.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol_61_5_freq = Ethanol_61_5.iloc[:,0]
Ethanol_61_5_EpsR = Ethanol_61_5.iloc[:,3]
Ethanol_61_5_EpsI = Ethanol_61_5.iloc[:,4]
Ethanol_61_5_Sigma = Ethanol_61_5.iloc[:,5]
Ethanol_61_5_TanD = Ethanol_61_5.iloc[:,6]

Temperatures = [-47.5, -44.4, -39, -36.2, -31.8, -25.4, -22.4, -16.1, -13.2, -9.2, 3.5, 6.2, 13.6, 21.3, 25.1, 30.6, 35.6, 40, 45.4, 52.8, 61.5]
Ethanol_freq = [Ethanol_n47_5_freq, Ethanol_n44_4_freq, Ethanol_n39_freq, Ethanol_n36_2_freq, Ethanol_n31_8_freq, Ethanol_n25_4_freq, Ethanol_n22_4_freq, Ethanol_n16_1_freq, Ethanol_n13_2_freq, Ethanol_n9_2_freq, Ethanol_3_5_freq, Ethanol_6_2_freq, Ethanol_13_6_freq, Ethanol_21_3_freq, Ethanol_25_1_freq, Ethanol_30_6_freq, Ethanol_35_6_freq, Ethanol_40_freq, Ethanol_45_4_freq, Ethanol_52_8_freq, Ethanol_61_5_freq]
Ethanol_EpsR = [Ethanol_n47_5_EpsR, Ethanol_n44_4_EpsR, Ethanol_n39_EpsR, Ethanol_n36_2_EpsR, Ethanol_n31_8_EpsR, Ethanol_n25_4_EpsR, Ethanol_n22_4_EpsR, Ethanol_n16_1_EpsR, Ethanol_n13_2_EpsR, Ethanol_n9_2_EpsR, Ethanol_3_5_EpsR, Ethanol_6_2_EpsR, Ethanol_13_6_EpsR, Ethanol_21_3_EpsR, Ethanol_25_1_EpsR, Ethanol_30_6_EpsR, Ethanol_35_6_EpsR, Ethanol_40_EpsR, Ethanol_45_4_EpsR, Ethanol_52_8_EpsR, Ethanol_61_5_EpsR]
Ethanol_EpsI = [Ethanol_n47_5_EpsI, Ethanol_n44_4_EpsI, Ethanol_n39_EpsI, Ethanol_n36_2_EpsI, Ethanol_n31_8_EpsI, Ethanol_n25_4_EpsI, Ethanol_n22_4_EpsI, Ethanol_n16_1_EpsI, Ethanol_n13_2_EpsI, Ethanol_n9_2_EpsI, Ethanol_3_5_EpsI, Ethanol_6_2_EpsI, Ethanol_13_6_EpsI, Ethanol_21_3_EpsI, Ethanol_25_1_EpsI, Ethanol_30_6_EpsI, Ethanol_35_6_EpsI, Ethanol_40_EpsI, Ethanol_45_4_EpsI, Ethanol_52_8_EpsI, Ethanol_61_5_EpsI]
Ethanol_Sigma = [Ethanol_n47_5_Sigma, Ethanol_n44_4_Sigma, Ethanol_n39_Sigma, Ethanol_n36_2_Sigma, Ethanol_n31_8_Sigma, Ethanol_n25_4_Sigma, Ethanol_n22_4_Sigma, Ethanol_n16_1_Sigma, Ethanol_n13_2_Sigma, Ethanol_n9_2_Sigma, Ethanol_3_5_Sigma, Ethanol_6_2_Sigma, Ethanol_13_6_Sigma, Ethanol_21_3_Sigma, Ethanol_25_1_Sigma, Ethanol_30_6_Sigma, Ethanol_35_6_Sigma, Ethanol_40_Sigma, Ethanol_45_4_Sigma, Ethanol_52_8_Sigma, Ethanol_61_5_Sigma]
Ethanol_TanD = [Ethanol_n47_5_TanD, Ethanol_n44_4_TanD, Ethanol_n39_TanD, Ethanol_n36_2_TanD, Ethanol_n31_8_TanD, Ethanol_n25_4_TanD, Ethanol_n22_4_TanD, Ethanol_n16_1_TanD, Ethanol_n13_2_TanD, Ethanol_n9_2_TanD, Ethanol_3_5_TanD, Ethanol_6_2_TanD, Ethanol_13_6_TanD, Ethanol_21_3_TanD, Ethanol_25_1_TanD, Ethanol_30_6_TanD, Ethanol_35_6_TanD, Ethanol_40_TanD, Ethanol_45_4_TanD, Ethanol_52_8_TanD, Ethanol_61_5_TanD]

#####################
# ------Plots------ #
#####################

def linear(x,a,b):
    return a*x + b

def epsR(f, eps_inf, eps_s, tao) : 
    return eps_inf + (eps_s - eps_inf)/(1 + (2 * np.pi * f*tao)**2)

def epsI(f, eps_diff, tao) : 
    return eps_diff*2 * np.pi * f*tao/(1 + (2 * np.pi * f*tao)**2)

def cole_cole(eps_R, R, eps_R_0) :
    return np.sqrt(R**2 - (eps_R-eps_R_0)**2)

maxfreqs = []

for T in Temperatures:
    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[15, 1], bounds=([14, 0], [40, np.inf]))
    eps_diff, tao = popt

    maxfreq = freq[np.argmax(epsI(freq, eps_diff, tao))]
    maxfreqs.append(maxfreq)

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,8),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsI, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_freq_vs_EpsI.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax,fig = u.create_figure_and_apply_format((8,8),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Ethanol_EpsR[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsR, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_freq_vs_EpsR.pdf")

#freq vs EpsR and EpsI with fit

taos_R = []
taos_I = []

for T in Temperatures:

    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Ethanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()

    popt_R, pcov_R = curve_fit(epsR, freq, EpsR, p0=[3, 19, 1], bounds=([2, 17, 0], [6, 40, np.inf]))
    eps_inf_R, eps_s_R, tao_R = popt_R
    f_R = np.linspace(50, 3000, 1000)
    y_R = epsR(f_R, eps_inf_R, eps_s_R, tao_R)
    taos_R.append(tao_R)

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[17, 1], bounds=([16, 0], [30, np.inf]))
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
    fig.savefig(f"TP_Micro-ondes/Figures/Ethanol_freq_vs_Eps_{T}.pdf")

    # Plot Cole-Cole

    xlabel = r"$\epsilon_r^{'}$"
    ylabel = r"$\epsilon_r^{''}$"
    ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(EpsR, EpsI, label=r"$\frac{\epsilon_r''}{\epsilon_r'}$", marker='x', s=10)
    ax.plot(epsR(f_R, eps_inf_R, eps_s_R, tao_R), epsI(f_R, eps_diff_I, tao_I),color = 'black', label=r"Fit $\frac{\epsilon_r''}{\epsilon_r'}$", linestyle='--')
    
    eps_r_max = eps_diff_I / 2

    x_max = (eps_s_R + eps_inf_R) / 2

    ax.plot([x_max, x_max], [0, eps_r_max], color='red', linestyle='--', label=r'$\frac{\epsilon_s - \epsilon_\infty}{2} = $' + f"{eps_r_max:.2f}")

    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Ethanol_Cole_Cole_{T}.pdf")

# Plot tao vs T
xlabel = "Temperature [°C]"
ylabel = r"$\tau$ [s]"
ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
ax.scatter(Temperatures, taos_R, label=r"$\tau$ with $\epsilon_r'$", marker='x', s=10)
# ax.scatter(Temperatures, taos_I, label=r"$\tau$ with $\epsilon_r''$", marker='x', s=10)
u.set_legend_properties(ax, fontsize=20)
plt.tight_layout()
fig.savefig(f"TP_Micro-ondes/Figures/Ethanol_tao_vs_T.pdf")

#Energy 
k = 1.38064852e-23

dt = 1

T = np.array(Temperatures) + 273.15
w = 2*np.pi*np.array(maxfreqs)
dts = dt*1/(np.array(T)**2)

T_neg = np.array(Temperatures[:10]) + 273.15
w_neg = 2*np.pi*np.array(maxfreqs[:10])
dts_neg = dt*1/(np.array(T_neg)**2)

T_pos = np.array(Temperatures[10:]) + 273.15
w_pos = 2*np.pi*np.array(maxfreqs[10:])
dts_pos = dt*1/(np.array(T_pos)**2)

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

ax.set_xlim(2.9e-3, 4.5e-3)
ax.set_ylim(6,10)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_Energy_of_activation.pdf")

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

ax.set_xlim(2.9e-3, 3.7e-3)
ax.set_ylim(6,10)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_Energy_of_activation_pos.pdf")

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

ax.set_xlim(3.7e-3, 4.5e-3)
ax.set_ylim(6,10)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_Energy_of_activation_neg.pdf")
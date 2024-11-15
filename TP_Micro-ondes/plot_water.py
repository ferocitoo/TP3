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

# 10°C
Water_10 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 10 deg.C 2024-Oct-18 15_42_54.txt", delimiter='\t', decimal = ".",header = 10)

Water_10_freq = Water_10.iloc[:,0]
Water_10_EpsR = Water_10.iloc[:,3]
Water_10_EpsI = Water_10.iloc[:,4]
Water_10_Sigma = Water_10.iloc[:,5]
Water_10_TanD = Water_10.iloc[:,6]

# 11.6°C
Water_11_6 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 11.6 deg.C 2024-Oct-18 15_40_44.txt", delimiter='\t', decimal = ".",header = 10)

Water_11_6_freq = Water_11_6.iloc[:,0]
Water_11_6_EpsR = Water_11_6.iloc[:,3]
Water_11_6_EpsI = Water_11_6.iloc[:,4]
Water_11_6_Sigma = Water_11_6.iloc[:,5]
Water_11_6_TanD = Water_11_6.iloc[:,6]

# 13.3°C
Water_13_3 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 13.3 deg.C 2024-Oct-18 15_39_43.txt", delimiter='\t', decimal = ".",header = 10)

Water_13_3_freq = Water_13_3.iloc[:,0]
Water_13_3_EpsR = Water_13_3.iloc[:,3]
Water_13_3_EpsI = Water_13_3.iloc[:,4]
Water_13_3_Sigma = Water_13_3.iloc[:,5]
Water_13_3_TanD = Water_13_3.iloc[:,6]

# 18.3°C
Water_18_3 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 18.3 deg.C 2024-Oct-18 15_34_21.txt", delimiter='\t', decimal = ".",header = 10)

Water_18_3_freq = Water_18_3.iloc[:,0]
Water_18_3_EpsR = Water_18_3.iloc[:,3]
Water_18_3_EpsI = Water_18_3.iloc[:,4]
Water_18_3_Sigma = Water_18_3.iloc[:,5]
Water_18_3_TanD = Water_18_3.iloc[:,6]

# 20.2°C
Water_20_2 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 20.2 deg.C 2024-Oct-18 15_31_19.txt", delimiter='\t', decimal = ".",header = 10)

Water_20_2_freq = Water_20_2.iloc[:,0]
Water_20_2_EpsR = Water_20_2.iloc[:,3]
Water_20_2_EpsI = Water_20_2.iloc[:,4]
Water_20_2_Sigma = Water_20_2.iloc[:,5]
Water_20_2_TanD = Water_20_2.iloc[:,6]

# 24.5°C
Water_24_5 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 24.5 deg.C 2024-Oct-18 11_49_29.txt", delimiter='\t', decimal = ".",header = 10)

Water_24_5_freq = Water_24_5.iloc[:,0]
Water_24_5_EpsR = Water_24_5.iloc[:,3]
Water_24_5_EpsI = Water_24_5.iloc[:,4]
Water_24_5_Sigma = Water_24_5.iloc[:,5]
Water_24_5_TanD = Water_24_5.iloc[:,6]

# 38.7°C
Water_38_7 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 38.7 deg.C 2024-Oct-18 12_12_48.txt", delimiter='\t', decimal = ".",header = 10)

Water_38_7_freq = Water_38_7.iloc[:,0]
Water_38_7_EpsR = Water_38_7.iloc[:,3]
Water_38_7_EpsI = Water_38_7.iloc[:,4]
Water_38_7_Sigma = Water_38_7.iloc[:,5]
Water_38_7_TanD = Water_38_7.iloc[:,6]

# 43°C
Water_43 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 43 deg.C 2024-Oct-18 12_18_51.txt", delimiter='\t', decimal = ".",header = 10)

Water_43_freq = Water_43.iloc[:,0]
Water_43_EpsR = Water_43.iloc[:,3]
Water_43_EpsI = Water_43.iloc[:,4]
Water_43_Sigma = Water_43.iloc[:,5]
Water_43_TanD = Water_43.iloc[:,6]

# 48°C
Water_48 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 48 deg.C 2024-Oct-18 12_25_11.txt", delimiter='\t', decimal = ".",header = 10)

Water_48_freq = Water_48.iloc[:,0]
Water_48_EpsR = Water_48.iloc[:,3]
Water_48_EpsI = Water_48.iloc[:,4]
Water_48_Sigma = Water_48.iloc[:,5]
Water_48_TanD = Water_48.iloc[:,6]

# 53.5°C
Water_53_5 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 53.5 deg.C 2024-Oct-18 12_40_30.txt", delimiter='\t', decimal = ".",header = 10)

Water_53_5_freq = Water_53_5.iloc[:,0]
Water_53_5_EpsR = Water_53_5.iloc[:,3]
Water_53_5_EpsI = Water_53_5.iloc[:,4]
Water_53_5_Sigma = Water_53_5.iloc[:,5]
Water_53_5_TanD = Water_53_5.iloc[:,6]

# 58°C
Water_58 = pd.read_csv("/workspaces/TP3/TP_Micro-ondes/Datas/water/DAK 12 Water 58 deg.C 2024-Oct-18 12_48_53.txt", delimiter='\t', decimal = ".",header = 10)

Water_58_freq = Water_58.iloc[:,0]
Water_58_EpsR = Water_58.iloc[:,3]
Water_58_EpsI = Water_58.iloc[:,4]
Water_58_Sigma = Water_58.iloc[:,5]
Water_58_TanD = Water_58.iloc[:,6]

Temperatures = [10, 11.6, 13.3, 18.3, 20.2, 24.5, 38.7, 43, 48, 53.5, 58]
Water_freq = [Water_10_freq, Water_11_6_freq, Water_13_3_freq, Water_18_3_freq, Water_20_2_freq, Water_24_5_freq, Water_38_7_freq, Water_43_freq, Water_48_freq, Water_53_5_freq, Water_58_freq]
Water_EpsR = [Water_10_EpsR, Water_11_6_EpsR, Water_13_3_EpsR, Water_18_3_EpsR, Water_20_2_EpsR, Water_24_5_EpsR, Water_38_7_EpsR, Water_43_EpsR, Water_48_EpsR, Water_53_5_EpsR, Water_58_EpsR]
Water_EpsI = [Water_10_EpsI, Water_11_6_EpsI, Water_13_3_EpsI, Water_18_3_EpsI, Water_20_2_EpsI, Water_24_5_EpsI, Water_38_7_EpsI, Water_43_EpsI, Water_48_EpsI, Water_53_5_EpsI, Water_58_EpsI]
Water_Sigma = [Water_10_Sigma, Water_11_6_Sigma, Water_13_3_Sigma, Water_18_3_Sigma, Water_20_2_Sigma, Water_24_5_Sigma, Water_38_7_Sigma, Water_43_Sigma, Water_48_Sigma, Water_53_5_Sigma, Water_58_Sigma]
Water_TanD = [Water_10_TanD, Water_11_6_TanD, Water_13_3_TanD, Water_18_3_TanD, Water_20_2_TanD, Water_24_5_TanD, Water_38_7_TanD, Water_43_TanD, Water_48_TanD, Water_53_5_TanD, Water_58_TanD]

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

for T in Temperatures:
    freq = Water_freq[Temperatures.index(T)].to_numpy()
    EpsI = Water_EpsI[Temperatures.index(T)].to_numpy()

    popt, _ = curve_fit(epsI, freq, EpsI, p0=[15, 1], bounds=([14, 0], [30, np.inf]))
    eps_diff, tao = popt

    maxfreq = freq[np.argmax(epsI(freq, eps_diff, tao))]
    maxfreqs.append(maxfreq)


#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Water_freq[Temperatures.index(T)].to_numpy()
    EpsI = Water_EpsI[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsI, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Water_freq_vs_EpsI.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

for T in Temperatures:
    
    freq = Water_freq[Temperatures.index(T)].to_numpy()
    EpsR = Water_EpsR[Temperatures.index(T)].to_numpy()

    ax.scatter(freq, EpsR, label=f"{T}°C", marker = 'x',s = 5)
    
u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Water_freq_vs_EpsR.pdf")

# Energy 

k = 1.38064852e-23

dt = 1

T = np.array(Temperatures) + 273.15
w = 2*np.pi*np.array(maxfreqs)
dts = dt*1/(np.array(T)**2)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

ax.errorbar(1/T,np.log(w),xerr=dts,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

popt, pcov = curve_fit(linear, 1/T, np.log(w))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

Ea = -slope*k/1.60217662e-19
dslope = dslope*k/1.60217662e-19
print(f"Ea = {Ea} +/- {dslope} ")

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Water_Energy_of_activation.pdf")
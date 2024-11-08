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



#Ethanol -6.3°C
Ethanol__n6_3 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -6.3 deg.C 2024-Nov-01 15_04_19.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n6_3__freq = Ethanol__n6_3.iloc[:,0]
Ethanol__n6_3__EpsR = Ethanol__n6_3.iloc[:,3]
Ethanol__n6_3__EpsI = Ethanol__n6_3.iloc[:,4]
Ethanol__n6_3__Sigma = Ethanol__n6_3.iloc[:,5]
Ethanol__n6_3__TanD = Ethanol__n6_3.iloc[:,6]

#Ethanol -7.1°C
Ethanol__n7_1 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -7.1 deg.C 2024-Nov-01 15_07_02.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n7_1__freq = Ethanol__n7_1.iloc[:,0]
Ethanol__n7_1__EpsR = Ethanol__n7_1.iloc[:,3]
Ethanol__n7_1__EpsI = Ethanol__n7_1.iloc[:,4]
Ethanol__n7_1__Sigma = Ethanol__n7_1.iloc[:,5]
Ethanol__n7_1__TanD = Ethanol__n7_1.iloc[:,6]

#Ethanol -13°C
Ethanol__n13 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -13 deg.C 2024-Nov-01 15_11_44.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n13__freq = Ethanol__n13.iloc[:,0]
Ethanol__n13__EpsR = Ethanol__n13.iloc[:,3]
Ethanol__n13__EpsI = Ethanol__n13.iloc[:,4]
Ethanol__n13__Sigma = Ethanol__n13.iloc[:,5]
Ethanol__n13__TanD = Ethanol__n13.iloc[:,6]

#Ethanol -14°C
Ethanol__n14 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -14 deg.C 2024-Nov-01 15_15_35.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n14__freq = Ethanol__n14.iloc[:,0]
Ethanol__n14__EpsR = Ethanol__n14.iloc[:,3]
Ethanol__n14__EpsI = Ethanol__n14.iloc[:,4]
Ethanol__n14__Sigma = Ethanol__n14.iloc[:,5]
Ethanol__n14__TanD = Ethanol__n14.iloc[:,6]

#Ethanol -17°C
Ethanol__n17 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -17 deg.C 2024-Nov-01 15_19_35.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n17__freq = Ethanol__n17.iloc[:,0]
Ethanol__n17__EpsR = Ethanol__n17.iloc[:,3]
Ethanol__n17__EpsI = Ethanol__n17.iloc[:,4]
Ethanol__n17__Sigma = Ethanol__n17.iloc[:,5]
Ethanol__n17__TanD = Ethanol__n17.iloc[:,6]

#Ethanol -18.4°C
Ethanol__n18_4 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -18.4 deg.C 2024-Nov-01 15_24_34.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n18_4__freq = Ethanol__n18_4.iloc[:,0]
Ethanol__n18_4__EpsR = Ethanol__n18_4.iloc[:,3]
Ethanol__n18_4__EpsI = Ethanol__n18_4.iloc[:,4]
Ethanol__n18_4__Sigma = Ethanol__n18_4.iloc[:,5]
Ethanol__n18_4__TanD = Ethanol__n18_4.iloc[:,6]

#Ethanol -23°C
Ethanol__n23 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -23 deg.C 2024-Nov-01 15_29_40.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n23__freq = Ethanol__n23.iloc[:,0]
Ethanol__n23__EpsR = Ethanol__n23.iloc[:,3]
Ethanol__n23__EpsI = Ethanol__n23.iloc[:,4]
Ethanol__n23__Sigma = Ethanol__n23.iloc[:,5]
Ethanol__n23__TanD = Ethanol__n23.iloc[:,6]

#Ethanol -25°C
Ethanol__n25 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol -25 deg.C 2024-Nov-01 15_35_30.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__n25__freq = Ethanol__n25.iloc[:,0]
Ethanol__n25__EpsR = Ethanol__n25.iloc[:,3]
Ethanol__n25__EpsI = Ethanol__n25.iloc[:,4]
Ethanol__n25__Sigma = Ethanol__n25.iloc[:,5]
Ethanol__n25__TanD = Ethanol__n25.iloc[:,6]




# All ethanol at different temperature in the same array
# Temperatures = [3.5,6.2,10.5,13.6,21.2]
# Ethanol_freq = [Ethanol__3_5__freq,Ethanol__6_2__freq,Ethanol__10_5__freq,Ethanol__13_6__freq,Ethanol__21_2__freq]
# Ethanol_EpsR = [Ethanol__3_5__EpsR,Ethanol__6_2__EpsR,Ethanol__10_5__EpsR,Ethanol__13_6__EpsR,Ethanol__21_2__EpsR]
# Ethanol_EpsI = [Ethanol__3_5__EpsI,Ethanol__6_2__EpsI,Ethanol__10_5__EpsI,Ethanol__13_6__EpsI,Ethanol__21_2__EpsI]
# Ethanol_Sigma = [Ethanol__3_5__Sigma,Ethanol__6_2__Sigma,Ethanol__10_5__Sigma,Ethanol__13_6__Sigma,Ethanol__21_2__Sigma]
# Ethanol_TanD = [Ethanol__3_5__TanD,Ethanol__6_2__TanD,Ethanol__10_5__TanD,Ethanol__13_6__TanD,Ethanol__21_2__TanD]

Temperatures = [-25,-23,-18.4,-17,-14,-13,-7.1,-6.3,3.5,6.2,10.5,13.6,21.2]
Ethanol_freq = [Ethanol__n25__freq,Ethanol__n23__freq,Ethanol__n18_4__freq,Ethanol__n17__freq,Ethanol__n14__freq,Ethanol__n13__freq,Ethanol__n7_1__freq,Ethanol__n6_3__freq,Ethanol__3_5__freq,Ethanol__6_2__freq,Ethanol__10_5__freq,Ethanol__13_6__freq,Ethanol__21_2__freq]
Ethanol_EpsR = [Ethanol__n25__EpsR,Ethanol__n23__EpsR,Ethanol__n18_4__EpsR,Ethanol__n17__EpsR,Ethanol__n14__EpsR,Ethanol__n13__EpsR,Ethanol__n7_1__EpsR,Ethanol__n6_3__EpsR,Ethanol__3_5__EpsR,Ethanol__6_2__EpsR,Ethanol__10_5__EpsR,Ethanol__13_6__EpsR,Ethanol__21_2__EpsR]
Ethanol_EpsI = [Ethanol__n25__EpsI,Ethanol__n23__EpsI,Ethanol__n18_4__EpsI,Ethanol__n17__EpsI,Ethanol__n14__EpsI,Ethanol__n13__EpsI,Ethanol__n7_1__EpsI,Ethanol__n6_3__EpsI,Ethanol__3_5__EpsI,Ethanol__6_2__EpsI,Ethanol__10_5__EpsI,Ethanol__13_6__EpsI,Ethanol__21_2__EpsI]
Ethanol_Sigma = [Ethanol__n25__Sigma,Ethanol__n23__Sigma,Ethanol__n18_4__Sigma,Ethanol__n17__Sigma,Ethanol__n14__Sigma,Ethanol__n13__Sigma,Ethanol__n7_1__Sigma,Ethanol__n6_3__Sigma,Ethanol__3_5__Sigma,Ethanol__6_2__Sigma,Ethanol__10_5__Sigma,Ethanol__13_6__Sigma,Ethanol__21_2__Sigma]
Ethanol_TanD = [Ethanol__n25__TanD,Ethanol__n23__TanD,Ethanol__n18_4__TanD,Ethanol__n17__TanD,Ethanol__n14__TanD,Ethanol__n13__TanD,Ethanol__n7_1__TanD,Ethanol__n6_3__TanD,Ethanol__3_5__TanD,Ethanol__6_2__TanD,Ethanol__10_5__TanD,Ethanol__13_6__TanD,Ethanol__21_2__TanD]






#####################
# ------Plots------ #
#####################

# --- Ethanol --- #

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

maxfreqs = []

i=0
for T in Temperatures:
    
    freq = Ethanol_freq[i].to_numpy()
    EpsR = Ethanol_EpsR[i].to_numpy()
    EpsI = Ethanol_EpsI[i].to_numpy()
    
    maxfreq = freq[np.argmax(EpsI)]
    maxfreqs.append(maxfreq)
    
    ax.scatter(freq,EpsI,label=f"{T}°C",s=2)

    i+=1
    
u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol__3_5_freq_vs_EpsR_and_EpsI.pdf")


#Energy of activation

#we have the relation : ln(w) = ln(w0) - E_a/(k*T), with w 2pi times the max frequencies. w0 is a constant. We are looking for E_a by plotting ln(w) = f(1/T), and making a fit to find the slope, which is -E_a/k. ln(w0) is the intercept.    

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [K^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

T = np.array(Temperatures) + 273.15
w = 2*np.pi*np.array(maxfreqs)

ax.scatter(1/T,np.log(w),label="Data",marker="+",color = "blue",s=100)

ax.text(3.65e-3,7.95,r"$\times 10^{-3}$",fontsize=13)

slope, intercept = np.polyfit(1/T, np.log(w), 1)
x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#blotzmann k constantt
k = 1.38064852e-23

#Ea in eV
Ea = -slope*k/1.60217662e-19
print(f"Ea = {Ea} ")

# ax.set_xlim(3.3e-3,3.63e-3)
# ax.set_ylim(8,8.8)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol__3_5_Energy_of_activation.pdf")



#################################
#################################
#################################
########TODO######################
#pour le plot d'énergie d'activation, fit sur les températures négatives, et sur les températures positives appart. Car calibration différente
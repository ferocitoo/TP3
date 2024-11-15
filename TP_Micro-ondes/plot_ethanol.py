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



#Ethanol 20.5°C
Ethanol__20_5 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 20.5 deg.C 2024-Nov-08 10_18_18.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__20_5__freq = Ethanol__20_5.iloc[:,0]
Ethanol__20_5__EpsR = Ethanol__20_5.iloc[:,3]
Ethanol__20_5__EpsI = Ethanol__20_5.iloc[:,4]
Ethanol__20_5__Sigma = Ethanol__20_5.iloc[:,5]
Ethanol__20_5__TanD = Ethanol__20_5.iloc[:,6]

#Ethanol 23.7°C
Ethanol__23_7 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 23.7 deg.C 2024-Nov-08 10_22_03.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__23_7__freq = Ethanol__23_7.iloc[:,0]
Ethanol__23_7__EpsR = Ethanol__23_7.iloc[:,3]
Ethanol__23_7__EpsI = Ethanol__23_7.iloc[:,4]
Ethanol__23_7__Sigma = Ethanol__23_7.iloc[:,5]
Ethanol__23_7__TanD = Ethanol__23_7.iloc[:,6]


#Ethanol 26.3°C
Ethanol__26_3 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 26.3 deg.C 2024-Nov-08 10_24_30.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__26_3__freq = Ethanol__26_3.iloc[:,0]
Ethanol__26_3__EpsR = Ethanol__26_3.iloc[:,3]
Ethanol__26_3__EpsI = Ethanol__26_3.iloc[:,4]
Ethanol__26_3__Sigma = Ethanol__26_3.iloc[:,5]
Ethanol__26_3__TanD = Ethanol__26_3.iloc[:,6]


#Ethanol 30.6°C
Ethanol__30_6 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 30.6 deg.C 2024-Nov-08 10_28_06.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__30_6__freq = Ethanol__30_6.iloc[:,0]
Ethanol__30_6__EpsR = Ethanol__30_6.iloc[:,3]
Ethanol__30_6__EpsI = Ethanol__30_6.iloc[:,4]
Ethanol__30_6__Sigma = Ethanol__30_6.iloc[:,5]
Ethanol__30_6__TanD = Ethanol__30_6.iloc[:,6]


#Ethanol 34.0°C
Ethanol__34_0 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 34 deg.C 2024-Nov-08 10_30_41.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__34_0__freq = Ethanol__34_0.iloc[:,0]
Ethanol__34_0__EpsR = Ethanol__34_0.iloc[:,3]
Ethanol__34_0__EpsI = Ethanol__34_0.iloc[:,4]
Ethanol__34_0__Sigma = Ethanol__34_0.iloc[:,5]
Ethanol__34_0__TanD = Ethanol__34_0.iloc[:,6]

#Ethanol 37°C
Ethanol__37 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 37 deg.C 2024-Nov-08 10_33_16.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__37__freq = Ethanol__37.iloc[:,0]
Ethanol__37__EpsR = Ethanol__37.iloc[:,3]
Ethanol__37__EpsI = Ethanol__37.iloc[:,4]
Ethanol__37__Sigma = Ethanol__37.iloc[:,5]
Ethanol__37__TanD = Ethanol__37.iloc[:,6]

#Ethanol 40.5°C
Ethanol__40_5 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 40.5 deg.C 2024-Nov-08 10_36_35.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__40_5__freq = Ethanol__40_5.iloc[:,0]
Ethanol__40_5__EpsR = Ethanol__40_5.iloc[:,3]
Ethanol__40_5__EpsI = Ethanol__40_5.iloc[:,4]
Ethanol__40_5__Sigma = Ethanol__40_5.iloc[:,5]
Ethanol__40_5__TanD = Ethanol__40_5.iloc[:,6]

#Ethanol 44.5°C
Ethanol__44_5 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 44.5 deg.C 2024-Nov-08 10_40_05.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__44_5__freq = Ethanol__44_5.iloc[:,0]
Ethanol__44_5__EpsR = Ethanol__44_5.iloc[:,3]
Ethanol__44_5__EpsI = Ethanol__44_5.iloc[:,4]
Ethanol__44_5__Sigma = Ethanol__44_5.iloc[:,5]
Ethanol__44_5__TanD = Ethanol__44_5.iloc[:,6]

#Ethanol 47.4°C
Ethanol__47_4 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 47.4 deg.C 2024-Nov-08 10_43_06.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__47_4__freq = Ethanol__47_4.iloc[:,0]
Ethanol__47_4__EpsR = Ethanol__47_4.iloc[:,3]
Ethanol__47_4__EpsI = Ethanol__47_4.iloc[:,4]
Ethanol__47_4__Sigma = Ethanol__47_4.iloc[:,5]
Ethanol__47_4__TanD = Ethanol__47_4.iloc[:,6]

#Ethanol 50°C
Ethanol__50 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 50 deg.C 2024-Nov-08 10_46_15.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__50__freq = Ethanol__50.iloc[:,0]
Ethanol__50__EpsR = Ethanol__50.iloc[:,3]
Ethanol__50__EpsI = Ethanol__50.iloc[:,4]
Ethanol__50__Sigma = Ethanol__50.iloc[:,5]
Ethanol__50__TanD = Ethanol__50.iloc[:,6]

#Ethanol 52.4°C
Ethanol__52_4 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 52.4 deg.C 2024-Nov-08 10_49_00.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__52_4__freq = Ethanol__52_4.iloc[:,0]
Ethanol__52_4__EpsR = Ethanol__52_4.iloc[:,3]
Ethanol__52_4__EpsI = Ethanol__52_4.iloc[:,4]
Ethanol__52_4__Sigma = Ethanol__52_4.iloc[:,5]
Ethanol__52_4__TanD = Ethanol__52_4.iloc[:,6]

#Ethanol 56.0°C
Ethanol__56 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 56 deg.C 2024-Nov-08 10_55_11.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__56__freq = Ethanol__56.iloc[:,0]
Ethanol__56__EpsR = Ethanol__56.iloc[:,3]
Ethanol__56__EpsI = Ethanol__56.iloc[:,4]
Ethanol__56__Sigma = Ethanol__56.iloc[:,5]
Ethanol__56__TanD = Ethanol__56.iloc[:,6]



Temperatures = [-25,-23,-18.4,-17,-14,-13,-7.1,-6.3,3.5,6.2,10.5,13.6,20.5,23.7,26.3,30.6,34,37,40.5,44.5,47.4,50,52.4,56]
Ethanol_freq = [Ethanol__n25__freq,Ethanol__n23__freq,Ethanol__n18_4__freq,Ethanol__n17__freq,Ethanol__n14__freq,Ethanol__n13__freq,Ethanol__n7_1__freq,Ethanol__n6_3__freq,Ethanol__3_5__freq,Ethanol__6_2__freq,Ethanol__10_5__freq,Ethanol__13_6__freq,Ethanol__20_5__freq,Ethanol__23_7__freq,Ethanol__26_3__freq,Ethanol__30_6__freq,Ethanol__34_0__freq,Ethanol__37__freq,Ethanol__40_5__freq,Ethanol__44_5__freq,Ethanol__47_4__freq,Ethanol__50__freq,Ethanol__52_4__freq,Ethanol__56__freq]

Ethanol_EpsR = [Ethanol__n25__EpsR,Ethanol__n23__EpsR,Ethanol__n18_4__EpsR,Ethanol__n17__EpsR,Ethanol__n14__EpsR,Ethanol__n13__EpsR,Ethanol__n7_1__EpsR,Ethanol__n6_3__EpsR,Ethanol__3_5__EpsR,Ethanol__6_2__EpsR,Ethanol__10_5__EpsR,Ethanol__13_6__EpsR,Ethanol__20_5__EpsR,Ethanol__23_7__EpsR,Ethanol__26_3__EpsR,Ethanol__30_6__EpsR,Ethanol__34_0__EpsR,Ethanol__37__EpsR,Ethanol__40_5__EpsR,Ethanol__44_5__EpsR,Ethanol__47_4__EpsR,Ethanol__50__EpsR,Ethanol__52_4__EpsR,Ethanol__56__EpsR]

Ethanol_EpsI = [Ethanol__n25__EpsI,Ethanol__n23__EpsI,Ethanol__n18_4__EpsI,Ethanol__n17__EpsI,Ethanol__n14__EpsI,Ethanol__n13__EpsI,Ethanol__n7_1__EpsI,Ethanol__n6_3__EpsI,Ethanol__3_5__EpsI,Ethanol__6_2__EpsI,Ethanol__10_5__EpsI,Ethanol__13_6__EpsI,Ethanol__20_5__EpsI,Ethanol__23_7__EpsI,Ethanol__26_3__EpsI,Ethanol__30_6__EpsI,Ethanol__34_0__EpsI,Ethanol__37__EpsI,Ethanol__40_5__EpsI,Ethanol__44_5__EpsI,Ethanol__47_4__EpsI,Ethanol__50__EpsI,Ethanol__52_4__EpsI,Ethanol__56__EpsI]



#####################
# ------Plots------ #
#####################

# --- Ethanol --- #

# #freq vs EpsR and EpsI
# xlabel = "Frequency [MHz]"
# ylabel = r"$\epsilon_r^{''}$"

# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# maxfreqs = []

# i=0
# for T in Temperatures:
    
#     freq = Ethanol_freq[i].to_numpy()
#     EpsR = Ethanol_EpsR[i].to_numpy()
#     EpsI = Ethanol_EpsI[i].to_numpy()
    
#     maxfreq = freq[np.argmax(EpsI)]
#     maxfreqs.append(maxfreq)
    
#     ax.scatter(freq,EpsI,label=f"{T}°C",s=2)

#     i+=1
    
# u.set_legend_properties(ax,fontsize=25)

# plt.tight_layout()

# fig.savefig("TP_Micro-ondes/Figures/Ethanol__3_5_freq_vs_EpsR_and_EpsI.pdf")




#define a linear function for the curve fit 
def linear(x,a,b):
    return a*x + b

def epsR(f, eps_inf, eps_s, tao) : 
    return eps_inf + (eps_s - eps_inf)/(1 + (2 * np.pi * f*tao)**2)

def epsI(f, eps_inf, eps_s, tao) : 
    return (eps_s - eps_inf)*2 * np.pi * f*tao/(1 + (2 * np.pi * f*tao)**2)

def cole_cole(eps_R, R, eps_R_0) :
    return np.sqrt(R**2 - (eps_R-eps_R_0)**2)



eps_s_array = []
deps_s_array = []
eps_inf_array = []
deps_inf_array = []



# Cole Cole plot
for T in Temperatures: 

    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Ethanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()

    xlabel = r"$\epsilon_r^{'}$"
    ylabel = r"$\epsilon_r^{''}$"
    ax, fig = u.create_figure_and_apply_format((10, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(EpsR, EpsI, label=f"Measurements for T = {T}°C", marker='x', s=10, color='blue')
    
    
    popt, pcov = curve_fit(cole_cole, EpsR, EpsI, p0=[13,17],maxfev=10000000)
    R,eps_R_0 = popt
    dR,deps_R_0 = np.sqrt(np.diag(pcov))
    
    
    eps_s = eps_R_0 - R
    deps_s = dR + deps_R_0
    eps_inf = eps_R_0 + R
    deps_inf = dR + deps_R_0
    
    eps_s_array.append(eps_s)
    deps_s_array.append(deps_s)
    eps_inf_array.append(eps_inf)
    deps_inf_array.append(deps_inf)
    
    ax.vlines(eps_s,0,30,linestyles='--',label=rf'$\epsilon_s = {eps_s:.2f} \pm {deps_s:.2f}$',color = "black")
    ax.vlines(eps_inf,0,30,linestyles='-.',label=rf'$\epsilon_\infty = {eps_inf:.1f} \pm 0.1$' ,color = "black")
    
    x = np.linspace(0, 35, 1000)
    y = cole_cole(x, R, eps_R_0)
    ax.plot(x, y, color='red', linestyle='--', label = r"Fit : $\epsilon_r^{''} = \sqrt{R^2 - (\epsilon_r^' - \epsilon_\infty)^2}$")

    
    ax.set_xlim(0, 35)
    ax.set_ylim(-5,20)
    
    #set the axis as equal scale
    ax.set_aspect('equal')
    
    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Ethanol_Cole_Cole_{T}.pdf")




# for i in range(len(Temperatures)):
#     freq = Propanol_freq[i].to_numpy()
#     EpsI = Propanol_EpsI[i].to_numpy()

#     maxfreq = freq[np.argmax(EpsI)]
#     maxfreqs.append(maxfreq)
maxfreqs = []

taos = []
dtaos = []

for T in Temperatures:
    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()
    
    eps_inf = eps_inf_array[Temperatures.index(T)]
    eps_s = eps_s_array[Temperatures.index(T)]

    popt, pcov = curve_fit(lambda f, tao : epsI(f, eps_inf, eps_s, tao), freq, EpsI, p0=[-0.1],bounds=([-np.inf], [0]))
    tao = popt
    dtao = np.sqrt(np.diag(pcov))
    
    taos.append(tao)
    dtaos.append(dtao)
    
    # print(f"eps_inf = {eps_inf} \pm {deps_inf}")
    # print(f"eps_s = {eps_s} \pm {deps_s}")
    # print(f"tao = {tao} \pm {dtao}")

    maxfreq = freq[np.argmax(epsI(freq, eps_inf, eps_s, tao))]
    maxfreqs.append(maxfreq)

print(taos)

#freq vs EpsR and EpsI
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

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
for T in Temperatures:
    freq = Ethanol_freq[Temperatures.index(T)].to_numpy()
    EpsR = Ethanol_EpsR[Temperatures.index(T)].to_numpy()
    EpsI = Ethanol_EpsI[Temperatures.index(T)].to_numpy()

    eps_s = eps_s_array[Temperatures.index(T)]
    eps_inf = eps_inf_array[Temperatures.index(T)]
    tao = taos[Temperatures.index(T)]

    # popt_R, pcov_R = curve_fit(epsR, freq, EpsR, p0=[1, 1, 1])
    # eps_inf_R, eps_s_R, tao_R = popt_R
    f_R = np.linspace(50, 3000, 1000)
    y_R = epsR(f_R, eps_s, eps_inf, tao)

    # popt_I, pcov_I = curve_fit(epsI, freq, EpsI, p0=[1, 1, 1])
    # eps_inf_I, eps_s_I, tao_I = popt_I
    f_I = np.linspace(50, 3000, 1000)
    y_I = epsI(f_I, eps_inf, eps_s, tao)


    # Plot EpsR and EpsI on the same graph
    xlabel = "Frequency [MHz]"
    ylabel = r"$\epsilon_r$"
    ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
    ax.scatter(freq, EpsR, label=f"{T}°C $\epsilon_r'$ Data", marker='x', s=10, color = 'blue')
    ax.plot(f_R, y_R, label=f"{T}°C $\epsilon_r'$ Fit", color = 'red', linestyle="--")
    ax.scatter(freq, EpsI, label=f"{T}°C $\epsilon_r''$ Data", marker='x', s=10, color = 'orange')
    ax.plot(f_I, y_I, label=f"{T}°C $\epsilon_r''$ Fit", color = 'black', linestyle="--")
    u.set_legend_properties(ax, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"TP_Micro-ondes/Figures/Ethanol_freq_vs_Eps_{T}.pdf")



#Energy of activation

#we have the relation : ln(w) = ln(w0) - E_a/(k*T), with w 2pi times the max frequencies. w0 is a constant. We are looking for E_a by plotting ln(w) = f(1/T), and making a fit to find the slope, which is -E_a/k. ln(w0) is the intercept.  

#positive temperatures

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [mK^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

T = np.array(Temperatures[8:]) + 273.15  
w = 2*np.pi*np.array(maxfreqs[8:])

dT = 1
dTs = dT * 1/T**2

ax.errorbar(1/T,np.log(w),xerr=dTs,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

# ax.scatter(1/T,np.log(w),label="Data",marker="+",color = "blue",s=100)

popt, pcov = curve_fit(linear, 1/T, np.log(w))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))

x = np.linspace(1/400,1/200,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#blotzmann k constant
k = 1.38064852e-23

#Ea in eV
Ea = -slope*k/1.60217662e-19
print(f"Ea = {Ea} \pm {dslope*k/1.60217662e-19} ")

# ax.set_xlim(3.3e-3,3.63e-3)
# ax.set_ylim(8,8.8)

ax.set_xlim(3.25e-3,3.75e-3)
ax.set_ylim(7.9,8.8)

u.set_legend_properties(ax,fontsize=25)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_Energy_of_activation_positive.pdf")



#negative temperatures

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=r"$1/T [mK^{-1}]$", ylabel=r"$\ln(\omega) [a.u.]$")

T = np.array(Temperatures[:8]) + 273.15  
w = 2*np.pi*np.array(maxfreqs[:8])

dT = 1
dTs = dT * 1/T**2

ax.errorbar(1/T,np.log(w),xerr=dTs,label="Data",marker="^",color = "blue",linestyle="None",markersize = 10, capsize = 5, capthick = 1)

# ax.scatter(1/T,np.log(w),label="Data",marker="+",color = "blue",s=100)

popt, pcov = curve_fit(linear, 1/T, np.log(w))
slope, intercept = popt
dslope, dintercept = np.sqrt(np.diag(pcov))

x = np.linspace(1/200,1/400,100)
u.x_axis_divide(ax,1e-3)
y = slope*x + intercept
ax.plot(x,y,label="Fit",color="red",linestyle="--")

#blotzmann k constant
k = 1.38064852e-23

#Ea in eV
Ea = -slope*k/1.60217662e-19
print(f"Ea = {Ea} \pm {dslope*k/1.60217662e-19} ")

u.set_legend_properties(ax,fontsize=25)

ax.set_xlim(3.7e-3,4e-3)
ax.set_ylim(7.5,8.2)

plt.tight_layout()

fig.savefig("TP_Micro-ondes/Figures/Ethanol_Energy_of_activation_negative.pdf")







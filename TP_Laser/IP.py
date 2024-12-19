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

R_int = 19.953
dR_int = 0.004

R_ext = 999
dR_ext = 1

alpha = 1.17*1e-1

dT = 0.2

# Importer les données

# 9.5
T_9_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_9_5.csv", delimiter="\t", decimal=".")
T_9_5_I = T_9_5.iloc[:,0]/R_int
T_9_5_P = T_9_5.iloc[:,1]/(alpha*R_ext)

T_12_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_12_5.csv", delimiter="\t", decimal=".")
T_12_5_I = T_12_5.iloc[:,0]/R_int
T_12_5_P = T_12_5.iloc[:,1]/(alpha*R_ext)

T_15_3 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_15_3.csv", delimiter="\t", decimal=".")
T_15_3_I = T_15_3.iloc[:,0]/R_int
T_15_3_P = T_15_3.iloc[:,1]/(alpha*R_ext)

T_17_9 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_17_9.csv", delimiter="\t", decimal=".")
T_17_9_I = T_17_9.iloc[:,0]/R_int
T_17_9_P = T_17_9.iloc[:,1]/(alpha*R_ext)

T_20_3 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_20_3.csv", delimiter="\t", decimal=".")
T_20_3_I = T_20_3.iloc[:,0]/R_int
T_20_3_P = T_20_3.iloc[:,1]/(alpha*R_ext)

T_22_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_22_5.csv", delimiter="\t", decimal=".")
T_22_5_I = T_22_5.iloc[:,0]/R_int
T_22_5_P = T_22_5.iloc[:,1]/(alpha*R_ext)

T_25_2 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_25_2.csv", delimiter="\t", decimal=".")
T_25_2_I = T_25_2.iloc[:,0]/R_int
T_25_2_P = T_25_2.iloc[:,1]/(alpha*R_ext)

T_27_8 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_27_8.csv", delimiter="\t", decimal=".")
T_27_8_I = T_27_8.iloc[:,0]/R_int
T_27_8_P = T_27_8.iloc[:,1]/(alpha*R_ext)

T_30_2 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_30_2.csv", delimiter="\t", decimal=".")
T_30_2_I = T_30_2.iloc[:,0]/R_int
T_30_2_P = T_30_2.iloc[:,1]/(alpha*R_ext)

T_32_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_32_5.csv", delimiter="\t", decimal=".")
T_32_5_I = T_32_5.iloc[:,0]/R_int
T_32_5_P = T_32_5.iloc[:,1]/(alpha*R_ext)

T_35_3 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_35_3.csv", delimiter="\t", decimal=".")
T_35_3_I = T_35_3.iloc[:,0]/R_int
T_35_3_P = T_35_3.iloc[:,1]/(alpha*R_ext)

T_37_6 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_37_6.csv", delimiter="\t", decimal=".")
T_37_6_I = T_37_6.iloc[:,0]/R_int
T_37_6_P = T_37_6.iloc[:,1]/(alpha*R_ext)

T_40_0 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/New_T/T_40_0.csv", delimiter="\t", decimal=".")
T_40_0_I = T_40_0.iloc[:,0]/R_int
T_40_0_P = T_40_0.iloc[:,1]/(alpha*R_ext)

Ts = [9.5, 12.5, 15.3, 17.9, 20.3, 22.5, 25.2, 27.8, 30.2, 32.5, 35.3, 37.6, 40.0]
T_Is = [T_9_5_I, T_12_5_I, T_15_3_I, T_17_9_I, T_20_3_I, T_22_5_I, T_25_2_I, T_27_8_I, T_30_2_I, T_32_5_I, T_35_3_I, T_37_6_I, T_40_0_I]
T_Ps = [T_9_5_P, T_12_5_P, T_15_3_P, T_17_9_P, T_20_3_P, T_22_5_P, T_25_2_P, T_27_8_P, T_30_2_P, T_32_5_P, T_35_3_P, T_37_6_P, T_40_0_P]

T_K = [T + 273.15 for T in Ts]

dIs = [0.01/R_int - (T_Is[i]/R_int**2)*0.004 for i in range(len(Ts))]
dPs = [0.01/(alpha*R_ext) - (T_Ps[i]/(alpha*R_ext)**2)*1 for i in range(len(Ts))] 

# Plot

def linear_fit(x, a, b):
    return a * x + b

def find_intersection(popt_1):
    a1, b1 = popt_1
    x_intersect = (b1) / (-a1)
    y_intersect = a1 * x_intersect + b1
    return x_intersect, y_intersect

# Plot de toutes les courbes

xlabel = r"$I$ (A)"
ylabel = r"$P$ (W)"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

for T in Ts:
    i = Ts.index(T)
    ax.plot(T_Is[i], T_Ps[i], label=f"T = {T} °C")

u.y_axis_divide(ax, 0.0001)
ax.annotate("1e$-4$", xy=(0.0, 1.03), xycoords='axes fraction', fontsize=10, ha='left', va='top')

u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Laser/Figures/IP_T_New.pdf")

# Plot chacune avec les fits

I_th = []
dI_ths = []
dI_ths_log = []

T = Ts[0]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[70:80], P[70:80])
P_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

ax, fig = u.create_figure_and_apply_format((8, 6), xlabel=xlabel, ylabel=ylabel)
ax.plot(I, P, label=f"T = {T} °C")
ax.hlines(0, min(I), max(I), color='r', linestyle='--', label='P = 0')
ax.plot(I, P_fit, color = 'g', linestyle = '--', label='fit')
ax.plot(intersection[0], intersection[1], 'bx', label=r'$I_{th}$')

u.y_axis_divide(ax, 0.0001)
ax.annotate("1e$-4$", xy=(0.0, 1.03), xycoords='axes fraction', fontsize=10, ha='left', va='top')

ax.set_ylim(min(P) - 2*1e-5, max(P) + 2*1e-5)

u.set_legend_properties(ax, fontsize=20)
plt.tight_layout()
fig.savefig(f"TP_Laser/Figures/IP_T_{T}_fit.pdf")

T = Ts[1]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[80:90], P[80:90])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[2]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[70:80], P[70:80])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[3]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[100:110], P[100:110])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[4]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[90:100], P[90:100])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[5]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[120:130], P[120:130])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[6]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[110:120], P[110:120])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[7]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[100:110], P[100:110])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[8]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[80:90], P[80:90])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

T = Ts[9]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[100:110], P[100:110])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))
                  
T = Ts[10]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[110:120], P[110:120])
P_2_fit = linear_fit(I, *popt_2)

a, b = popt_2
da, db = np.sqrt(np.diag(pcov2))

dI_th = np.abs((b/(a**2)))*da + np.abs((1/-a))*db

intersection = find_intersection(popt_2)
I_th.append(intersection[0])
dI_ths.append(dI_th)
dI_ths_log.append(np.abs((dI_th/intersection[0])))

# Plot de log I_th en fonction de T

xlabel = r"$T$ (K)"
ylabel = r"$\log(I_{th})$"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

popt, pcov = curve_fit(linear_fit, T_K[:len(I_th)], np.log(I_th))
a, b = popt
da, db = np.sqrt(np.diag(pcov))

T_fit = np.linspace(min(T_K[:len(I_th)]), max(T_K[:len(I_th)]), 100)
I_th_log_fit = linear_fit(T_fit, a, b)

ax.plot(T_fit, I_th_log_fit, color='red', linestyle='--', label=rf"Fit: $\log(I_{{th}}) = {a:.2f} T + ({b:.2f})$")
ax.errorbar(T_K[:len(I_th)], np.log(I_th), xerr=dT, yerr=dI_ths_log, marker='x', linestyle='', label=r"$\log(I_{th})$")

u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Laser/Figures/I_th_T_New.pdf")

print (rf"T_0 = {1/a} +- {da/a**2} K")
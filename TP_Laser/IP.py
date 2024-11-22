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

# 9.4
T_9_4 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_9_4.csv", delimiter="\t", decimal=".")
T_9_4_I = T_9_4.iloc[:,0]/R_int
T_9_4_P = T_9_4.iloc[:,1]/(alpha*R_ext)

# 12.5
T_12_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_12_5.csv", delimiter="\t", decimal=".")
T_12_5_I = T_12_5.iloc[:,0]/R_int
T_12_5_P = T_12_5.iloc[:,1]/(alpha*R_ext)

# 15.2
T_15_2 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_15_2.csv", delimiter="\t", decimal=".")
T_15_2_I = T_15_2.iloc[:,0]/R_int
T_15_2_P = T_15_2.iloc[:,1]/(alpha*R_ext)

# 17.7
T_17_7 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_17_7.csv", delimiter="\t", decimal=".")
T_17_7_I = T_17_7.iloc[:,0]/R_int
T_17_7_P = T_17_7.iloc[:,1]/(alpha*R_ext)

# 20.2
T_20_2 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_20_2.csv", delimiter="\t", decimal=".")
T_20_2_I = T_20_2.iloc[:,0]/R_int
T_20_2_P = T_20_2.iloc[:,1]/(alpha*R_ext)

# 22.5
T_22_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_22_5.csv", delimiter="\t", decimal=".")
T_22_5_I = T_22_5.iloc[:,0]/R_int
T_22_5_P = T_22_5.iloc[:,1]/(alpha*R_ext)

# 25.6
T_25_6 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_25_6.csv", delimiter="\t", decimal=".")
T_25_6_I = T_25_6.iloc[:,0]/R_int
T_25_6_P = T_25_6.iloc[:,1]/(alpha*R_ext)

# 27.8
T_27_8 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_27_8.csv", delimiter="\t", decimal=".")
T_27_8_I = T_27_8.iloc[:,0]/R_int
T_27_8_P = T_27_8.iloc[:,1]/(alpha*R_ext)

# 30
T_30 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_30_0.csv", delimiter="\t", decimal=".")
T_30_I = T_30.iloc[:,0]/R_int
T_30_P = T_30.iloc[:,1]/(alpha*R_ext)

# 32.5
T_32_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_32_5.csv", delimiter="\t", decimal=".")
T_32_5_I = T_32_5.iloc[:,0]/R_int
T_32_5_P = T_32_5.iloc[:,1]/(alpha*R_ext)

# 35
T_35 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_35_0.csv", delimiter="\t", decimal=".")
T_35_I = T_35.iloc[:,0]/R_int
T_35_P = T_35.iloc[:,1]/(alpha*R_ext)

# 37.5
T_37_5 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_37_5.csv", delimiter="\t", decimal=".")
T_37_5_I = T_37_5.iloc[:,0]/R_int
T_37_5_P = T_37_5.iloc[:,1]/(alpha*R_ext)

# 40.1
T_40_1 = pd.read_csv("/workspaces/TP3/TP_Laser/Datas/T_40_1.csv", delimiter="\t", decimal=".")
T_40_1_I = T_40_1.iloc[:,0]/R_int
T_40_1_P = T_40_1.iloc[:,1]/(alpha*R_ext)

Ts = [9.4, 12.5, 15.2, 17.7, 20.2, 22.5, 25.6, 27.8, 30, 32.5, 35, 37.5, 40.1]
T_Is = [T_9_4_I, T_12_5_I, T_15_2_I, T_17_7_I, T_20_2_I, T_22_5_I, T_25_6_I, T_27_8_I, T_30_I, T_32_5_I, T_35_I, T_37_5_I, T_40_1_I]
T_Ps = [T_9_4_P, T_12_5_P, T_15_2_P, T_17_7_P, T_20_2_P, T_22_5_P, T_25_6_P, T_27_8_P, T_30_P, T_32_5_P, T_35_P, T_37_5_P, T_40_1_P]

T_K = [T + 273.15 for T in Ts]

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

fig.savefig("TP_Laser/Figures/IP_T.pdf")

# Plot chacune avec les fits

I_th = []
dI_ths = []
dI_ths_log = []

T = Ts[0]
i = Ts.index(T)
I = T_Is[i]
P = T_Ps[i]

popt_2, pcov2 = curve_fit(linear_fit, I[130:150], P[130:150])
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

popt_2, pcov2 = curve_fit(linear_fit, I[150:170], P[150:170])
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

popt_2, pcov2 = curve_fit(linear_fit, I[140:150], P[140:150])
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

popt_2, pcov2 = curve_fit(linear_fit, I[190:200], P[190:200])
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

popt_2, pcov2 = curve_fit(linear_fit, I[170:180], P[170:180])
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

popt_2, pcov2 = curve_fit(linear_fit, I[220:230], P[220:230])
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

popt_2, pcov2 = curve_fit(linear_fit, I[240:250], P[240:250])
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

popt_2, pcov2 = curve_fit(linear_fit, I[280:290], P[280:290])
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

fig.savefig("TP_Laser/Figures/I_th_T.pdf")
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os

module_name = "utils_v2"
file_path = "/workspaces/TP-Chaos/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd




#Import the data
#----config1-----#
C1 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_1.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C1_t=C1.iloc[:,0]
C1_theta=C1.iloc[:,1]
C1_thetadot=C1.iloc[:,2]

#----config2-----#
#3 périodes mieux
C2_1 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_2_1.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C2_1_t=C2_1.iloc[:,0]
C2_1_theta=C2_1.iloc[:,1]
C2_1_thetadot=C2_1.iloc[:,2]

#3 pértiodes inversé
C2_2 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_2_2.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C2_2_t=C2_2.iloc[:,0]
C2_2_theta=C2_2.iloc[:,1]
C2_2_thetadot=C2_2.iloc[:,2]

#1 période
C2_3 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_2_3.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C2_3_t=C2_3.iloc[:,0]
C2_3_theta=C2_3.iloc[:,1]
C2_3_thetadot=C2_3.iloc[:,2]

#----config3-----#
C3_1 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_3_1.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C3_1_t=C3_1.iloc[:,0]
C3_1_theta=C3_1.iloc[:,1]
C3_1_thetadot=C3_1.iloc[:,2]

C3_2 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_3_2.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C3_2_t=C3_2.iloc[:,0]
C3_2_theta=C3_2.iloc[:,1]
C3_2_thetadot=C3_2.iloc[:,2]

C3_3 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_3_3.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C3_3_t=C3_3.iloc[:,0]
C3_3_theta=C3_3.iloc[:,1]
C3_3_thetadot=C3_3.iloc[:,2]

#----config4-----#
#1 période
C4 = pd.read_csv("TP_Chaos/Datas_Moteur/Config_4.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",")  # Adjust delimiter if necessary
C4_t=C4.iloc[:,0]
C4_theta=C4.iloc[:,1]
C4_thetadot=C4.iloc[:,2]



#-----------config 1------------#

#----plot----#
t = C1_t
theta = C1_theta
thetadot = C1_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(theta,thetadot,color="black",linestyle=':', linewidth=0.5)
ax.plot(theta,thetadot,color="black")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_phase_space.pdf")

#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

ax.plot(positive_freqs, positive_fft, color="black")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_spectral_analysis.pdf")



